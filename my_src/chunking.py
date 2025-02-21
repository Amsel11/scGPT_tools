import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import scanpy as sc
from pathlib import Path
import h5py
from typing import Optional, Union, List
from dataclasses import dataclass
import warnings
from tqdm.auto import tqdm
import gc
import logging
from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing parameters"""
    n_hvgs: int = 2000
    min_cells: int = 3
    min_genes: int = 200
    max_genes: int = 7000
    min_counts: int = 500
    max_counts: int = 30000
    max_mt_percent: float = 20.0
    chunk_size: int = 5000
    n_threads: int = 4

class ScGPTPreprocessor:
    def __init__(self, config: PreprocessingConfig = None):
        self.config = config or PreprocessingConfig()
        self._setup_logging()
        # Initialize storage for HVG and scaling information
        self._hvg_genes = None
        self._scaler = None
    
    def _setup_logging(self):
        """Configure logging with memory tracking"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
    
    def preprocess_file(self, 
                       input_path: Union[str, Path], 
                       output_path: Union[str, Path],
                       metadata_path: Optional[str] = None) -> ad.AnnData:
        """Main preprocessing function"""
        self.logger.info(f"Starting preprocessing of {input_path}")
        
        # Load data in backed mode
        adata = self._load_data(input_path, metadata_path)
        
        # Perform initial QC
        adata = self._quality_control(adata)
        
        # Process data in chunks
        adata = self._process_in_chunks(adata)
        
        # Save processed data
        self._save_data(adata, output_path)
        
        return adata
    
    def _load_data(self, input_path: Union[str, Path], metadata_path: Optional[str]) -> ad.AnnData:
        """Load data with optional external metadata"""
        self.logger.info("Loading data...")
        
        # Use backed mode for large files
        adata = ad.read_h5ad(input_path, backed='r')
        
        # Load additional metadata if provided
        if metadata_path:
            metadata = pd.read_csv(metadata_path, index_col=0)
            adata.obs = adata.obs.join(metadata)
        
        return adata
    
    def _quality_control(self, adata: ad.AnnData) -> ad.AnnData:
        """Perform quality control and filtering"""
        self.logger.info("Performing quality control...")
        
        # Calculate QC metrics
        sc.pp.calculate_qc_metrics(
            adata,
            qc_vars=['MT-', 'mt-'],
            inplace=True,
            percent_top=None,
            log1p=False
        )
        
        # Filter cells
        cell_mask = (
            (adata.obs.n_genes_by_counts >= self.config.min_genes) &
            (adata.obs.n_genes_by_counts <= self.config.max_genes) &
            (adata.obs.total_counts >= self.config.min_counts) &
            (adata.obs.total_counts <= self.config.max_counts) &
            (adata.obs.pct_counts_mt <= self.config.max_mt_percent)
        )
        
        # Filter genes
        gene_mask = np.array((adata.X > 0).sum(axis=0)).flatten() >= self.config.min_cells
        
        # Apply filters
        adata = adata[cell_mask, gene_mask].copy()
        
        self.logger.info(f"Retained {adata.n_obs} cells and {adata.n_vars} genes after QC")
        return adata
    
    def _process_in_chunks(self, adata: ad.AnnData) -> ad.AnnData:
        """Process the AnnData object in chunks to manage memory usage"""
        self.logger.info("Starting chunked processing...")
        
        # Calculate number of chunks
        n_cells = adata.n_obs
        n_chunks = (n_cells + self.config.chunk_size - 1) // self.config.chunk_size
        self.logger.info(f"Processing {n_cells} cells in {n_chunks} chunks")
        
        # Initialize list to store processed chunks
        processed_chunks = []
        
        # Process each chunk
        for chunk_idx in tqdm(range(n_chunks), desc="Processing chunks"):
            # Calculate chunk boundaries
            start_idx = chunk_idx * self.config.chunk_size
            end_idx = min((chunk_idx + 1) * self.config.chunk_size, n_cells)
            
            # Load chunk into memory
            self.logger.info(f"Loading chunk {chunk_idx + 1}/{n_chunks} (cells {start_idx}-{end_idx})")
            chunk = adata[start_idx:end_idx].copy()
            
            try:
                # Process this chunk
                processed_chunk = self._process_single_chunk(chunk)
                processed_chunks.append(processed_chunk)
                
                # Clear memory
                del chunk
                gc.collect()
                
            except Exception as e:
                self.logger.error(f"Error processing chunk {chunk_idx}: {str(e)}")
                raise
        
        # Combine all processed chunks
        self.logger.info("Concatenating processed chunks...")
        try:
            final_adata = ad.concat(
                processed_chunks,
                axis=0,
                join='outer',
                merge='same'
            )
            
            # Add processing info to uns
            final_adata.uns['scgpt'] = {
                'preprocessing_config': vars(self.config),
                'hvg_score': final_adata.var['dispersions_norm'].to_dict() if 'dispersions_norm' in final_adata.var else None,
                'scaling_params': {
                    'mean': self._scaler.mean_.tolist(),
                    'scale': self._scaler.scale_.tolist()
                } if self._scaler is not None else None
            }
            
            # Clear memory
            del processed_chunks
            gc.collect()
            
        except Exception as e:
            self.logger.error(f"Error concatenating chunks: {str(e)}")
            raise
        
        return final_adata
    
    def _process_single_chunk(self, chunk: ad.AnnData) -> ad.AnnData:
        """Process a single chunk of data"""
        # Normalize this chunk
        sc.pp.normalize_total(chunk, target_sum=1e4)
        sc.pp.log1p(chunk)
        
        # If this is the first chunk, calculate HVG scores
        if self._hvg_genes is None:
            sc.pp.highly_variable_genes(
                chunk,
                n_top_genes=self.config.n_hvgs,
                flavor='seurat_v3',
                batch_key=None
            )
            self._hvg_genes = chunk.var.index[chunk.var.highly_variable].copy()
        
        # Keep only HVG genes (same genes for all chunks)
        chunk = chunk[:, self._hvg_genes].copy()
        
        # Scale the data
        if self._scaler is None:
            self._scaler = StandardScaler(with_mean=True, with_std=True)
            chunk.layers['scaled'] = self._scaler.fit_transform(chunk.X.toarray())
        else:
            chunk.layers['scaled'] = self._scaler.transform(chunk.X.toarray())
        
        return chunk
    
    def _save_data(self, adata: ad.AnnData, output_path: Union[str, Path]):
        """Save processed data"""
        self.logger.info(f"Saving processed data to {output_path}")
        adata.write_h5ad(output_path, compression='gzip')

# Example usage
if __name__ == "__main__":
    # Configure preprocessing
    config = PreprocessingConfig(
        n_hvgs=2000,
        chunk_size=5000,
        n_threads=4
    )
    
    # Initialize preprocessor
    preprocessor = ScGPTPreprocessor(config)
    
    # Process data
    adata = preprocessor.preprocess_file(
        input_path=r"C:\Users\annel\OneDrive\Documenten\Machine Learning\scGPT\data\1m_cells.h5ad",
        output_path=r"C:\Users\annel\OneDrive\Documenten\Machine Learning\scGPT\data\processed_for_scgpt.h5ad",
        metadata_path="additional_metadata.csv"  # Optional
    )