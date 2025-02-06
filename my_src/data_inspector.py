import scanpy as sc
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import os
import sys
import scipy.sparse  # Added for sparsity calculation
from .my_utils import check_annotation_keys

class DataInspector:
    """Interactive data inspection and preparation for scGPT embedding"""
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.metadata_info = None
        self.gene_col = None
        self.cell_type_col = None
        self.hvg_col = None
        self.adata_sample = None
        
    def inspect_sample(self, sample_size: int = 1000) -> Dict:
        """Load and inspect a data sample using a careful, exploratory approach"""
        print(f"Loading and analyzing sample of {sample_size} cells...")
        
        # Let's start by thoroughly examining the file structure
        with h5py.File(self.file_path, 'r') as f:
            print("\nAnalyzing file structure...")
            
            # First, let's see what groups are at the root level
            root_groups = list(f.keys())
            print("\nRoot level groups:", root_groups)
            
            # Helper function to explore groups recursively
            def explore_group(group, prefix=''):
                """Recursively explore and print group structure"""
                for key in group.keys():
                    item = group[key]
                    if isinstance(item, h5py.Dataset):
                        print(f"{prefix}Dataset: {key}")
                        print(f"{prefix}  Shape: {item.shape}")
                        print(f"{prefix}  Type: {item.dtype}")
                    elif isinstance(item, h5py.Group):
                        print(f"{prefix}Group: {key}")
                        explore_group(item, prefix + '  ')
            
            print("\nDetailed structure:")
            explore_group(f)
            
            # Now let's try to find our data dimensions
            n_cells = None
            n_genes = None
            
            # Check various possible locations for dimensions
            if 'X' in f:
                if isinstance(f['X'], h5py.Dataset):
                    print("\nFound X as a dataset")
                    n_cells, n_genes = f['X'].shape
                elif isinstance(f['X'], h5py.Group):
                    print("\nFound X as a group")
                    if 'shape' in f['X'].attrs:
                        n_cells, n_genes = f['X'].attrs['shape']
                    elif 'indptr' in f['X']:
                        print("Found sparse matrix structure")
                        n_cells = len(f['X']['indptr']) - 1
                        if 'shape' in f['X']:
                            n_genes = f['X']['shape'][1]
            
            # Try alternative sources if we still don't have dimensions
            if n_cells is None and 'obs' in f:
                for key in f['obs'].keys():
                    if isinstance(f['obs'][key], h5py.Dataset):
                        n_cells = len(f['obs'][key])
                        print(f"\nInferred number of cells from obs/{key}: {n_cells}")
                        break
            
            if n_genes is None and 'var' in f:
                for key in f['var'].keys():
                    if isinstance(f['var'][key], h5py.Dataset):
                        n_genes = len(f['var'][key])
                        print(f"Inferred number of genes from var/{key}: {n_genes}")
                        break
            
            print("\nFile exploration complete")
            print(f"Found dimensions: {n_cells} cells × {n_genes} genes")
            
            # Now we can try loading our sample using scanpy
            try:
                print("\nAttempting to load data with scanpy...")
                adata = sc.read_h5ad(
                    self.file_path,
                    backed='r',
                    chunk_size=sample_size
                )
                
                # Take our sample
                self.adata_sample = adata[:sample_size].copy()
                print(f"Successfully loaded sample of {sample_size} cells")
                
                # Process the sample
                self.metadata_info = check_annotation_keys(self.adata_sample, verbose=True)
                self._identify_key_columns()
                
                return self.get_summary()
                
            except Exception as e:
                print(f"\nError during loading: {str(e)}")
                print("The file structure might be more complex than expected.")
                print("Would you like to see a detailed analysis of a specific group?")
                return None
                
    def _fallback_load(self, sample_size: int) -> Dict:
        """Alternative loading strategy for when the main approach fails"""
        try:
            print("\nTrying minimal loading approach...")
            # Try loading with absolute minimal features
            adata = sc.read_h5ad(
                self.file_path,
                backed='r',
                chunk_size=sample_size
            )
            
            # Take just what we need
            self.adata_sample = adata[:sample_size].copy()
            print("Successfully loaded minimal sample")
            
            self.metadata_info = check_annotation_keys(self.adata_sample, verbose=True)
            self._identify_key_columns()
            
            return self.get_summary()
        
        except Exception as e:
            raise RuntimeError(f"All loading attempts failed. Last error: {str(e)}")
    def _identify_key_columns(self):
        """Identify key columns in the data, with informative messaging"""
        print("\nIdentifying key columns:")
        
        # Gene column identification
        self.gene_col = self._find_column(
            self.metadata_info['gene_keys'], 
            self.adata_sample.var.columns,
            "gene"
        )
        
        # Cell type column identification
        self.cell_type_col = self._find_column(
            self.metadata_info['cell_type_keys'], 
            self.adata_sample.obs.columns,
            "cell type"
        )
        
        # HVG column identification
        self.hvg_col = self._find_column(
            self.metadata_info['hvg_keys'], 
            self.adata_sample.var.columns,
            "HVG"
        )
    
    def _find_column(self, possible_keys: List[str], available_cols, col_type: str) -> Optional[str]:
        """Helper method to find columns with clear messaging
        
        Args:
            possible_keys: List of possible column names to check
            available_cols: Available columns in the dataset
            col_type: Type of column being searched (for messaging)
            
        Returns:
            Found column name or None
        """
        for key in possible_keys:
            if key in available_cols:
                print(f"✓ Found {col_type} column: {key}")
                return key
        print(f"ℹ No {col_type} column found among {possible_keys}")
        return None
    
    def get_summary(self) -> Dict:
        """Get comprehensive summary of identified columns and data properties"""
        summary = {
            'gene_column': self.gene_col,
            'cell_type_column': self.cell_type_col,
            'hvg_column': self.hvg_col,
            'n_genes': self.adata_sample.n_vars,
            'n_cells_in_sample': self.adata_sample.n_obs,
        }
        
        # Calculate sparsity if the matrix is sparse
        if scipy.sparse.issparse(self.adata_sample.X):
            summary['sparsity'] = 1.0 - (self.adata_sample.X.nnz / 
                                       (self.adata_sample.n_obs * self.adata_sample.n_vars))
        else:
            summary['sparsity'] = None
            print("\nNote: Data matrix is not sparse. Sparsity calculation skipped.")
        
        return summary