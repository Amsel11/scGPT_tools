#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utility functions for scGPT scripts
"""

import json
import os
from pathlib import Path
import pprint
from html import escape
import pandas as pd
import scanpy as sc
import h5py
import scipy.sparse as sparse
from anndata import AnnData
import logging
import sys
from datetime import datetime
import anndata as ad
import torch

def setup_directories(repo_path = None, data_path = None, save_path = None, model_path = None):
    if repo_path is None:
        repo_dir = Path.cwd().absolute()
    else: 
        repo_dir = Path(repo_path)
        
    """Set up necessary directories and return their paths"""
    data_dir = Path(data_path).absolute() if data_path else (repo_dir / "data")
    save_dir = Path(save_path).absolute() if save_path else (repo_dir / "save")
    model_dir = Path(model_path).absolute() if model_path else (repo_dir / "models")
    
    repo_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
        
    directories = {
        "repo_dir": str(repo_dir),
        "data_dir": str(data_dir),
        "save_dir": str(save_dir),
        "model_dir": str(model_dir)
    }
    return repo_dir, data_dir, save_dir, model_dir, directories

def download_cellxgene_data(url, output_dir, file_name=None):
    os.makedirs(output_dir, exist_ok=True)
    if file_name is not None:
        filename = file_name
    else:
        filename = url.split('/')[-1]
    file_path = os.path.join(output_dir, filename)
    if os.path.exists(file_path):
        print(f"File {filename} already exists in {output_dir}")
        return file_path
    
    try:
        print(f"Downloading {filename} from {url} to {output_dir}")
        response = requests.get(url, stream=True, timeout=30)
        
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(file_path, 'wb') as file, tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive chunks
                    file.write(chunk)
                    pbar.update(len(chunk))
        
        print(f"Download completed: {file_path}")
        return file_path
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        if os.path.exists(file_path):
            os.remove(file_path)  # Remove partial download
        return None



def test_embed_config(adata=None, config_path=None, metadata_path=None):
    """
    ONLY test configuration parsing without ANY model loading or embedding
    
    Args:
        adata: Optional AnnData object (only used to check column names)
        config_path: Path to config file
        metadata_path: Path to metadata file
        
    Returns:
        dict: The configuration that would be used
    """
    
    print("🔍 TESTING CONFIGURATION ONLY - NO MODEL WILL BE LOADED")
    
    # Resolve config path
    script_dir = Path(__file__).parent
    config_paths_to_try = [
        Path(config_path) if config_path else None,
        script_dir / "scGPT_embed_config.json",
        Path("scGPT_embed_config.json")
    ]
    config_paths_to_try = [p for p in config_paths_to_try if p]
    
    # Try each path
    config = None
    for path in config_paths_to_try:
        if path.exists():
            print(f"📄 Found config at: {path}")
            with open(path, 'r') as f:
                config = json.load(f)
            break
    
    if not config:
        print(f"❌ Config file not found. Tried: {[str(p) for p in config_paths_to_try]}")
        return None
        
    # Check metadata
    metadata = None
    if metadata_path:
        try:
            metadata_path = Path(metadata_path)
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                print(f"📄 Loaded metadata from: {metadata_path}")
                print("\nMetadata content:")
                pprint.pprint(metadata)
            else:
                print(f"❌ Metadata file not found: {metadata_path}")
        except Exception as e:
            print(f"❌ Error loading metadata: {e}")
    
    # Print config
    print("\nConfig that would be used:")
    pprint.pprint(config)
    
    # If we have adata, check column compatibility
    if adata is not None:
        print("\nChecking data compatibility:")
        gene_col = config["data"]["gene_col"]
        
        print(f"- Available var columns: {list(adata.var.columns)}")
        print(f"- Gene column setting: {gene_col}")
        
        if gene_col in adata.var.columns:
            print(f"✅ '{gene_col}' exists as a column")
        else:
            print(f"❌ '{gene_col}' NOT found in var columns")
    
    return config

def build_config(args, metadata=None):
    """Build a centralized configuration from args and config file"""
    # Start with a copy of args as a dictionary
    config = vars(args).copy()
    
    # If a config file is specified, load and merge it
    if args.config_file and os.path.exists(args.config_file):
        try:
            with open(args.config_file, 'r') as f:
                file_config = json.load(f)
                
            # Update config with file values (except command-line overrides)
            for key, value in file_config.items():
                if key not in config or config[key] is None:
                    config[key] = value
                    print(f"Using config value for {key}: {value}")
        except Exception as e:
            print(f"Error loading config file: {e}")

    # Update essential parameters from metadata if available
    if metadata and not metadata.get('error'):
        # Set gene_col if not already specified
        if not config['gene_col'] and metadata.get('found_keys_by_category', {}).get('gene_keys'):
            gene_keys = metadata['found_keys_by_category']['gene_keys']
            # Try common gene column names first
            priority = ["feature_name", "gene_symbol", "gene_name", "ensembl_id"]
            for key in priority:
                if key in gene_keys:
                    config['gene_col'] = key
                    print(f"Using detected gene column: {key}")
                    break
            
            # Fallback to first available
            if not config['gene_col'] and gene_keys:
                config['gene_col'] = gene_keys[0]
                print(f"Using detected gene column: {config['gene_col']}")
        
        # Set cell_type_col for classification
        if not config.get('cell_type_col') and metadata.get('found_keys_by_category', {}).get('cell_type_keys'):
            config['cell_type_col'] = metadata['found_keys_by_category']['cell_type_keys'][0]
            print(f"Using detected cell type column: {config['cell_type_col']}")
        
        # Set batch_key for batch-aware operations
        if not config.get('batch_key') and metadata.get('found_keys_by_category', {}).get('batch_keys'):
            config['batch_key'] = metadata['found_keys_by_category']['batch_keys'][0]
            print(f"Using detected batch column: {config['batch_key']}")

    return config

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test configuration utilities")
    parser.add_argument('--config', type=str, help='Config file path')
    parser.add_argument('--metadata', type=str, help='Metadata file path')
    args = parser.parse_args()
    
    test_embed_config(config_path=args.config, metadata_path=args.metadata)


def generate_basic_html_report(analysis_text, found_keys, adata, output_path):
    """
    Generate a simple HTML report with the analysis text and metadata.
    No visualizations, just nicely formatted text.
    """
    # Format the analysis text for HTML
    html_analysis = escape(analysis_text).replace('\n', '<br>').replace('  ', '&nbsp;&nbsp;')
    
    # Create basic metadata table
    metadata_table = "<table border='1' cellpadding='5' cellspacing='0'>\n"
    metadata_table += "<tr><th>Category</th><th>Found Keys</th></tr>\n"
    
    for category, keys in found_keys.items():
        category_name = category.replace('_', ' ').title()
        keys_list = ", ".join(keys) if keys else "None found"
        metadata_table += f"<tr><td>{category_name}</td><td>{keys_list}</td></tr>\n"
    
    metadata_table += "</table>"
    
    # Create dataset summary
    dataset_summary = f"<h3>Dataset Summary</h3>"
    dataset_summary += f"<p>Dimensions: {adata.n_obs} cells × {adata.n_vars} genes</p>"
    
    # Create HTML file
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>scGPT Analysis: {Path(output_path).stem}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
            pre {{ background-color: #f5f5f5; padding: 15px; overflow-x: auto; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ text-align: left; padding: 12px; }}
            th {{ background-color: #f2f2f2; }}
            .analysis {{ background-color: #f9f9f9; padding: 20px; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <h1>scGPT Data Analysis Report</h1>
        
        {dataset_summary}
        
        <h2>Found Metadata Keys</h2>
        {metadata_table}
        
        <h2>Detailed Analysis</h2>
        <div class="analysis">
            {html_analysis}
        </div>
    </body>
    </html>
    """
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return output_path


# loading utils functions for loading subset of data
def _load_anndata(path, subset=None):
    """Internal function to load anndata with subsetting"""
    # Load subset using h5py for memory efficiency
    start_row = subset.get('start_row', 0)
    n_rows = subset.get('n_rows', None)
    obs_columns = subset.get('obs_columns', None)
    
    with h5py.File(path, "r") as f:
        # Determine total rows
        total_rows = len(f["X"]["indptr"]) - 1
        if n_rows is None:
            n_rows = total_rows - start_row
            
        print(f"Loading subset: rows {start_row}-{start_row+n_rows} of {total_rows}")

        # Load components
        data, indices, indptr = _load_csr_matrix_components(f, start_row, n_rows)
        var_df = _load_var_metadata(f)
        obs_df = _load_obs_metadata(f, start_row, n_rows, obs_columns)

        # Create sparse matrix
        X_subset = sparse.csr_matrix(
            (data, indices, indptr), shape=(n_rows, len(var_df))
        )
        
        # Create the AnnData object
        adata_subset = AnnData(X=X_subset, obs=obs_df, var=var_df)
        
        # Load obsm data including embeddings
        if "obsm" in f:
            for obsm_key in f["obsm"].keys():
                # Get the data for the selected rows
                if isinstance(f["obsm"][obsm_key], h5py.Dataset):
                    obsm_data = f["obsm"][obsm_key][start_row:start_row+n_rows]
                    adata_subset.obsm[obsm_key] = obsm_data
                else:
                    print(f"Warning: Could not load {obsm_key} from obsm, not a Dataset")

    return adata_subset

def _load_csr_matrix_components(f, start_row, n_rows):
    """Helper function to load CSR matrix components from h5ad file."""
    indptr = f["X"]["indptr"][start_row : start_row + n_rows + 1]
    start_idx, end_idx = indptr[0], indptr[-1]

    data = f["X"]["data"][start_idx:end_idx]
    indices = f["X"]["indices"][start_idx:end_idx]
    indptr = indptr - start_idx  # Adjust indptr to start at 0

    return data, indices, indptr


def _load_var_metadata(f):
    """Helper function to load variable (gene) metadata."""
    var_dict = {}
    for key in f["var"].keys():
        item = f["var"][key]
        if isinstance(item, h5py.Dataset):
            var_dict[key] = item[:]
        elif isinstance(item, h5py.Group) and "categories" in item and "codes" in item:
            categories = [
                cat.decode("utf-8") if isinstance(cat, bytes) else cat
                for cat in item["categories"][:]
            ]
            codes = item["codes"][:]
            var_dict[key] = pd.Categorical.from_codes(codes, categories=categories)

    var_df = pd.DataFrame(var_dict)

    # Convert bytes to strings
    for col in var_df.columns:
        if var_df[col].dtype == object:
            var_df[col] = var_df[col].apply(
                lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
            )

    if "feature_name" in var_df:
        var_df.index = var_df["feature_name"]

    return var_df


def _load_obs_metadata(f, start_row, n_rows, obs_columns=None):
    """Helper function to load observation (cell) metadata."""
    selected_obs_keys = obs_columns if obs_columns else list(f["obs"].keys())
    obs_dict = {}

    for key in selected_obs_keys:
        if key not in f["obs"]:
            continue

        item = f["obs"][key]
        if isinstance(item, h5py.Dataset):
            obs_dict[key] = item[start_row : start_row + n_rows]
        elif isinstance(item, h5py.Group) and "categories" in item and "codes" in item:
            categories = [
                cat.decode("utf-8") if isinstance(cat, bytes) else cat
                for cat in item["categories"][:]
            ]
            codes = item["codes"][start_row : start_row + n_rows]
            obs_dict[key] = pd.Categorical.from_codes(codes, categories=categories)

    return pd.DataFrame(obs_dict)


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
    

def save_metadata_json(found_keys, output_path):
    """Save metadata to JSON file in a standard format"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(found_keys, f, indent=4, ensure_ascii=False)
    return output_path

def setup_logging(log_dir, log_file=None, log_level=logging.INFO, disable_file_logging=False):
    """Setup logging for scGPT scripts"""
    # Ensure log directory exists (this will be default the output directory)
    log_dir = Path(log_dir).absolute()
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger('scGPT_pipeline')
    logger.setLevel(log_level)

    logger.propagate = False
    
    # Remove existing handlers if any
    if logger.handlers:
        logger.handlers = []
    
    # Create file handler ONLY if file logging is not disabled
    if not disable_file_logging:
        # Create log filename if not provided
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"scGPT_pipeline_{timestamp}.log"
        
        log_path = log_dir / log_file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_path}")
    else:
        logger.info("File logging disabled")
    
    # Create console handler (always added)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    logger.info("Logging initialized")
    return logger
    
class AnnDataChunker:
    """
    A class for loading a subset of an AnnData object from an h5ad file for use in multi-core processing.
    
    Parameters
    ----------
    file_path : str or Path
        Path to the h5ad file
    obs_columns : list of str or None
        List of observation columns to load. If None, loads all columns.
    
    Examples
    --------
    >>> with AnnDataChunker('data.h5ad', ['cell_type', 'condition']) as chunker:
    ...     chunk = chunker.load_subset(start_row=0, n_rows=1000)
    """
    def __init__(self, file_path, obs_columns):
        if not isinstance(file_path, (str, Path)):
            raise TypeError("file_path must be a string or Path object")
        if obs_columns is not None and not isinstance(obs_columns, (list, tuple)):
            raise TypeError("obs_columns must be None or a list/tuple of strings")
        
        self.file_path = Path(file_path)
        self.obs_columns = obs_columns
        self._file = None
        self._obs_df = None
        self._var_df = None
        self._total_rows = None

    def __len__(self):
        if self._total_rows is None:
            raise RuntimeError("File not opened. Use 'with' statement or call open() first")
        return self._total_rows

    def _load_obs_metadata(self, f, obs_columns=None):
        """Helper function to load all observation (cell) metadata."""
        selected_obs_keys = obs_columns if obs_columns else list(f["obs"].keys())
        obs_dict = {}

        for key in selected_obs_keys:
            if key not in f["obs"]:
                continue

            item = f["obs"][key]
            if isinstance(item, h5py.Dataset):
                obs_dict[key] = item[:]  # Load entire array
            elif isinstance(item, h5py.Group) and "categories" in item and "codes" in item:
                categories = [
                    cat.decode("utf-8") if isinstance(cat, bytes) else cat
                    for cat in item["categories"][:]
                ]
                codes = item["codes"][:]  # Load all codes
                obs_dict[key] = pd.Categorical.from_codes(codes, categories=categories)

        return pd.DataFrame(obs_dict)

    def open(self):
        """
        Open the h5ad file in read mode using h5py.
        Returns self for method chaining.
        """
        if self._file is None:
            self._file = h5py.File(self.file_path, "r")
            self._obs_df = self._load_obs_metadata(self._file, self.obs_columns)  # Using class method
            self._var_df = _load_var_metadata(self._file)
            self._total_rows = len(self._obs_df)

        return self

    def close(self):
        """
        Close the h5py file.
        """
        if self._file is not None:
            self._file.close()
            self._file = None
            del self._obs_df
            del self._var_df
            self._obs_df = None
            self._var_df = None

    def __enter__(self):
        """
        Context manager entry point.
        """
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit point.
        """
        self.close()

    @property
    def obs(self):
        if self._obs_df is None:
            raise RuntimeError("File not opened")
        return self._obs_df

    @property
    def var(self):
        if self._var_df is None:
            raise RuntimeError("File not opened")
        return self._var_df
    

    def load_subset(self, start_row, n_rows, valid_indices=None):
        """
        Load a subset of rows from the h5ad file.

        Args:
            start_row: Starting row index
            n_rows: Number of rows to load
            valid_indices: Optional list of column indices to keep. If provided,
                         will filter the expression matrix to only these columns.
        """
        if not isinstance(start_row, int) or start_row < 0:
            raise ValueError("start_row must be a non-negative integer")
        if not isinstance(n_rows, int) or n_rows <= 0:
            raise ValueError("n_rows must be a positive integer")
            
        total_rows = len(self._obs_df)
        if start_row >= total_rows:
            raise ValueError(f"start_row ({start_row}) exceeds total rows ({total_rows})")
        if start_row + n_rows > total_rows:
            n_rows = total_rows - start_row
            print(f"Warning: Requested rows exceed total rows. Adjusting n_rows to {n_rows}")
        
        if self._file is None:
            raise RuntimeError("File not opened")

        data, indices, indptr = _load_csr_matrix_components(
            self._file, 
            start_row, 
            n_rows, 
            valid_indices
        )

        # Create sparse matrix with appropriate shape
        n_cols = len(valid_indices) if valid_indices is not None else len(self._var_df)
        X_subset = sparse.csr_matrix(
            (data, indices, indptr), 
            shape=(n_rows, n_cols)
        )

        # Subset the obs DataFrame for the requested rows
        obs_subset = self._obs_df.iloc[start_row:start_row + n_rows]
        
        # Filter var DataFrame if needed
        var_df = self._var_df.iloc[valid_indices] if valid_indices is not None else self._var_df

        return ad.AnnData(X=X_subset, obs=obs_subset, var=var_df)

    def load_torch_csr_matrix(self, start_row, n_rows, valid_indices=None):
        """
        Load a subset of rows from the h5ad file as a torch CSR matrix.
        """
        if self._file is None:
            raise RuntimeError("File not opened")
            
        data, indices, indptr = _load_csr_matrix_components(self._file, start_row, n_rows, valid_indices)
        # Convert to float32 and ensure indices are long
        data = torch.from_numpy(data).float()
        indices = torch.from_numpy(indices).long()
        indptr = torch.from_numpy(indptr).long()
        
        # Use the filtered number of columns when valid_indices is provided
        n_cols = len(valid_indices) if valid_indices is not None else len(self._var_df)
        return torch.sparse_csr_tensor(indptr, indices, data, (n_rows, n_cols))

    @property
    def is_open(self):
        """Check if the file is currently open."""
        return self._file is not None

    def iter_chunks(self, chunk_size, valid_indices=None):
        """
        Iterator that yields chunks of the AnnData object.
        
        Args:
            chunk_size: Number of rows to include in each chunk
            
        Yields:
            AnnData: Chunk of the data with chunk_size rows (or fewer for the last chunk)
            
        Raises:
            RuntimeError: If the file is not opened
        """
        if self._file is None:
            raise RuntimeError("File not opened. Use 'with' statement or call open() first")
            
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
            
        total_rows = len(self._obs_df)
        start_row = 0
        
        while start_row < total_rows:
            # Calculate the actual chunk size (might be smaller for the last chunk)
            current_chunk_size = min(chunk_size, total_rows - start_row)
            
            # Load and yield the chunk
            chunk = self.load_subset(start_row, current_chunk_size, valid_indices=valid_indices)
            yield chunk
            
            # Move to next chunk
            start_row += chunk_size


def load_subset_anndata(file_path, start_row=0, n_rows=None, obs_columns=None):
    """
    Load a subset of rows from an h5ad file as an AnnData object efficiently.

    Args:
        file_path: Path to h5ad file
        start_row: Starting row index
        n_rows: Number of rows to load
        obs_columns: List of obs (cell metadata) columns to include. If None, includes all.

    Returns:
        AnnData object with the subset of data and selected obs metadata.
    """
    with h5py.File(file_path, "r") as f:
        # Determine total rows and number of rows to load
        total_rows = len(f["X"]["indptr"]) - 1
        if n_rows is None:
            n_rows = total_rows - start_row

        # Load components
        data, indices, indptr = _load_csr_matrix_components(f, start_row, n_rows)
        var_df = _load_var_metadata(f)
        obs_df = _load_obs_metadata(f, start_row, n_rows, obs_columns)

        # Create sparse matrix
        X_subset = sparse.csr_matrix(
            (data, indices, indptr), shape=(n_rows, len(var_df))
        )

    return ad.AnnData(X=X_subset, obs=obs_df, var=var_df)


def _get_matrix_n_cols(f):
    """
    Helper function to get the number of columns from an h5ad file matrix.
    
    Args:
        f: h5py File object
    
    Returns:
        int: Number of columns in the matrix
        
    Raises:
        KeyError: If neither 'X' nor 'raw/X' dataset is found in the H5AD file
    """
    if "X" in f:
        if isinstance(f["X"], h5py.Dataset):
            return f["X"].shape[1]
        else:  # It's a group containing CSR matrix data
            return f["X"].attrs["shape"][1]
    elif "raw/X" in f:
        if isinstance(f["raw/X"], h5py.Dataset):
            return f["raw/X"].shape[1]
        else:
            return f["raw/X"].attrs["shape"][1]
    else:
        raise KeyError("Could not find 'X' dataset in the H5AD file.")

def _load_csr_matrix_components(f, start_row, n_rows, valid_indices=None):
    """
    Helper function to load CSR matrix components from h5ad file.
    
    Args:
        f: h5py File object
        start_row: Starting row index
        n_rows: Number of rows to load
        valid_indices: Optional list of column indices to keep
    """
    # Load the basic CSR components
    indptr = f["X"]["indptr"][start_row : start_row + n_rows + 1]
    start_idx, end_idx = indptr[0], indptr[-1]
    indices = f["X"]["indices"][start_idx:end_idx]
    data = f["X"]["data"][start_idx:end_idx]
    
    # Adjust indptr relative to start_idx
    indptr = indptr - start_idx
    
    if valid_indices is not None:
        n_cols = _get_matrix_n_cols(f)
        mat = sparse.csr_matrix((data, indices, indptr), shape=(n_rows, n_cols))
        
        # Get columns to keep
        mat = mat[:, valid_indices]
        
        return mat.data, mat.indices, mat.indptr
        
    return data, indices, indptr


def _load_var_metadata(f):
    """Helper function to load variable (gene) metadata."""
    var_dict = {}
    for key in f["var"].keys():
        item = f["var"][key]
        if isinstance(item, h5py.Dataset):
            var_dict[key] = item[:]
        elif isinstance(item, h5py.Group) and "categories" in item and "codes" in item:
            categories = [
                cat.decode("utf-8") if isinstance(cat, bytes) else cat
                for cat in item["categories"][:]
            ]
            codes = item["codes"][:]
            var_dict[key] = pd.Categorical.from_codes(codes, categories=categories)

    var_df = pd.DataFrame(var_dict)

    # Convert bytes to strings
    for col in var_df.columns:
        if var_df[col].dtype == object:
            var_df[col] = var_df[col].apply(
                lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
            )

    if "feature_name" in var_df:
        var_df.index = var_df["feature_name"]

    return var_df


def _load_obs_metadata(f, start_row, n_rows, obs_columns=None):
    """Helper function to load observation (cell) metadata."""
    selected_obs_keys = obs_columns if obs_columns else list(f["obs"].keys())
    obs_dict = {}

    for key in selected_obs_keys:
        if key not in f["obs"]:
            continue

        item = f["obs"][key]
        if isinstance(item, h5py.Dataset):
            obs_dict[key] = item[start_row : start_row + n_rows]
        elif isinstance(item, h5py.Group) and "categories" in item and "codes" in item:
            categories = [
                cat.decode("utf-8") if isinstance(cat, bytes) else cat
                for cat in item["categories"][:]
            ]
            codes = item["codes"][start_row : start_row + n_rows]
            obs_dict[key] = pd.Categorical.from_codes(codes, categories=categories)

    return pd.DataFrame(obs_dict)


