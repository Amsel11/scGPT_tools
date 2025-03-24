#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utility functions for scGPT scripts
"""

import json
from pathlib import Path
import pprint
from html import escape
import pandas as pd
import scanpy as sc
import h5py
import scipy.sparse as sparse
from anndata import AnnData

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
        
    
    return repo_dir, data_dir, save_dir, model_dir

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