#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data embedding and analysis for scGPT model, using scGPT model. 

This script:
1. Loads AnnData (h5ad) files with caching for faster repeated access
2. Detects cell type, gene, batch and other annotations automatically
3. Extracts metadata needed for scGPT embedding
4. Saves analysis results as text and structured metadata as JSON 
5. Optionally generates an HTML report with the analysis results and metadata

Key usage is to prepare data for the embedding step in scGPT, which will implemented right after. 
"""

import sys
import os
import argparse
import scanpy as sc
import pandas as pd
import numpy as np
import torch
import h5py
import scipy
#import anndata
from anndata import AnnData
from pathlib import Path
import json
import logging
import io
from contextlib import redirect_stdout
from datetime import datetime
import pickle
from html import escape
import anndata as ad
import tqdm
import scipy.sparse as sparse
import scgpt as scg
from utils import _load_anndata
from utils import setup_directories
# === SETUP AND DATA LOADING ===

def load_h5ad(path, save_dir=None, subset=None, force_reload=False):
    """Load h5ad file with optional subsetting and caching
    
    Args:
        path: Path to h5ad file
        save_dir: Directory to store cache (optional)
        subset: Dict with keys 'start_row', 'n_rows', 'obs_columns' or None for full dataset
        force_reload: Whether to force reload and ignore cache
    
    Returns:
        AnnData object
    """
    import time
    from pathlib import Path
    
    # Start timing
    start_time = time.time()
    
    if save_dir is None:
        print(f"No save_dir specified, caching disabled")
        return _load_anndata(path, subset)
    
    # Modify cache filename if using subset
    if subset:
        start = subset.get('start_row', 0)
        n_rows = subset.get('n_rows', 'end')
        cache_suffix = f"_subset_{start}-{n_rows}"

    
    # Load data from original file
    print(f"Loading data from {path}{' (subset)' if subset else ''}")
    adata = _load_anndata(path, subset)
    
    return adata



# === EMBEDDING FUNCTION ===
"""
Using scGPT model with their own embedding function to embed the data

input: AnnData object
output: AnnData object with embedded data in .obsm['X_scGPT']

Needs the following input 
- model_dir: path to the model directory
- config_path: path to the config file
- gene_col: column name of the gene names in the .var dataframe
- batch_size: batch size for the embedding

def embed_data(adata, model_dir, config_path = 'config/scGPT_embedder_config.json'):
    Embed data using scGPT model
    Preprocess anndata and embed the data using the model.

    Args:
        adata_or_file (Union[AnnData, PathLike]): The AnnData object or the path to the
            AnnData object.
        model_dir (PathLike): The path to the model directory.
        gene_col (str): The column in adata.var that contains the gene names.
        max_length (int): The maximum length of the input sequence. Defaults to 1200.
        batch_size (int): The batch size for inference. Defaults to 64.
        obs_to_save (Optional[list]): The list of obs columns to save in the output adata.
            Useful for retaining meta data to output. Defaults to None.
        device (Union[str, torch.device]): The device to use. Defaults to "cuda".
        use_fast_transformer (bool): Whether to use flash-attn. Defaults to True.
        return_new_adata (bool): Whether to return a new AnnData object. If False, will
            add the cell embeddings to a new :attr:`adata.obsm` with key "X_scGPT".

    Returns:
        AnnData: The AnnData object with the cell embeddings.
"""

def embed_data(adata, config_path='scGPT_embed_config.json', metadata_path=None, **kwargs):
    """
    Embed data using scGPT model with intelligent parameter detection
    
    Args:
        adata (AnnData): The AnnData object to embed
        config_path (str): Path to the base embedding config JSON
        metadata_path (str): Path to the dataset metadata JSON (optional)
        **kwargs: Parameters to override from config
        
    Returns:
        AnnData: The AnnData object with embeddings in .obsm
    """
    # Resolve config path more robustly
    script_dir = Path(__file__).parent
    config_paths_to_try = [
        Path(config_path),  # Try as provided
        script_dir / config_path,  # Try relative to script directory
        script_dir / "scGPT_embed_config.json",  # Try default name in script dir
    ]
    
    # Try each possible path
    config_file = None
    for path in config_paths_to_try:
        if path.exists():
            config_file = path
            break
    
    if not config_file:
        raise FileNotFoundError(f"Config file not found. Tried: {[str(p) for p in config_paths_to_try]}")
    
    # Load base config
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        print(f"Loaded config from {config_file}")
    except Exception as e:
        print(f"Error loading config: {e}")
        raise
    
    # Integrate metadata if provided
    if metadata_path:
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"Loaded metadata from {metadata_path}")
            
            # Intelligent parameter selection based on metadata
            
            # 1. Use the first detected gene column if not explicitly overridden
            if "gene_col" not in kwargs and metadata.get("gene_keys"):
                config["data"]["gene_col"] = metadata["gene_keys"][1]
                print(f"Using detected gene column: {config['data']['gene_col']}")
            
            if "model_dir" not in kwargs and metadata.get("directories", {}).get("model_dir"):
                config["model"]["model_dir"] = metadata["directories"]["model_dir"]
                print(f"Using detected model directory: {config['model']['model_dir']}")
            
            # 2. Include important columns in obs_to_save
            obs_to_save = []
            if metadata.get("cell_type_keys"):
                obs_to_save.extend(metadata["cell_type_keys"])
            if metadata.get("batch_keys"):
                obs_to_save.extend(metadata["batch_keys"])
            
            if obs_to_save:
                config["embedding"]["obs_to_save"] = obs_to_save
                print(f"Will preserve these columns: {', '.join(obs_to_save)}")
                
        except Exception as e:
            print(f"Warning: Could not load metadata: {e}")
    
    # Override with any explicit parameters
    for section in config:
        if isinstance(config[section], dict):
            config[section].update({k: v for k, v in kwargs.items() if k in config[section]})
    
    # Also allow flat overrides
    for k, v in kwargs.items():
        for section in config:
            if isinstance(config[section], dict) and k in config[section]:
                config[section][k] = v
    
    # Extract parameters for embedding
    model_params = config["model"]
    data_params = config["data"]
    embed_params = config["embedding"]
    
    # Model directory handling
    model_dir = model_params["model_dir"]
    
    print(f"Embedding {adata.shape[0]} cells using model in {model_dir}")
    print(f"Using gene column: {data_params['gene_col']}")
    print(f"Batch size: {data_params['batch_size']}")
    print(f"Device: {config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')}")
    
    # Call scGPT embedding function THIS IS THE IMPORTANT PART THE REST IS JUST PREP
    embed_adata = scg.tasks.embed_data(
        adata,
        model_dir,
        gene_col=data_params["gene_col"],
        batch_size=data_params["batch_size"],
        max_length=model_params.get("max_seq_len", 1200),
        obs_to_save=embed_params.get("obs_to_save"),
        device=config.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
        use_fast_transformer=embed_params.get("use_flash_attention", True),
        return_new_adata=False
    )
    
    # Rename embedding key if specified
    output_key = embed_params.get("output_key", "X_scGPT")
    if output_key != "X_scGPT" and "X_scGPT" in adata.obsm:
        adata.obsm[output_key] = adata.obsm["X_scGPT"].copy()
    
    print(f"Embedding complete. Shape: {adata.obsm[output_key].shape}")
    return embed_adata

from utils import test_embed_config

# === SAVE FUNCTION === # 

def save_adata(adata, output_dir, input_file, date_str = datetime.now().strftime("%Y%m%d")):
    """Save the adata object to a file with scGPT embeddings"""
    name = Path(input_file).stem
    if 'X_scGPT' in adata.obsm:
        print(f"Saving embedded adata object to {output_dir}")
        # Create output filename
        adata_path = output_dir / f"{name}_scGPT_embed_{date_str}.h5ad"
        
        # Store original filename in metadata if available
        if 'filename' not in adata.uns:
            adata.uns['original_file'] = input_file
            
        # Save file
        adata.write_h5ad(adata_path)
        print(f"Saved embedded data to {adata_path}")
        return adata_path
    else:
        print(f"No scGPT embeddings found in adata.obsm['X_scGPT'].")
        print(f"Original file {input_file} not saved in {output_dir}.")
        return None



# === MAIN FUNCTION ===# 

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='scGPT data loading and metadata extraction')
    parser.add_argument('--input_file', type=str, default='data/Derived_Embryoid_Bodies.h5ad', 
                       help='Path to input h5ad file')
    parser.add_argument('--force_reload', action='store_true', 
                       help='Force reload data and ignore cache')
    parser.add_argument('--testing', action='store_true', default=True, 
                       help='Use test output directory')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Print additional details about the dataset')
    parser.add_argument('--test_config', action='store_true', default= True,
                       help='Test configuration parsing without running embedding')

    
    # Updated subsetting arguments
    subset_group = parser.add_argument_group('Data subsetting options')
    subset_group.add_argument('--subset', type=int, default=None,
                       help='Load a subset of N cells (shorthand for --n_rows)')
    subset_group.add_argument('--start_row', type=int, default=0,
                       help='Starting row index for subset loading')
    subset_group.add_argument('--n_rows', type=int, default=None,
                       help='Number of rows to load (None = all remaining rows)')
    subset_group.add_argument('--obs_columns', type=str, nargs='+', default=None,
                       help='Space-separated list of observation columns to include')
    
    args = parser.parse_args()
    
    print("Running scGPT dataloader...")
    
    # Setup directories
    repo_dir, data_dir, save_dir, model_dir = setup_directories()
    
    # Add repo to path if needed
    if str(repo_dir) not in sys.path:
        sys.path.append(str(repo_dir))
    
    # Create output directory
    base_name = Path(args.input_file).stem
    date_str = datetime.now().strftime("%Y%m%d")
    
    if args.testing:
        output_dir = save_dir / "test_output"
    else:
        existing = [x for x in save_dir.iterdir() if x.is_dir() and x.name.startswith(f"{base_name}_{date_str}")]
        number = len(existing) + 1
        output_dir = save_dir / f"{base_name}_{date_str}_{number:02d}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load data with subsetting
    subset = None
    
    # Handle --subset as a shortcut for --n_rows
    if args.subset is not None:
        args.n_rows = args.subset
    
    # Create subset dictionary if any subsetting options specified
    if args.n_rows is not None or args.obs_columns is not None:
        subset = {
            'start_row': args.start_row,
            'n_rows': args.n_rows,
            'obs_columns': args.obs_columns
        }
        print(f"Loading data subset: start={args.start_row}, rows={args.n_rows or 'all'}")
        if args.obs_columns:
            print(f"Including only these obs columns: {', '.join(args.obs_columns)}")
    
    # Call the unified loading function with optional subsetting
    adata = load_h5ad(
        args.input_file, 
        save_dir=output_dir,  # Enable caching in the output directory
        subset=subset,
        force_reload=args.force_reload
    )
    print(f"Loaded dataset: {adata.shape[0]} cells × {adata.shape[1]} genes")

    # Path to the metadata created by scGPT_dataloader.py
    metadata_path = output_dir / f"{base_name}_metadata.json"

    if args.test_config:
        # Path to the metadata created by scGPT_dataloader.py        
        print("\n TESTING CONFIGURATION WITHOUT EMBEDDING")
        test_embed_config(
            adata, 
            config_path="scripts/scGPT_embed_config.json",
            metadata_path=metadata_path if metadata_path.exists() else None
        )
        return
    else:
        # Embed with automatic metadata integration
        embed_data(
            adata, 
            config_path="scripts/scGPT_embed_config.json",
            metadata_path=metadata_path if metadata_path.exists() else None
        )
        
        #save to new file with {filename}_scGPT_embed_{date_str}.h5ad
        save_adata(adata, output_dir, args.input_file)
   


if __name__ == '__main__':
    main()