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


# === EMBEDDING FUNCTION ===

def embed_adata(adata, config=None, config_path=None, output_dir=None):
    """
    Wrapper function to embed AnnData using scGPT with configuration from pipeline
    
    Args:
        adata: AnnData object to embed
        config: Dictionary containing configuration (from build_config)
        config_path: Path to config file (optional)
        output_dir: Directory to save outputs
        
    Returns:
        AnnData object with embeddings in obsm
    """
    # If both config and config_path are None, use default config
    if config is None and config_path is None:
        print("No config provided, using default settings")
        return embed_data(
            adata,
            model_dir="models/scGPT_human",
            gene_col="feature_name",
            batch_size=64,
            device="cuda" if torch.cuda.is_available() else "cpu",
            use_flash_attention=False,
            output_key="X_scGPT"
        )
    
    # If config is provided (from build_config in pipeline), use it
    if config is not None:
        print(f"Using provided config with {len(config)} parameters")
        return embed_data(
            adata,
            model_dir=config.get("model_dir", "models/scGPT_human"),
            gene_col=config.get("gene_col", "feature_name"),
            batch_size=config.get("batch_size", 64),
            max_length=config.get("max_seq_len", 1200),
            device=config.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
            use_flash_attention=config.get("use_flash_attention", False),
            output_key=config.get("output_key", "X_scGPT"),
        )
    
    # If only config_path is provided, process it
    print(f"Loading config from {config_path}")
    import json
    with open(config_path, 'r') as f:
        file_config = json.load(f)
    
    # Extract parameters from nested config structure
    model_params = file_config.get('model', {})
    data_params = file_config.get('data', {})
    embedding_params = file_config.get('embedding', {})
    
    return embed_data(
        adata,
        model_dir=model_params.get('model_dir', "models/scGPT_human"),
        gene_col=data_params.get('gene_col', "feature_name"),
        batch_size=data_params.get('batch_size', 64),
        max_length=model_params.get('max_seq_len', 1200),
        device=file_config.get('device', "cuda" if torch.cuda.is_available() else "cpu"),
        use_flash_attention=embedding_params.get('use_flash_attention', False),
        output_key=embedding_params.get('output_key', "X_scGPT"),
    )

# === SAVE FUNCTION === # 

def save_adata(adata, output_dir, input_file, date_str=None):
    """Save the adata object to a file with scGPT embeddings"""
    if date_str is None:
        date_str = datetime.now().strftime("%Y%m%d")
        
    name = Path(input_file).stem
    if 'X_scGPT' in adata.obsm:
        print(f"Saving embedded adata object to {output_dir}")
        # Create output filename
        adata_path = Path(output_dir) / f"{name}_scGPT_embed_{date_str}.h5ad"
        
        # Store original filename in metadata if available
        if 'filename' not in adata.uns:
            adata.uns['original_file'] = str(input_file)
            
        # Save file
        adata.write_h5ad(adata_path)
        print(f"Saved embedded data to {adata_path}")
        return adata_path
    else:
        print(f"No scGPT embeddings found in adata.obsm['X_scGPT'].")
        print(f"Original file {input_file} not saved in {output_dir}.")
        return None

# === TEST FUNCTION ===

def test_embed_config(adata, config_path=None, metadata_path=None, **kwargs):
    """Test configuration preparation without running embedding"""
    config = prepare_embed_config(config_path, metadata_path, **kwargs)
    print("\nTEST CONFIGURATION:")
    print("===================")
    print(f"Model dir: {config['model']['model_dir']}")
    print(f"Gene column: {config['data']['gene_col']}")
    print(f"Batch size: {config['data']['batch_size']}")
    print(f"Max sequence length: {config['model'].get('max_seq_len')}")
    print(f"Output key: {config['embedding']['output_key']}")
    print(f"Columns to preserve: {config['embedding'].get('obs_to_save')}")
    print(f"Device: {config.get('device')}")
    print("===================")
    return config

# === MAIN FUNCTION ===# 

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='scGPT data loading and embedding')
    parser.add_argument('--input_file', type=str, required=True, 
                       help='Path to input h5ad file')
    parser.add_argument('--config_file', type=str, default=None,
                       help='Path to config JSON file')
    parser.add_argument('--metadata_file', type=str, default=None,
                       help='Path to metadata JSON file')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory')
    parser.add_argument('--model_dir', type=str, default=None,
                       help='Model directory')
    parser.add_argument('--gene_col', type=str, default=None,
                       help='Column name for gene identifiers')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size for inference')
    parser.add_argument('--test_config', action='store_true',
                       help='Test configuration without running embedding')

    
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
            config_path=args.config_file,
            metadata_path=args.metadata_file,
            **kwargs
        )
        return
    else:
        # Embed with automatic metadata integration
        embed_data(
            adata, 
            config_path=args.config_file,
            metadata_path=args.metadata_file,
            **kwargs
        )
        
        #save to new file with {filename}_scGPT_embed_{date_str}.h5ad
        save_adata(adata, output_dir, args.input_file)
   


if __name__ == '__main__':
    main()