#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data loading and metadata extraction for scGPT model for embedding and analysis of data (preparation for scGPT)

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
# === SETUP AND DATA LOADING ===

def setup_directories():
    """Set up necessary directories and return their paths"""
    repo_dir = Path.cwd().absolute()
    data_dir = repo_dir / "data"
    save_dir = repo_dir / "save"
    
    data_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    return repo_dir, data_dir, save_dir

def load_h5ad(path, save_dir=None, subset=None, force_reload=False):
    """Load h5ad file with optional subsetting (no caching)
    
    Args:
        path: Path to h5ad file
        save_dir: Ignored parameter (kept for compatibility)
        subset: Dict with keys 'start_row', 'n_rows', 'obs_columns' or None for full dataset
        force_reload: Ignored parameter (kept for compatibility)
    
    Returns:
        AnnData object
    """
    import time
    from pathlib import Path
    
    # Start timing
    start_time = time.time()
    
    # Simple loading without any caching
    print(f"Loading data from {path}{' (subset)' if subset else ''}")
    adata = _load_anndata(path, subset)
    print(f"Data loaded in {time.time()-start_time:.2f}s with shape {adata.shape}")
    
    return adata

def _load_anndata(path, subset=None):
    """Internal function to load anndata with or without subsetting"""
    if subset is None:
        # Load full dataset
        return sc.read_h5ad(path)
    else:
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

        return AnnData(X=X_subset, obs=obs_df, var=var_df)

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

#two functions to cache and load for easier data loading: 
def cache_adata(adata, output_dir, filename_base):
    cache_path = output_dir / f"{filename_base}_cache.h5ad"
    print(f"Caching AnnData object to {cache_path}")
    adata.write(cache_path)
    return cache_path

def load_cached_adata(output_dir, filename_base):
    cache_path = output_dir / f"{filename_base}_cache.h5ad"
    if cache_path.exists():
        print(f"Loading cached AnnData from {cache_path}")
        return sc.read_h5ad(cache_path), True
    return None, False

# === ANNOTATION DETECTION AND ANALYSIS ===

def get_examples_and_counts(series, n=5):
    """Get n unique examples and value counts from a series"""
    if hasattr(series, 'unique'):
        examples = series.unique()[:n]
        counts = series.value_counts()
        n_unique = len(series.unique())
        return examples, counts, n_unique
    return series.iloc[:n], None, len(series.unique())

def check_keys(adata, key_list, category_name, location, show_examples=True, special_handling=None):
    """
    Unified function to check for keys in AnnData object.
    
    Args:
        adata: AnnData object
        key_list: List of possible keys to check
        category_name: Name of the category (for display purposes)
        location: Where to look ('obs', 'var', or 'var_hvg')
        show_examples: Whether to display examples of found keys
        special_handling: Optional function to handle special cases
        
    Returns:
        list: Found keys
    """
    found = []
    print(f"\nChecking {category_name} in .{location.split('_')[0]}:")
    
    # For var locations, always show index name
    if location.startswith('var'):
        print(f"Var index name: {adata.var.index.name or 'None'}")
    
    # Determine which dataframe to check
    if location.startswith('obs'):
        df = adata.obs
        check_index = False
    else:  # var or var_hvg
        df = adata.var
        check_index = True
    
    # Check each key
    for key in key_list:
        # Check in columns
        if key in df.columns:
            found.append(key)
            
            # Special handling for HVG boolean columns
            if location == 'var_hvg' and df[key].dtype == bool:
                n_items = df[key].sum()
                total = len(df[key])
                print(f"\n✓ Found '{key}' (boolean) with {n_items}/{total} marked as {category_name}")
                print(f"Percentage: {(n_items/total)*100:.2f}%")
            # Regular column handling with examples
            elif show_examples:
                examples, counts, n_unique = get_examples_and_counts(df[key])
                print(f"\n✓ Found '{key}' with {n_unique} unique values")
                print(f"Examples:")
                if isinstance(examples, (list, np.ndarray)):
                    for i, ex in enumerate(examples, 1):
                        print(f"  {i}. {ex}")
                else:
                    print(f"  {examples}")
        
        # Check index name (only for var location)
        elif check_index and df.index.name == key:
            found.append(key)
            if show_examples:
                examples, counts, n_unique = get_examples_and_counts(df.index)
                print(f"\n✓ Found '{key}' as index with {n_unique} unique values")
                print(f"Examples:")
                for i, ex in enumerate(examples, 1):
                    print(f"  {i}. {ex}")
    
    # If no keys found, show available columns
    if not found:
        print(f"\n⚠️ No well-known {category_name} found in .{location.split('_')[0]}")
        print(f"Available .{location.split('_')[0]} columns: {', '.join(df.columns)}")
    
    # Apply any additional special handling
    if special_handling and callable(special_handling):
        special_handling(adata, found)
    
    return found

def check_annotation_keys(adata):
    """
    Detect commonly used keys for cell types, genes, batches, and HVGs.
    Returns a dictionary of found keys.
    """
    # Define key categories with their possible variations
    key_categories = {
        'cell_type_keys': [
            'cell_type', 'celltype', 'CellType', 'cell.type', 'cell_type_label',
            'cell_ontology_class', 'cluster', 'cluster_label', 'louvain', 'leiden', 
            'annotation', 'cell_annotation', 'predicted_celltype', 'cell_identity', 
            'cell_id', 'type', 'subtype'
        ],
        'gene_keys': [
            'gene_name', 'gene_names', 'gene_symbol', 'symbol', 'gene_id', 
            'gene_short_name', 'ensembl_id', 'ensembl', 'gene', 'genes',
            'feature_name', 'feature_id', 'name', 'index'
        ],
        'hvg_keys': [
            'highly_variable', 'highly_variable_genes', 'hvg', 'HVG',
            'variable_genes', 'variable_features', 'highly_variable_rank',
            'highly_variable_scores', 'dispersions_norm', 'dispersions'
        ],
        'batch_keys': [
            'batch', 'batch_id', 'batch_label', 'batch_condition', 'batch_index',
            'batch_name', 'batch_number', 'batch_group', 'batch_category',
            'batch_identifier', 'batch_id_label', 'batch_id_name', 'batch_id_number',
            'batch_id_group', 'batch_id_category', 'donor_id', 'donor', 'sample', 'sample_id'
        ]
    }
    
    # Initialize results dictionary
    found_keys = {category: [] for category in key_categories}
    
    # Special handling for comparing multiple cell types
    def compare_cell_types(adata, found_cell_types):
        if len(found_cell_types) > 1:
            print("\nComparison of cell type annotations:")
            print(f"Found {len(found_cell_types)} different cell type annotations.")
            for key in found_cell_types:
                n_types = len(adata.obs[key].unique())
                print(f"• '{key}' has {n_types} unique values")
    
    # Check for each category using the generalized function
    found_keys['cell_type_keys'] = check_keys(
        adata, 
        key_categories['cell_type_keys'], 
        'cell type annotations', 
        'obs',
        special_handling=compare_cell_types
    )
    
    found_keys['gene_keys'] = check_keys(
        adata, 
        key_categories['gene_keys'], 
        'gene annotations', 
        'var'
    )
    
    found_keys['hvg_keys'] = check_keys(
        adata, 
        key_categories['hvg_keys'], 
        'highly variable genes', 
        'var_hvg'
    )
    
    found_keys['batch_keys'] = check_keys(
        adata, 
        key_categories['batch_keys'], 
        'batch annotations', 
        'obs'
    )
    
    # Additional metadata checks
    print("\nChecking additional metadata:")
    # List of potential AnnData attributes to check
    anndata_attributes = ['uns', 'layers', 'obsm', 'varm', 'obsp', 'varp']
    
    for attr in anndata_attributes:
        if hasattr(adata, attr):
            attr_obj = getattr(adata, attr)
            if isinstance(attr_obj, dict):
                print(f"Available .{attr} keys: {list(attr_obj.keys())}")
            else:
                print(f"Available .{attr}: {attr_obj}")
    
    return found_keys

# === OUTPUT AND REPORTING ===

def save_metadata_json(found_keys, output_path):
    """Save metadata to JSON file in a standard format"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(found_keys, f, indent=4, ensure_ascii=False)
    return output_path

def print_summary(adata, found_keys, output_dir, metadata_file, output):
    """Print a concise summary of the loaded data"""
    print("\n" + "="*80)
    print("DATASET SUMMARY:")
    print(f"• Dimensions: {adata.n_obs} cells × {adata.n_vars} genes")
    
    if found_keys['cell_type_keys']:
        key = found_keys['cell_type_keys'][0]
        n_types = len(adata.obs[key].unique())
        print(f"• Cell types: {n_types} unique types found using '{key}'")
        examples = list(adata.obs[key].value_counts().head(3).index)
        print(f"  Top types: {', '.join(map(str, examples))}")
    
    if found_keys['batch_keys']:
        key = found_keys['batch_keys'][0]
        n_batches = len(adata.obs[key].unique())
        print(f"• Batches: {n_batches} batches found using '{key}'")
    
    if found_keys['gene_keys']:
        key = found_keys['gene_keys'][0]
        print(f"• Gene identifiers found: '{key}'")
    
    if found_keys['hvg_keys']:
        key = found_keys['hvg_keys'][0]
        if adata.var[key].dtype == bool:
            n_hvg = adata.var[key].sum()
            print(f"• Highly variable genes: {n_hvg} genes marked using '{key}'")
    
    print(f"\nMetadata saved to: {metadata_file}")
    print(f"Analysis saved to: {output}")
    print("="*80)



# === MAIN FUNCTION ===

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='scGPT data loading and metadata extraction')
    parser.add_argument('--input_file', type=str, default='data/7af3a87a-c148-4988-a7cd-f33666ffd883.h5ad', 
                       help='Path to input h5ad file')
    parser.add_argument('--reload', action='store_true', 
                       help='Force reload data and ignore cache')
    parser.add_argument('--testing', action='store_true', default=False, 
                       help='Use test output directory')
    parser.add_argument('--output_mode', type=str, choices=['terminal', 'file', 'both'],
                       default='both', help='Where to output analysis information')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Print additional details about the dataset')
    parser.add_argument('--html', action='store_true', default=False,
                       help='Generate HTML report instead of text')
    
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
    repo_dir, data_dir, save_dir = setup_directories()
    directories = {
        "repo_dir": str(repo_dir),
        "data_dir": str(data_dir),
        "save_dir": str(save_dir)
    }
    
    # Add repo to path if needed
    if str(repo_dir) not in sys.path:
        sys.path.append(str(repo_dir))
    
    if args.testing:
        output_dir = save_dir / "test_output"
    else:
        # Create output directory
        base_name = Path(args.input_file).stem
        date_str = datetime.now().strftime("%Y%m%d")
        existing = [x for x in save_dir.iterdir() if x.is_dir() and x.name.startswith(f"{base_name}_{date_str}")]
        number = len(existing) + 1
        output_dir = save_dir / f"{base_name}_{date_str}_{number:02d}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Prepare subsetting options
    subset = None
    if args.subset is not None:
        args.n_rows = args.subset
    
    if args.n_rows is not None or args.obs_columns is not None:
        subset = {
            'start_row': args.start_row,
            'n_rows': args.n_rows,
            'obs_columns': args.obs_columns
        }
        print(f"Loading data subset: start={args.start_row}, rows={args.n_rows or 'all'}")
        if args.obs_columns:
            print(f"Including only these obs columns: {', '.join(args.obs_columns)}")
    
    # Try to load cached data if not explicitly asked to reload
    adata = None
    base_name = Path(args.input_file).stem
    
    if not args.reload:
        try:
            cache_path = output_dir / f"{base_name}_cache.h5ad"
            if cache_path.exists():
                print("Loading data from cache...")
                adata = sc.read_h5ad(cache_path)
                print(f"Loaded dataset from cache: {adata.shape[0]} cells × {adata.shape[1]} genes")
        except Exception as e:
            print(f"Error loading cache: {e}")
            adata = None
    
    # If no cache or reload requested, load from original file
    if adata is None:
        print("Loading data from original file...")
        adata = load_h5ad(
            args.input_file, 
            save_dir=output_dir,
            subset=subset
        )
        print(f"Loaded dataset from file: {adata.shape[0]} cells × {adata.shape[1]} genes")
        
        # Save for future use
        cache_path = output_dir / f"{base_name}_cache.h5ad"
        print(f"Saving data cache to: {cache_path}")
        adata.write(cache_path)
    
    # Capture annotation analysis output
    output_buffer = io.StringIO()
    with redirect_stdout(output_buffer):
        found_keys = check_annotation_keys(adata)
         
        # Additional verbose information if requested
        if args.verbose:
            print("\n----- ADDITIONAL DATASET DETAILS -----")
            if 'n_genes' not in adata.obs and 'n_counts' not in adata.obs:
                print("Computing basic QC metrics...")
                sc.pp.calculate_qc_metrics(adata, inplace=True)

            if 'n_genes' in adata.obs:
                print(f"Genes per cell: min={adata.obs.n_genes.min()}, "
                      f"median={adata.obs.n_genes.median()}, "
                      f"max={adata.obs.n_genes.max()}")
            
            if 'n_counts' in adata.obs:
                print(f"UMI counts per cell: min={adata.obs.n_counts.min()}, "
                      f"median={adata.obs.n_counts.median()}, "
                      f"max={adata.obs.n_counts.max()}")
                
            # Check sparsity
            if scipy.sparse.issparse(adata.X):
                sparsity = 1.0 - (adata.X.nnz / (adata.n_obs * adata.n_vars))
                print(f"Data matrix sparsity: {sparsity:.4f} "
                      f"({sparsity*100:.1f}% zeros)")
    
    # Get captured output as string
    analysis_output = output_buffer.getvalue()
    
    # Handle different output modes
    if args.output_mode in ['terminal', 'both']:
        print(analysis_output)
    
    if args.output_mode in ['file', 'both']:
        if args.html:
            # HTML report
            from utils import generate_basic_html_report
            html_file = output_dir / f"{base_name}_analysis.html"
            generate_basic_html_report(analysis_output, found_keys, adata, html_file)
            output = html_file
            print(f"Saved HTML report to {html_file}")
        else:
            # Text report
            analysis_file = output_dir / f"{base_name}_analysis.txt"
            with open(analysis_file, 'w', encoding='utf-8') as f:
                f.write(analysis_output)
            print(f"Saved detailed analysis to {analysis_file}")
            output = analysis_file

    # Save found keys as JSON metadata with directories included
    metadata_file = output_dir / f"{base_name}_metadata.json"
    found_keys['directories'] = directories
    save_metadata_json(found_keys, metadata_file)

    # Print summary
    print_summary(adata, found_keys, output_dir, metadata_file, output)
    
    return adata, found_keys

if __name__ == '__main__':
    main()