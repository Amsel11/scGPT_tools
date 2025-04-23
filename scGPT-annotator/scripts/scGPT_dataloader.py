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

def split_query_ref_standalone(adata, method='batch', batch_key=None, test_size=0.2, 
                             random_state=42, verbose=True):
    """
    Split a single AnnData object into query and reference datasets.
    
    Args:
        adata: AnnData object to split
        method: Splitting method ('batch', 'kfold', or 'random')
        batch_key: Column in adata.obs containing batch information
        test_size: Proportion of data to use as query
        random_state: Random seed for reproducibility
        verbose: Whether to print information about the split
        
    Returns:
        ref_adata, query_adata: Split datasets
    """
    # Verify batch_key exists if needed for 'batch' method
    if method == 'batch':
        if batch_key not in adata.obs:
            raise ValueError(f"Batch key '{batch_key}' not found in adata.obs")
        if verbose:
            print(f"Found {len(adata.obs[batch_key].unique())} batches using key '{batch_key}'")
    
    # Different splitting methods
    if method == 'batch':
        # Split by keeping batches together
        from sklearn.model_selection import GroupShuffleSplit
        
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(adata.X, groups=adata.obs[batch_key]))
        
        # Create the split datasets
        ref_adata = adata[train_idx].copy()
        query_adata = adata[test_idx].copy()
        
        # Report the batches in each set
        if verbose:
            query_batches = query_adata.obs[batch_key].unique()
            ref_batches = ref_adata.obs[batch_key].unique()
            print(f"Reference set: {len(ref_adata)} cells from {len(ref_batches)} batches")
            print(f"Query set: {len(query_adata)} cells from {len(query_batches)} batches")
            print(f"Query batches: {', '.join(map(str, query_batches))}")
    
    elif method == 'random':
        # Simple random split ignoring batches
        from sklearn.model_selection import train_test_split
        
        train_idx, test_idx = train_test_split(
            range(adata.n_obs), 
            test_size=test_size,
            random_state=random_state
        )
        
        ref_adata = adata[train_idx].copy()
        query_adata = adata[test_idx].copy()
        
        if verbose:
            print(f"Random split: {len(ref_adata)} reference cells, {len(query_adata)} query cells")
    
    else:
        raise ValueError(f"Unsupported split method: {method}. Use 'batch' or 'random'")
    
    return ref_adata, query_adata
  

# === ANNOTATION DETECTION AND ANALYSIS ===

def analysis_meta(file_path, save=False, output_dir=None):
    """
    Analyze metadata in an H5 file and save results to a JSON file.
    
    Args:
        file_path: Path to the H5 file
        save: Whether to save the results to a JSON file (Boolean)
        output_dir: Directory to save the JSON file (default: current directory)

    Returns: 
        Dictionary containing the metadata of the h5py file
    """
    logger = logging.getLogger('scGPT_pipeline')
    logger.info(f"Analyzing metadata for {file_path}")
    
    # Define key category patterns #THIS CAN BE EXTENDED OR IN THE FUTURE BE PUT IN A CONFIG FILE
    key_categories = { 
        'cell_type_keys': ['cell_type', 'celltype', 'cell.type', 'subtype', 'cell_type_label', 'cell_id', 'CELL_ID', 'cellID', 'CellID'],
        'gene_keys': ['feature_name', 'gene_symbol', 'gene_name', 'gene_id', 'ensg', 'ensembl_id'],
        'batch_keys': ['batch', 'donor', 'sample', 'replicate', 'experiment', 'dataset'],
        'hvg_keys': ['highly_variable', 'hvg', 'variable_gene', 'dispersion'],
        'embedding_keys': ['x_umap', 'x_tsne', 'x_pca', 'umap', 'tsne', 'pca'],
        'qc_keys': ['n_genes', 'n_counts', 'percent_mito', 'pct_mito']
    }
    
    # Map of key types to their H5 groups
    key_layers = {
        'cell_type_keys': 'obs',
        'gene_keys': 'var',
        'batch_keys': 'obs',
        'hvg_keys': 'var',
        'embedding_keys': 'obsm',
        'qc_keys': 'obs'
    }

    # Initialize results as a dictionary
    results = { 
        'file_name': Path(file_path).name.split('.')[0],
        'file_path': str(file_path),
        'dimensions': {},
        'file_layers': [],
        # Initialize results with simple lists instead of complex structures
        'found_keys_by_category': {cat: [] for cat in key_categories},
        'obs_keys': [],
        'var_keys': [],
        'obsm_keys': [],
        'varm_keys': [],
        'uns_keys': [],
        'obsp_keys': [],
        'varp_keys': [],
        'found_obs_keys': [],
        'found_var_keys': []
    }

    # Load the data
    try:
        logger.info(f"Loading data from {file_path}")
        with h5py.File(file_path, 'r') as f:
            # Find all available layers
            available_layers = []
            for layer in ['obs', 'var', 'uns', 'obsm', 'varm', 'obsp', 'varp']:
                if layer in f:
                    available_layers.append(layer)
                    # Store all keys from this layer
                    results[f'{layer}_keys'] = list(f[layer].keys())
            
            results['file_layers'] = available_layers
            
            # Create lookup dictionaries for case-insensitive matching
            layer_keys = {}
            for layer in available_layers:
                # Create a mapping from lowercase key to original key
                layer_keys[layer] = {k.lower(): k for k in f[layer].keys()}
            
            # Find keys by category
            for category, patterns in key_categories.items():
                layer = key_layers[category]
                if layer not in layer_keys:
                    continue
                
                lowercase_patterns = [p.lower() for p in patterns]
                
                for lowercase_key, original_key in layer_keys[layer].items():
                    if any(pattern in lowercase_key for pattern in lowercase_patterns):
                        # Simply add the name to our list
                        results['found_keys_by_category'][category].append(original_key)
                        
                        # Also store key metadata in a separate structure if needed
                        key_info = {
                            'name': original_key,
                            'layer': layer
                        }
                        
                        # Extract additional info if in obs or var
                        if layer in ['obs', 'var']:
                            item = f[layer][original_key]
                            if layer == 'obs':
                                if original_key not in results['found_obs_keys']:
                                    results['found_obs_keys'].append(original_key)
                            else:  # layer == 'var'
                                if original_key not in results['found_var_keys']:
                                    results['found_var_keys'].append(original_key)
                            if isinstance(item, h5py.Dataset):
                                key_info['dtype'] = str(item.dtype)
                                key_info['is_categorical'] = False
                                # Try to get sample values
                                try:
                                    key_info['sample_values'] = item[:5].tolist()
                                except:
                                    pass
                            elif isinstance(item, h5py.Group) and 'categories' in item:
                                key_info['dtype'] = 'categorical'
                                key_info['is_categorical'] = True
                                key_info['n_categories'] = len(item['categories'])
                                try:
                                    categories = [cat.decode('utf-8') if isinstance(cat, bytes) else cat 
                                                for cat in item['categories'][:min(5, len(item['categories']))]]
                                    key_info['sample_categories'] = categories
                                except:
                                    pass
                        
                        # Store detailed info in a separate structure
                        if 'key_details' not in results:
                            results['key_details'] = {}
                        results['key_details'][original_key] = key_info
        
        # Find dimensions
        if 'X' in f:
            logger.debug("Processing expression matrix dimensions")
            if isinstance(f['X'], h5py.Group) and 'shape' in f['X'].attrs:
                shape = f['X'].attrs['shape']
                results['dimensions']['n_cells'] = int(shape[0])
                results['dimensions']['n_genes'] = int(shape[1])
                logger.info(f"Found sparse matrix with {results['dimensions']['n_cells']} cells and {results['dimensions']['n_genes']} genes")
                if 'data' in f['X']:
                    results['dimensions']['n_nonzero'] = len(f['X']['data'])
                    sparsity = 100 * (1 - (results['dimensions']['n_nonzero'] / 
                                        (results['dimensions']['n_cells'] * results['dimensions']['n_genes'])))
                    results['dimensions']['sparsity_percent'] = round(sparsity, 2)
                    logger.info(f"Matrix sparsity: {results['dimensions']['sparsity_percent']}%")
            elif isinstance(f['X'], h5py.Dataset):
                shape = f['X'].shape
                results['dimensions']['n_cells'] = int(shape[0])
                results['dimensions']['n_genes'] = int(shape[1])
                logger.info(f"Found dense matrix with {results['dimensions']['n_cells']} cells and {results['dimensions']['n_genes']} genes")
            
        # Log summary information 
        logger.info(f"Found {len(results['file_layers'])} file layers")
        for category, keys in results['found_keys_by_category'].items():
            if keys:  # Only log non-empty categories
                logger.info(f"Found {len(keys)} {category}: {', '.join(keys[:5])}" + 
                           ("..." if len(keys) > 5 else ""))
                
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        results['error'] = f"File not found: {file_path}"
        return results
    except OSError as e:
        logger.error(f"Error opening file {file_path}: {str(e)}")
        results['error'] = f"Error opening file: {str(e)}"
        return results
    except Exception as e:
        logger.error(f"Unexpected error analyzing {file_path}: {str(e)}", exc_info=True)
        results['error'] = f"Unexpected error: {str(e)}"
        return results
    
    # Save results to JSON if requested
    if save:
        import json
        try:
            # Determine save location
            if output_dir:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                save_path = Path(output_dir) / f"{results['file_name']}_metadata.json"
            else:
                save_path = f"{results['file_name']}_metadata.json"
                
            logger.info(f"Saving metadata to {save_path}")
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=4, default=str)
            logger.info(f"Metadata saved successfully to {save_path}")
        except PermissionError:
            logger.error(f"Permission denied when saving to {save_path}. Check file permissions.")
        except IOError as e:
            logger.error(f"I/O error when saving metadata: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error when saving metadata: {str(e)}", exc_info=True)
            
    logger.info("Metadata analysis completed")
    return results

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
    #print(f"Analysis saved to: {output}")
    print("="*80)



# === MAIN FUNCTION ===

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='scGPT data loading and metadata extraction')
    parser.add_argument('--query_file', type=str, default='data/7af3a87a-c148-4988-a7cd-f33666ffd883.h5ad', 
                       help='Path to input h5ad file')
    parser.add_argument('--force_reload', action='store_true', 
                       help='Force reload data and ignore cache')
    parser.add_argument('--testing', action='store_true', default=False, 
                       help='Use test output directory')                                 #change this one to True for testing, or in terminal
    parser.add_argument('--output_mode', type=str, choices=['terminal', 'file', 'both'], #terminal: output to terminal, file: output to file, both: output to both
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
    from utils import setup_directories
    repo_dir, data_dir, save_dir, output_dir, model_dir, directories = setup_directories()
    
    # Add repo to path if needed
    if str(repo_dir) not in sys.path:
        sys.path.append(str(repo_dir))
    
    if args.testing:
        output_dir = save_dir / "test_output" #you could also change this name if you like 
    else:
        # Create output directory
        base_name = Path(args.input_file).stem
        date_str = datetime.now().strftime("%Y%m%d")
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
    
    # Capture annotation analysis output
    output_buffer = io.StringIO()
    with redirect_stdout(output_buffer):
        found_keys = check_annotation_keys(adata)
         
        # Additional verbose information if requested
        if args.verbose:
            print("\n----- ADDITIONAL DATASET DETAILS ----- (Not complete yet, can be added! )")
            if 'n_genes' not in adata.obs and 'n_counts' not in adata.obs:
                print("Computing basic QC metrics...")
                # Compute some basic stats
                sc.pp.calculate_qc_metrics(adata, inplace=True)

            #this needs to be fixed 
            if 'n_genes' in adata.obs:
                print(f"Genes per cell: min={adata.obs.n_genes.min()}, "
                      f"median={adata.obs.n_genes.median()}, "
                      f"max={adata.obs.n_genes.max()}")
                
            
            #this too     
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
            from utils import generate_basic_html_report #import the custom html report function from utils.py
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

    # Save found keys as JSON metadata
    metadata_file = output_dir / f"{base_name}_metadata.json"
    found_keys['directories'] = directories
    save_metadata_json(found_keys, metadata_file)
    # Print summary
    print_summary(adata, found_keys, output_dir, metadata_file, output)
    
    return adata, found_keys

if __name__ == '__main__':
    main()