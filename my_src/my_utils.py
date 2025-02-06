from pathlib import Path
from shutil import move
import pickle
import requests
import zipfile
import gzip
import scanpy as sc
import pandas as pd
import numpy as np
import anndata as ad
import scipy
import scipy.sparse

def check_annotation_keys(adata, verbose=True):
    """
    Check for common variations of cell type, gene, and HVG annotations in AnnData object.
    Shows multiple examples and counts for each found key if verbose=True.
    Provides feedback when no known keys are found.
    

    Args:
        adata: AnnData object
    
    Returns:
        dict: Found keys for cell types, gene names, and HVG
    """
    # Common variations of cell type labels in obs
    cell_type_keys = [
        'cell_type', 'celltype', 'CellType',
        'cell.type', 'cell_type_label',
        'cell_ontology_class',
        'cluster', 'cluster_label',
        'louvain', 'leiden',
        'annotation', 'cell_annotation',
        'predicted_celltype',
        'cell_identity', 'cell_id',
        'type', 'subtype'
    ]
    
    # Common variations of gene names in var
    gene_keys = [
        'gene_name', 'gene_names',
        'gene_symbol', 'symbol',
        'gene_id', 'gene_short_name',
        'ensembl_id', 'ensembl',
        'gene', 'genes',
        'feature_name', 'feature_id',
        'name', 'index'
    ]
    
    # Common variations of HVG annotations
    hvg_keys = [
        'highly_variable', 'highly_variable_genes',
        'hvg', 'HVG', 
        'variable_genes', 'variable_features',
        'highly_variable_rank', 'highly_variable_scores',
        'dispersions_norm', 'dispersions'
    ]
    
    # Rest of your code remains the same...
    
    found_keys = {
        'cell_type_keys': [],
        'gene_keys': [],
        'hvg_keys': []
    }
    
    def get_examples_and_counts(series, n=5):
        """Get n unique examples and value counts from a series"""
        if hasattr(series, 'unique'):
            examples = series.unique()[:n]
            counts = series.value_counts()
            n_unique = len(series.unique())
            return examples, counts, n_unique
        return series.iloc[:n], None, len(series.unique())
    
    # Check obs columns for cell type keys
    print("\nChecking cell type annotations in .obs:")
    for key in cell_type_keys:
        if key in adata.obs.columns:
            examples, counts, n_unique = get_examples_and_counts(adata.obs[key])
            print(f"\n✓ Found '{key}' with {n_unique} unique values")
            print(f"Examples:")
            for i, ex in enumerate(examples, 1):
                print(f"  {i}. {ex}")
            print(f"Top counts:")
            for label, count in counts.head().items():
                print(f"  {label}: {count} cells")
            found_keys['cell_type_keys'].append(key)
    
    if not found_keys['cell_type_keys']:
        print("\n⚠️ No well-known cell type key found in .obs")
        print("Available .obs columns:")
        for col in adata.obs.columns:
            print(f"  - {col}")
    
    # Check var columns and index for gene keys
    print("\nChecking gene annotations in .var:")
    for key in gene_keys:
        if key in adata.var.columns:
            examples, counts, n_unique = get_examples_and_counts(adata.var[key])
            print(f"\n✓ Found '{key}' with {n_unique} unique values")
            print(f"Examples:")
            for i, ex in enumerate(examples, 1):
                print(f"  {i}. {ex}")
            found_keys['gene_keys'].append(key)
        elif adata.var.index.name == key:
            examples, counts, n_unique = get_examples_and_counts(adata.var.index)
            print(f"\n✓ Found '{key}' as index with {n_unique} unique values")
            print(f"Examples:")
            for i, ex in enumerate(examples, 1):
                print(f"  {i}. {ex}")
            found_keys['gene_keys'].append(key)
    
    if not found_keys['gene_keys']:
        print("\n⚠️ No well-known gene key found in .var")
        print("Available .var columns:")
        for col in adata.var.columns:
            print(f"  - {col}")
        print("\nVar index name:", adata.var.index.name or "None")
    
    # Check for HVG annotations
    print("\nChecking highly variable gene annotations in .var:")
    hvg_found = False
    for key in hvg_keys:
        if key in adata.var.columns:
            hvg_found = True
            if adata.var[key].dtype == bool:
                n_hvg = adata.var[key].sum()
                total = len(adata.var[key])
                print(f"\n✓ Found '{key}' (boolean) with {n_hvg}/{total} genes marked as highly variable")
                print(f"Percentage: {(n_hvg/total)*100:.2f}%")
            else:
                examples, counts, n_unique = get_examples_and_counts(adata.var[key])
                print(f"\n✓ Found '{key}' with {n_unique} unique values")
                print(f"Examples:")
                for i, ex in enumerate(examples, 1):
                    print(f"  {i}. {ex}")
            found_keys['hvg_keys'].append(key)
    
    if not hvg_found:
        print("\n⚠️ No HVG annotations found in .var columns")
    
    # Create a DataFrame to display cell type keys side by side
    if len(found_keys['cell_type_keys']) > 1:
        print("\nComparison of cell type annotations:")
        cell_type_dfs = []
        for key in found_keys['cell_type_keys']:
            value_counts = adata.obs[key].value_counts()
            df = pd.DataFrame({
                key: value_counts.values,
                f'{key}_count': value_counts.index
            })
            cell_type_dfs.append(df)
        
        comparison_df = pd.concat(cell_type_dfs, axis=1)
        print(comparison_df)
    
    # Additional metadata checks
    print("\nChecking additional metadata:")
    if hasattr(adata, 'uns'):
        print("Available .uns keys:", list(adata.uns.keys()))
    if hasattr(adata, 'layers'):
        print("Available .layers:", list(adata.layers.keys()))
    
    return found_keys