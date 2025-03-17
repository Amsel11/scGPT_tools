#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data classification and annotation with scGPT model. 

This script:
1. Loads AnnData (h5ad) files with embeddings 
2. Classifies the data using different classifier methods
3. Extracts metadata needed for scGPT embedding
4. Saves the  
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

#for classification 
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import StratifiedGroupKFold

# Add conditional import for LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available. Install with: pip install lightgbm")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Faiss not available. Install with: pip install faiss-cpu")

#class for the scGPT classifier 
class scGPTAnnotator: 
    # initialize the classifier
    def __init__(self, query_adata = None, ref_adata = None, embedding_key = 'X_scGPT'):
        self.query_adata = query_adata
        self.ref_adata = ref_adata
        self.embedding_key = embedding_key 
        self.classifier = None #the type of  classifier
        self.trained = False #start with an untrained classifier
    
    def init_classifier(self, classifier_name, **kwargs):
        """Initialize a classifier by name."""
        print(f"Initializing classifier: {classifier_name}")
        classifier_name = classifier_name.lower()
        
        # Map classifier names to their constructors
        classifier_map = {
            'randomforest': RandomForestClassifier,
            'randomforestclassifier': RandomForestClassifier,
            'knn': KNeighborsClassifier,
            'kneighborsclassifier': KNeighborsClassifier,
        }
        
        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            classifier_map['lightgbm'] = lgb.LGBMClassifier
        
        # Get the constructor or default to RandomForest
        classifier_class = classifier_map.get(classifier_name, RandomForestClassifier)
        
        # Print message if using default
        if classifier_name not in classifier_map:
            print(f"Unknown classifier: {classifier_name}, using RandomForest instead")
        elif classifier_name == 'lightgbm' and not LIGHTGBM_AVAILABLE:
            print("LightGBM not available. Using RandomForest instead.")
        
        # Show the selected parameters
        print(f"Classifier parameters: {kwargs}")
        
        # Set the classifier attribute
        self.classifier = classifier_class(**kwargs)
        print(f"Initialized {type(self.classifier).__name__}")
        return self.classifier
    
    def check_embeddings(self,adata):
        try: 
            with h5py.File(adata, 'r') as f:
                if 'obsm' not in f:
                    return False, "No 'obsm' group found in file {adata}" #Returns boolean value and error message to be printed 
                
                if 'X_scGPT' not in f['obsm']:  
                    return False, "No embeddings found in adata.obsm['X_scGPT']"
                
                return True, f"Found embeddings with shape: {f['obsm']['X_scGPT'].shape}"
            
        except Exception as e:
            return False, f"Error checking for embeddings: {e}"
    
    def set_query_data(self, adata):
        self.query_adata = adata

    def set_ref_data(self, adata):
        self.ref_adata = adata

    # Improved Split Query/Reference Function
    def split_query_ref(self, adata, method='batch', batch_key='batch', test_size=0.8, 
                       fold_idx=0, n_splits=5, random_state=42, verbose=True):
        """
        Split a single AnnData object into query and reference datasets.
        
        Args:
            adata: AnnData object to split
            method: Splitting method ('batch', 'kfold', or 'random')
            batch_key: Column in adata.obs containing batch information
            test_size: Proportion of data to use as query (for 'batch' and 'random' methods)
            fold_idx: Which fold to use as query (for 'kfold' method)
            n_splits: Number of folds to create (for 'kfold' method)
            random_state: Random seed for reproducibility
            verbose: Whether to print information about the split
            
        Returns:
            self: For method chaining
        """
        # Verify batch_key exists if needed
        if method in ['batch', 'kfold']:
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
            self.ref_adata = adata[train_idx].copy()
            self.query_adata = adata[test_idx].copy()
            
            # Report the batches in each set
            query_batches = self.query_adata.obs[batch_key].unique()
            ref_batches = self.ref_adata.obs[batch_key].unique()
            
            if verbose:
                print(f"Reference set: {len(self.ref_adata)} cells from {len(ref_batches)} batches")
                print(f"Query set: {len(self.query_adata)} cells from {len(query_batches)} batches")
                print(f"Query batches: {', '.join(map(str, query_batches))}")
        
        elif method == 'kfold':
            # Split using stratified k-fold cross-validation
            from sklearn.model_selection import StratifiedGroupKFold
            
            # Ensure fold_idx is valid
            if fold_idx < 0 or fold_idx >= n_splits:
                raise ValueError(f"fold_idx must be between 0 and {n_splits-1}")
            
            # Get a stratification variable if available
            if 'cell_type' in adata.obs:
                strat_var = adata.obs['cell_type']
            else:
                # Use batch as stratification if no cell type
                strat_var = adata.obs[batch_key]
            
            # Create the k-fold splitter
            kfold = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            
            # Get the specified fold
            for i, (train_idx, test_idx) in enumerate(
                kfold.split(adata.X, strat_var, groups=adata.obs[batch_key])
            ):
                if i == fold_idx:
                    self.ref_adata = adata[train_idx].copy()
                    self.query_adata = adata[test_idx].copy()
                    break
            
            # Report the batches in each set
            query_batches = self.query_adata.obs[batch_key].unique()
            ref_batches = self.ref_adata.obs[batch_key].unique()
            
            if verbose:
                print(f"Using fold {fold_idx} of {n_splits}")
                print(f"Reference set: {len(self.ref_adata)} cells from {len(ref_batches)} batches")
                print(f"Query set: {len(self.query_adata)} cells from {len(query_batches)} batches")
                print(f"Query batches: {', '.join(map(str, query_batches))}")
        
        elif method == 'random':
            # Simple random split ignoring batches
            from sklearn.model_selection import train_test_split
            
            train_idx, test_idx = train_test_split(
                range(adata.n_obs), 
                test_size=test_size,
                random_state=random_state
            )
            
            self.ref_adata = adata[train_idx].copy()
            self.query_adata = adata[test_idx].copy()
            
            if verbose:
                print(f"Random split: {len(self.ref_adata)} reference cells, {len(self.query_adata)} query cells")
        
        else:
            raise ValueError(f"Unknown split method: {method}. Use 'batch', 'kfold', or 'random'")
        
        # Verify the embedding key exists in both split datasets
        for dataset_name, dataset in [("reference", self.ref_adata), ("query", self.query_adata)]:
            if self.embedding_key not in dataset.obsm:
                print(f"Warning: Embedding key '{self.embedding_key}' not found in {dataset_name} dataset")
        
        return self

    # train the classifier
    def train_classifier(self, classifier_name, cell_type_col = 'cell_type', batch_key = None, **kwargs):
        """Train a classifier on reference data."""
        print(f"\n==== STARTING CLASSIFIER TRAINING ====")
        print(f"Classifier: {classifier_name}")
        print(f"Cell type column: {cell_type_col}")
        print(f"Batch key: {batch_key}")
        
        # Initialize the classifier
        print(f"Initializing classifier...")
        self.classifier = self.init_classifier(classifier_name, **kwargs)
        print(f"Initialized {type(self.classifier).__name__} classifier")
        
        # Need reference data (can be done by splitting the query adata)
        if self.ref_adata is None:
            print("ERROR: Reference data not set")
            raise ValueError("Reference data not set. Please call set_ref_data() first.")
        
        print(f"Reference data shape: {self.ref_adata.shape}")
        
        # Check the column (this could go in the general check function)
        if cell_type_col not in self.ref_adata.obs.columns:
            print(f"ERROR: Cell type column '{cell_type_col}' not found in reference data")
            print(f"Available columns: {list(self.ref_adata.obs.columns)}")
            raise ValueError(f"Cell type {cell_type_col} not found in reference data.")
        
        # Extract features and labels
        print(f"Extracting features from {self.embedding_key}...")
        try:
            X_train = pd.DataFrame(self.ref_adata.obsm[self.embedding_key])
            print(f"Feature matrix shape: {X_train.shape}")
        except KeyError:
            print(f"ERROR: Embedding key '{self.embedding_key}' not found in reference data")
            print(f"Available embeddings: {list(self.ref_adata.obsm.keys())}")
            raise
        
        print(f"Extracting labels from {cell_type_col}...")
        y_train = self.ref_adata.obs[cell_type_col]
        unique_labels = y_train.unique()
        print(f"Found {len(unique_labels)} unique cell types: {unique_labels[:5]}{'...' if len(unique_labels) > 5 else ''}")
        
        # Track metadata for later use
        self.cell_type_col = cell_type_col
        self.classifier_type = classifier_name
        
        # If batch_key is provided, use batch-aware training
        if batch_key and batch_key in self.ref_adata.obs:
            unique_batches = self.ref_adata.obs[batch_key].unique()
            n_batches = len(unique_batches)
            
            print(f"Batch information: {n_batches} unique batches found")
            print(f"Batches: {unique_batches}")
            
            if n_batches < 2:
                print(f"Warning: Only {n_batches} batch found. Switching to standard training.")
                print(f"Fitting classifier on {len(X_train)} samples...")
                self.classifier.fit(X_train, y_train)
                print(f"Classifier training complete")
            else:
                print(f"Using batch-aware training with key: {batch_key} ({n_batches} batches)")
                from sklearn.model_selection import GroupKFold
                
                # Use GroupKFold to ensure batch separation in cross-validation
                n_splits = min(5, n_batches)
                print(f"Setting up {n_splits}-fold cross-validation...")
                gkf = GroupKFold(n_splits=n_splits)
                groups = self.ref_adata.obs[batch_key]
                
                # Initialize metrics if not already present
                if not hasattr(self, 'metrics'):
                    self.metrics = {}
                
                # Compute cross-validated metrics
                print(f"Running cross-validation...")
                from sklearn.model_selection import cross_validate
                cv_results = cross_validate(
                    self.classifier, X_train, y_train, 
                    groups=groups,
                    scoring=['accuracy', 'f1_weighted'],
                    cv=gkf,
                    return_estimator=True,
                    verbose=1  # Add verbosity to see progress
                )
                
                # Store metrics
                self.metrics['cv_accuracy'] = cv_results['test_accuracy'].mean()
                self.metrics['cv_f1'] = cv_results['test_f1_weighted'].mean()
                
                print(f"Cross-validation results:")
                print(f"  Mean accuracy: {self.metrics['cv_accuracy']:.4f}")
                print(f"  Mean F1 score: {self.metrics['cv_f1']:.4f}")
                print(f"  Individual fold accuracies: {cv_results['test_accuracy']}")
                
                # Use the best estimator
                best_idx = np.argmax(cv_results['test_f1_weighted'])
                self.classifier = cv_results['estimator'][best_idx]
                print(f"Selected best classifier from fold {best_idx+1} with F1: {cv_results['test_f1_weighted'][best_idx]:.4f}")
        else:
            # Standard training
            print(f"No batch information provided or found. Using standard training.")
            print(f"Fitting classifier on {len(X_train)} samples...")
            self.classifier.fit(X_train, y_train)
            print(f"Classifier training complete")
        
        # Also add prints to init_classifier
        self.trained = True
        print(f"==== CLASSIFIER TRAINING FINISHED ====\n")
        return self.classifier

    
    def predict(self, adata, pred_cell_col = 'pred_cell_type', store_probs = True, return_adata = False): 
        """Predict cell types for query data."""
        print(f"\n==== STARTING PREDICTION ====")
        
        if not self.trained: 
            print("ERROR: Classifier not trained")
            raise ValueError("Classifier must be trained before prediction")
        
        print(f"Predicting with {self.classifier_type} classifier")
        
        pred_adata = adata if adata is not None else self.query_adata
        print(f"Prediction data shape: {pred_adata.shape}")

        # Extract features that we want to predict on
        print(f"Extracting features from {self.embedding_key}...")
        try:
            X_pred = pd.DataFrame(pred_adata.obsm[self.embedding_key])
            print(f"Feature matrix shape: {X_pred.shape}")
        except KeyError:
            print(f"ERROR: Embedding key '{self.embedding_key}' not found in prediction data")
            print(f"Available embeddings: {list(pred_adata.obsm.keys())}")
            raise
        
        # Make predictions using the trained classifier
        print(f"Making predictions on {len(X_pred)} cells...")
        y_pred = self.classifier.predict(X_pred)
        
        # Add the predictions to the AnnData object
        pred_adata.obs[pred_cell_col] = y_pred
        print(f"Predictions stored in .obs['{pred_cell_col}']")
        
        # Prediction probabilities
        if store_probs and hasattr(self.classifier, 'predict_proba'):
            print(f"Computing prediction probabilities...")
            probs = self.classifier.predict_proba(X_pred)
            class_names = list(self.classifier.classes_)
            print(f"Storing probabilities for {len(class_names)} cell types")
            
            # Store probabilities for each class
            for i, cell_type in enumerate(class_names):
                pred_adata.obs[f"{pred_cell_col}_prob_{cell_type}"] = probs[:, i]
            
            print(f"Probability columns added with prefix '{pred_cell_col}_prob_'")
        
        print(f"==== PREDICTION COMPLETE ====\n")
        
        if return_adata:
            return pred_adata
        else:
            return y_pred


    #run the whole thing 
    def evaluate(self, adata): 
        return adata
    



#setup directories 
def setup_directories():
    """Set up necessary directories and return their paths
    
    TO BE ADDED:
    - Absolute paths if needed 
    - inputs for if specified paths are needed (either from config or from command line) 
    """
    repo_dir = Path.cwd().absolute()
    data_dir = repo_dir / "data"
    save_dir = repo_dir / "save"
    
    data_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    return repo_dir, data_dir, save_dir


#load in data

#ensure that there are embeddings -> in main? 
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


#save the results 

#generate a report 

#generate visualisations 
    #dataframe of embeddings
    #scatter plot of embeddings
    #PCA plot of embeddings
    #t-SNE plot of embeddings
    #UMAP plot of embeddings
    #heatmap of embeddings
    #correlation matrix of embeddings
    #heatmap of correlation matrix

#Benchmarking 
#   - compare with other classifiers
#   - compare with other clustering methods
#   - compare with other dimensionality reduction methods


# === MAIN FUNCTION === # 


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='scGPT data loading and metadata extraction')
    parser.add_argument('--query_file', type=str, default='data/Derived_Embryoid_Bodies_all_embeds.h5ad', 
                       help='Path to input h5ad file')
    parser.add_argument('--ref_file', type=str, default=None,
                       help='Path to input h5ad file (defaults to query_file if not specified)')
    parser.add_argument('--force_reload', action='store_true', 
                       help='Force reload data and ignore cache')
    parser.add_argument('--testing', action='store_true', default=True, 
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
    
    print("Running scGPT classifier...")
    
    # Setup directories
    repo_dir, data_dir, save_dir = setup_directories()
    
    # Add repo to path if needed
    if str(repo_dir) not in sys.path:
        sys.path.append(str(repo_dir))
    
    # Create output directory
    base_name = Path(args.query_file).stem
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

    #load metadata
    metadata_path = output_dir / f"Derived_Embryoid_Bodies_metadata.json" #change this back this is just for testing
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    annotator = scGPTAnnotator(embedding_key='X_scGPT') #initialize the annotator
    has_query_embeddings, query_message = annotator.check_embeddings(args.query_file) #check if the query data has embeddings
    if not has_query_embeddings:
        print(f"Error: {query_message}")
        print("Cannot proceed without embeddings in query file.")
        sys.exit(1)

    print(f"Query file: {query_message}")
    query_adata = load_h5ad(args.query_file, save_dir=output_dir, subset=subset, force_reload=args.force_reload) #load the query data
    print(query_adata)
    annotator.set_query_data(query_adata) #set the query data

    # Get cell type and batch keys from metadata
    cell_type_key = metadata['cell_type_keys'][0] if metadata.get('cell_type_keys') else None #to make sure it doesn't error
    batch_key = metadata['batch_keys'][1] if metadata.get('batch_keys') else None #to make sure it doesn't error if it's not there 

    print(f"Using cell type key: {cell_type_key}")
    print(f"Using batch key: {batch_key}")

    # Handle reference data - either from file or by splitting query
    if args.ref_file is not None:
        # External reference file provided
        has_ref_embeddings, ref_message = annotator.check_embeddings(args.ref_file)
        if not has_ref_embeddings:
            print(f"Warning: Reference file doesn't have embeddings: {ref_message}")
            print("Falling back to split from query data.")
            
            # Split query using batch key from metadata
            if batch_key and batch_key in query_adata.obs:
                print(f"Splitting query data using {batch_key} information")
                print(f"Available batches: {query_adata.obs[batch_key].unique()}")
                annotator.split_query_ref(query_adata, method='batch', batch_key=batch_key)
            else:
                print("No valid batch key. Using random split.")
                annotator.split_query_ref(query_adata, method='random')
        else:
            # Reference file has embeddings, use it
            print(f"Reference file: {ref_message}")
            ref_adata = load_h5ad(args.ref_file, save_dir=output_dir, subset=subset, force_reload=args.force_reload)
            annotator.set_ref_data(ref_adata)
    else:
        # No reference file provided, split query data
        print("No reference file provided. Creating reference from query data.")
        #there needs to be an extra check that the query data has cell_type information 
        
        # Use batch key from metadata
        if batch_key and batch_key in query_adata.obs:
            if len(query_adata.obs[batch_key].unique()) > 1:
                print(f"Splitting query data using {batch_key} information")
                print(f"Available batches: {query_adata.obs[batch_key].unique()}")
                annotator.split_query_ref(query_adata, method='batch', batch_key=batch_key)
            else:
                print(f"Only one {batch_key} found. Using random split.")
                annotator.split_query_ref(query_adata, method='random')
        else:
            print("No valid batch key in metadata. Using random split.")
            annotator.split_query_ref(query_adata, method='random')

    # Verify we have what we need
    if annotator.query_adata is None:
        print("Error: No query data available.")
        sys.exit(1)
    if annotator.ref_adata is None:
        print("Error: No reference data available. Cannot proceed with classification.")
        sys.exit(1)

    print(f"Ready for cell type annotation using {cell_type_key} with {len(annotator.ref_adata)} reference cells and {len(annotator.query_adata)} query cells")
    #name for predicted cell type column
    pred_cell_type_key = 'pred_cell_type'
    
    annotator.train_classifier(classifier_name='randomforest', cell_type_col=cell_type_key, batch_key=batch_key)
    predicted_adata = annotator.predict(annotator.query_adata, pred_cell_col=pred_cell_type_key, store_probs=True, return_adata=True)
    print(predicted_adata.obs[[cell_type_key, pred_cell_type_key]].head())

    results_path = output_dir / f"Derived_Embryoid_Bodies_pred_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5ad"
    predicted_adata.write_h5ad(results_path)
    print(f"Predicted results saved to {results_path}")




if __name__ == "__main__":
    main()




