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
import matplotlib.pyplot as plt
import seaborn as sns

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

##=== CLASSIFIER ===# 

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
    
    def set_query_data(self, adata): #this is the data that we are using to predict on. When not provided, we can set it here 
        self.query_adata = adata

    def set_ref_data(self, adata): #this is the data that we are using to train on. When not provided, we can set it here. Can come from split query data
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
            from sklearn.model_selection import GroupShuffleSplit #only import when this is needed 
            
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
        
        elif method == 'unannotated':
            self.ref_adata = adata[~adata.obs['cell_type'].isin(["", "to be determined", "NaN", np.nan])].copy() # the ~makes it the opposite 
            self.query_adata = adata[adata.obs['cell_type'].isin(["", "to be determined", "NaN", np.nan])].copy()

            if verbose:
                print(f"Unannotated split: {len(self.ref_adata)} reference cells, {len(self.query_adata)} query cells")
                print(f"There are {len(self.query_adata)} cells with missing/invalid cell type annotations in the query set, and {len(self.ref_adata)} cells with valid cell type annotations in the reference set")
        
        else:
            raise ValueError(f"Unknown split method: {method}. Use 'batch', 'kfold', or 'random'")
        
        # Verify the embedding key exists in both split datasets
        for dataset_name, dataset in [("reference", self.ref_adata), ("query", self.query_adata)]:
            if self.embedding_key not in dataset.obsm:
                print(f"Warning: Embedding key '{self.embedding_key}' not found in {dataset_name} dataset")
        
        return self

    # train the classifier
    def train_classifier(self, classifier_name='randomforest', cell_type_col=None, batch_key=None, save_path=None):
        """Train a classifier using reference data embeddings"""
        print(f"\n==== STARTING CLASSIFIER TRAINING ====")
        print(f"Classifier: {classifier_name}")
        print(f"Cell type column: {cell_type_col}")
        print(f"Batch key: {batch_key}")
        
        # Initialize the classifier we will use: 
        print(f"Initializing classifier...")
        self.classifier = self.init_classifier(classifier_name)
        print(f"Initialized {type(self.classifier).__name__} classifier")
        
        # Need reference data, if not provided, we split the query data based on the provided splitting method
        if self.ref_adata is None:
            print("ERROR: Reference data not set")
            raise ValueError("Reference data not set. Please call set_ref_data() first.")
        
        print(f"Reference data shape: {self.ref_adata.shape}")
        
        # Check the column (this could go in the general check function) --> yeah move his ass to the check function, at least for the query data 
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
        self.cell_type_col = cell_type_col #this is the column that we are using to predict on, and was extracted from the dataloader
        self.classifier_type = classifier_name  #this is the type of classifier we are using
        
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
        
        # Save classifier if a path is provided
        if save_path is not None:
            print(f"Saving classifier to {save_path}")
            self.save_classifier(save_path)
        
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
    
    def save_classifier(self, output_path):
        """Save the trained classifier to disk."""
        if not self.trained or not hasattr(self, 'classifier'):
            print("ERROR: No trained classifier to save")
            return False
        
        import pickle
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the classifier
        with open(output_path, 'wb') as f:
            pickle.dump(self.classifier, f)
        print(f"Classifier saved to {output_path}")
        return True

    def load_classifier(self, model_path):
        """Load a previously trained classifier."""
        import pickle
        
        try:
            with open(model_path, 'rb') as f:
                self.classifier = pickle.load(f)
            self.trained = True
            print(f"Loaded classifier from {model_path}")
            return True
        except Exception as e:
            print(f"ERROR: Failed to load classifier: {e}")
            return False

    def evaluate_with_visuals(self, adata=None, y_pred=None, y_true=None, valid_classes=None, 
                         cell_type_col=None, pred_cell_col='pred_cell_type', 
                         figsize=(12, 10), cmap='viridis'):

        from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
        
        # Get the data we need depending on our inputs
        if y_true is None or y_pred is None:
            if adata is None:
                adata = self.query_adata
            
            # Use provided column names or fall back to stored values
            y_true_col = cell_type_col if cell_type_col is not None else self.cell_type_col
            y_true = adata.obs[y_true_col]
            y_pred = adata.obs[pred_cell_col]
        
        # Convert to pandas Series for consistent handling
        y_true = pd.Series(y_true)
        y_pred = pd.Series(y_pred)
        valid_idx = ~(y_true.isna() | y_pred.isna())
        y_true = y_true[valid_idx]
        y_pred = y_pred[valid_idx]
        
        if valid_classes is None:
            valid_classes = sorted(set(y_true) & set(y_pred))
        
        # Calculate metrics using the sklearn metrics library: 
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro',labels=valid_classes, zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', labels=valid_classes, zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', labels=valid_classes, zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', labels=valid_classes, zero_division=0)
        
        # Printsss
        print(f"\nEvaluation Results:")
        print(f"Samples: {len(y_true)}")
        print(f"Classes: {len(valid_classes)}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1: {f1_macro:.4f}")
        print(f"Weighted F1: {f1_weighted:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        
        
        adata.uns["classifier_metrics"] = {
            'accuracy': float(accuracy),
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted),
            'precision': float(precision),
            'recall': float(recall),
            'num_classes': len(valid_classes),
            'n_samples': len(y_true)
        }
        print(f"Metrics stored in adata.uns['classifier_metrics']")
        

        # visualisations down here:  This should be split later ! 
        #1. confusion matrix
        plt.figure(figsize=figsize)
        cm = confusion_matrix(y_true, y_pred, labels=valid_classes)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        #customisable if needed
        ax = sns.heatmap(cm_normalized, annot=False, cmap=cmap, xticklabels=valid_classes, yticklabels=valid_classes)
        plt.title("Normalized Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
        
        #2. Create UMAP visualization if available
        if 'X_umap' in adata.obsm: #(it is for this one, otherwise we need to create it)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            
            # Plot true labels
            scatter1 = ax1.scatter(adata.obsm['X_umap'][:, 0], adata.obsm['X_umap'][:, 1], 
                    c=adata.obs[cell_type_col].astype('category').cat.codes, 
                    cmap=cmap, s=1, alpha=0.7)
            ax1.set_title(f"UMAP - True Cell Types ({cell_type_col})")
            
            # Plot predicted labels
            scatter2 = ax2.scatter(adata.obsm['X_umap'][:, 0], adata.obsm['X_umap'][:, 1], 
                    c=adata.obs[pred_cell_col].astype('category').cat.codes, 
                    cmap=cmap, s=1, alpha=0.7)
            ax2.set_title(f"UMAP - Predicted Cell Types ({pred_cell_col})")
            
            plt.tight_layout()
            plt.show()
        else:
            print("UMAP embeddings not found in adata.obsm['X_umap']. Skipping UMAP visualization.")
        
        #final results
        results = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision': precision,
            'recall': recall,
            'num_classes': len(valid_classes),
            'n_samples': len(y_true)
        }
        
        return valid_classes, adata, results

    #save the results
    def save_results(self, adata, results_path = None):
        """Save results to a file."""
        if adata is None:
            print("ERROR: No data to save")
            return
        adata.write_h5ad(results_path)
        print(f"Results saved to {results_path}")

    def add_top_n_predictions(self, adata, n=3, pred_cell_col='pred_cell_type', cell_type_col='cell_type'):
        """show of the top n (default 3) the prediction scores of the classifier for choosing the cell type, for better comparison
        and interpretation of the results. Als compares with ground truth (if provided). Scores the probabilities of the other cells
        in a probability matrix in adata.uns 
        """	

        # Get probability columns
        prob_cols = [col for col in adata.obs.columns if col.startswith(f'{pred_cell_col}_prob_')]
        
        # Extract cell type names
        cell_types = [col.replace(f'{pred_cell_col}_prob_', '') for col in prob_cols]
        
        # Create probability matrix
        prob_matrix = np.zeros((adata.n_obs, len(cell_types)))
        for i, col in enumerate(prob_cols):
            prob_matrix[:, i] = adata.obs[col].values
        
        # Store full probability matrix in obsm
        adata.obsm['cell_type_probabilities'] = prob_matrix
        adata.uns['cell_type_probability_classes'] = cell_types
        
        # For each cell, add top N predictions to obs
        for i in range(adata.n_obs):
            # Get indices of top N probabilities
            top_indices = np.argsort(prob_matrix[i])[-n:][::-1]
            
            # Add top N types and probabilities to obs
            for rank, idx in enumerate(top_indices):
                cell_type = cell_types[idx]
                probability = prob_matrix[i, idx]
                
                # Add as new columns (rank+1 to start numbering at 1)
                adata.obs.loc[adata.obs.index[i], f"top{rank+1}_type"] = cell_type
                adata.obs.loc[adata.obs.index[i], f"top{rank+1}_prob"] = probability
        
        # Calculate if true cell type is in top N predictions
        if cell_type_col in adata.obs.columns:
            in_top_n = []
            for i in range(adata.n_obs):
                true_type = adata.obs[cell_type_col].iloc[i]
                if pd.isna(true_type):
                    in_top_n.append(False)
                    continue
                    
                top_types = [adata.obs[f"top{j+1}_type"].iloc[i] for j in range(n)]
                in_top_n.append(true_type in top_types)
            
            adata.obs['true_in_top_n'] = in_top_n
            valid_idx = ~adata.obs[cell_type_col].isna()
            top_n_accuracy = sum(in_top_n) / sum(valid_idx)
            print(f"Top-{n} accuracy: {top_n_accuracy:.4f}")
        
        print(f"Full probability matrix stored in adata.obsm['cell_type_probabilities']")
        print(f"Top {n} predictions stored in adata.obs columns")
        
        return adata


## === ENVIRONMENT and DATA LOADING === # 
    

#setup directories  
def setup_directories(): #this could also be done in the utils, and imported. Also -- needs to be able to use
    #relative and absolute paths. It's too narrow right now 
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
    if subset is None:
        adata = sc.read_h5ad(path)
    else:
        from utils import _load_anndata
        adata = _load_anndata(path, subset)
    print(f"Data loaded in {time.time()-start_time:.2f}s with shape {adata.shape}")
    
    return adata


## === BENCHMARKING === # 

#   - compare with other classifiers
#   - compare with other clustering methods
#   - compare with other dimensionality reduction methods

def test_existing_results(file_path, cell_type_col='cell_type', pred_cell_col='pred_cell_type'):
    """
    Test evaluation functions on existing results file without retraining.
    
    Args:
        file_path: Path to existing results h5ad file
        cell_type_col: Column name for true cell type
        pred_cell_col: Column name for predicted cell type
    """
    print(f"Loading existing results from {file_path}")
    import scanpy as sc
    
    # Load the existing results
    adata = sc.read_h5ad(file_path)
    print(f"Loaded data with shape {adata.shape}")
    
    # Create an annotator instance for using the evaluation methods
    annotator = scGPTAnnotator(embedding_key='X_scGPT')
    
    # Track metadata for method compatibility
    annotator.cell_type_col = cell_type_col
    
    # Add top N predictions analysis
    adata = annotator.add_top_n_predictions(adata, n=3, 
                                          pred_cell_col=pred_cell_col, 
                                          cell_type_col=cell_type_col)
    
    # Evaluate with visualizations
    valid_classes, adata, results = annotator.evaluate_with_visuals(
        adata, 
        cell_type_col=cell_type_col, 
        pred_cell_col=pred_cell_col
    )
    
    # Save the updated results with analysis
    import os
    from pathlib import Path
    from datetime import datetime
    
    # Create an output filename based on the input
    input_path = Path(file_path)
    output_dir = input_path.parent
    output_name = f"{input_path.stem}_analyzed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5ad"
    output_path = output_dir / output_name
    
    # Save results
    adata.write_h5ad(output_path)
    print(f"Updated results saved to {output_path}")
    
    return adata, results

# Add this to the bottom of your script or call it directly
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze existing scGPT classification results')
    parser.add_argument('--results_file', type=str, required=True,
                       help='Path to existing results h5ad file')
    parser.add_argument('--cell_type_col', type=str, default='cell_type',
                       help='Column name for true cell type')
    parser.add_argument('--pred_cell_col', type=str, default='pred_cell_type',
                       help='Column name for predicted cell type')
    
    args = parser.parse_args()
    
    # Just run the test function on existing results
    test_existing_results(args.results_file, 
                        cell_type_col=args.cell_type_col,
                        pred_cell_col=args.pred_cell_col)


## === REPORT === # 



# === MAIN FUNCTION === # 


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='scGPT data loading and metadata extraction')

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

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
    
    # Arguments for analysis-only mode (new functionality)
    analyze_parser = subparsers.add_parser('analyze', help='Analyze existing results without retraining')
    analyze_parser.add_argument('--results_file', type=str, required=True,
                        help='Path to existing results h5ad file')
    analyze_parser.add_argument('--cell_type_col', type=str, default='cell_type',
                        help='Column name for true cell type')
    analyze_parser.add_argument('--pred_cell_col', type=str, default='pred_cell_type',
                        help='Column name for predicted cell type')
    
    args = parser.parse_args()
    
    # If no command specified, default to 'train'
    if args.command is None:
        args.command = 'train'
        
    if args.command == 'analyze':
        # Run the analysis-only workflow
        print("Running analysis on existing results...")
        test_existing_results(args.results_file, 
                             cell_type_col=args.cell_type_col,
                             pred_cell_col=args.pred_cell_col)
    else:
        # Run the full training and prediction workflow (your existing code)
        print("Running scGPT classifier training and prediction pipeline...")
        
        # Your existing code goes here
        # Setup directories
        repo_dir, data_dir, save_dir = setup_directories()
    
    
    
        # Setup directories
        
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
        base_name = Path(args.query_file).stem
        metadata_path = output_dir / f"{base_name}_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Check if metadata exists
        if not metadata_path.exists():
            print(f"Warning: No metadata found at {metadata_path}")
            print("Proceeding without metadata - some features may be limited")
            metadata = {}
        else:
            print(f"Loading metadata from {metadata_path}")
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
        pred_cell_type_key = 'pred_cell_type' #this could be done differently - better input or in config file

        #train the classifier. Standard random forest classifier but can be changed to other classifiers
        annotator.train_classifier(classifier_name='randomforest', cell_type_col=cell_type_key, batch_key=batch_key)

        #predict the cell types
        predicted_adata = annotator.predict(annotator.query_adata, pred_cell_col=pred_cell_type_key, store_probs=True, return_adata=True)
        print(predicted_adata.obs[[cell_type_key, pred_cell_type_key]].head())
        #evaluate the results
        valid_classes, predicted_adata = annotator.evaluate(predicted_adata, y_true=cell_type_key, y_pred=pred_cell_type_key)

        annotator.add_top_n_predictions(predicted_adata, n=3, pred_cell_col=pred_cell_type_key, cell_type_col=cell_type_key)

        # Evaluate with visualizations
        valid_classes, predicted_adata, results = annotator.evaluate_with_visuals(
            predicted_adata, 
            cell_type_col=cell_type_key, 
            pred_cell_col=pred_cell_type_key
        )
        
        #save the results
        results_path = output_dir / f"Derived_Embryoid_Bodies_pred_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5ad"
        annotator.save_results(predicted_adata, results_path)
        print(f"Predicted results saved to {results_path}")




if __name__ == "__main__":
    main()


