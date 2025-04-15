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

    def evaluate_with_visuals(self, adata, y_pred='pred_cell_type', y_true='cell_type'):
        """
        Evaluate prediction results and generate visualizations.
        
        Args:
            adata: AnnData object with predictions
            y_pred: Column name for predicted labels
            y_true: Column name for true labels
            
        Returns:
            valid_classes: List of valid class labels
            adata: Updated AnnData object
            results: Dictionary with evaluation metrics
        """
        import numpy as np
        import pandas as pd
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Make sure required columns exist
        if y_true not in adata.obs.columns:
            print(f"ERROR: True label column '{y_true}' not found in data")
            return [], adata, {"error": f"Missing true label column: {y_true}"}
            
        if y_pred not in adata.obs.columns:
            print(f"ERROR: Prediction column '{y_pred}' not found in data")
            return [], adata, {"error": f"Missing prediction column: {y_pred}"}
        
        # Filter out rows with missing labels
        valid_mask = ~adata.obs[y_true].isna() & ~adata.obs[y_pred].isna()
        if valid_mask.sum() == 0:
            print(f"ERROR: No valid data points with both true and predicted labels")
            return [], adata, {"error": "No valid data points"}
        
        # Get valid samples for evaluation
        y_true_values = adata.obs[y_true][valid_mask].values
        y_pred_values = adata.obs[y_pred][valid_mask].values
        
        # Get unique classes that appear in both true and predicted labels
        true_classes = set(y_true_values)
        pred_classes = set(y_pred_values)
        valid_classes = sorted(list(true_classes.intersection(pred_classes)))
        
        # Debug information
        print(f"True classes: {len(true_classes)}, Predicted classes: {len(pred_classes)}")
        print(f"Common classes: {len(valid_classes)}")
        
        if len(valid_classes) == 0:
            print(f"WARNING: No overlapping cell types between true and predicted labels!")
            print(f"True labels: {true_classes}")
            print(f"Predicted labels: {pred_classes}")
            # Return minimal results to avoid crashing
            results = {
                "samples": valid_mask.sum(),
                "classes": 0,
                "accuracy": 0.0,
                "macro_f1": np.nan,
                "weighted_f1": np.nan,
                "precision": np.nan,
                "recall": np.nan,
                "error": "No overlapping cell types found"
            }
            
            # Store metrics in adata
            adata.uns['classifier_metrics'] = results
            print(f"Metrics stored in adata.uns['classifier_metrics']")
            
            return [], adata, results
        
        # Calculate metrics
        # Use only valid rows for evaluation
        accuracy = accuracy_score(y_true_values, y_pred_values)
        
        try:
            macro_f1 = f1_score(y_true_values, y_pred_values, average='macro', labels=valid_classes)
            weighted_f1 = f1_score(y_true_values, y_pred_values, average='weighted', labels=valid_classes)
            precision = precision_score(y_true_values, y_pred_values, average='weighted', labels=valid_classes)
            recall = recall_score(y_true_values, y_pred_values, average='weighted', labels=valid_classes)
        except Exception as e:
            print(f"WARNING: Error calculating metrics: {e}")
            macro_f1 = weighted_f1 = precision = recall = np.nan
        
        # Store metrics
        results = {
            "samples": valid_mask.sum(),
            "classes": len(valid_classes),
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1) if not np.isnan(macro_f1) else None,
            "weighted_f1": float(weighted_f1) if not np.isnan(weighted_f1) else None,
            "precision": float(precision) if not np.isnan(precision) else None,
            "recall": float(recall) if not np.isnan(recall) else None
        }
        
        # Print results
        print("Evaluation Results:")
        print(f"Samples: {results['samples']}")
        print(f"Classes: {results['classes']}")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Macro F1: {results['macro_f1'] if results['macro_f1'] is not None else 'nan'}")
        print(f"Weighted F1: {results['weighted_f1'] if results['weighted_f1'] is not None else 'nan'}")
        print(f"Precision: {results['precision'] if results['precision'] is not None else 'nan'}")
        print(f"Recall: {results['recall'] if results['recall'] is not None else 'nan'}")
        
        # Store metrics in adata
        adata.uns['classifier_metrics'] = results
        print(f"Metrics stored in adata.uns['classifier_metrics']")
        
        # Create confusion matrix if we have enough data and classes
        if len(valid_classes) > 1 and valid_mask.sum() > 1:
            try:
                cm = confusion_matrix(y_true_values, y_pred_values, labels=valid_classes)
                
                # Create visualization
                plt.figure(figsize=(12, 10))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=valid_classes, yticklabels=valid_classes)
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title('Confusion Matrix')
                plt.tight_layout()
                
                # Save plot in adata
                if 'figures' not in adata.uns:
                    adata.uns['figures'] = {}
                adata.uns['figures']['confusion_matrix'] = plt
                
                print("Confusion matrix created and stored in adata.uns['figures']['confusion_matrix']")
                
            except Exception as e:
                print(f"WARNING: Error creating confusion matrix: {e}")
        
        return valid_classes, adata, results
    #save the results
    def save_results(self, adata, results_path = None):
        """Save results to a file."""
        if adata is None:
            print("ERROR: No data to save")
            return
        adata.write_h5ad(results_path)
        print(f"Results saved to {results_path}")
    def add_top_n_predictions(self, adata, n=5, pred_cell_col='pred_cell_type', cell_type_col='cell_type'):
        """Show the top n prediction scores of the classifier for choosing the cell type.
        
        Adds top predictions to adata.obs, stores probability matrix in adata.obsm,
        and adds a clean dataframe with prediction data in adata.uns.
        
        Args:
            adata: AnnData object with predictions
            n: Number of top predictions to store (default 5)
            pred_cell_col: Column name for predicted cell type
            cell_type_col: Column name for true cell type (if available)
            
        Returns:
            adata: Updated AnnData object
        """
        # Basic input validation
        if pred_cell_col not in adata.obs.columns:
            print(f"Warning: '{pred_cell_col}' not found in data")
            return adata
            
        # Get probability columns
        prob_cols = [col for col in adata.obs.columns if col.startswith(f'{pred_cell_col}_prob_')]
        
        if not prob_cols:
            print(f"Warning: No probability columns found with prefix '{pred_cell_col}_prob_'")
            return adata
        
        print(f"Found {len(prob_cols)} probability columns")
        
        # Extract cell type names
        cell_types = [col.replace(f'{pred_cell_col}_prob_', '') for col in prob_cols]
        
        # Create probability matrix
        prob_matrix = np.zeros((adata.n_obs, len(cell_types)))
        for i, col in enumerate(prob_cols):
            prob_matrix[:, i] = adata.obs[col].values
        
        # Store full probability matrix in obsm
        adata.obsm['cell_type_probabilities'] = prob_matrix
        adata.uns['cell_type_probability_classes'] = cell_types
        
        # Get confidence score (probability of top prediction)
        top_probabilities = np.max(prob_matrix, axis=1)
        adata.obs['prediction_confidence'] = top_probabilities
        
        # For each cell, add top N predictions to obs
        print(f"Adding top {n} predictions for {adata.n_obs} cells...")
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
            top_n_accuracy = sum(in_top_n) / sum(valid_idx) if sum(valid_idx) > 0 else 0
            print(f"Top-{n} accuracy: {top_n_accuracy:.4f} ({sum(in_top_n)} / {sum(valid_idx)} cells)")
        
        # Create the DataFrame for adata.uns
        columns = []
        if cell_type_col in adata.obs.columns:
            columns.append(cell_type_col)
        
        columns.append(pred_cell_col)
        columns.append('prediction_confidence')
        
        for i in range(1, n+1):
            columns.extend([f'top{i}_type', f'top{i}_prob'])
        
        if 'true_in_top_n' in adata.obs.columns:
            columns.append('true_in_top_n')
        
        # Store this clean DataFrame in uns for easy access
        prediction_df = adata.obs[columns].copy()
        adata.uns['prediction_data'] = prediction_df
        
        print(f"Prediction data stored in adata.uns['prediction_data']")
        
        return adata

    def evaluate_comprehensive(self, adata, pred_cell_col='pred_cell_type', cell_type_col='cell_type'):
        """
        Comprehensive evaluation of cell type predictions with standard metrics and confidence analysis.
        
        Args:
            adata: AnnData object with predictions (must have prediction_data in adata.uns)
            pred_cell_col: Column with predicted cell types
            cell_type_col: Column with true cell types
            
        Returns:
            adata: Updated AnnData object with evaluation results
        """
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Basic validation
        if cell_type_col not in adata.obs.columns:
            print(f"Error: True label column '{cell_type_col}' not found in data")
            return adata
            
        if pred_cell_col not in adata.obs.columns:
            print(f"Error: Prediction column '{pred_cell_col}' not found in data")
            return adata
            
        if 'prediction_data' not in adata.uns:
            print("Warning: No prediction_data found. Run add_top_n_predictions first.")
            return adata
        
        # Filter valid cells with non-null true labels
        valid_mask = ~adata.obs[cell_type_col].isna()
        valid_count = valid_mask.sum()
        
        if valid_count == 0:
            print("No cells with valid cell type labels found")
            return adata
            
        print(f"Evaluating {valid_count} cells with true cell type labels")
        
        # Get true and predicted labels for valid cells
        y_true = adata.obs[cell_type_col][valid_mask]
        y_pred = adata.obs[pred_cell_col][valid_mask]
        
        # Calculate overall metrics
        accuracy = accuracy_score(y_true, y_pred)
        class_report = classification_report(y_true, y_pred, output_dict=True)
        
        # Get unique classes and confusion matrix
        cell_types = sorted(y_true.unique())
        cm = confusion_matrix(y_true, y_pred, labels=cell_types)
        
        # Create evaluation results structure
        adata.uns['evaluation_results'] = {
            'overall': {
                'accuracy': float(accuracy),
                'weighted_avg_precision': float(class_report['weighted avg']['precision']),
                'weighted_avg_recall': float(class_report['weighted avg']['recall']),
                'weighted_avg_f1': float(class_report['weighted avg']['f1-score'])
            },
            'confusion_matrix': {
                'matrix': cm.tolist(),
                'labels': cell_types
            }
        }
        
        # Add per-class metrics
        per_class_metrics = []
        for cell_type in cell_types:
            metrics = class_report.get(cell_type, {})
            if metrics:
                per_class_metrics.append({
                    'cell_type': cell_type,
                    'precision': float(metrics['precision']),
                    'recall': float(metrics['recall']),
                    'f1': float(metrics['f1-score']),
                    'support': int(metrics['support'])
                })
        
        adata.uns['evaluation_results']['per_class'] = per_class_metrics
        
        # Calculate metrics by confidence level
        confidence_analysis = []
        confidence_bins = [0, 0.25, 0.5, 0.75, 0.9, 1.0]
        
        prediction_df = adata.uns['prediction_data']
        for i in range(len(confidence_bins)-1):
            lower = confidence_bins[i]
            upper = confidence_bins[i+1]
            
            # Filter cells in this confidence range
            mask = ((prediction_df['prediction_confidence'] > lower) & 
                    (prediction_df['prediction_confidence'] <= upper) &
                    (~prediction_df[cell_type_col].isna()))
                
            bin_df = prediction_df[mask]
            
            if len(bin_df) > 0:
                bin_accuracy = (bin_df[cell_type_col] == bin_df[pred_cell_col]).mean()
                confidence_analysis.append({
                    'confidence_range': f"{lower:.2f}-{upper:.2f}",
                    'cell_count': len(bin_df),
                    'percent_of_total': len(bin_df) / len(prediction_df) * 100,
                    'accuracy': float(bin_accuracy)
                })
        
        adata.uns['evaluation_results']['confidence_analysis'] = confidence_analysis
        
        # Print summary
        print("\nClassification Performance Summary:")
        print(f"Overall accuracy: {accuracy:.4f}")
        print(f"Weighted avg precision: {class_report['weighted avg']['precision']:.4f}")
        print(f"Weighted avg recall: {class_report['weighted avg']['recall']:.4f}")
        print(f"Weighted avg F1: {class_report['weighted avg']['f1-score']:.4f}")
        
        # Print top 5 classes by support
        top_classes = sorted(per_class_metrics, key=lambda x: x['support'], reverse=True)[:5]
        print("\nTop 5 Cell Type Performance:")
        for metrics in top_classes:
            print(f"{metrics['cell_type']} (n={metrics['support']}): " +
                f"Precision={metrics['precision']:.4f}, " +
                f"Recall={metrics['recall']:.4f}, " +
                f"F1={metrics['f1']:.4f}")
        
        # Create confusion matrix visualization if not too large
        if len(cell_types) <= 20:  # Only create visualization for reasonable size
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=cell_types, yticklabels=cell_types)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            plt.tight_layout()
            
            # Store plot in adata
            adata.uns['evaluation_results']['confusion_matrix_plot'] = plt
        
        return adata

    





## === BENCHMARKING === # 

#   - compare with other classifiers
#   - compare with other clustering methods
#   - compare with other dimensionality reduction methods


