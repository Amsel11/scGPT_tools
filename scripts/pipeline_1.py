#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pipeline for the scGPT model. 

This script:
1. Loads AnnData (h5ad) files
2. Detects cell type, gene, batch and other annotations automatically
3. Extracts metadata needed for scGPT embedding
4. Uses metadata to intelligently configure the embedding process
"""

import argparse
import os
from re import I
import sys
import json
import time
from pathlib import Path
import datetime
from datetime import datetime
import logging
import torch
import h5py
import numpy as np
import pandas as pd
import traceback
import scanpy as sc
import scipy.sparse

# Import the scripts from this repository
import scGPT_dataloader
import scGPT_embedder
from utils import add_dict_to_argparser
from utils import str2bool
from utils import build_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def create_argparser():
    defaults = {
        # Model configuration
        "model_dir": "data/scGPT_Human",
        "checkpoint_name": "best_model.pt",
        "embedding_dim": 512,
        "num_heads": 8,
        "num_layers": 12,
        "max_seq_len": 1200,
        
        # Data configuration
        "gene_col": None,  # Will be detected from metadata if not specified
        "batch_size": 64,
        "max_genes": 2000,
        "use_highly_variable": False,
        "cell_type_col": None,
        "batch_key": None,
        
        # Other settings 
        "output_dir": None,
        "evaluate": False,
        "classifier_type": "randomforest",
        "n_top_predictions": 5,
        "pred_cell_type_key": "pred_cell_type",
    }

    parser = argparse.ArgumentParser(description='scGPT pipeline')
    parser.add_argument('--query_file', default="data/Derived Embryoid Bodies.h5ad",
                      help='Path to the query file')
    parser.add_argument('--ref_file', default = None, help = 'Path to the reference file')
    parser.add_argument('--config_file', default="scripts/scGPT_embed_config.json", 
                      help='Path to the JSON config file')
    parser.add_argument('--classifier_file', default=None, help='Path to the classifier file')
    parser.add_argument('--analysis', action='store_true', default=False,
                      help='Run analysis step')
    parser.add_argument('--embed', action='store_true', default=False,
                      help='Run embedding step')
    parser.add_argument('--classify', action='store_true', default=False,
                      help='Run classification step')
    parser.add_argument('--evaluate', action='store_false', help='Path to the evaluation file')
    parser.add_argument('--testing', action='store_true', default=False,
                      help='Use testing directory')
    parser.add_argument('--disable_file_logging', action='store_true', default=False,
                      help='Disable file logging')
    parser.add_argument('--download_data', action='store_true', default=False,
                      help='Download data from cellxgene')

    add_dict_to_argparser(parser, defaults)

    return parser

def fix_reserved_column_names(adata):
    """
    Fix reserved column names in AnnData object that would cause errors when saving.
    
    Args:
        adata: AnnData object to fix
        
    Returns:
        Fixed AnnData object
    """
    # Check var DataFrame for reserved column names
    reserved_names = ['_index', '_i', '_ref']
    
    # Fix var DataFrame
    if hasattr(adata, 'var') and isinstance(adata.var, pd.DataFrame):
        for reserved in reserved_names:
            if reserved in adata.var.columns:
                # Rename the column
                new_name = f"renamed_{reserved}"
                print(f"Warning: Renaming reserved column name '{reserved}' to '{new_name}' in var DataFrame")
                adata.var = adata.var.rename(columns={reserved: new_name})
    
    # Fix obs DataFrame
    if hasattr(adata, 'obs') and isinstance(adata.obs, pd.DataFrame):
        for reserved in reserved_names:
            if reserved in adata.obs.columns:
                # Rename the column
                new_name = f"renamed_{reserved}"
                print(f"Warning: Renaming reserved column name '{reserved}' to '{new_name}' in obs DataFrame")
                adata.obs = adata.obs.rename(columns={reserved: new_name})

    return results


def main():
    #setup parsing
    parser = create_argparser()
    args = parser.parse_args()



    print (f"args: {args}")
    base_name = Path(args.query_file).stem
    logging.info(f"base_name: {base_name}")


    from utils import build_config, setup_directories
    repo_dir, data_dir, save_dir, model_dir, directories = setup_directories()
    logging.info("Starting scGPT pipeline")
    logging.info(f"the directories are: {directories}")
    
    # Create output directory - the if statement is only for testing 
    if args.testing:
        output_dir = save_dir / f"test_output_{datetime.now().strftime('%Y%m%d')}"
        output_dir = save_dir / f"test_output_{datetime.now().strftime('%Y%m%d')}"
        if args.testing:
            # Pass testing flag to setup_logging to disable file output
            args.disable_file_logging = True
    else:
        date_str = datetime.now().strftime("%Y%m%d")
        existing = [x for x in save_dir.iterdir() if x.is_dir() and x.name.startswith(f"{base_name}_{date_str}")]
        number = len(existing) + 1
        output_dir = save_dir / f"{base_name}_{date_str}_{number:02d}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    query_file = args.query_file 
    print(query_file)

    if args.download_data:
        from utils import download_cellxgene_v2
        file_paths = download_cellxgene_v2(output_dir=output_dir, data_dir=data_dir)
        query_file = file_paths[0]
        
        logging.info(f"Using Cellxgene V2: query_file: {query_file}")

    #setup logging
    from utils import setup_logging
    logger = setup_logging(output_dir, disable_file_logging=args.disable_file_logging)
    logger.info("Starting scGPT pipeline")

    config = build_config(args) #build initial config from command line arguments and from the defaults. 
    initial_config_path = output_dir / "initial_config.json"
    with open(initial_config_path, 'w') as f:
        json.dump(config, f, indent=4)
    logger.info(f"Initial config saved to {initial_config_path}")
    logger.info(f"Initial config: {config}") #just for testing now

    # Log some key config values
    logger.info(f"Initial config values:")
    logger.info(f"  model_dir: {config.get('model_dir')}")
    logger.info(f"  gene_col: {config.get('gene_col')}")
    logger.info(f"  batch_size: {config.get('batch_size')}")
    logger.info(f"  device: {config.get('device')}")
    logger.info(f"  cell_type_col: {config.get('cell_type_col')}")
    logger.info(f"  batch_key: {config.get('batch_key')}")

    Metadata = None

    # Step 1: Analysis (if enabled)
    if args.analysis:
        logger.info("=== STEP 1: Running analysis ===")
        try:
            from scGPT_dataloader import analysis_meta
            metadata = analysis_meta(query_file, save=True, output_dir=output_dir)
            
            if metadata is None:
                logger.error("Metadata analysis returned None")
                metadata = {}  # Initialize to empty dict to avoid NoneType errors
            
            if metadata.get('error'):
                logger.error(f"Analysis failed: {metadata['error']}")
                if not getattr(args, 'force_continue', False):
                    return 1
                logger.warning("Continuing despite analysis failure (force_continue)")
            else:
                logger.info("Analysis completed successfully")
                
                # Update config with metadata
                config = build_config(args, metadata)
                
                # Log key config values
                logger.info(f"Config after metadata analysis:")
                logger.info(f"  gene_col: {config.get('gene_col')}")
                logger.info(f"  cell_type_col: {config.get('cell_type_col')}")
                logger.info(f"  batch_key: {config.get('batch_key')}")
                logger.info(f"  device: {config.get('device')}")
                
                # Save the config for reference
                config_path = output_dir / "pipeline_config.json"
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                logger.info(f"Configuration saved to {config_path}")
                logger.info(f"Config: {config}")
        except Exception as e:
            logger.error(f"Unexpected error during analysis: {e}", exc_info=True)
            if not getattr(args, 'force_continue', False):
                return 1
            logger.warning("Continuing despite analysis error (force_continue)")
    else:
        logger.info("Analysis step skipped (not enabled)")
    
    # Extract found keys from metadata if analysis was run
    if args.analysis and not metadata.get('error'):
        important_obs_keys = metadata.get('found_obs_keys', [])
        important_var_keys = metadata.get('found_var_keys', [])
        logger.info(f"Found {len(important_obs_keys)} observation keys and {len(important_var_keys)} variable keys")
        logger.debug(f"Observation keys: {important_obs_keys}")
        logger.debug(f"Variable keys: {important_var_keys}")
    else:
        important_obs_keys = config.get('found_obs_keys', [])
        logger.info(f"Important observation keys: {important_obs_keys}")
        important_var_keys = config.get('found_var_keys', [])
        logger.info(f"Important variable keys: {important_var_keys}")
        logger.warning("Metadata available from config file - will load those columns")

    
    #load the data
    logger.info("=== STEP 2: Loading data ===")
    try:
        # Load the complete file directly with scanpy instead of using chunks
        adata = sc.read_h5ad(query_file)
        
        # Check for empty rows
        row_sums = adata.X.sum(axis=1)
        if scipy.sparse.issparse(row_sums):
            row_sums = row_sums.A1
        zero_expr_cells = np.where(row_sums == 0)[0]
        
        if len(zero_expr_cells) > 0:
            logger.warning(f"Found {len(zero_expr_cells)} cells with zero expression - removing these cells")
            adata = adata[~np.isin(np.arange(adata.n_obs), zero_expr_cells), :]
        
        logger.info(f"Loaded AnnData with {adata.n_obs} cells and {adata.n_vars} genes")
        
        # Check if embeddings exist
        if 'X_scGPT' in adata.obsm:
            logger.info(f"Found X_scGPT embeddings of shape {adata.obsm['X_scGPT'].shape}")
        else:
            logger.info(f"Available obsm keys: {list(adata.obsm.keys())}")
            if not args.embed and not args.classify:
                logger.warning("No embeddings found in data - consider running with --embed")
        
        logger.info("First 5 rows of adata.obs:")
        logger.info("\n" + str(adata.obs.head()))
        logger.info("First 5 rows of adata.var:")
        logger.info("\n" + str(adata.var.head()))
    except Exception as e:
        logger.error(f"Error loading AnnData: {e}")
        traceback.print_exc()
        sys.exit(1)

    
    # The config comes from either:
    # 1. Command line args processed earlier in the script
    # 2. Default config file (scripts/scGPT_embed_config.json)
    # 3. User-provided config file via --config argument
    # The final config is stored in output_dir/pipeline_config.json

    
    


    # Step 3: Embedding (if enabled)
    if args.embed:
        import scgpt as scg

        logger.info("=== STEP 3: Running embedding ===")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        config["device"] = device
        print(f"Using device from config: {config['device']}")

        logger.info("Current configuration:")
        #for key, value in config.items():
         #   logger.info(f"  {key}: {value}")
            
        gene_col = config.get("gene_col")
        batch_size = config.get("batch_size", 64)  # Default to 64 if not specified
        model_dir = config.get("model_dir")
        
        if not gene_col:
            logger.warning("No gene_col specified in config - will use index")
            logger.warning("Please provide gene_col in config or command line")
            return 1
        if not model_dir:
            logger.error("No model_dir specified in config - required for embedding/classification")
            return 1
        
        logger.debug(f"Using gene_col: {gene_col}")
        logger.debug(f"Using batch_size: {batch_size}")
        logger.debug(f"Using model_dir: {model_dir}")


        if device == "cuda":
            logger.info("Using GPU for embedding")
            embedding_adata = scg.tasks.embed_data(
                adata,
                gene_col=gene_col,
                batch_size=batch_size,
                model_dir= model_dir ,
                device="cuda"
            )

            if embedding_adata.var.index.name in embedding_adata.var.columns:
                embedding_adata.var.index.name = f"{embedding_adata.var.index.name}_index"

            embedding_file = os.path.join(output_dir, "embeddings.h5ad")
            logger.info(f"Saving embeddings to {embedding_file}")
            try:
                # Fix any reserved column names before saving
                embedding_adata = fix_reserved_column_names(embedding_adata)
                embedding_adata.write_h5ad(embedding_file)
                logger.info(f"Embeddings saved to {embedding_file}")
                adata = embedding_adata
            except Exception as e:
                logger.error(f"Failed to save embeddings: {str(e)}")
                traceback.print_exc()
                if not getattr(args, 'force_continue', False):
                    return 1
                logger.warning("Continuing despite embedding save error (force_continue)")
        else:
            logger.info("No GPU available, cannot do embedding.")

    if args.classify:
        logger.info("=== STEP 4: Running classification ===")
        from scGPT_classifier import scGPTAnnotator
        
        # Get the embedding key and verify it exists
        embedding_key = config.get("embedding_key", "X_scGPT")
        logger.info(f"Using embedding key: {embedding_key}")
        logger.info(f"all keys in adata {adata}")
        
        # Ensure embeddings exist in the data
        if embedding_key not in adata.obsm_keys():
            logger.error(f"Embedding key '{embedding_key}' not found in data")
            logger.error(f"Available keys: {list(adata.obsm_keys())}")
            logger.error("Run with --embed first to generate embeddings")
            return 1
            
        # Initialize annotator with correct embedding key
        annotator = scGPTAnnotator(embedding_key=embedding_key)
        
        # Set the query data (which has embeddings)
        annotator.set_query_data(adata)
        
        # Get batch key from config - this is now more reliable after our changes
        batch_key = config.get('batch_key')
        if batch_key:
            logger.info(f"Using batch key from config: {batch_key}")
            # Check if it exists in the data
            if batch_key not in adata.obs.columns:
                logger.warning(f"Batch key '{batch_key}' not found in data columns")
                logger.warning(f"Available columns: {list(adata.obs.columns)}")
                batch_key = None
        
        # If batch key is set and has enough unique values, use it for splitting
        if batch_key and batch_key in adata.obs and len(adata.obs[batch_key].unique()) > 2:
            logger.info(f"Splitting using batch key: {batch_key}")
            logger.info(f"Available batches: {adata.obs[batch_key].unique()}")
            annotator.split_query_ref(adata, method='batch', batch_key=batch_key)
        else:
            if not batch_key:
                logger.info("No valid batch key found")
            elif batch_key in adata.obs:
                logger.info(f"Not enough unique values in batch key: {len(adata.obs[batch_key].unique())}")
            logger.info("Using random split for reference/query")
            annotator.split_query_ref(adata, method='random')
        
        # Verify both query and reference have embeddings
        if embedding_key not in annotator.ref_adata.obsm_keys():
            logger.error(f"Reference data missing embedding key: {embedding_key}")
            return 1
            
        if embedding_key not in annotator.query_adata.obsm_keys():
            logger.error(f"Query data missing embedding key: {embedding_key}")
            return 1
            
        logger.info(f"Reference data shape: {annotator.ref_adata.shape}")
        logger.info(f"Query data shape: {annotator.query_adata.shape}")

        # Get cell type key directly from config - much more reliable now
        cell_type_key = config.get('cell_type_col')
        
        # Double-check it exists in the data
        if cell_type_key and cell_type_key in adata.obs.columns:
            logger.info(f"Using cell type key from config: {cell_type_key}")
        else:
            if cell_type_key:
                logger.warning(f"Configured cell type key '{cell_type_key}' not found in data columns")
            
            # Try to find an appropriate column in the data
            possible_keys = ['cell_type', 'celltype', 'CellType', 'cell.type', 'cell_ontology_class', 
                           'cell_ontology_term', 'cluster', 'leiden', 'louvain']
            
            for key in possible_keys:
                if key in adata.obs.columns:
                    cell_type_key = key
                    logger.info(f"Found cell type key in data: {cell_type_key}")
                    break
            
            # If still not found, show error
            if not cell_type_key:
                logger.error("Could not determine cell type column!")
                logger.error(f"Available columns: {list(adata.obs.columns)}")
                logger.error("Please specify cell_type_col in config or command line.")
                return 1
        
        logger.info(f"Using cell type key: {cell_type_key}")
        
        # Handle reference data
        if args.ref_file is not None:
            # External reference file provided
            has_ref_embeddings, ref_message = annotator.check_embeddings(args.ref_file)
            if not has_ref_embeddings:
                logger.warning(f"Reference file doesn't have embeddings: {ref_message}")
                logger.warning("Falling back to split from query data.")
                
                # We already split the data above, so no need to do it again
                logger.info("Using previously created split")
            else:
                # Reference file has embeddings, use it
                logger.info(f"Loading reference file with embeddings: {ref_message}")
                ref_adata = sc.read_h5ad(args.ref_file)
                annotator.set_ref_data(ref_adata)
        else:
            # We already split the data above, so just log that
            logger.info("Using previously created split from query data")

        # Verify we have what we need
        if annotator.query_adata is None:
            logger.error("No query data available.")
            return 1
        if annotator.ref_adata is None:
            logger.error("No reference data available. Cannot proceed with classification.")
            return 1

        # Check if cell type key exists in reference data
        if cell_type_key not in annotator.ref_adata.obs.columns:
            logger.error(f"Cell type key '{cell_type_key}' not found in reference data")
            logger.error(f"Available columns: {list(annotator.ref_adata.obs.columns)}")
            return 1
            
        logger.info(f"Ready for cell type annotation using {cell_type_key}")
        logger.info(f"Reference data: {len(annotator.ref_adata)} cells with {len(annotator.ref_adata.obs[cell_type_key].unique())} unique cell types")
        logger.info(f"Query data: {len(annotator.query_adata)} cells")
        
        # Name for predicted cell type column
        pred_cell_type_key = config.get('pred_cell_type_key', 'pred_cell_type')
        
        # Train the classifier
        classifier_type = config.get('classifier_type', 'knn')
        logger.info(f"Training {classifier_type} classifier")
        annotator.train_classifier(classifier_name=classifier_type, cell_type_col=cell_type_key, batch_key=batch_key)
        annotator.save_classifier(output_dir / f"{classifier_type}_classifier_{base_name}.pkl")


        # Predict the cell types
        logger.info("Predicting cell types")
        predicted_adata = annotator.predict(annotator.query_adata, pred_cell_col=pred_cell_type_key, store_probs=True, return_adata=True)
        logger.info("Prediction complete")
        logger.info(f"Sample of predictions:\n{predicted_adata.obs[[cell_type_key, pred_cell_type_key]].head()}")
        
        # Evaluate the results
        logger.info("Evaluating prediction results")
        valid_classes, predicted_adata, results = annotator.evaluate_with_visuals(predicted_adata, y_pred=pred_cell_type_key, y_true=cell_type_key)

        # Add top N predictions
        n_predictions = config.get('n_top_predictions', 3)
        logger.info(f"Adding top {n_predictions} predictions")
        annotator.add_top_n_predictions(predicted_adata, n=n_predictions, pred_cell_col=pred_cell_type_key, cell_type_col=cell_type_key)


 
        # Save the results
        filename_base = os.path.basename(args.query_file).split('.')[0] if args.query_file else "scGPT_results"
        results_path = output_dir / f"{filename_base}_pred_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5ad"
        logger.info(f"Saving prediction results to {results_path}")
        annotator.save_results(predicted_adata, results_path)
        logger.info(f"Predicted results saved to {results_path}")

        
        


if __name__ == '__main__':
    main()
