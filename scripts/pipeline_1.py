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

        #classifier settings
        "classifier_type": "randomforest",
        "n_top_predictions": 5,
        "pred_cell_type_key": "pred_cell_type",
    }

    parser = argparse.ArgumentParser(
        description='scGPT Cell Type Annotation Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  # Show defaults in help
    )
    
    # Create argument groups for better organization
    io_group = parser.add_argument_group('Input/Output Options')
    io_group.add_argument('--query_file', required=True,
                       help='Path to the query h5ad file containing cells to analyze/annotate. Required - pipeline cannot run without input data.')
    io_group.add_argument('--ref_file', default=None, 
                       help='Path to reference h5ad file with annotated cells (optional)')
    io_group.add_argument('--config_file', default="scripts/scGPT_embed_config.json", 
                       help='Path to the JSON configuration file')
    io_group.add_argument('--classifier_file', default=None, 
                       help='Path to a pre-trained classifier file (optional)')
    io_group.add_argument('--output_dir', default=None,
                       help='Directory to save output files (default: auto-generated)')
    
    # Pipeline steps
    steps_group = parser.add_argument_group('Pipeline Steps')
    steps_group.add_argument('--analysis', action='store_true', default=False,
                       help='Run analysis step (extract metadata and detect cell types/genes)')
    steps_group.add_argument('--embed', action='store_true', default=False,
                       help='Run embedding step (generate scGPT embeddings)')
    steps_group.add_argument('--classify', action='store_true', default=False,
                       help='Run classification step (predict cell types)')
    steps_group.add_argument('--evaluate', action='store_true', default=True,
                       help='Evaluate classification performance (when ground truth is available)')
    steps_group.add_argument('--all', action='store_true', default=False,
                       help='Run all pipeline steps (analysis, embed, classify, evaluate)')
    
    # Model settings
    model_group = parser.add_argument_group('Model Settings')
    model_group.add_argument('--model_dir', default=defaults["model_dir"],
                       help='Directory containing the scGPT model files')
    model_group.add_argument('--checkpoint_name', default=defaults["checkpoint_name"],
                       help='Name of the model checkpoint file')
    model_group.add_argument('--embedding_dim', type=int, default=defaults["embedding_dim"],
                       help='Dimension of the scGPT embeddings')
    model_group.add_argument('--device', default=None,
                       help='Device to use (cuda or cpu, default: auto-detect)')
    
    # Data settings
    data_group = parser.add_argument_group('Data Settings')
    data_group.add_argument('--gene_col', default=defaults["gene_col"],
                       help='Column name in adata.var for gene identifiers (auto-detected if not specified)')
    data_group.add_argument('--cell_type_col', default=defaults["cell_type_col"],
                       help='Column name in adata.obs for cell type annotations (auto-detected if not specified)')
    data_group.add_argument('--batch_key', default=defaults["batch_key"],
                       help='Column name in adata.obs for batch annotations (auto-detected if not specified)')
    data_group.add_argument('--batch_size', type=int, default=defaults["batch_size"],
                       help='Batch size for model inference')
    data_group.add_argument('--max_genes', type=int, default=defaults["max_genes"],
                       help='Maximum number of genes to use')
    
    # Classification settings
    class_group = parser.add_argument_group('Classification Settings')
    class_group.add_argument('--classifier_type', default=defaults["classifier_type"],
                       choices=['randomforest', 'knn', 'svm', 'lightgbm'],
                       help='Type of classifier to use for cell type prediction')
    class_group.add_argument('--n_top_predictions', type=int, default=defaults["n_top_predictions"],
                       help='Number of top predictions to include in results')
    class_group.add_argument('--pred_cell_type_key', default=defaults["pred_cell_type_key"],
                       help='Column name to use for predicted cell types in output')
    
    # Other options
    other_group = parser.add_argument_group('Other Options')
    other_group.add_argument('--testing', action='store_true', default=False,
                       help='Run in testing mode (uses test output directory)')
    other_group.add_argument('--disable_file_logging', action='store_true', default=False,
                       help='Disable logging to file')
    other_group.add_argument('--download_data', action='store_true', default=False,
                       help='Download example data from cellxgene')
    other_group.add_argument('--force_continue', action='store_true', default=False,
                       help='Force continue pipeline even if errors occur in earlier steps')
    
    # Add the remaining default values
    remaining_defaults = {k: v for k, v in defaults.items() 
                         if k not in ['model_dir', 'checkpoint_name', 'embedding_dim',
                                     'gene_col', 'batch_size', 'max_genes', 'cell_type_col',
                                     'batch_key', 'output_dir', 'classifier_type',
                                     'n_top_predictions', 'pred_cell_type_key']}
    add_dict_to_argparser(parser, remaining_defaults)

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

    return adata


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
    
    if args.all:
        args.analysis = True
        args.embed = True
        args.classify = True
        args.evaluate = True


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

    # Show CLI banner
    print("\n" + "="*80)
    print(f"scGPT Cell Type Annotation Pipeline (v0.1.0)")
    print("="*80)
    print(f"Query file: {args.query_file}")
    if args.ref_file:
        print(f"Reference file: {args.ref_file}")
    print(f"Output directory: {output_dir}")
    steps = []
    if args.analysis:
        steps.append("analysis")
    if args.embed:
        steps.append("embed")
    if args.classify:
        steps.append("classify")
    if args.evaluate:
        steps.append("evaluate")
    print(f"Steps: {' '.join(steps)}")
    print("-"*80 + "\n")

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

        if annotator.query_adata is None:
            logger.error("No query data available.")
            return 1
        if annotator.ref_adata is None:
            logger.error("No reference data available. Cannot proceed with classification.")
            return 1

        if cell_type_key not in annotator.ref_adata.obs.columns:
            logger.error(f"Cell type key '{cell_type_key}' not found in reference data")
            logger.error(f"Available columns: {list(annotator.ref_adata.obs.columns)}")
            return 1
            
        logger.info(f"Ready for cell type annotation using {cell_type_key}")
        logger.info(f"Reference data: {len(annotator.ref_adata)} cells with {len(annotator.ref_adata.obs[cell_type_key].unique())} unique cell types")
        logger.info(f"Query data: {len(annotator.query_adata)} cells")
        
        pred_cell_type_key = config.get('pred_cell_type_key', 'pred_cell_type')
        
        # Train the classifier
        classifier_type = config.get('classifier_type', 'knn') #also in the defaults
        logger.info(f"Training {classifier_type} classifier") #this is the type of classifier we are using
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

    # After loading adata, add these debug statements:
    logger.info("=== Data Content Debug ===")
    logger.info(f"Columns in adata.obs: {list(adata.obs.columns)}")
    if cell_type_key in adata.obs.columns:
        logger.info(f"Unique values in {cell_type_key}: {adata.obs[cell_type_key].unique()}")
        logger.info(f"Number of cells with cell type: {adata.obs[cell_type_key].notna().sum()}")
    else:
        logger.error(f"Cell type column '{cell_type_key}' not found in data!")
        logger.info("Available columns are:")
        for col in adata.obs.columns:
            logger.info(f"  - {col}: {adata.obs[col].nunique()} unique values")

    # Before classification, add:
    if args.classify:
        logger.info("=== Classification Debug ===")
        logger.info(f"Reference data columns: {list(annotator.ref_adata.obs.columns)}")
        logger.info(f"Query data columns: {list(annotator.query_adata.obs.columns)}")
        if cell_type_key:
            if cell_type_key in annotator.ref_adata.obs.columns:
                logger.info(f"Reference unique cell types: {annotator.ref_adata.obs[cell_type_key].unique()}")
            else:
                logger.error(f"Cell type key '{cell_type_key}' not in reference data!")

    # Show completion message
    print("\n" + "="*80)
    print(f"Pipeline completed successfully!")
    if args.classify and 'results_path' in locals():
        print(f"Results saved to: {results_path}")
    print("="*80 + "\n")

        
        


if __name__ == '__main__':
    main()
