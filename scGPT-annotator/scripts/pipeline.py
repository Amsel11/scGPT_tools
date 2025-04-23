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
from utils import test_embed_config
from cellxgene_download import download_cellxgene_v2

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Add the conversion function after imports
def convert_np_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_np_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_np_types(obj.tolist())
    else:
        return obj

def create_argparser():
    defaults = {
        # Model configuration
        "model_dir": "/root/scGPT_dir/scGPT/data/scGPT_Human",
        "checkpoint_name": "best_model.pt",
        "embedding_dim": 512,
        "num_heads": 8,
        "num_layers": 12,
        "max_seq_len": 1200,
        "query_file": r"C:\Users\annel\OneDrive\Documenten\Machine Learning\scGPT\scGPT-annotator\data\Derived_Embryoid_Bodies_all_embeds.h5ad",
        
        # Data configuration
        "gene_col": None,  # Will be detected from metadata if not specified
        "batch_size": 64,
        "max_genes": 2000,
        "use_highly_variable": False,
        "cell_type_col": None,
        "batch_key": None,
        
        # Other settings 
        "output_dir": None,
        "classifier_file": None,
        "classifier": "randomforest",
        "n_top_predictions": 5,
        "pred_cell_type_key": "pred_cell_type",
    }

    parser = argparse.ArgumentParser(
        description='scGPT Cell Type Annotation Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  # Show defaults in help
    )
    
    # Create argument groups for better organization
    io_group = parser.add_argument_group('Input/Output Options')
    io_group.add_argument('--query_file', default=None,
                       help='Path to the query h5ad file containing cells to analyze/annotate. Required unless --download_data is used.')
    io_group.add_argument('--ref_file', default=None, 
                       help='Path to reference h5ad file with annotated cells (optional)')
    io_group.add_argument('--config_file', default="scripts/scGPT_embed_config.json", 
                       help='Path to the JSON configuration file')
    io_group.add_argument('--classifier_file', default=None,
                       help='Path to a pre-trained classifier file (optional)')
    io_group.add_argument('--output_dir', default=None,
                       help='Directory to save output files (default: auto-generated)')
    io_group.add_argument('--verbose', action='store_true', default=False,
                       help='Run in verbose mode (print more information)')
    
    # Pipeline steps
    steps_group = parser.add_argument_group('Pipeline Steps')
    steps_group.add_argument('--analysis', action='store_true', default=False,
                       help='Run analysis step (extract metadata and detect cell types/genes)')
    steps_group.add_argument('--embed', action='store_true', default=False,
                       help='Run embedding step (generate scGPT embeddings)')
    steps_group.add_argument('--classify', action='store_true', default=False,
                       help='Run classification step (predict cell types)')
    steps_group.add_argument('--evaluate', action='store_true', default=False,
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
    class_group.add_argument('--classifier', default=defaults["classifier"],
                       choices=['randomforest', 'knn', 'svm', 'lightgbm'],
                       help='Type of classifier to use for cell type prediction')
    class_group.add_argument('--n_top_predictions', type=int, default=defaults["n_top_predictions"],
                       help='Number of top predictions to include in results')
    class_group.add_argument('--pred_cell_type_key', default=defaults["pred_cell_type_key"],
                       help='Column name to use for predicted cell types in output')
    
    # Other options
    other_group = parser.add_argument_group('Other Options')
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
                                     'batch_key', 'output_dir', 'classifier',
                                     'n_top_predictions', 'pred_cell_type_key']}
    add_dict_to_argparser(parser, remaining_defaults)

    return parser

def fix_reserved_column_names(adata):
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

def embed_step(adata, config):
    """Embed data using config overrides or defaults."""
    logger = logging.getLogger("scGPT")
    
    # Embeddings specific config
    embed_config = {
        'gene_col': config.get('gene_col'),
        'batch_size': config.get('batch_size'),
        'max_length': config.get('max_length'),
        'device': config.get('device'),
        'use_fast_transformer': config.get('use_fast_transformer'),
        'return_new_adata': config.get('return_new_adata'),
        'obs_to_save': config.get('obs_to_save'),
        # DO NOT include model_dir here
    }
    
    # Remove None values to use defaults
    embed_config = {k: v for k, v in embed_config.items() if v is not None}
    
    from scGPT_embedder import scGPTEmbedder
    embedder = scGPTEmbedder(
        model_dir=config.get('model_dir', "/root/scGPT_dir/scGPT/data/scGPT_Human"),
        config=embed_config
    )
    
    return embedder.embed_data(adata)


def main():
    #setup the default parsing
    parser = create_argparser()
    args = parser.parse_args()

    if args.all:
        args.analysis = True
        args.embed = True
        args.classify = True
        args.evaluate = True

    #check the args if verbose is on 
    if args.verbose:
        print(f"args: {args}")  



    from utils import build_config, setup_directories
    repo_dir, data_dir, save_dir, model_dir, directories = setup_directories()
    logging.info("Starting scGPT pipeline")
    logging.info(f"the directories are: {directories}")

 

    # First handle the query file and download logic
    if args.query_file:
        # User specified a query file directly
        query_file = args.query_file
        print(f"query_file: {query_file}")
        output_dir = save_dir / f"cellxgene_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{query_file}"
        if not Path(query_file).exists():
            logging.warning(f"Query file does not exist: {query_file}")
            if args.download_data:
                logging.info("Will download data since query file doesn't exist.")
                need_download = True
            else:
                logging.error("Exiting due to missing query file.")
                return 1
        else:
            logging.info(f"Using specified query file: {query_file}")
            need_download = False
    else:
        # No query file specified
        if args.download_data:
            logging.info("No query file specified, will download data.")
            need_download = True
            query_file = None  # Will be set after download
        else:
            logging.error("No query file specified and --download_data not used. Nothing to analyze.")
            return 1

    # Download data if needed
    if need_download:
        logging.info("=== STEP 0: Downloading example data ===")
        # Use a fixed directory for downloads
        download_dir = data_dir / "cellxgene_v2"
        download_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Download the files
            file_paths = download_cellxgene_v2(output_dir=download_dir)
            
            if not file_paths:
                logging.error("No files were downloaded.")
                return 1
            
            # If no query file was specified, use the first downloaded file
            if query_file is None:
                query_file = str(file_paths[0])
                output_dir = save_dir / f"cellxgene_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file_paths[0].stem}"
                logging.info(f"Using downloaded file as query: {query_file}")
            
            # List all available files
            logging.info(f"All available files in {download_dir}:")
            for i, path in enumerate(sorted(download_dir.glob("*.h5ad"))):
                logging.info(f"[{i+1}] {path}")
            
        except Exception as e:
            logging.error(f"Error downloading data: {str(e)}")
            return 1

    # Now check if we have a query file
    if query_file is None:
        logging.error("No query file specified. Use --query_file or --download_data")
        return 1
    
    
    # Now get the base name
    base_name = Path(query_file).stem

    #create the output directory for this run 
    if output_dir is None:
        date_str = datetime.now().strftime("%Y%m%d")
        existing = [x for x in save_dir.iterdir() if x.is_dir() and x.name.startswith(f"{date_str}_{base_name}")]
        number = len(existing) + 1
        output_dir = Path(str(save_dir)) / f"{base_name}_{date_str}_{number:02d}"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Created output directory in {save_dir}: {output_dir}")

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



    # Show CLI banner
    print("\n" + "="*80)
    logger.info(f"scGPT Cell Type Annotation Pipeline (v0.1.0)")
    print("="*80)
    print(f"Query file: {args.query_file}")
    if args.ref_file:
        print(f"Reference file: {args.ref_file}")
    print(f"Model directory: {config.get('model_dir')}")
    print(f"Save directory: {save_dir}")
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

    
    #load the data after analysis is done 
    logger.info("=== STEP 2: Loading data ===")
    from utils import AnnDataChunker
    try:
        obs_columns = important_obs_keys if args.analysis else None
    except:
        obs_columns = None
    try:
        with AnnDataChunker(query_file, obs_columns=obs_columns) as chunker:
            total_rows = len(chunker)        
            adata = chunker.load_subset(start_row=0, n_rows=total_rows)  #change this to something else if you want to load  less data
            # Load the complete file directly with scanpy instead of using chunks
        #adata = sc.read_h5ad(query_file)
        
        # Check for empty rows
        row_sums = adata.X.sum(axis=1)
        if scipy.sparse.issparse(row_sums):
            row_sums = row_sums.A1
        zero_expr_cells = np.where(row_sums == 0)[0]
        
        if len(zero_expr_cells) > 0:
            logger.warning(f"Found {len(zero_expr_cells)} cells with zero expression - removing these cells")
            adata = adata[~np.isin(np.arange(adata.n_obs), zero_expr_cells), :]
        
        logger.info(f"Loaded AnnData with {adata.n_obs} cells and {adata.n_vars} genes")
        
        # Check if embeddings exist (important for the classifier) 
        if 'X_scGPT' in adata.obsm:
            logger.info(f"Found X_scGPT embeddings of shape {adata.obsm['X_scGPT'].shape}")
            args.embed = False #we don't need to run embedding if it already exists
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


    #check if the data is empty
    if adata.n_obs == 0:
        logger.error("No data loaded - please check your query file")
        sys.exit(1)

    if args.embed:
        device = config.get("device", "cuda")
        if device != "cuda":
            logger.error("No GPU found - can't embed")
            return 1

        logger.info("Using GPU for embedding")
        
        # Try embedding -- see if it fials 
        try:
            adata_embed = embed_step(adata, config)
            
            # Check if embeddings were generated
            if 'X_scGPT' not in adata_embed.obsm:
                raise ValueError("X_scGPT not found in embedded data")
                
            logger.info(f"Successfully generated embeddings with shape {adata_embed.obsm['X_scGPT'].shape}")
            
            # Update main adata object
            adata = adata_embed
            
            # Handle embedding key
            embedding_key = config.get('embedding_key', 'X_scGPT')
            if embedding_key != 'X_scGPT':
                adata.obsm[embedding_key] = adata.obsm['X_scGPT']
                logger.info(f"Copied embeddings to new key: {embedding_key}")
                
        except Exception as e:
            logger.error(f"Embedding failed: {str(e)}")
            if not args.force_continue:
                return 1
            logger.warning("Continuing despite embedding failure (force_continue)")
            
        try:
            output_file = output_dir / f"{base_name}_embeddings.h5ad"
            adata.write_h5ad(output_file)
            logger.info(f"Saved embedded data to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save embedded data: {str(e)}")
            if not args.force_continue:
                return 1
            logger.warning("Continuing despite save failure (force_continue)")
        

    embedding_key = config.get('embedding_key', 'X_scGPT')


    if args.classify:
        logger.info("=== STEP 4: Running classification ===")
        from scGPT_classifier import scGPTAnnotator
        
        # Get the embedding key and verify it exists
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
        if config.get('classifier_file') is None:
            classifier = config.get('classifier', 'knn')
            logger.info("the classifier is: ", classifier)
            logger.info(f"Training {classifier} classifier")
            annotator.train_classifier(classifier_name=classifier, cell_type_col=cell_type_key, batch_key=batch_key)
            annotator.save_classifier(output_dir / f"{classifier}_classifier_{base_name}.pkl")
        else:
            logger.info(f"Loading classifier from {config.get('classifier_file')}")
            annotator.load_classifier(config.get('classifier_file'))

        # Predict the cell types
        logger.info("Predicting cell types")
        predicted_adata = annotator.predict(annotator.query_adata, pred_cell_col=pred_cell_type_key, store_probs=True, return_adata=True)
        logger.info("Prediction complete")
        logger.info(f"Sample of predictions:\n{predicted_adata.obs[[cell_type_key, pred_cell_type_key]].head()}")
        
        # Add top N predictions
        n_predictions = config.get('n_top_predictions', 3)
        logger.info(f"Adding top {n_predictions} predictions")
        annotator.add_top_n_predictions(predicted_adata, n=n_predictions, pred_cell_col=pred_cell_type_key, cell_type_col=cell_type_key)

        logger.info(f"Prediction data stored in adata.uns['prediction_data']")
        logger.info(predicted_adata.uns['prediction_data'])

        # Save the results
        results_path = output_dir / f"{base_name}_pred_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5ad"
        logger.info(f"Saving prediction results to {results_path}")
        annotator.save_results(predicted_adata, results_path)
        logger.info(f"Predicted results saved to {results_path}")

        
        #logger.info(f"Predicted data saved to {output_dir / 'predicted_adata.h5ad'}")
        #predicted_adata.write_h5ad(output_dir / 'predicted_adata.h5ad')

    
    # Step 5: Evaluate the results (if enabled) 
    if args.evaluate:
        from scGPT_classifier import scGPTAnnotator
        logger.info("=== STEP 5: Evaluating prediction results ===")
        
        try:
            # First, verify we have the necessary data for evaluation
            if cell_type_key not in predicted_adata.obs.columns:
                logger.warning(f"Cannot evaluate: missing true cell type column '{cell_type_key}'")
                logger.warning("Skipping evaluation step")
            elif pred_cell_type_key not in predicted_adata.obs.columns:
                logger.warning(f"Cannot evaluate: missing predicted cell type column '{pred_cell_type_key}'")
                logger.warning("Skipping evaluation step")
            else:
                # Check if we have non-null values
                valid_mask = ~predicted_adata.obs[cell_type_key].isna() & ~predicted_adata.obs[pred_cell_type_key].isna()
                valid_count = valid_mask.sum()
                
                if valid_count == 0:
                    logger.warning("Cannot evaluate: no valid cells with both true and predicted cell types")
                    logger.warning("Skipping evaluation step")
                else:
                    logger.info(f"Evaluating {valid_count} cells with both true and predicted cell types")
                    
                    # Debug information about classes
                    true_classes = set(predicted_adata.obs[cell_type_key][valid_mask].unique())
                    pred_classes = set(predicted_adata.obs[pred_cell_type_key][valid_mask].unique())
                    common_classes = true_classes.intersection(pred_classes)
                    
                    logger.info(f"True classes: {len(true_classes)}, Predicted classes: {len(pred_classes)}")
                    logger.info(f"Common classes: {len(common_classes)}")
                    
                    # Initialize annotator
                    annotator = scGPTAnnotator(embedding_key=embedding_key)
                    
                    # Run evaluation with better error handling
                    valid_classes, predicted_adata, results = annotator.evaluate_with_visuals(
                        predicted_adata, 
                        y_pred=pred_cell_type_key, 
                        y_true=cell_type_key
                    )
                    
                    # Save the evaluation results even if there's an error
                    logger.info(f"Saving evaluation results to {output_dir / 'evaluation_results.json'}")
                    try:
                        with open(output_dir / 'evaluation_results.json', 'w') as f:
                            # Convert NumPy types before saving
                            converted_results = convert_np_types(results)
                            json.dump(converted_results, f, indent=2)
                    except Exception as e:
                        logger.error(f"Error saving results to JSON: {str(e)}")
                        logger.info("Saving in simpler format without indent")
                        try:
                            # Fallback to simpler JSON format
                            with open(output_dir / 'evaluation_results.json', 'w') as f:
                                converted_results = convert_np_types(results)
                                json.dump(converted_results, f)
                        except Exception as e2:
                            logger.error(f"Failed to save results: {str(e2)}")
                    
                    # Only continue with top-N analysis if we have valid classes
                    if len(valid_classes) > 0:
                        # Display top-N predictions table in log
                        n_predictions = config.get('n_top_predictions', 3)
                        logger.info(f"\n=== Top {n_predictions} Predictions Analysis ===")
                        
                        # Create a dataframe focusing on true labels, predicted labels, and top predictions
                        top_cols = [cell_type_key, pred_cell_type_key]
                        for i in range(1, n_predictions+1):
                            col_name = f"top{i}_type"
                            if col_name in predicted_adata.obs.columns:
                                top_cols.append(col_name)
                                top_cols.append(f"top{i}_prob")
                        
                        # Add true_in_top_n if it exists
                        if 'true_in_top_n' in predicted_adata.obs.columns:
                            top_cols.append('true_in_top_n')
                            in_top_n_pct = predicted_adata.obs['true_in_top_n'].mean() * 100
                            logger.info(f"True cell type found in top {n_predictions} predictions: {in_top_n_pct:.2f}%")
                        
                        # Display a sample of the prediction table if we have the columns
                        if all(col in predicted_adata.obs.columns for col in top_cols):
                            pred_sample = predicted_adata.obs[top_cols].head(10)
                            logger.info(f"Sample of top predictions (first 10 cells):\n{pred_sample}")
                            
                            # Save a CSV with all predictions for easier analysis
                            csv_path = output_dir / f"top_{n_predictions}_prediction_results.csv"
                            predicted_adata.obs[top_cols].to_csv(csv_path)
                            logger.info(f"Full top-{n_predictions} prediction table saved to {csv_path}")
                        else:
                            logger.warning("Missing expected top prediction columns - skipping prediction table display")
        
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            logger.error(traceback.format_exc())
            logger.warning("Continuing despite evaluation error")
    else:
        logger.info("Evaluation step skipped (not enabled)")
 


    # Show completion message
    print("\n" + "="*80)
    print(f"Pipeline completed successfully!")
    print(f"Log saved to the output directory {output_dir}")
    if args.classify and 'results_path' in locals():
        print(f"Results saved to: {results_path}")
    print("="*80 + "\n")

        
        


if __name__ == '__main__':
    main()
