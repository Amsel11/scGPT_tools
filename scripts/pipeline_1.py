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
        "model_dir": "models/scGPT_human",
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
        "output_dir": None
    }

    parser = argparse.ArgumentParser(description='scGPT pipeline')
    parser.add_argument('--query_file', default="data/Derived_Embryoid_Bodies.h5ad", #change this later to a required and leave empty
                      help='Path to the query file')
    parser.add_argument('--ref_file', default = None, help = 'Path to the reference file')
    parser.add_argument('--config_file', default="scripts/scGPT_embed_config.json", 
                      help='Path to the JSON config file')
    parser.add_argument('--analysis', action='store_true', default=True,
                      help='Run analysis step')
    parser.add_argument('--embed', action='store_true', default=False,
                      help='Run embedding step')
    parser.add_argument('--classify', action='store_true', default=False,
                      help='Run classification step')
    parser.add_argument('--testing', action='store_true', default=True,
                      help='Use testing directory')
    parser.add_argument('--disable_file_logging', action='store_true', default=False,
                      help='Disable file logging')

    add_dict_to_argparser(parser, defaults)

    return parser


def test_config():
    """Test the configuration system"""
    print("\n=== TESTING CONFIGURATION SYSTEM ===")
    
    # 1. Test with command-line args only
    print("\nTest 1: Command-line args only")
    parser = create_argparser()
    test_args = parser.parse_args(['--gene_col', 'feature_name', '--batch_size', '96'])
    config = build_config(test_args)
    print(f"Config from args:")
    print(f"  gene_col: {config.get('gene_col')}")
    print(f"  batch_size: {config.get('batch_size')}")
    print(f"  model_dir: {config.get('model_dir')}")
    
    # 2. Test with config file
    print("\nTest 2: With config file")
    test_args = parser.parse_args(['--config_file', 'scripts/test_config.json'])
    config = build_config(test_args)
    print(f"Config with file:")
    print(f"  model_dir: {config.get('model_dir')}")
    print(f"  batch_size: {config.get('batch_size')}")
    print(f"  max_seq_len: {config.get('max_seq_len')}")
    
    # 3. Test with metadata
    print("\nTest 3: With metadata")
    test_args = parser.parse_args([])
    
    # Create mock metadata
    metadata = {
        'found_keys_by_category': {
            'gene_keys': ['ensembl_id', 'feature_name'],
            'cell_type_keys': ['cell_type', 'cell_type_ontology_term_id'],
            'batch_keys': ['donor_id', 'sample']
        }
    }
    
    config = build_config(test_args, metadata)
    print(f"Config with metadata:")
    print(f"  gene_col: {config.get('gene_col')}")
    print(f"  cell_type_col: {config.get('cell_type_col')}")
    print(f"  batch_key: {config.get('batch_key')}")
    
    # 4. Test priority (command-line args > config file > metadata)
    print("\nTest 4: Testing priority")
    test_args = parser.parse_args([
        '--gene_col', 'my_gene_col',
        '--config_file', 'scripts/test_config.json'
    ])
    config = build_config(test_args, metadata)
    print(f"Config with priority test:")
    print(f"  gene_col: {config.get('gene_col')}  # Should be 'my_gene_col' from command line")
    print(f"  model_dir: {config.get('model_dir')}  # Should be from config file")
    print(f"  batch_key: {config.get('batch_key')}  # Should be from metadata")
    
    print("\n=== CONFIGURATION TESTING COMPLETE ===")



def main():
    #setup parsing
    parser = create_argparser()
    args = parser.parse_args()

    from utils import build_config, setup_directories
    repo_dir, data_dir, save_dir, model_dir, directories = setup_directories()
    
    # Create output directory - the if statement is only for testing 
    if args.testing:
        output_dir = save_dir / f"test_output_{datetime.now().strftime('%Y%m%d')}"
        output_dir = save_dir / f"test_output_{datetime.now().strftime('%Y%m%d')}"
        if args.testing:
            # Pass testing flag to setup_logging to disable file output
            args.disable_file_logging = True
    else:
        base_name = Path(args.query_file).stem
        date_str = datetime.now().strftime("%Y%m%d")
        existing = [x for x in save_dir.iterdir() if x.is_dir() and x.name.startswith(f"{base_name}_{date_str}")]
        number = len(existing) + 1
        output_dir = save_dir / f"{base_name}_{date_str}_{number:02d}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    query_file = args.query_file 
    print(query_file)

    #setup logging
    from utils import setup_logging
    logger = setup_logging(output_dir, disable_file_logging=args.disable_file_logging)
    logger.info("Starting scGPT pipeline")

    initial_config = build_config(args)
    initial_config_path = output_dir / "initial_config.json"
    with open(initial_config_path, 'w') as f:
        json.dump(initial_config, f, indent=4)
    logger.info(f"Initial config saved to {initial_config_path}")

    # Log some key config values
    logger.info(f"Initial config values:")
    logger.info(f"  model_dir: {initial_config.get('model_dir')}")
    logger.info(f"  gene_col: {initial_config.get('gene_col')}")
    logger.info(f"  batch_size: {initial_config.get('batch_size')}")
    logger.info(f"  device: {initial_config.get('device')}")

    Metadata = None

    # Step 1: Analysis (if enabled)
    if args.analysis:
        logger.info("=== STEP 1: Running analysis ===")
        try:
            from scGPT_dataloader import analysis_meta
            metadata = analysis_meta(query_file, save=True, output_dir=output_dir)
            
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
        important_obs_keys = []
        important_var_keys = []
        logger.warning("No metadata available - will load all columns")

    
    #load the data
    logger.info("=== STEP 2: Loading data ===")
    from utils import AnnDataChunker
    try:
        obs_columns = important_obs_keys if args.analysis else None
    except:
        obs_columns = None

    with AnnDataChunker(query_file, obs_columns=obs_columns) as chunker:
        total_rows = len(chunker)        
        adata = chunker.load_subset(start_row=0, n_rows=total_rows)  #change this to something else if you want to load  less data
        logger.info(f"Loaded AnnData with {adata.n_obs} cells and {adata.n_vars} genes")
        logger.info("First 5 rows of adata.obs:")
        logger.info("\n" + str(adata.obs.head()))

    
    # The config comes from either:
    # 1. Command line args processed earlier in the script
    # 2. Default config file (scripts/scGPT_embed_config.json)
    # 3. User-provided config file via --config argument
    # The final config is stored in output_dir/pipeline_config.json
    


    # Step 3: Embedding (if enabled)
    if args.embed:
        import scgpt as scg

        logger.info("=== STEP 3: Running embedding ===")
        device =  "cuda" if torch.cuda.is_available() else "cpu",

        logger.info("Current configuration:")
        #for key, value in config.items():
         #   logger.info(f"  {key}: {value}")
            
        gene_col = config.get("gene_col")
        batch_size = config.get("batch_size", 64)  # Default to 64 if not specified
        model_dir = config.get("model_dir")
        
        if not gene_col:
            logger.warning("No gene_col specified in config - will use index")
        if not model_dir:
            logger.error("No model_dir specified in config - required for embedding/classification")
        
        logger.debug(f"Using gene_col: {gene_col}")
        logger.debug(f"Using batch_size: {batch_size}")
        logger.debug(f"Using model_dir: {model_dir}")


        if device == "cuda":
            logger.info("Using GPU for embedding")
            embedding_adata = scg.tasks.embed_data(
                adata,
                gene_col=gene_col,
                batch_size=batch_size,
                model_dir=model_dir,
                device="cuda"
            )
            embedding_file = os.path.join(output_dir, "embeddings.h5ad")
            logger.info(f"Saving embeddings to {embedding_file}")
            try:
                embedding_adata.write(embedding_file)
                logger.info("Successfully saved embeddings")
            except Exception as e:
                logger.error(f"Failed to save embeddings: {e}", exc_info=True)
                if not getattr(args, 'force_continue', False):
                    return 1
                logger.warning("Continuing despite embedding save error (force_continue)")
        else:
            logger.info("No GPU available, cannot do embedding.")

    if args.classify:
        logger.info("=== STEP 4: Running classification ===")
        from scGPT_classifier import scGPTAnnotator
        annotator = scGPTAnnotator(model_dir= config["model_dir"], embedding_key= "X_scGPT") #config needs to be configured, and embeding key needs to come from config
        has_query_embeddings, query_message = annotator.check_embeddings(args.query_file)
        if not has_query_embeddings:
            logger.error(f"Error: {query_message}")
            logger.error("Cannot proceed without embeddings in query file.")
            sys.exit(1)
        
        print(f"Query file: {query_message}")
        annotator.set_query_data(adata) #set the query data

        cell_type_key = metadata['cell_type_keys'][0] if metadata.get('cell_type_keys') else None #to make sure it doesn't error
        batch_key = metadata['batch_keys'][1] if metadata.get('batch_keys') else None #to make sure it doesn't error if it's not there 

        print(f"Using cell type key: {cell_type_key}")
        print(f"Using batch key: {batch_key}")

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
                    annotator.split_query_ref(adata, method='random')
            else:
                # Reference file has embeddings, use it
                print(f"Reference file: {ref_message}")
                ref_adata = chunker.load_subset(start_row=0, n_rows=total_rows)
                annotator.set_ref_data(ref_adata)
        else:
            # No reference file provided, split query data
            print("No reference file provided. Creating reference from query data.")
            #there needs to be an extra check that the query data has cell_type information 
            
            # Use batch key from metadata
            if batch_key and batch_key in adata.obs:
                if len(adata.obs[batch_key].unique()) > 1:
                    print(f"Splitting query data using {batch_key} information")
                    print(f"Available batches: {adata.obs[batch_key].unique()}")
                    annotator.split_query_ref(adata, method='batch', batch_key=batch_key)
                else:
                    print(f"Only one {batch_key} found. Using random split.")
                    annotator.split_query_ref(adata, method='random')
            else:
                print("No valid batch key in metadata. Using random split.")
                annotator.split_query_ref(adata, method='random')

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

        
        


if __name__ == '__main__':
    main()
