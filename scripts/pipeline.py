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
    import torch

    # Import the scripts from this repository
    import scGPT_dataloader
    import scGPT_embedder
    from utils import add_dict_to_argparser
    from utils import str2bool

def create_argparser():
    # Default configuration for everything
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
        "use_highly_variable": True,
        
        # Other settings
        "output_dir": None
    }
    
    parser = argparse.ArgumentParser(description='scGPT pipeline')
    
    # Main arguments
    parser.add_argument('--input_file', default="data/Derived_Embryoid_Bodies.h5ad", 
                      help='Path to the input file')
    parser.add_argument('--config_file', default="scripts/scGPT_embed_config.json", 
                      help='Path to the JSON config file')
    parser.add_argument('--analysis', action='store_true', default=True,
                      help='Run analysis step')
    parser.add_argument('--embed', action='store_true', default=False,
                      help='Run embedding step')
    parser.add_argument('--testing', action='store_true', default=True,
                      help='Use testing directory')
    
    # Add all default configs as optional arguments
    add_dict_to_argparser(parser, defaults)
    
    return parser



def build_config(args, metadata=None):
    """Build final configuration using args and metadata"""
    # Start with args as dictionary
    config = vars(args).copy()
    
    if metadata:
        if config['gene_col'] is None and metadata['gene_keys']:
            config['gene_col'] = metadata['gene_keys'][0] #this can be changed to use another variable gene
        
        # Add observation columns to save
        config['obs_to_save'] = []
        if metadata.get('cell_type_keys'):
            config['obs_to_save'].extend(metadata['cell_type_keys'])
        if metadata.get('batch_keys'):
            config['obs_to_save'].extend(metadata['batch_keys'])
        if metadata.get('hvg_keys'):
            config['obs_to_save'].extend(metadata['hvg_keys'])
        
        if 'model_dir' in metadata['directories']:
            config['model_dir'] = metadata['directories']['model_dir']

    
    return config

def main():
    # Parse arguments using the create_argparser function and defaults 
    parser = create_argparser()
    args = parser.parse_args()
    
    # Setup directories
    from utils import setup_directories
    repo_dir, data_dir, save_dir, model_dir, directories = setup_directories()
    
    # Create output directory
    if args.testing:
        output_dir = save_dir / f"test_output_{datetime.now().strftime('%Y%m%d')}"
    else:
        base_name = Path(args.input_file).stem
        date_str = datetime.now().strftime("%Y%m%d")
        existing = [x for x in save_dir.iterdir() if x.is_dir() and x.name.startswith(f"{base_name}_{date_str}")]
        number = len(existing) + 1
        output_dir = save_dir / f"{base_name}_{date_str}_{number:02d}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load data
    adata = scGPT_dataloader.load_h5ad(args.input_file, save_dir=save_dir, subset=None, force_reload=False)
    print(f"Loaded data: {adata.shape[0]} cells × {adata.shape[1]} genes")
    
    # Metadata file path
    base_name = Path(args.input_file).stem
    metadata_file = output_dir / f"{base_name}_metadata.json"
    found_keys = None
    
    # Analysis mode
    if args.analysis:
        print("Running analysis...")
        found_keys = scGPT_dataloader.check_annotation_keys(adata)
        scGPT_dataloader.save_metadata_json(found_keys, metadata_file)
        scGPT_dataloader.print_summary(adata, found_keys, output_dir, metadata_file, None)
    
    # Embedding mode
    if args.embed:
        print("Preparing for embedding...")
        
        # Configuration priority:
        # 1. Command line args (already in args)
        # 2. Load metadata if available
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    found_keys = json.load(f)
                print(f"Loaded metadata from {metadata_file}")
            except Exception as e:
                print(f"Error loading metadata: {e}")
        
        # 3. Build initial config from args and metadata
        config = build_config(args, found_keys)
        
        # 4. Override with config file if provided
        if args.config_file and os.path.exists(args.config_file):
            try:
                with open(args.config_file, 'r') as f:
                    file_config = json.load(f)
                # Update config with file values but don't override metadata
                for k, v in file_config.items():
                    # Only use config file value if not set by metadata
                    if k == 'gene_col' and config.get('gene_col') is not None:
                        continue  # Skip if already set by metadata
                    if k == 'obs_to_save' and config.get('obs_to_save'):
                        continue  # Skip if already set by metadata
                    config[k] = v
                print(f"Applied configuration from {args.config_file}")
            except Exception as e:
                print(f"Error loading config file: {e}")
        
        # Ensure output_dir is set
        config['output_dir'] = str(output_dir)
        
        # Print final configuration
        print("\nEmbedding with configuration:")
        for key in ['model_dir', 'gene_col', 'batch_size', 'max_seq_len']:
            if key in config:
                print(f"  {key}: {config[key]}")
        
        # Run embedding with the final config
        print("Starting embedding process...")
        scGPT_embedder.embed_data(adata, config_path=args.config_file, metadata_path=metadata_file, **config)
        print("Embedding complete!")

if __name__ == '__main__':
    main()