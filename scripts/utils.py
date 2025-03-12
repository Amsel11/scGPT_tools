#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utility functions for scGPT scripts
"""

import json
from pathlib import Path
import pprint

def test_embed_config(adata=None, config_path=None, metadata_path=None):
    """
    ONLY test configuration parsing without ANY model loading or embedding
    
    Args:
        adata: Optional AnnData object (only used to check column names)
        config_path: Path to config file
        metadata_path: Path to metadata file
        
    Returns:
        dict: The configuration that would be used
    """
    
    print("🔍 TESTING CONFIGURATION ONLY - NO MODEL WILL BE LOADED")
    
    # Resolve config path
    script_dir = Path(__file__).parent
    config_paths_to_try = [
        Path(config_path) if config_path else None,
        script_dir / "scGPT_embed_config.json",
        Path("scGPT_embed_config.json")
    ]
    config_paths_to_try = [p for p in config_paths_to_try if p]
    
    # Try each path
    config = None
    for path in config_paths_to_try:
        if path.exists():
            print(f"📄 Found config at: {path}")
            with open(path, 'r') as f:
                config = json.load(f)
            break
    
    if not config:
        print(f"❌ Config file not found. Tried: {[str(p) for p in config_paths_to_try]}")
        return None
        
    # Check metadata
    metadata = None
    if metadata_path:
        try:
            metadata_path = Path(metadata_path)
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                print(f"📄 Loaded metadata from: {metadata_path}")
                print("\nMetadata content:")
                pprint.pprint(metadata)
            else:
                print(f"❌ Metadata file not found: {metadata_path}")
        except Exception as e:
            print(f"❌ Error loading metadata: {e}")
    
    # Print config
    print("\nConfig that would be used:")
    pprint.pprint(config)
    
    # If we have adata, check column compatibility
    if adata is not None:
        print("\nChecking data compatibility:")
        gene_col = config["data"]["gene_col"]
        
        print(f"- Available var columns: {list(adata.var.columns)}")
        print(f"- Gene column setting: {gene_col}")
        
        if gene_col in adata.var.columns:
            print(f"✅ '{gene_col}' exists as a column")
        else:
            print(f"❌ '{gene_col}' NOT found in var columns")
    
    return config

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test configuration utilities")
    parser.add_argument('--config', type=str, help='Config file path')
    parser.add_argument('--metadata', type=str, help='Metadata file path')
    args = parser.parse_args()
    
    test_embed_config(config_path=args.config, metadata_path=args.metadata)