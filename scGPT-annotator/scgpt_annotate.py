#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scGPT Cell Type Annotator

This is the main entry point for annotating single-cell RNA-seq data using scGPT embeddings.
"""

import os
import sys
import logging
from pathlib import Path
import traceback

def setup_environment():
    script_path = Path(__file__).resolve()
    root_dir = script_path.parent
    
    paths_to_add = [
        str(root_dir),                    # Main project directory
        str(root_dir / "scripts"),        # Scripts directory
    ]
    
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    return root_dir

#some simple logging setup 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("scGPT_annotate")

#this calls the scGPT annotator pipeline
def main():
    root_dir = setup_environment()
    logger.info("Environment setup complete")
    
    try:
        logger.info("Importing pipeline module...")
        from scripts.pipeline_1 import main as pipeline_main
        logger.info("Successfully imported pipeline module")
        return pipeline_main()
        
    except ModuleNotFoundError as e:
        logger.error(f"Error importing required modules: {e}")
        logger.error(f"Current working directory: {os.getcwd()}")
        logger.error(f"Python path: {sys.path}")
        logger.error(f"Project root directory: {root_dir}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())