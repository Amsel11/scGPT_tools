"""
scGPT Cell Classifier - Main Entry Point

This script provides a unified command-line interface for the single-cell classification pipeline.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Import component modules
from utils import setup_directories
import scGPT_dataloader as dataloader
import scGPT_embedder as embedder
from scGPT_classifier import scGPTAnnotator

def create_argparser():
    """Create command-line argument parser"""
    parser = argparse.ArgumentParser(description="scGPT Cell Classifier")
    
    # Data loading and preprocessing
    parser.add_argument('--query_file', type=str, required=True, help="Path to input file of single cell data")
    parser.add_argument('--ref_file', type=str, required=True, help="Path to reference file of single cell data")
    parser.add_argument('--output_dir', type=str, default=f'output_{datetime.now().strftime("%Y%m%d")}', help="Output directory for results")

    
    


        
