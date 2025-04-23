#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data embedding and analysis for scGPT model, using scGPT model. 

This script:
1. Loads AnnData (h5ad) files with caching for faster repeated access
2. Detects cell type, gene, batch and other annotations automatically
3. Extracts metadata needed for scGPT embedding
4. Saves analysis results as text and structured metadata as JSON 
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
from utils import _load_anndata
from utils import setup_directories
from scgpt.tasks.cell_emb import embed_data as scgpt_embed_data
from typing import Optional, Union, Dict


# === TEST FUNCTION ===




class scGPTEmbedder:
    """Simple wrapper for scGPT embedding functionality."""
    
    def __init__(self, model_dir: Union[str, Path], config: Optional[Dict] = None):
        """
        Initialize embedder with model directory and optional config overrides.
        
        Args:
            model_dir: Directory containing model files
            config: Optional configuration to override defaults from cell_emb.py
        """
        self.model_dir = Path(model_dir)
        self.config = config or {}  # Use empty dict if no config provided
        self.logger = logging.getLogger("scGPT")
    
    def validate_setup(self, adata: sc.AnnData) -> bool:
        """Validate setup before embedding."""
        self.logger.info("Validating setup for embedding generation...")
        
        # Check model files
        required_files = ['vocab.json', 'args.json', 'best_model.pt']
        for file in required_files:
            if not (self.model_dir / file).exists():
                raise ValueError(f"Required file {file} not found in {self.model_dir}")
        
        # Check AnnData
        if adata.n_obs == 0 or adata.X is None:
            raise ValueError("Invalid AnnData: empty or missing expression matrix")
            
        # Check gene column if specified in config
        gene_col = self.config.get('gene_col')
        if gene_col and gene_col != 'index' and gene_col not in adata.var.columns:
            raise ValueError(f"Gene column '{gene_col}' not found in adata.var")
        
        self.logger.info("Validation passed!")
        return True
    
    def embed_data(self, adata: sc.AnnData) -> sc.AnnData:
        """Generate embeddings using config or defaults from cell_emb.py"""
        self.validate_setup(adata)
        
        try:
            # Use config values where provided, cell_emb.py defaults otherwise
            embedded_adata = scgpt_embed_data(
                adata_or_file=adata,
                model_dir=self.model_dir,
                **self.config  # Pass any config overrides
            )
            
            self.logger.info(f"Generated embeddings with shape: {embedded_adata.obsm['X_scGPT'].shape}")
            return embedded_adata
            
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {str(e)}")
            raise
