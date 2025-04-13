#continual training and finetuning -- what if we have to upscale 

#imports
# %%
import copy
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from typing import List, Tuple, Dict, Union, Optional
import warnings
import pandas as pd
# from . import asyn
import pickle
import torch
from anndata import AnnData
import scanpy as sc
import scvi
import seaborn as sns
import numpy as np
import wandb
from scipy.sparse import issparse
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)
from sklearn.metrics import confusion_matrix

sys.path.insert(0, "../")
import scgpt as scg
from scgpt.model import TransformerModel, AdversarialDiscriminator
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.loss import (
    masked_mse_loss,
    masked_relative_error,
    criterion_neg_log_bernoulli,
)
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.preprocess import Preprocessor
from scgpt import SubsetsBatchSampler
from scgpt.utils import set_seed, category_str2int, eval_scib_metrics

sc.set_figure_params(figsize=(6, 6))
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings('ignore')

import argparse
from utils import add_dict_to_argparser
from utils import str2bool
from utils import build_config
from utils import setup_logging
from utils import AnnDataChunker

##

#what do we need? 
def create_argparser():
    defaults = dict(
        seed=0,
        dataset_name="ms",
        do_train=True,
        load_model="../save/scGPT_human",
        mask_ratio=0.0,
        epochs=10,
        n_bins=51,
        MVC=False,  # Masked value prediction for cell embedding
        ecs_thres=0.0,  # Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable
        dab_weight=0.0,
        lr=1e-4,
        batch_size=32,
        layer_size=128,
        nlayers=4,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        nhead=4,  # number of heads in nn.MultiheadAttention
        dropout=0.2,  # dropout probability
        schedule_ratio=0.9,  # ratio of epochs for learning rate schedule
        save_eval_interval=5,
        fast_transformer=True,
        pre_norm=False,
        amp=True,  # Automatic Mixed Precision
        include_zero_gene=False,
        freeze=False,  # freeze
        DSBN=False,  # Domain-spec batchnorm
    )
    
    parser = argparse.ArgumentParser(description='scGPT training and evaluation script')
    add_dict_to_argparser(parser, defaults)

    return parser


#a config file and defualts
#good logging
#data to train on 
#   #I could do an extra function here to load the data which is related to the scGPTd dataloader
#   #But, there is of course also already their own scGPT dataloader 
#data to test on 

def load_and_process_data_in_chunks(file_path, model_dir, n_bins, gene_column="ensembl_id", chunk_size=10000):
    """
    Load and preprocess a large dataset in chunks
    
    Args:
        file_path: Path to the h5ad file
        model_dir: Path to the pretrained model directory
        n_bins: Number of bins for preprocessing
        gene_column: Column in var containing gene identifiers to match with model vocab
        chunk_size: Number of cells to process at once
    """
    logger = logging.getLogger("scGPT_pipeline")
    logger.info(f"Loading and processing data in chunks from {file_path}")
    logger.info(f"Using gene column: {gene_column}")
    
    # Load vocab from the model
    vocab_file = Path(model_dir) / "vocab.json"
    if not vocab_file.exists():
        raise FileNotFoundError(f"Vocabulary file not found at {vocab_file}")
    
    with open(vocab_file, "r") as f:
        vocab_dict = json.load(f)
    
    gene_ids = vocab_dict.get("gene_ids", [])
    logger.info(f"Loaded vocabulary with {len(gene_ids)} genes")
    
    # Check format of gene_ids in vocabulary
    sample_gene_id = gene_ids[0] if gene_ids else ""
    logger.info(f"Sample gene ID from model vocabulary: {sample_gene_id}")
    
    # Set up the preprocessor
    preprocessor = Preprocessor(
        use_key="X",
        filter_gene_by_counts=False,
        filter_cell_by_counts=False,
        normalize_total=1e4,
        log1p=True,
        binning=n_bins,
        result_binned_key="X_binned"
    )
    
    # Process in chunks
    with AnnDataChunker(file_path, obs_columns=None) as chunker:
        # Get var dataframe 
        var_df = chunker.var
        
        # Verify the gene column exists
        if gene_column not in var_df.columns and gene_column != "index":
            available_columns = list(var_df.columns)
            logger.error(f"Gene column '{gene_column}' not found. Available columns: {available_columns}")
            raise ValueError(f"Gene column '{gene_column}' not found in dataset")
        
        # Get gene values and clean them if needed (e.g., remove byte prefix)
        if gene_column == "index":
            gene_values = var_df.index
        else:
            gene_values = var_df[gene_column]
            # Clean gene values if they're bytes (like b'ENSG00000000003')
            if isinstance(gene_values.iloc[0], str) and gene_values.iloc[0].startswith("b'"):
                gene_values = gene_values.apply(lambda x: x[2:-1] if x.startswith("b'") else x)
        
        # Create a mapping for matching
        logger.info(f"Creating gene matching map using column: {gene_column}")
        var_df["in_vocab"] = [gene in gene_ids for gene in gene_values]
        valid_indices = np.where(var_df["in_vocab"])[0]
        
        logger.info(f"Found {len(valid_indices)} matching genes out of {len(var_df)}")
        if len(valid_indices) < 0.5 * len(gene_ids):
            logger.warning("Less than 50% of model genes found in dataset. Check gene identifier format!")
            
            # Show some examples to help diagnose mismatches
            sample_data_genes = list(gene_values[:5])
            sample_vocab_genes = gene_ids[:5]
            logger.warning(f"Sample dataset genes: {sample_data_genes}")
            logger.warning(f"Sample vocabulary genes: {sample_vocab_genes}")
            
        # Process each chunk
        processed_chunks = []
        total_rows = len(chunker)
        
        for chunk_idx, start_row in enumerate(range(0, total_rows, chunk_size)):
            current_chunk_size = min(chunk_size, total_rows - start_row)
            logger.info(f"Processing chunk {chunk_idx+1} starting at row {start_row} with {current_chunk_size} cells")
            
            # Load this subset with only the valid gene columns
            chunk = chunker.load_subset(start_row, current_chunk_size, valid_indices=valid_indices)
            
            # Apply preprocessing to this chunk
            preprocessor(chunk, batch_key=None)
            
            # Store processed chunk
            processed_chunks.append(chunk)
            
            logger.info(f"Finished processing chunk {chunk_idx+1}")
        
        logger.info(f"Completed processing {len(processed_chunks)} chunks")
        return processed_chunks, gene_ids, vocab_dict

#need to load the model (obv) 
#and how we need to load the model 

#we are going to base this mostly on the tutorial for cell annotation fine tuning 


def main():
    from utils import setup_logging
    setup_logging()

    logging = logging.getLogger("scGPT continual training")

    parser = create_argparser()
    args = parser.parse_args()

    print(f"Running with parameters:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    from utils import build_config, setup_directories
    repo_dir, data_dir, save_dir, model_dir, directories = setup_directories()
    logging.info("Starting scGPT training")
    logging.info(f"the directories are: {directories}")

    # Load the model config
    from utils import build_model_config
    model_config = build_model_config(args)

    # %% validate settings
    assert model_config.get("input_style") in ["normed_raw", "log1p", "binned"]
    assert model_config.get("output_style") in ["normed_raw", "log1p", "binned"]
    assert model_config.get("input_emb_style") in ["category", "continuous", "scaling"]
    if model_config.get("input_style") == "binned":
        if model_config.get("input_emb_style") == "scaling":
            raise ValueError("input_emb_style `scaling` is not supported for binned input.")
    elif model_config.get("input_style") == "log1p" or model_config.get("input_style") == "normed_raw":
        if model_config.get("input_emb_style") == "category":
            raise ValueError(
                "input_emb_style `category` is not supported for log1p or normed_raw input."
            )

    if model_config.get("input_emb_style") == "category":
        mask_value = model_config.get("n_bins") + 1
        pad_value = model_config.get("n_bins")  # for padding gene expr values
        n_input_bins = model_config.get("n_bins") + 2
    else:
        mask_value = -1
        pad_value = -2
        n_input_bins = model_config.get("n_bins")

    if model_config.get("ADV") and model_config.get("DAB"):
        raise ValueError("ADV and DAB cannot be both True.")
    DAB_separate_optim = True if model_config.get("DAB") > 1 else False





if __name__ == "__main__":
    main()


