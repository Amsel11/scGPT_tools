#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Continual Pretraining Script for scGPT
"""

import argparse
import logging
import os
import torch
import scanpy as sc
import numpy as np
from pathlib import Path
import scgpt as scg
from scgpt.model import TransformerModel, AdversarialDiscriminator
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.loss import (
    masked_mse_loss,
    masked_relative_error,
    criterion_neg_log_bernoulli,
)
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from typing import Dict, Tuple, Union, List, Optional


from scgpt.preprocess import Preprocessor
from scgpt import SubsetsBatchSampler
from scgpt.utils import set_seed, category_str2int, eval_scib_metrics
import datetime
import warnings
import json
from scipy.sparse import issparse
from sklearn.model_selection import train_test_split
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

sc.set_figure_params(figsize=(6, 6))
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings('ignore')

# Import your existing utilities
from utils import (
    setup_directories,
    build_config,
    build_model_config,
    setup_logging,
    add_dict_to_argparser,
    str2bool
)

class PretrainingError(Exception):
    """Base exception class for pretraining errors."""
    pass

class DatasetError(PretrainingError):
    """Exception raised for errors in dataset processing."""
    pass

class ModelConfigError(PretrainingError):
    """Exception raised for model configuration errors."""
    pass

class TrainingError(PretrainingError):
    """Exception raised for errors during training."""
    pass

def create_argument_parser():
    """Create argument parser with sensible defaults for continual pretraining."""
    defaults = dict(
        # Basic settings
        seed=42,
        do_train=False,
        do_eval=False,
        
        # Path settings
        load_model="/root/scGPT_dir/scGPT/data/scGPT_Human",
        output_dir=None,  # Will be set automatically if None
        
        # Data processing
        mask_ratio=0.15,
        include_zero_gene=True,
        max_seq_len=3001,
        n_bins=51,
        data_is_raw=False,
        filter_gene_by_counts=False,
        preprocess=False,
        
        # Input/output representation
        input_style="binned",  # "normed_raw", "log1p", or "binned"
        output_style="binned",  # "normed_raw", "log1p", or "binned"
        input_emb_style="continuous",  # "category", "continuous", or "scaling"
        cell_emb_style="cls",  # "avg-pool", "w-pool", or "cls"
        
        # Training objectives
        MLM=False,  # Masked language modeling 
        CLS=True,   # Cell type classification objective
        ADV=False,  # Adversarial training for batch correction
        CCE=False,  # Contrastive cell embedding
        MVC=True,   # Masked value prediction for cell embedding
        ECS=True,   # Elastic cell similarity objective
        ecs_thres=0.75,  # Threshold for elastic cell similarity
        DAB=False,  # Domain adaptation by reverse backpropagation
        dab_weight=0.0,
        INPUT_BATCH_LABELS=False,  # Have these help MLM and MVC, while not to classifier
        mvc_decoder_style="inner product",  # Decoder style for MVC
        adv_E_delay_epochs=0,  # Delay adversarial training on encoder
        adv_D_delay_epochs=0,  # Delay adversarial training on discriminator
        explicit_zero_prob=None,  # Will be set automatically based on MLM and include_zero_gene
        do_sample_in_train=False,  # Sample the bernoulli in training
        per_seq_batch_sample=False,  # Whether to sample per sequence batch
        
        # Model architecture
        fast_transformer=False,
        fast_transformer_backend="flash",  # "linear" or "flash"
        layer_size=512,  # Embedding dimension
        nlayers=12,      # Number of transformer layers
        nhead=8,         # Number of attention heads
        dropout=0.1,     # Dropout probability
        
        # Training settings
        batch_size=32,
        eval_batch_size=32,
        epochs=10,
        lr=5e-5,
        lr_ADV=1e-3,     # Learning rate for discriminator
        weight_decay=0.01,
        max_grad_norm=1.0,
        gradient_accumulation_steps=1,
        schedule_interval=1,  # Scheduling interval
        schedule_ratio=0.9,   # Scheduler gamma value
        
        # Optimization settings
        amp=True,  # Automatic Mixed Precision
        
        # Chunking settings
        chunk_size=50000,  # Number of cells per chunk
        save_chunks=False, # Whether to save preprocessed chunks to disk
        
        # Logging
        log_interval=100,  # Log every N steps
        save_eval_interval=1,  # Save and evaluate every N epochs
        do_eval_scib_metrics=True,  # Whether to evaluate scIB metrics
        disable_file_logging=True,
    )

    parser = argparse.ArgumentParser(description='Continual Pretraining for scGPT')
    parser.add_argument('--query_file', default='/root/scGPT_dir/scGPT/data/Derived Embryoid Bodies.h5ad', 
                        help='Path to the query dataset file'),
    parser.add_argument('--ref_file', default=None, 
                        help='Path to the reference dataset file')
    parser.add_argument('--config_file', default=None, help="Config file with info")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./scGPT_continual_training",  # Add a default value
        help="Directory to save model checkpoints and logs"
    )

    add_dict_to_argparser(parser, defaults)
    
    return parser

def validate_and_setup_config(config):
    """
    Validate configuration settings and set derived values.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Updated config with derived settings
    """
    # Validate input/output styles
    valid_input_styles = ["normed_raw", "log1p", "binned"]
    valid_output_styles = ["normed_raw", "log1p", "binned"]
    valid_input_emb_styles = ["category", "continuous", "scaling"]
    
    if config['input_style'] not in valid_input_styles:
        raise ValueError(f"input_style must be one of {valid_input_styles}")
    
    if config['output_style'] not in valid_output_styles:
        raise ValueError(f"output_style must be one of {valid_output_styles}")
    
    if config['input_emb_style'] not in valid_input_emb_styles:
        raise ValueError(f"input_emb_style must be one of {valid_input_emb_styles}")
    
    # Check for incompatible configurations
    if config['input_style'] == "binned" and config['input_emb_style'] == "scaling":
        raise ValueError("input_emb_style `scaling` is not supported for binned input.")
    
    if (config['input_style'] == "log1p" or config['input_style'] == "normed_raw") and config['input_emb_style'] == "category":
        raise ValueError(
            "input_emb_style `category` is not supported for log1p or normed_raw input."
        )

    # Set values based on input embedding style
    if config['input_emb_style'] == "category":
        config['mask_value'] = config['n_bins'] + 1
        config['pad_value'] = config['n_bins']  # for padding gene expr values
        config['n_input_bins'] = config['n_bins'] + 2
    else:
        config['mask_value'] = -1
        config['pad_value'] = -2
        config['n_input_bins'] = config['n_bins']

    # Validate adversarial settings
    if config.get('ADV', False) and config.get('DAB', False):
        raise ValueError("ADV and DAB cannot be both True.")
    
    # Set DAB separate optimizer flag
    config['DAB_separate_optim'] = True if config.get('DAB', 0) > 1 else False
    
    # Set ECS based on threshold
    config['ECS'] = config.get('ecs_thres', 0) > 0
    
    # Special tokens
    if 'pad_token' not in config:
        config['pad_token'] = "<pad>"
        
    if 'special_tokens' not in config:
        config['special_tokens'] = [config['pad_token'], "<cls>", "<eoc>"]
    
    # Handle zero genes
    if 'explicit_zero_prob' not in config:
        config['explicit_zero_prob'] = config.get('MLM', False) and config.get('include_zero_gene', False)
        
    if 'do_sample_in_train' not in config:
        config['do_sample_in_train'] = False and config['explicit_zero_prob']
    
    # Set input layer key based on input style
    config['input_layer_key'] = {
        "normed_raw": "X_normed",
        "log1p": "X_log1p",
        "binned": "X_binned"
    }.get(config['input_style'])
    
    # Set default values for essential parameters if not present
    defaults = {
        'max_seq_len': 3001,
        'batch_size': 32,
        'eval_batch_size': 32,
        'gradient_accumulation_steps': 1,
        'max_grad_norm': 1.0,
        'dab_weight': 0.0,
        'log_interval': 100,
        'save_eval_interval': 1,
        'per_seq_batch_sample': False
    }
    
    for key, value in defaults.items():
        if key not in config:
            config[key] = value
    
    return config

def load_data(input_file):
    """
    Load data from h5ad file with error handling.
    
    Args:
        input_file: Path to input h5ad file
        
    Returns:
        AnnData object
    """
    try:
        logging.info(f"Loading data from {input_file}")
        adata = sc.read_h5ad(input_file)
        logging.info(f"Loaded data with {adata.n_obs} cells and {adata.n_vars} genes")
        return adata
    except Exception as e:
        logging.error(f"Failed to load data from {input_file}: {str(e)}")
        raise

def preprocess_data(adata, config):
    """
    Preprocess AnnData object using the scGPT Preprocessor.
    
    Args:
        adata: AnnData object
        config: Configuration dictionary
        
    Returns:
        Preprocessed AnnData object
    """
    logging.info("Starting data preprocessing")
    
    try:
        preprocessor = Preprocessor(
            use_key="X",  # the key in adata.layers to use as raw data
            filter_gene_by_counts=config['filter_gene_by_counts'],  # step 1
            filter_cell_by_counts=False,  # step 2
            normalize_total=1e4,  # 3. whether to normalize the raw data and to what sum
            result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
            log1p=config['data_is_raw'],  # 4. whether to log1p the normalized data
            result_log1p_key='X_log1p',
            subset_hvg=False,  # 5. whether to subset the raw data to highly variable genes
            hvg_flavor='seurat_v3' if config['data_is_raw'] else "cell_ranger",
            binning=config['n_bins'],  # 6. whether to bin the raw data and to what number of bins
            result_binned_key="X_binned",  # the key in adata.layers to store the binned data
        )

        # Apply preprocessing
        preprocessor(adata)
        logging.info("Data preprocessing completed successfully")
        return adata
    except Exception as e:
        logging.error(f"Error during preprocessing: {str(e)}")
        raise

def analyze_ms_data(adata, adata_test, cell_type_key, batch_key):
    """
    Analyze and preprocess multiple sclerosis dataset.
    
    Args:
        adata: Main AnnData object
        adata_test: Test AnnData object
        cell_type_key: Key for cell type information
        batch_key: Key for batch information
        
    Returns:
        Combined and preprocessed AnnData object
    """
    logging.info("Analyzing MS dataset")
    
    try:
        # Set cell type categories
        adata.obs["celltype"] = adata.obs[cell_type_key].astype("category")
        adata_test.obs["celltype"] = adata_test.obs[cell_type_key].astype("category")
        
        # Set batch IDs
        adata.obs["batch_id"] = adata.obs["str_batch"] = "0"
        adata_test.obs["batch_id"] = adata_test.obs["str_batch"] = "1"
        
        # Set gene name indices
        if "gene_name" in adata.var.columns:
            adata.var.set_index(adata.var["gene_name"], inplace=True)
        
        if "gene_name" in adata_test.var.columns:
            adata_test.var.set_index(adata_test.var["gene_name"], inplace=True)
        
        # Concatenate datasets
        adata = adata.concatenate(adata_test, batch_key="str_batch")
        
        # Create category codes
        adata.obs["batch_id"] = adata.obs["str_batch"].astype("category").cat.codes.values
        adata.obs["celltype_id"] = adata.obs["celltype"].astype("category").cat.codes.values
        
        logging.info(f"MS data analysis complete: {adata.n_obs} cells, {adata.n_vars} genes")
        return adata
    except Exception as e:
        logging.error(f"Error during MS data analysis: {str(e)}")
        raise

def tokenize_data(adata, config):
    """
    Tokenize AnnData object for model input.
    
    Args:
        adata: AnnData object
        config: Configuration dictionary
        
    Returns:
        Tokenized data
    """
    logging.info("Tokenizing data")
    
    try:
        tokenized_data = tokenize_and_pad_batch(
            adata,
            config['gene_ids'],
            max_len=config['max_seq_len'],
            vocab=config['vocab'],
            pad_token=config['pad_token'],
            pad_value=config['pad_value'],
            append_cls=True,
            include_zero_gene=config['include_zero_gene']
        )
        logging.info(f"Tokenization complete: {tokenized_data['genes'].shape[0]} samples, {tokenized_data['genes'].shape[1]} features")
        return tokenized_data
    except Exception as e:
        logging.error(f"Error during tokenization: {str(e)}")
        raise

def prepare_data(
    tokenized_train, 
    tokenized_valid, 
    train_batch_labels,
    valid_batch_labels,
    train_celltype_labels,
    valid_celltype_labels,
    mask_ratio, 
    mask_value, 
    pad_value, 
    epoch=0,  # Added default value for epoch
    sort_seq_batch=False
) -> Tuple[Dict[str, torch.Tensor]]:
    masked_values_train = random_mask_value(
        tokenized_train["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    masked_values_valid = random_mask_value(
        tokenized_valid["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    print(
        f"random masking at epoch {epoch:3d}, ratio of masked values in train: ",
        f"{(masked_values_train == mask_value).sum() / (masked_values_train - pad_value).count_nonzero():.4f}",
    )

    input_gene_ids_train, input_gene_ids_valid = (
        tokenized_train["genes"],
        tokenized_valid["genes"],
    )
    input_values_train, input_values_valid = masked_values_train, masked_values_valid
    target_values_train, target_values_valid = (
        tokenized_train["values"],
        tokenized_valid["values"],
    )

    tensor_batch_labels_train = torch.from_numpy(train_batch_labels).long()
    tensor_batch_labels_valid = torch.from_numpy(valid_batch_labels).long()

    tensor_celltype_labels_train = torch.from_numpy(train_celltype_labels).long()
    tensor_celltype_labels_valid = torch.from_numpy(valid_celltype_labels).long()

    if sort_seq_batch:  # TODO: update to random pick seq source in each traning batch
        train_sort_ids = np.argsort(train_batch_labels)
        input_gene_ids_train = input_gene_ids_train[train_sort_ids]
        input_values_train = input_values_train[train_sort_ids]
        target_values_train = target_values_train[train_sort_ids]
        tensor_batch_labels_train = tensor_batch_labels_train[train_sort_ids]
        tensor_celltype_labels_train = tensor_celltype_labels_train[train_sort_ids]

        valid_sort_ids = np.argsort(valid_batch_labels)
        input_gene_ids_valid = input_gene_ids_valid[valid_sort_ids]
        input_values_valid = input_values_valid[valid_sort_ids]
        target_values_valid = target_values_valid[valid_sort_ids]
        tensor_batch_labels_valid = tensor_batch_labels_valid[valid_sort_ids]
        tensor_celltype_labels_valid = tensor_celltype_labels_valid[valid_sort_ids]

    train_data_pt = {
        "gene_ids": input_gene_ids_train,
        "values": input_values_train,
        "target_values": target_values_train,
        "batch_labels": tensor_batch_labels_train,
        "celltype_labels": tensor_celltype_labels_train,
    }
    valid_data_pt = {
        "gene_ids": input_gene_ids_valid,
        "values": input_values_valid,
        "target_values": target_values_valid,
        "batch_labels": tensor_batch_labels_valid,
        "celltype_labels": tensor_celltype_labels_valid,
    }

    return train_data_pt, valid_data_pt


# data_loader
def prepare_dataloader(
    data_pt: Dict[str, torch.Tensor],
    batch_size: int,
    per_seq_batch_sample: bool = False,  # Added parameter with default value
    shuffle: bool = False,
    intra_domain_shuffle: bool = False,
    drop_last: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    if num_workers == 0:
        num_workers = min(len(os.sched_getaffinity(0)), batch_size // 2)

    dataset = SeqDataset(data_pt)

    if per_seq_batch_sample:
        # find the indices of samples in each seq batch
        subsets = []
        batch_labels_array = data_pt["batch_labels"].numpy()
        for batch_label in np.unique(batch_labels_array):
            batch_indices = np.where(batch_labels_array == batch_label)[0].tolist()
            subsets.append(batch_indices)
        data_loader = DataLoader(
            dataset=dataset,
            batch_sampler=SubsetsBatchSampler(
                subsets,
                batch_size,
                intra_subset_shuffle=intra_domain_shuffle,
                inter_subset_shuffle=shuffle,
                drop_last=drop_last,
            ),
            num_workers=num_workers,
            pin_memory=True,
        )
        return data_loader

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
    )
    return data_loader

# dataset
class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


def train(model, loader, config, epoch):
    """
    Train the model for one epoch.
    
    Args:
        model: The model to train
        loader: DataLoader containing training data
        config: Configuration dictionary with all parameters
        epoch: Current epoch number
    
    Returns:
        None
    """
    import time
    
    model.train()
    
    # Unpack needed values from config for cleaner code
    device = config['device']
    vocab = config['vocab']
    pad_token = config['pad_token']
    mask_value = config['mask_value']
    log_interval = config['log_interval']
    optimizer = config['optimizer']
    scheduler = config['scheduler']
    scaler = config['scaler']
    criterion = config['criterion']
    criterion_cls = config['criterion_cls']
    criterion_dab = config['criterion_dab']
    logger = config['logger']
    
    # Initialize tracking variables
    (
        total_loss,
        total_mse,
        total_cls,
        total_cce,
        total_mvc,
        total_ecs,
        total_dab,
        total_adv_E,
        total_adv_D,
        total_zero_log_prob,
        total_mvc_zero_log_prob,
    ) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    total_error = 0.0
    start_time = time.time()

    num_batches = len(loader)
    for batch, batch_data in enumerate(loader):
        input_gene_ids = batch_data["gene_ids"].to(device)
        input_values = batch_data["values"].to(device)
        input_target_values = batch_data["target_values"].to(device)
        batch_labels = batch_data["batch_labels"].to(device)
        celltype_labels = batch_data["celltype_labels"].to(device)

        src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
        with torch.cuda.amp.autocast(enabled=config['amp']):
            output_dict = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=batch_labels if config['INPUT_BATCH_LABELS'] or config.get('DSBN', False) else None,
                CLS=config['CLS'],
                CCE=config['CCE'],
                MVC=config['MVC'],
                ECS=config['ECS'],
                do_sample=config['do_sample_in_train'],
            )

            masked_positions = input_values.eq(mask_value)  # the positions to predict
            loss = 0.0
            metrics_to_log = {}
            
            # MLM loss
            if config['MLM']:
                loss_mse = criterion(
                    output_dict["mlm_output"], input_target_values, masked_positions
                )
                loss = loss + loss_mse
                metrics_to_log = {"train/mse": loss_mse.item()}
                total_mse += loss_mse.item()
            
            # Zero probability loss (for sparse data)
            if config['explicit_zero_prob']:
                loss_zero_log_prob = config['criterion_neg_log_bernoulli'](
                    output_dict["mlm_zero_probs"], input_target_values, masked_positions
                )
                loss = loss + loss_zero_log_prob
                metrics_to_log.update({"train/nzlp": loss_zero_log_prob.item()})
                total_zero_log_prob += loss_zero_log_prob.item()
            
            # Cell type classification loss
            if config['CLS']:
                loss_cls = criterion_cls(output_dict["cls_output"], celltype_labels)
                loss = loss + loss_cls
                metrics_to_log.update({"train/cls": loss_cls.item()})
                total_cls += loss_cls.item()

                error_rate = 1 - (
                    (output_dict["cls_output"].argmax(1) == celltype_labels)
                    .sum()
                    .item()
                ) / celltype_labels.size(0)
                total_error += error_rate
            
            # Contrastive cell embedding loss
            if config['CCE']:
                loss_cce = 10 * output_dict["loss_cce"]
                loss = loss + loss_cce
                metrics_to_log.update({"train/cce": loss_cce.item()})
                total_cce += loss_cce.item()
            
            # Masked value prediction loss
            if config['MVC']:
                loss_mvc = criterion(
                    output_dict["mvc_output"], input_target_values, masked_positions
                )
                loss = loss + loss_mvc
                metrics_to_log.update({"train/mvc": loss_mvc.item()})
                total_mvc += loss_mvc.item()
            
            # MVC with zero probability loss
            if config['MVC'] and config['explicit_zero_prob']:
                loss_mvc_zero_log_prob = config['criterion_neg_log_bernoulli'](
                    output_dict["mvc_zero_probs"], input_target_values, masked_positions
                )
                loss = loss + loss_mvc_zero_log_prob
                metrics_to_log.update({"train/mvc_nzlp": loss_mvc_zero_log_prob.item()})
                total_mvc_zero_log_prob += loss_mvc_zero_log_prob.item()
            
            # Elastic cell similarity loss
            if config['ECS']:
                loss_ecs = 10 * output_dict["loss_ecs"]
                loss = loss + loss_ecs
                metrics_to_log.update({"train/ecs": loss_ecs.item()})
                total_ecs += loss_ecs.item()
            
            # Domain adaptation by backpropagation loss
            if config['DAB']:
                loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)
                loss = loss + config['dab_weight'] * loss_dab
                metrics_to_log.update({"train/dab": loss_dab.item()})
                total_dab += loss_dab.item()

        # Backward pass and optimization
        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        
        # Gradient clipping
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.get('max_grad_norm', 1.0),
                error_if_nonfinite=False if scaler.is_enabled() else True,
            )
            if len(w) > 0:
                logger.warning(
                    f"Found infinite gradient. This may be caused by the gradient "
                    f"scaler. The current scale is {scaler.get_scale()}. This warning "
                    "can be ignored if no longer occurs after autoscaling of the scaler."
                )
                
        scaler.step(optimizer)
        scaler.update()


        # Adversarial training if enabled
        if config['ADV']:
            # Get discriminator from config
            discriminator = config['discriminator']
            criterion_adv = config['criterion_adv']
            optimizer_D = config['optimizer_D']
            optimizer_E = config['optimizer_E']
            
            # Rerun the model for adversarial training
            output_dict = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=batch_labels if config['INPUT_BATCH_LABELS'] or config.get('DSBN', False) else None,
                CLS=config['CLS'],
                CCE=config['CCE'],
                MVC=config['MVC'],
                ECS=config['ECS'],
                do_sample=config['do_sample_in_train'],
            )

            # TRAINING DISCRIMINATOR
            loss_adv_D = criterion_adv(
                discriminator(output_dict["cell_emb"].detach()), batch_labels
            )
            if epoch > config['adv_D_delay_epochs']:
                discriminator.zero_grad()
                loss_adv_D.backward()
                optimizer_D.step()

            # TRAINING ENCODER
            loss_adv_E = -criterion_adv(
                discriminator(output_dict["cell_emb"]), batch_labels
            )
            # NOTE: the loss is negative here because we want to maximize
            # the cross_entropy_loss, in other words, disguise against the discriminator
            if epoch > config['adv_E_delay_epochs']:
                model.zero_grad()
                discriminator.zero_grad()
                loss_adv_E.backward()
                optimizer_E.step()
                
            total_adv_E += loss_adv_E.item() 
            total_adv_D += loss_adv_D.item()


        total_loss += loss.item()
        
        # Log batch statistics
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_mse = total_mse / log_interval if config['MLM'] else 0.0
            cur_cls = total_cls / log_interval if config['CLS'] else 0.0
            cur_cce = total_cce / log_interval if config['CCE'] else 0.0
            cur_mvc = total_mvc / log_interval if config['MVC'] else 0.0
            cur_ecs = total_ecs / log_interval if config['ECS'] else 0.0
            cur_dab = total_dab / log_interval if config['DAB'] else 0.0
            cur_adv_E = total_adv_E / log_interval if config['ADV'] else 0.0
            cur_adv_D = total_adv_D / log_interval if config['ADV'] else 0.0
            cur_zero_log_prob = total_zero_log_prob / log_interval if config['explicit_zero_prob'] else 0.0
            cur_mvc_zero_log_prob = total_mvc_zero_log_prob / log_interval if config['MVC'] and config['explicit_zero_prob'] else 0.0
            cur_error = total_error / log_interval if config['CLS'] else 0.0
            
            # Build log message
            log_msg = (
                f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:5.2f} | "
            )
            
            # Add component-specific metrics
            if config['MLM']:
                log_msg += f"mse {cur_mse:5.2f} | "
                
            if config['CLS']:
                log_msg += f"cls {cur_cls:5.2f} | err {cur_error:5.2f} | "
                
            if config['CCE']:
                log_msg += f"cce {cur_cce:5.2f} | "
                
            if config['MVC']:
                log_msg += f"mvc {cur_mvc:5.2f} | "
                
            if config['ECS']:
                log_msg += f"ecs {cur_ecs:5.2f} | "
                
            if config['DAB']:
                log_msg += f"dab {cur_dab:5.2f} | "
                
            if config['ADV']:
                log_msg += f"adv_E {cur_adv_E:5.2f} | adv_D {cur_adv_D:5.2f} | "
                
            if config['explicit_zero_prob']:
                log_msg += f"nzlp {cur_zero_log_prob:5.2f} | "
                
            if config['MVC'] and config['explicit_zero_prob']:
                log_msg += f"mvc_nzlp {cur_mvc_zero_log_prob:5.2f} | "
            
            # Remove trailing separator
            log_msg = log_msg.rstrip("| ")
            
            logger.info(log_msg)
            
            # Reset counters
            total_loss = 0
            total_mse = 0
            total_cls = 0
            total_cce = 0
            total_mvc = 0
            total_ecs = 0
            total_dab = 0
            total_adv_E = 0
            total_adv_D = 0
            total_zero_log_prob = 0
            total_mvc_zero_log_prob = 0
            total_error = 0
            start_time = time.time()



def evaluate(model, loader, config, epoch, return_raw=False):
    """
    Evaluate the model on the evaluation data.
    
    Args:
        model: The model to evaluate
        loader: DataLoader containing evaluation data
        config: Configuration dictionary
        epoch: Current epoch number
        return_raw: Whether to return raw predictions
        
    Returns:
        Evaluation metrics or raw predictions
    """
    device = config['device']
    vocab = config['vocab']
    pad_token = config['pad_token']
    
    model.eval()
    total_loss = 0.0
    total_error = 0.0
    total_dab = 0.0
    total_num = 0
    predictions = []
    
    with torch.no_grad():
        for batch_data in loader:
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            target_values = batch_data["target_values"].to(device)
            batch_labels = batch_data["batch_labels"].to(device)
            celltype_labels = batch_data["celltype_labels"].to(device)

            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            with torch.cuda.amp.autocast(enabled=config['amp']):
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels if config.get('INPUT_BATCH_LABELS', False) or config.get('DSBN', False) else None,
                    CLS=config.get('CLS', True),  # Default to True for evaluation
                    CCE=False,  # Always False for evaluation
                    MVC=False,  # Always False for evaluation
                    ECS=False,  # Always False for evaluation
                    do_sample=config.get('do_sample_in_train', False),
                )
                output_values = output_dict["cls_output"]
                loss = config['criterion_cls'](output_values, celltype_labels)

                if config.get('DAB', False):
                    loss_dab = config['criterion_dab'](output_dict["dab_output"], batch_labels)
                    total_dab += loss_dab.item() * len(input_gene_ids)

            total_loss += loss.item() * len(input_gene_ids)
            accuracy = (output_values.argmax(1) == celltype_labels).sum().item()
            total_error += (1 - accuracy / len(input_gene_ids)) * len(input_gene_ids)
            total_num += len(input_gene_ids)
            preds = output_values.argmax(1).cpu().numpy()
            predictions.append(preds)


    if return_raw:
        return np.concatenate(predictions, axis=0)

    return total_loss / total_num, total_error / total_num

# %% inference
def test(model, adata, config):
    """
    Test the model on new data.
    
    Args:
        model: The trained model
        adata: AnnData object with test data
        config: Configuration dictionary
        
    Returns:
        Predictions, labels, and results dictionary
    """
    device = config['device']
    vocab = config['vocab']
    pad_token = config['pad_token']
    pad_value = config['pad_value']
    mask_value = config['mask_value']
    input_layer_key = config['input_layer_key']
    gene_ids = config['gene_ids']
    max_seq_len = config['max_seq_len']
    include_zero_gene = config['include_zero_gene']
    
    # Get counts from the specified layer
    all_counts = (
        adata.layers[input_layer_key].A
        if issparse(adata.layers[input_layer_key])
        else adata.layers[input_layer_key]
    )

    # Get cell type and batch labels
    celltypes_labels = adata.obs["celltype_id"].tolist()
    celltypes_labels = np.array(celltypes_labels)

    batch_ids = adata.obs["batch_id"].tolist() if "batch_id" in adata.obs else np.zeros(adata.n_obs)
    batch_ids = np.array(batch_ids)

    # Tokenize test data
    tokenized_test = tokenize_and_pad_batch(
        all_counts,
        gene_ids,
        max_len=max_seq_len,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=True,
        include_zero_gene=include_zero_gene,
    )

    # Apply random masking
    input_values_test = random_mask_value(
        tokenized_test["values"],
        mask_ratio=config['mask_ratio'],
        mask_value=mask_value,
        pad_value=pad_value,
    )

    # Prepare test data
    test_data_pt = {
        "gene_ids": tokenized_test["genes"],
        "values": input_values_test,
        "target_values": tokenized_test["values"],
        "batch_labels": torch.from_numpy(batch_ids).long(),
        "celltype_labels": torch.from_numpy(celltypes_labels).long(),
    }

    # Create dataloader
    test_loader = DataLoader(
        dataset=SeqDataset(test_data_pt),
        batch_size=config['eval_batch_size'],
        shuffle=False,
        drop_last=False,
        num_workers=min(len(os.sched_getaffinity(0)), config['eval_batch_size'] // 2),
        pin_memory=True,
    )

    # Get predictions
    model.eval()
    predictions = evaluate(
        model,
        loader=test_loader,
        config=config,
        epoch=0,  # Dummy epoch for testing
        return_raw=True,
    )

    # Compute metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    accuracy = accuracy_score(celltypes_labels, predictions)
    precision = precision_score(celltypes_labels, predictions, average="macro")
    recall = recall_score(celltypes_labels, predictions, average="macro")
    macro_f1 = f1_score(celltypes_labels, predictions, average="macro")

    # Log results
    logger = config.get('logger', logging.getLogger())
    logger.info(
        f"Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, "
        f"Macro F1: {macro_f1:.3f}"
    )

    # Return results
    results = {
        "test/accuracy": accuracy,
        "test/precision": precision,
        "test/recall": recall,
        "test/macro_f1": macro_f1,
    }

    return predictions, celltypes_labels, results



def main():
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Set up directories
    repo_dir, data_dir, save_dir, model_dir, directories = setup_directories()
    print(directories)
    print(args.output_dir)
    
    # Set up logging
    logger = setup_logging(args.output_dir)  # Use args instead of config
    
    # Analyze metadata from input file
    try:
        from scGPT_dataloader import analysis_meta
        metadata = analysis_meta(args.query_file, save=True, output_dir=args.output_dir)
        logger.info("Metadata analysis completed")
    except ImportError:
        logger.warning("Could not import analysis_meta function. Proceeding without metadata analysis.")
        metadata = None
    except Exception as e:
        logger.warning(f"Error in metadata analysis: {str(e)}. Proceeding without metadata.")
        metadata = None
    
    # Build a unified configuration
    config = build_config(args, metadata)
    logger.info(config)
    
    # Add model-specific configuration
    config.update(build_model_config(args))
    
    # Validate and set derived config values
    config = validate_and_setup_config(config)
    
    # Additional setup
    config['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config['scaler'] = torch.cuda.amp.GradScaler(enabled=config['amp'])
    config['logger'] = logger
    
    # Rest of the function remains the same
    
    from scGPT_dataloader import split_query_ref_standalone
    # Load datasets
    try:
        adata_query = load_data(args.query_file)
        
        if args.ref_file is not None:
            adata_ref = load_data(args.ref_file)
            adata_train = adata_ref
            adata_test = adata_query
        else:
            # Check if batch information is available
            if 'batch_id' in adata_query.obs:
                # Split data using batch-aware splitting
                logger.info("Using batch-aware splitting with batch_id")
                adata_train, adata_test = split_query_ref_standalone(
                    adata_query, 
                    method='batch', 
                    batch_key='batch_id', 
                    test_size=0.2, 
                    random_state=42
                )
            else:
                # Fall back to random splitting
                logger.info("No batch information found, using random splitting")
                adata_train, adata_test = split_query_ref_standalone(
                    adata_query, 
                    method='random', 
                    test_size=0.2, 
                    random_state=42
                )
            
        logger.info(f"Train set: {adata_train.n_obs} cells, {adata_train.n_vars} genes")
        logger.info(f"Test set: {adata_test.n_obs} cells, {adata_test.n_vars} genes")
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise
    
    
    # Preprocess the data
    if args.preprocess:
        try:
            adata_train = preprocess_data(adata_train, config)
            adata_test = preprocess_data(adata_test, config)
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            raise

    # Load vocabulary
    try:
        vocab_file = os.path.join(args.load_model, "vocab.json")
        logger.info(f"Looking for vocabulary file at: {vocab_file}")
        if not os.path.exists(vocab_file):
            logger.error(f"Vocabulary file not found. Please check the following:")
            logger.error(f"1. Model directory exists: {args.load_model}")
            logger.error(f"2. Vocabulary file exists in model directory")
            logger.error(f"3. Current working directory: {os.getcwd()}")
            raise FileNotFoundError(f"Vocab file not found at: {vocab_file}")
            
        logger.info(f"Loading vocabulary from {vocab_file}")
        vocab = GeneVocab.from_file(vocab_file)
        
        special_tokens = [config['pad_token'], "<cls>", "<eoc>"]
        for s in special_tokens:
            if s not in vocab:
                vocab.append_token(s)
                
        logger.info(f"Vocabulary size: {len(vocab)}")
    except Exception as e:
        logger.error(f"Error loading vocabulary: {str(e)}")
        raise
    
    # Load model configuration
    try:
        model_config_file = Path(config['load_model']) / "args.json"
        if not model_config_file.exists():
            raise FileNotFoundError(f"Model config file not found at: {model_config_file}")
            
        with open(model_config_file, "r") as f:
            model_configs = json.load(f)
            
        logger.info(f"Loaded model configuration from {model_config_file}")
    except Exception as e:
        logger.error(f"Error loading model configuration: {str(e)}")
        raise
    
    # Match genes with vocabulary
    try:
        # Ensure gene_col is defined
        if 'gene_col' not in locals():
            if "gene_name" in adata_train.var.columns:
                gene_col = "gene_name"
            else:
                gene_col = adata_train.var.index.name or "index"
                logger.warning(f"No gene column specified, using {gene_col}")
                
        if gene_col not in adata_train.var.columns:
            logger.warning(f"Gene column {gene_col} not in var index. Using index as gene names.")
            adata_train.var[gene_col] = adata_train.var.index
            adata_test.var[gene_col] = adata_test.var.index
        
        adata_train.var["id_in_vocab"] = [
            1 if gene in vocab else -1 for gene in adata_train.var[gene_col]
        ]
            
        # Also filter test data to match
        common_genes = adata_train.var.index
        if len(common_genes) < adata_test.n_vars:
            logger.info(f"Filtering test data to {len(common_genes)} common genes")
            adata_test = adata_test[:, common_genes]
    except Exception as e:
        logger.error(f"Error matching genes with vocabulary: {str(e)}")
        raise
    
    # Extract model configuration parameters
    embsize = model_configs.get("embsize", args.layer_size)
    nhead = model_configs.get("nheads", args.nhead)
    d_hid = model_configs.get("d_hid", args.layer_size * 4)
    nlayers = model_configs.get("nlayers", args.nlayers)
    n_layers_cls = model_configs.get("n_layers_cls", 1)


    #extract data parameters
    celltype = config['cell_type_col']
    gene_col = config['gene_col']
    batch_key = config['batch_key']
    
    # Prepare input data
    try:
        input_layer_key = {
            "normed_raw": "X_normed",
            "log1p": "X_normed",
            "binned": "X_binned",
        }[args.input_style]
        
        if input_layer_key not in adata_train.layers:
            avail_keys = list(adata_train.layers.keys())
            logger.error(f"Input layer key {input_layer_key} not in adata.layers. Available keys: {avail_keys}")
            raise KeyError(f"Input layer key {input_layer_key} not found in adata.layers")
        
        all_counts = (
            adata_train.layers[input_layer_key].A
            if issparse(adata_train.layers[input_layer_key])
            else adata_train.layers[input_layer_key]
        )
        # Ensure celltype and batch columns exist
        if celltype not in adata_train.obs.columns:
            logger.error("Cell type information missing. Please specify cell type column.")
            raise ValueError("Cell type information missing")
            
        if batch_key not in adata_train.obs.columns:
            logger.warning("Batch information missing. Setting all cells to batch 0.")
            adata_train.obs["batch_id"] = 0
            
        genes = adata_train.var[gene_col].tolist()
        celltypes_labels = adata_train.obs[celltype].tolist()
        celltypes_labels = np.array(celltypes_labels)
        
        batch_ids = adata_train.obs[batch_key].tolist()
        num_batch_types = len(set(batch_ids))
        batch_ids = np.array(batch_ids)
        
        logger.info(f"Input data prepared: {all_counts.shape[0]} cells, {all_counts.shape[1]} genes")
        logger.info(f"Number of cell types: {len(np.unique(celltypes_labels))}")
        logger.info(f"Number of batch types: {num_batch_types}")
    except Exception as e:
        logger.error(f"Error preparing input data: {str(e)}")
        raise
    
    # Train/validation split
    try:
        (
            train_data,
            valid_data,
            train_celltype_labels,
            valid_celltype_labels,
            train_batch_labels,
            valid_batch_labels,
        ) = train_test_split(
            all_counts, celltypes_labels, batch_ids, test_size=0.1, shuffle=True
        )
        
        logger.info(f"Train/validation split complete: {train_data.shape[0]} train samples, {valid_data.shape[0]} validation samples")
    except Exception as e:
        logger.error(f"Error during train/validation split: {str(e)}")
        raise
    
    # Prepare gene_ids and tokenization config
    try:
        vocab.set_default_index(vocab[config['pad_token']])
        gene_ids = np.array([vocab[gene] if gene in vocab else vocab[config['pad_token']] for gene in genes], dtype=int)
        config['gene_ids'] = gene_ids
        config['vocab'] = vocab
        config['max_len'] = args.max_seq_len
        
        # Tokenize data
        tokenized_train = tokenize_data(train_data, config)
        tokenized_valid = tokenize_data(valid_data, config)
        
        logger.info(
            f"Train set tokenized: {tokenized_train['genes'].shape[0]} samples, "
            f"feature length: {tokenized_train['genes'].shape[1]}"
        )
        logger.info(
            f"Valid set tokenized: {tokenized_valid['genes'].shape[0]} samples, "
            f"feature length: {tokenized_valid['genes'].shape[1]}"
        )
    except Exception as e:
        logger.error(f"Error during tokenization: {str(e)}")
        raise
    
    logger.info("Preprocessing and tokenization complete")

    config['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config['scaler'] = torch.cuda.amp.GradScaler(enabled=config['amp'])
    config['logger'] = logger
    config['criterion'] = masked_mse_loss
    config['criterion_cls'] = nn.CrossEntropyLoss()
    config['criterion_dab'] = nn.CrossEntropyLoss()
    config['criterion_neg_log_bernoulli'] = criterion_neg_log_bernoulli

    if config['do_train']:
        # Initialize the model
        ntokens = len(vocab)  # size of vocabulary
        model = TransformerModel(
            ntokens,
            embsize,
            nhead,
            d_hid,
            nlayers,
            nlayers_cls=n_layers_cls,
            n_cls=len(np.unique(celltypes_labels)),
            vocab=vocab,
            dropout=config['dropout'],
            pad_token=config['pad_token'],
            pad_value=config['pad_value'],
            do_mvc=config['MVC'],
            do_dab=config['DAB'],
            use_batch_labels=config.get('INPUT_BATCH_LABELS', False),
            num_batch_labels=num_batch_types,
            domain_spec_batchnorm=config.get('DSBN', False),
            input_emb_style=config['input_emb_style'],
            n_input_bins=config['n_input_bins'],
            cell_emb_style=config['cell_emb_style'],
            mvc_decoder_style=config.get('mvc_decoder_style', 'inner_product'),
            ecs_threshold=config['ecs_thres'],
            explicit_zero_prob=config['explicit_zero_prob'],
            use_fast_transformer=config['fast_transformer'],
            fast_transformer_backend=config['fast_transformer_backend'],
            pre_norm=config.get('pre_norm', False),
        )

        # Load pre-trained model if specified
        try:
            load_pretrained_model(args.load_model, model)
        except Exception as e:
            logger.error(f"Failed to load pretrained model: {str(e)}")
            raise

        # Move model to device
        model.to(config['device'])

        # Set up optimizer and scheduler
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config['lr'], eps=1e-4 if config['amp'] else 1e-8 
        )
        config['optimizer'] = optimizer
        
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, config['schedule_interval'], gamma=config['schedule_ratio']
        )
        config['scheduler'] = scheduler

        # Set up discriminator for adversarial training if enabled
        if config['ADV']:
            discriminator = AdversarialDiscriminator(
                d_model=embsize,
                n_cls=num_batch_types,
            ).to(config['device'])
            config['discriminator'] = discriminator
            
            criterion_adv = nn.CrossEntropyLoss()
            config['criterion_adv'] = criterion_adv
            
            optimizer_E = torch.optim.Adam(model.parameters(), lr=config['lr_ADV'])
            config['optimizer_E'] = optimizer_E
            
            scheduler_E = torch.optim.lr_scheduler.StepLR(
                optimizer_E, config['schedule_interval'], gamma=config['schedule_ratio']
            )
            config['scheduler_E'] = scheduler_E
            
            optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=config['lr_ADV'])
            config['optimizer_D'] = optimizer_D
            
            scheduler_D = torch.optim.lr_scheduler.StepLR(
                optimizer_D, config['schedule_interval'], gamma=config['schedule_ratio']
            )
            config['scheduler_D'] = scheduler_D

        # Set up DAB optimizer if needed
        if config.get('DAB_separate_optim', False):
            optimizer_dab = torch.optim.Adam(model.parameters(), lr=config['lr'])
            config['optimizer_dab'] = optimizer_dab
            
            scheduler_dab = torch.optim.lr_scheduler.StepLR(
                optimizer_dab, config['schedule_interval'], gamma=config['schedule_ratio']
            )
            config['scheduler_dab'] = scheduler_dab



        # Training loop
        best_val_loss = float("inf")
        best_model = None
        best_model_epoch = 0
        
        import time
        import copy

        for epoch in range(1, config['epochs'] + 1):
            epoch_start_time = time.time()
            
            # Prepare data for this epoch
            train_data_pt, valid_data_pt = prepare_data(
                tokenized_train,
                tokenized_valid,
                train_batch_labels,
                valid_batch_labels,
                train_celltype_labels,
                valid_celltype_labels,
                config['mask_ratio'],
                config['mask_value'],
                config['pad_value'],
                epoch=epoch,
                sort_seq_batch=config['per_seq_batch_sample']
            )
            
            # Create data loaders
            train_loader = prepare_dataloader(
                train_data_pt,
                batch_size=config['batch_size'],
                per_seq_batch_sample=config['per_seq_batch_sample'],
                shuffle=False,
                intra_domain_shuffle=True,
                drop_last=False,
            )
            
            valid_loader = prepare_dataloader(
                valid_data_pt,
                batch_size=config['eval_batch_size'],
                per_seq_batch_sample=False,
                shuffle=False,
                intra_domain_shuffle=False,
                drop_last=False,
            )

            # Train for this epoch
            if config['do_train']:
                train(
                    model,
                    train_loader,
                    config,
                    epoch
                )
            
            # Evaluate
            val_loss, val_err = evaluate(
                model,
                valid_loader,
                config,
                epoch
            )
            
            # Log results
            elapsed = time.time() - epoch_start_time
            logger.info("-" * 89)
            logger.info(
                f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
                f"valid loss/mse {val_loss:5.4f} | err {val_err:5.4f}"
            )
            logger.info("-" * 89)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(model)
                best_model_epoch = epoch
                logger.info(f"Best model with score {best_val_loss:5.4f}")
                
                # Save model checkpoint
                if config['output_dir']:
                    model_path = Path(config['output_dir']) / "best_model.pt"
                    torch.save(best_model.state_dict(), model_path)
                    logger.info(f"Saved best model to {model_path}")

            # Step schedulers
            scheduler.step()
            
            if config.get('DAB_separate_optim', False):
                config['scheduler_dab'].step()
                
            if config['ADV']:
                config['scheduler_D'].step()
                config['scheduler_E'].step()

        logger.info(f"Training complete. Best model was from epoch {best_model_epoch} with validation loss {best_val_loss:.4f}")
    
    # Use the best model for testing if available, otherwise use the current model
    if config['do_train']:
        # If we did training, use the best model if available
        test_model = best_model if 'best_model' in locals() else model
    else:
        # If we didn't do training, just use the loaded model
        test_model = model

    logger.info("Using model for testing...")
    
    # Test the model
    predictions, labels, results = test(test_model, adata_test, config)
    
    # Map predictions to cell types if possible
    if 'celltype' in adata_test.obs.columns:
        # Create a mapping from cell type IDs to names
        id2type = dict(enumerate(adata_test.obs['celltype'].cat.categories))
        adata_test.obs["predictions"] = [id2type.get(p, str(p)) for p in predictions]
        
        # Plot results if UMAP coordinates are available
        if 'X_umap' in adata_test.obsm and config.get('do_plot', True):
            try:
                import matplotlib.pyplot as plt
                
                # Get unique cell types for color palette
                celltypes = adata_test.obs['celltype'].cat.categories.tolist()
                
                # Create a color palette
                palette_ = plt.rcParams["axes.prop_cycle"].by_key()["color"]
                # Extend palette if needed
                while len(palette_) < len(celltypes):
                    palette_ = palette_ + palette_
                palette_ = {c: palette_[i] for i, c in enumerate(celltypes)}
                
                # Plot UMAP with cell types and predictions
                with plt.rc_context({"figure.figsize": (12, 5), "figure.dpi": (300)}):
                    fig, (ax1, ax2) = plt.subplots(1, 2)
                    
                    sc.pl.umap(adata_test, color="celltype", ax=ax1, show=False, palette=palette_)
                    ax1.set_title("True Cell Types")
                    
                    sc.pl.umap(adata_test, color="predictions", ax=ax2, show=False, palette=palette_)
                    ax2.set_title("Predicted Cell Types")
                    
                    fig.tight_layout()
                    
                    # Save figure
                    output_dir = Path(config['output_dir'])
                    plt.savefig(output_dir / "results_umap.png", dpi=300)
                    plt.close()
                    
                    logger.info(f"Saved UMAP visualization to {output_dir / 'results_umap.png'}")
                    

            except Exception as e:
                logger.warning(f"Could not create UMAP visualization: {str(e)}")
    
    # Save results
    if config['output_dir']:
        import pickle
        
        save_dict = {
            "predictions": predictions,
            "labels": labels,
            "results": results,
            "id_maps": id2type if 'id2type' in locals() else None
        }
        
        with open(Path(config['output_dir']) / "test_results.pkl", "wb") as f:
            pickle.dump(save_dict, f)
        
        logger.info(f"Saved test results to {Path(config['output_dir']) / 'test_results.pkl'}")


def save_recovery_checkpoint(
    model: nn.Module,
    epoch: int,
    batch_idx: int,
    metrics: Dict[str, float]
) -> None:
    """
    Save a recovery checkpoint during training.

    Args:
        model: The model being trained
        epoch: Current epoch number
        batch_idx: Current batch index
        metrics: Dictionary of training metrics

    Raises:
        TrainingError: If checkpoint saving fails
    """
    try:
        checkpoint = {
            'model_state': model.state_dict(),
            'epoch': epoch,
            'batch_idx': batch_idx,
            'metrics': metrics,
            'timestamp': datetime.datetime.now().isoformat()
        }
        torch.save(checkpoint, f'recovery_checkpoint_epoch_{epoch}_batch_{batch_idx}.pt')
    except Exception as e:
        raise TrainingError(f"Failed to save recovery checkpoint: {str(e)}")

def load_recovery_checkpoint(checkpoint_path: str) -> Dict:
    """
    Load a recovery checkpoint to resume training.

    Args:
        checkpoint_path: Path to the checkpoint file

    Returns:
        Dictionary containing checkpoint data

    Raises:
        TrainingError: If checkpoint loading fails
    """
    try:
        checkpoint = torch.load(checkpoint_path)
        logger = logging.getLogger("scGPT")
        logger.info(f"Resuming from checkpoint saved at {checkpoint['timestamp']}")
        return checkpoint
    except Exception as e:
        raise TrainingError(f"Failed to load recovery checkpoint: {str(e)}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}", exc_info=True)
        raise