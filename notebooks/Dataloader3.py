# %%
import torch
import numpy as np
import os
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
import shutil
from pathlib import Path

# %%
hyperparameter_defaults = dict(
    seed=0,
    dataset_name="ms",
    do_train=True,
    load_model="../save/scGPT_human",
    mask_ratio=0.0,
    epochs=10,
    n_bins=51,
    MVC=False, # Masked value prediction for cell embedding
    ecs_thres=0.0, # Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable
    dab_weight=0.0,
    lr=1e-4,
    batch_size=32,
    layer_size=128,
    nlayers=4,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nheads=4,  # number of heads in nn.MultiheadAttention
    dropout=0.2,  # dropout probability
    schedule_ratio=0.9,  # ratio of epochs for learning rate schedule
    save_eval_interval=5,
    fast_transformer=True,
    pre_norm=False,
    amp=True,  # Automatic Mixed Precision
    include_zero_gene = False,
    freeze = False, #freeze
    DSBN = False,  # Domain-spec batchnorm
)

# %%
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class ModelConfig:
    # Basic settings
    seed: int = 0
    dataset_name: str = "ms"
    do_train: bool = True
    load_model: Path = Path("../save/scGPT_human")
    
    # Training parameters
    mask_ratio: float = 0.0
    epochs: int = 10
    n_bins: int = 51
    lr: float = 1e-4
    batch_size: int = 32
    schedule_ratio: float = 0.9
    save_eval_interval: int = 5
    
    # Model architecture
    layer_size: int = 128
    nlayers: int = 4
    nheads: int = 4
    dropout: float = 0.2
    
    # Model features
    MVC: bool = False  # Masked value prediction for cell embedding
    ecs_thres: float = 0.0  # Elastic cell similarity objective
    dab_weight: float = 0.0
    fast_transformer: bool = True
    pre_norm: bool = False
    amp: bool = True  # Automatic Mixed Precision
    include_zero_gene: bool = False
    freeze: bool = False
    DSBN: bool = False  # Domain-spec batchnorm
    
    def __post_init__(self):
        # Convert load_model to Path if it's a string
        if isinstance(self.load_model, str):
            self.load_model = Path(self.load_model)
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'ModelConfig':
        """Create a ModelConfig instance from a dictionary"""
        return cls(**config_dict)
    
    def to_dict(self) -> dict:
        """Convert the config to a dictionary"""
        return {
            key: str(value) if isinstance(value, Path) else value 
            for key, value in self.__dict__.items()
        }
    
    def update(self, **kwargs):
        """Update config parameters"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")

# Example usage:
config = ModelConfig.from_dict(hyperparameter_defaults)

# Or create directly:
# config = ModelConfig(
#     dataset_name="ms",
#     layer_size=256,
#     # ... other parameters
# )

# %%
# settings for training
MLM = False  # whether to use masked language modeling, currently it is always on.
CLS = True  # celltype classification objective
ADV = False  # Adversarial training for batch correction
CCE = False  # Contrastive cell embedding objective
MVC = config.MVC  # Masked value prediction for cell embedding
ECS = config.ecs_thres > 0  # Elastic cell similarity objective
DAB = False  # Domain adaptation by reverse backpropagation, set to 2 for separate optimizer
INPUT_BATCH_LABELS = False  # TODO: have these help MLM and MVC, while not to classifier
input_emb_style = "continuous"  # "category" or "continuous" or "scaling"
cell_emb_style = "cls"  # "avg-pool" or "w-pool" or "cls"
adv_E_delay_epochs = 0  # delay adversarial training on encoder for a few epochs
adv_D_delay_epochs = 0
mvc_decoder_style = "inner product"
ecs_threshold = config.ecs_thres
dab_weight = config.dab_weight

explicit_zero_prob = MLM and include_zero_gene  # whether explicit bernoulli for zeros
do_sample_in_train = False and explicit_zero_prob  # sample the bernoulli in training

per_seq_batch_sample = False

# %%
print(config)
pad_value = -2
mask_value = -1
n_input_bins = config.n_bins + 2

# input/output representation
input_style = "binned"  # "normed_raw", "log1p", or "binned"
output_style = "binned"  # "normed_raw", "log1p", or "binned"

# Settings for input and preprocessing
pad_token = "<pad>"
special_tokens = [pad_token, "cls", "<eoc>"]
mask_ratio = config.mask_ratio

include_zero_gene = config.include_zero_gene
max_seq_len = 3001
n_bins = config.n_bins

#settings for evaluation
eval_batch_size = config.batch_size

# settings for the model
fast_transformer = config.fast_transformer
fast_transformer_backend = "flash"  # "linear" or "flash"
embsize = config.layer_size  # embedding dimension
d_hid = config.layer_size  # dimension of the feedforward network in TransformerEncoder
nlayers = config.nlayers  # number of TransformerEncoderLayer in TransformerEncoder
nheads = config.nheads  # number of heads in nn.MultiheadAttention
dropout = config.dropout  # dropout probabilit


# %%
# %% validate settings
assert input_style in ["normed_raw", "log1p", "binned"]
assert output_style in ["normed_raw", "log1p", "binned"]
assert input_emb_style in ["category", "continuous", "scaling"]
if input_style == "binned":
    if input_emb_style == "scaling":
        raise ValueError("input_emb_style `scaling` is not supported for binned input.")
elif input_style == "log1p" or input_style == "normed_raw":
    if input_emb_style == "category":
        raise ValueError(
            "input_emb_style `category` is not supported for log1p or normed_raw input."
        )

if input_emb_style == "category":
    mask_value = n_bins + 1
    pad_value = n_bins  # for padding gene expr values
    n_input_bins = n_bins + 2
else:
    mask_value = -1
    pad_value = -2
    n_input_bins = n_bins

if ADV and DAB:
    raise ValueError("ADV and DAB cannot be both True.")
DAB_separate_optim = True if DAB > 1 else False

print(input_style, output_style, input_emb_style, cell_emb_style)

# %%
from pathlib import Path
import scanpy as sc

#Input
dataset = Path("scGPT_data/ms/filtered_ms_adata.h5ad")
adata = sc.read(dataset)
print(adata.obs.select_dtypes(['category']).columns)
adata.obs["celltype"] = adata.obs["Factor Value[inferred cell type - authors labels]"].astype("category")

# %%
import numpy as np

print(adata.var.select_dtypes(['category']).columns) #show all the categories of the variables
adata.var.set_index(adata.var["gene_name"], inplace=True) #make sure it starts at 0 
#some conditions about the data
data_is_raw = False
filter_gene_by_counts = False

#make numerical id labels for the model
celltype_id_labels = adata.obs["celltype"].astype("category").cat.codes.values
celltypes = adata.obs["celltype"].unique()
print(celltypes)
print(celltype_id_labels)

num_types = len(np.unique(celltype_id_labels))
print(num_types)
id2type = dict(enumerate(adata.obs["celltype"].astype("category").cat.categories)) #mappping from id to celltype
adata.obs["celltype_id"] = celltype_id_labels
adata.var["gene_name"] = adata.var.index.tolist()

# Check the layer names (if data exists in different processing states)
print(adata.layers.keys())

# Check the main data matrix
print(f"Main matrix (adata.X) type: {type(adata.X)}")
print(f"Data shape: {adata.shape}")
print(f"Minimum value: {adata.X.min()}")
print(f"Maximum value: {adata.X.max()}")

# Check if there's any preprocessing info stored in .uns
print("\nPreprocessing info in .uns:")
print(adata.uns.keys())

# If data is log1p transformed, values should be between 0 and ~10
# Also check for negative values which shouldn't exist in log1p data
import numpy as np
print(f"Contains negative values: {np.any(adata.X < 0)}")
print(f"Data range: [{np.min(adata.X)}, {np.max(adata.X)}]")

# Calculate sum of counts per cell
sums_per_cell = adata.X.sum(axis=1)
print(f"Mean sum per cell: {np.mean(sums_per_cell)}")
print(f"Std of sums: {np.std(sums_per_cell)}")
# If normalized to 10000 counts per cell, mean should be close to 10000

# Calculate sum of counts per cell
sums_per_cell = adata.X.sum(axis=1)
print(f"Mean sum per cell: {np.mean(sums_per_cell)}")
print(f"Std of sums: {np.std(sums_per_cell)}")
# If normalized to 10000 counts per cell, mean should be close to 10000

# Check unique values - if binned, should have limited unique values
unique_vals = np.unique(adata.X)
print(f"Number of unique values: {len(unique_vals)}")
print(f"First few unique values: {unique_vals[:10]}")
# For sparse matrices, convert to dense array for variance calculation
# Use small chunks if memory is a concern
variances = np.var(adata.X.toarray(), axis=0)
# Or use scipy's sparse variance function
from scipy.sparse import issparse
if issparse(adata.X):
    variances = np.array((adata.X.power(2).mean(axis=0) - np.power(adata.X.mean(axis=0), 2)).tolist()[0])


# %%
import json
import scgpt as scg
from scgpt.tokenizer.gene_tokenizer import GeneVocab
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
#from scgpt.utils import set_seed, category_str2int, eval_scib_metricsimport 
logger = scg.logger

#configurate the model
human_model_dir = Path("scGPT_data/Human")
model_config_file = human_model_dir / "args.json" #load the model configuration
model_file = human_model_dir / "best_model.pt" #load the pretrained model
vocab_file = human_model_dir / "vocab.json" #load the vocabulary

#special tokens for padding and cls
# settings for input and preprocessing
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"] #end of cell

#generate a vocabubulary file
vocab = GeneVocab.from_file(vocab_file)
shutil.copy(vocab_file, "vocab.json") #make safety copy
for s in special_tokens: #make sure the padding and cls is there
    if s not in vocab:
        vocab.add_special_token(s)

#making sure all the words of the data are in the vocabulary
adata.var["id_in_vocab"] = [1 if gene in vocab 
                            else -1 for gene in adata.var["gene_name"]] #check if the gene is in the vocabulary
gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
#print how many genes are in the  vocab and how many are not
logger.info(
    f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
    f"in vocabulary of size {len(vocab)}."
)
adata = adata[:, adata.var["id_in_vocab"] >= 0] #filter the genes that are not in the vocabulary (only keep the ones that are in it)

#Just to check quickly: 
#print(adata)
#print (adata.var)

with open(model_config_file, "r") as f:
    model_configs = json.load(f) #load from the args.json file the configuration of the model

#load the parameters and override the ones from the configuration given in this script, the args.json has the correct ones used for the pre-training
embsize = model_configs["embsize"] 
nheads = model_configs["nheads"]
d_hid = model_configs["d_hid"]
nlayers = model_configs["nlayers"]
n_layers_cls = model_configs["n_layers_cls"]

# %%
print(embsize, nheads, d_hid, nlayers, n_layers_cls)

# %%
# set up the preprocessor, use the args to config the workflow. The preprocessor is a function from scgpt that preprocesses the data
preprocessor = Preprocessor(
    use_key="X",  # the key in adata.layers to use as raw data
    filter_gene_by_counts=filter_gene_by_counts,  # step 1
    filter_cell_by_counts=False,  # step 2
    normalize_total=1e4,  # 3. whether to normalize the raw data and to what sum
    result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
    log1p=data_is_raw,  # 4. whether to log1p the normalized data
    result_log1p_key="X_log1p",
    subset_hvg=False,  # 5. whether to subset the raw data to highly variable genes
    hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
    binning= config.n_bins,  # 6. whether to bin the raw data and to what number of bins
    result_binned_key="X_binned",  # the key in adata.layers to store the binned data
)

# Before preprocessing
print("Before preprocessing:")
print(f"Available layers: {list(adata.layers.keys())}")
print(f"Main matrix range: [{adata.X.min()}, {adata.X.max()}]")
print(f"Mean counts per cell: {adata.X.sum(axis=1).mean()}")

# Apply preprocessing
preprocessor(adata)

# After preprocessing
print("\nAfter preprocessing:")
print(f"Available layers: {list(adata.layers.keys())}")
if "X_normed" in adata.layers:
    print(f"Normalized data range: [{adata.layers['X_normed'].min()}, {adata.layers['X_normed'].max()}]")
    print(f"Mean counts per cell (normalized): {adata.layers['X_normed'].sum(axis=1).mean()}")
if "X_binned" in adata.layers:
    print(f"Binned data range: [{adata.layers['X_binned'].min()}, {adata.layers['X_binned'].max()}]")

# %%
def check_scgpt_preprocessing(adata, preprocessor):
    """
    Analyze the preprocessing steps for scGPT data
    """
    print("\n=== scGPT Preprocessing Analysis ===")
    
    # 1. Check available layers
    print("\n1. Available layers in adata:")
    for key in adata.layers.keys():
        print(f"Layer '{key}': shape {adata.layers[key].shape}")
    
    # 2. Raw Data Analysis
    print("\n2. Raw Data Analysis (X):")
    raw_data = adata.X
    print(f"Shape: {raw_data.shape}")
    print(f"Range: [{raw_data.min():.3f}, {raw_data.max():.3f}]")
    print(f"Mean: {raw_data.mean():.3f}")
    print(f"Sparsity: {(raw_data == 0).sum() / raw_data.size:.3f}")
    
    # 3. Check Normalization
    if "X_normed" in adata.layers:
        normed_data = adata.layers["X_normed"]
        print("\n3. Normalized Data Analysis:")
        print(f"Sum per cell (first 5): {normed_data.sum(axis=1)[:5]}")
        print(f"Expected sum per cell: {preprocessor.normalize_total}")
        print(f"Range: [{normed_data.min():.3f}, {normed_data.max():.3f}]")
    else:
        print("\n3. No normalized layer found (X_normed)")
    
    # 4. Check Log1p
    if "X_log1p" in adata.layers:
        log_data = adata.layers["X_log1p"]
        print("\n4. Log1p Data Analysis:")
        print(f"Range: [{log_data.min():.3f}, {log_data.max():.3f}]")
        print(f"Mean: {log_data.mean():.3f}")
        # Check if it's really log1p transformed
        if log_data.max() > raw_data.max():
            print("Warning: Log data maximum exceeds raw data maximum")
    else:
        print("\n4. No log1p layer found (X_log1p)")
    
    # 5. Check Binning
    if "X_binned" in adata.layers:
        binned_data = adata.layers["X_binned"]
        print("\n5. Binned Data Analysis:")
        unique_bins = np.unique(binned_data)
        print(f"Number of unique bins: {len(unique_bins)}")
        print(f"Bin values: {unique_bins}")
        if preprocessor.binning:
            print(f"Expected number of bins: {preprocessor.binning}")
    else:
        print("\n5. No binned layer found (X_binned)")
    
    # 6. Check Gene Filtering
    print("\n6. Gene Filtering:")
    print(f"Original number of genes: {adata.n_vars}")
    if hasattr(preprocessor, 'selected_genes_'):
        print(f"Number of selected genes: {len(preprocessor.selected_genes_)}")
    
    # 7. Visualize distributions
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(15, 5))
    
    # Raw data distribution
    plt.subplot(131)
    plt.hist(raw_data.shape[0], bins=50)
    plt.title('Raw Data Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    
    # Normalized data
    if "X_normed" in adata.layers:
        plt.subplot(132)
        plt.hist(adata.layers["X_normed"].flatten(), bins=50)
        plt.title('Normalized Data Distribution')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
    
    # Log1p data
    if "X_log1p" in adata.layers:
        plt.subplot(133)
        plt.hist(adata.layers["X_log1p"].flatten(), bins=50)
        plt.title('Log1p Data Distribution')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

    return {
        'layers_present': list(adata.layers.keys()),
        'raw_stats': {
            'shape': raw_data.shape,
            'min': raw_data.min(),
            'max': raw_data.max(),
            'mean': raw_data.mean()
        },
        'preprocessing_config': {
            'normalize_total': preprocessor.normalize_total,
            'log1p': preprocessor.log1p,
            'binning': preprocessor.binning
        }
    }

# Usage:
stats = check_scgpt_preprocessing(adata, preprocessor)

# %%
# 1. First create a properly formatted AnnData object
import anndata
import scipy.sparse as sp

# Convert your data to the right format if it's not already
if not isinstance(adata.X, (np.ndarray, sp.spmatrix)):
    matrix = adata.X.toarray() if sp.issparse(adata.X) else np.array(adata.X)
    adata = anndata.AnnData(X=matrix,
                           obs=adata.obs if hasattr(adata, 'obs') else None,
                           var=adata.var if hasattr(adata, 'var') else None)

# 2. Set up the preprocessor with the correct settings for already log-transformed data
preprocessor = Preprocessor(
    use_key="X",
    filter_gene_by_counts=True,
    filter_cell_by_counts=False,
    normalize_total=1e4,  # normalize to 10k counts
    result_normed_key="X_normed",
    log1p=False,  # data is already log-transformed
    subset_hvg=False,
    binning=51,
    result_binned_key="X_binned"
)

# 3. Apply preprocessing
try:
    adata_prep = preprocessor(adata)
    
    # Verify the preprocessing worked
    print("\nPreprocessing results:")
    print(f"Available layers: {list(adata_prep.layers.keys())}")
    if 'X_binned' in adata_prep.layers:
        binned = adata_prep.layers['X_binned']
        print(f"Binned data shape: {binned.shape}")
        print(f"Binned data range: [{binned.min()}, {binned.max()}]")
except Exception as e:
    print(f"Error during preprocessing: {str(e)}")

# %%
# 1. First let's check what type of data we have
print("Initial data check:")
print(f"Type of adata: {type(adata)}")
print(f"Type of adata.X: {type(adata.X)}")
print(f"Data shape: {adata.X.shape}")

# 2. Create a fresh AnnData object
import anndata
import numpy as np
import scipy.sparse as sp

# Make a proper copy of the data
try:
    X = adata.X.toarray() if sp.issparse(adata.X) else np.array(adata.X)
    adata_new = anndata.AnnData(
        X=X,
        obs=adata.obs.copy() if hasattr(adata, 'obs') else None,
        var=adata.var.copy() if hasattr(adata, 'var') else None
    )
    print("\nSuccessfully created new AnnData object")
    print(f"New data shape: {adata_new.shape}")
except Exception as e:
    print(f"Error creating new AnnData: {str(e)}")

# 3. Now try preprocessing with the new object
try:
    preprocessor = Preprocessor(
        use_key="X",
        filter_gene_by_counts=True,
        filter_cell_by_counts=False,
        normalize_total=1e4,
        result_normed_key="X_normed",
        log1p=False,  # data is already log-transformed
        subset_hvg=False,
        binning=51,
        result_binned_key="X_binned"
    )
    
    print("\nApplying preprocessing...")
    adata_processed = preprocessor(adata_new)
    
    if adata_processed is not None:
        print("\nPreprocessing successful!")
        print(f"Available layers: {list(adata_processed.layers.keys())}")
        if 'X_binned' in adata_processed.layers:
            print(f"Binned data shape: {adata_processed.layers['X_binned'].shape}")
    else:
        print("\nPreprocessing returned None")
except Exception as e:
    print(f"\nError during preprocessing: {str(e)}")

# 4. Let's also examine the gene matching
print("\nGene matching info:")
print(f"Number of genes in data: {adata.shape[1]}")
if hasattr(preprocessor, 'selected_genes_'):
    print(f"Number of selected genes: {len(preprocessor.selected_genes_)}")

# %%
# 1. Create fresh AnnData with dense matrix
adata_new = anndata.AnnData(
    X=adata.X.toarray(),  # Convert sparse to dense
    obs=adata.obs.copy(),
    var=adata.var.copy()
)

# 2. Set up preprocessor (modified for already log-transformed data)
preprocessor = Preprocessor(
    use_key="X",
    filter_gene_by_counts=False,  # Disable filtering since we want to keep gene count
    filter_cell_by_counts=False,
    normalize_total=1e4,
    result_normed_key="X_normed",
    log1p=False,  # Data is already log-transformed
    subset_hvg=False,
    binning=51,
    result_binned_key="X_binned"
)

# 3. Create data loader with correct settings
from scgpt.tokenizer import tokenize_and_pad_batch

def create_data_loader(adata_proc, batch_size=32):
    # Get gene names
    genes = adata_proc.var_names.tolist()
    
    # Create dataset
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(adata_proc.X, dtype=torch.float32),
        torch.tensor(pd.Categorical(adata_proc.obs['celltype']).codes, dtype=torch.long)
    )
    
    # Create loader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return loader

# %%
#in case there is no config of the model, we will get them from the genes we found in the anndata 
print (config.load_model is None)
if config.load_model is None:
    vocab = Vocab(
        VocabPybind(genes + special_tokens, None)
    )  # bidirectional lookup [gene <-> int]

#define the genes from the anndata
genes = adata.var["gene_name"].tolist()

#set the default index to padding
vocab.set_default_index(vocab["<pad>"]) #following the "grammatical rules" that it has to start with padding to ensure all the same length
gene_ids = np.array(vocab(genes), dtype=int) #create the gene ids from the vocabulary

# %%
class SeqDataset(Dataset): 
    def __init__(self, data, labels=None): #data is a dictionary of different tensors, all share same first dimension
        self.data = data # dictionary with keys: gene_ids, values

    def __len__(self):
        return self.data["gene_ids"].shape[0] #number of samples (first dimension of data)

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()} #return a tuple of data and labels of the sample
        #this is a dictionary
        

# %%
# Add these debug prints before model creation
print("Configuration values:")
print(f"embsize: {embsize}")
print(f"nhead: {nheads}")
print(f"d_hid: {d_hid}")
print(f"nlayers: {nlayers}")

# Then create model


# %% [markdown]
# # Load the pre-trained scGPT model

# %%
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLS = True  # Since we want to do classification
MVC = False  # Masked value prediction for cell embedding (not needed for inference)
DAB = False  # Not needed for inference
INPUT_BATCH_LABELS = False  # Not using batch labels
num_batch_types = 1  # Default value if not using batch labels
input_emb_style = "continuous"  # or whatever style was used in training
cell_emb_style = "cls"  # assuming using CLS token for cell embedding
mvc_decoder_style = "linear"  # default value
ecs_threshold = 0.0  # default value
explicit_zero_prob = False  # default value
mvc_decoder_style = "linear"  # default value

ntokens = len(vocab)

# Now initialize the model with all parameters
model = TransformerModel(
    ntokens,
    embsize,
    nheads,
    d_hid,
    nlayers,
    nlayers_cls=3,
    n_cls=num_types,  # This will use your 18 cell types
    vocab=vocab,
    dropout=dropout,
    pad_token=pad_token,
    pad_value=pad_value,
    do_mvc=MVC,
    do_dab=DAB,
    use_batch_labels=INPUT_BATCH_LABELS,
    num_batch_labels=num_batch_types,
    domain_spec_batchnorm=config.DSBN,
    input_emb_style=input_emb_style,
    n_input_bins=n_input_bins,
    cell_emb_style=cell_emb_style,
    mvc_decoder_style=mvc_decoder_style,
    ecs_threshold=ecs_threshold,
    explicit_zero_prob=explicit_zero_prob,
    use_fast_transformer=fast_transformer,
    fast_transformer_backend=fast_transformer_backend,
    pre_norm=config.pre_norm,
)

if config.load_model is not None:
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_file, map_location=device)
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

model.to(device)
model.eval()  # Add this line for inference mode

# %%
def print_model_config(model):
    """Print the configuration of a loaded model"""
    print("\n=== Model Configuration ===")
    
    # Get model state dict
    state_dict = model.state_dict()
    
    # Infer embedding size from encoder embedding layer
    emb_weight = state_dict.get('encoder.embedding.weight', None)
    if emb_weight is not None:
        print(f"Embedding size (inferred): {emb_weight.shape[1]}")
    
    # Correctly infer number of heads from input projection weight
    for key, value in state_dict.items():
        if 'self_attn.in_proj_weight' in key:
            # Shape is [3 * d_model * nhead, d_model]
            in_proj_shape = value.shape
            n_heads = in_proj_shape[0] // (3 * in_proj_shape[1])
            print(f"Number of heads (inferred): {n_heads}")
            break
    
    # Correctly count transformer layers
    unique_layers = set()
    for key in state_dict.keys():
        if 'transformer_encoder.layers.' in key:
            layer_num = int(key.split('.')[2])
            unique_layers.add(layer_num)
    print(f"Number of transformer layers (inferred): {len(unique_layers)}")

# Add this after model loading
print_model_config(model)

# %%
def debug_attention_params(model):
    """Debug attention parameters in detail"""
    print("\n=== Attention Parameter Analysis ===")
    state_dict = model.state_dict()
    
    for key, value in state_dict.items():
        if 'self_attn.in_proj_weight' in key:
            print(f"\nAnalyzing {key}:")
            print(f"Shape: {value.shape}")
            # For MultiheadAttention, in_proj_weight shape should be [3 * d_model, d_model]
            # where d_model = embsize = nhead * head_dim
            d_model = value.shape[1]  # 512
            total_proj = value.shape[0]  # 1536
            print(f"d_model (embedding dimension): {d_model}")
            print(f"Total projection dimension: {total_proj}")
            print(f"Implies number of heads: {(total_proj/3)/d_model * d_model/64}")  # Standard head_dim is usually 64
            break

    # Also check the actual MultiheadAttention parameters
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.MultiheadAttention):
            print(f"\nFound MultiheadAttention in {name}")
            print(f"Actual number of heads: {module.num_heads}")
            print(f"Embedding dimension: {module.embed_dim}")
            break

print (debug_attention_params(model))

# %%
criterion = masked_mse_loss
criterion_cls = nn.CrossEntropyLoss()

# %%
def evaluate(model: nn.Module, loader: DataLoader, early_stop: bool = True,  return_raw: bool = False) -> float:
    """
    Evaluate the model on the evaluation data.
    """
    model.eval()
    total_loss = 0.0
    total_error = 0.0
    total_dab = 0.0
    total_num = 0
    predictions = []
    
    # Add a batch counter
    total_batches = len(loader)
    print(f"Starting evaluation on {total_batches} batches...")
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(loader):
            # Print progress every few batches
            if early_stop and batch_idx >= 10:
                break
            if batch_idx % 10 == 0:
                print(f"Processing batch {batch_idx}/{total_batches}")
            
            # Verify input shapes
            print(f"Batch {batch_idx} shapes:")
            print(f"  gene_ids: {batch_data['gene_ids'].shape}")
            print(f"  values: {batch_data['values'].shape}")
            print(f"  target_values: {batch_data['target_values'].shape}")
            print(f"  celltypes_labels: {batch_data['celltypes_labels'].shape}")
            
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            target_values = batch_data["target_values"].to(device) 
            celltypes_labels = batch_data["celltypes_labels"].to(device)

            # Check for NaN values
            if torch.isnan(input_values).any():
                print(f"Warning: NaN values detected in input_values in batch {batch_idx}")
            
            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            
            try:
                with torch.cuda.amp.autocast(enabled=config.amp):
                    output_dict = model(
                        input_gene_ids,
                        input_values,
                        src_key_padding_mask=src_key_padding_mask,
                        batch_labels = None,
                        CLS=CLS,
                        CCE=False,
                        MVC=False,
                        ECS=False,
                        do_sample=do_sample_in_train,
                    )
                    
                    # Verify output shapes
                    print(f"  Output shapes:")
                    print(f"    cls_output: {output_dict['cls_output'].shape}")
                    
                    output_values = output_dict["cls_output"]
                    loss = criterion_cls(output_values, celltypes_labels)

                    # Print periodic updates about predictions
                    if batch_idx % 10 == 0:
                        pred_classes = output_values.argmax(1)
                        print(f"  Sample predictions: {pred_classes[:5]}")
                        print(f"  Sample true labels: {celltypes_labels[:5]}")
                        print(f"  Current batch loss: {loss.item():.4f}")

            except Exception as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                raise e

            total_loss += loss.item() * len(input_gene_ids)
            accuracy = (output_values.argmax(1) == celltypes_labels).sum().item()
            total_error += (1 - accuracy / len(input_gene_ids)) * len(input_gene_ids)
            total_num += len(input_gene_ids)
            preds = output_values.argmax(1).cpu().numpy()
            predictions.append(preds)

            # Print running accuracy every few batches
            if batch_idx % 10 == 0:
                current_acc = accuracy / len(input_gene_ids)
                print(f"  Current batch accuracy: {current_acc:.4f}")

    print("\nEvaluation completed!")
    print(f"Total samples processed: {total_num}")
    
    if return_raw:
        final_predictions = np.concatenate(predictions, axis=0)
        print(f"Final predictions shape: {final_predictions.shape}")
        return final_predictions

    final_loss = total_loss / total_num
    final_error = total_error / total_num
    print(f"Final average loss: {final_loss:.4f}")
    print(f"Final average error: {final_error:.4f}")
    return final_loss, final_error

# %%
def accuracy(model: nn.Module, loader: DataLoader, num_batches: int = 10) -> float:
    """
    Quick evaluation of the model on a few batches to check predictions.
    
    Args:
        model: The neural network model
        loader: DataLoader containing evaluation data
        num_batches: Number of batches to evaluate (default 10)
    """
    model.eval()
    total_accuracy = 0
    samples_seen = 0
    
    print(f"Evaluating {num_batches} batches...")
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(loader):
            if batch_idx >= num_batches:
                break
                
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            celltypes_labels = batch_data["celltypes_labels"].to(device)
            
            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            
            output_dict = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=None,
                CLS=CLS,
                CCE=False,
                MVC=False,
                ECS=False,
                do_sample=do_sample_in_train,
            )
            
            output_values = output_dict["cls_output"]
            predictions = output_values.argmax(1)
            
            # Calculate accuracy for this batch
            correct = (predictions == celltypes_labels).sum().item()
            batch_accuracy = correct / len(celltypes_labels)
            total_accuracy += correct
            samples_seen += len(celltypes_labels)
            
            # Print predictions vs actual for this batch
            print(f"\nBatch {batch_idx + 1}:")
            print("Predictions:", predictions[:5].cpu().numpy())
            print("True labels:", celltypes_labels[:5].cpu().numpy())
            print(f"Batch accuracy: {batch_accuracy:.4f}")
    
    # Print final accuracy
    final_accuracy = total_accuracy / samples_seen
    print(f"\nFinal accuracy over {samples_seen} samples: {final_accuracy:.4f}")
    
    return final_accuracy

# %%
# Test on 10 batches (default)
accuracy = evaluate(model, test_loader)

# Or test on fewer/more batches
accuracy = evaluate(model, test_loader, num_batches=5)

# %%
def evaluate_diagnostic(model: nn.Module, loader: DataLoader, num_batches: int = 3):
    """
    Diagnostic evaluation to understand prediction issues.
    """
    model.eval()
    device = next(model.parameters()).device
    print(f"Model is on device: {device}")
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(loader):
            if batch_idx >= num_batches:
                break
                
            print(f"\n=== Batch {batch_idx} Analysis ===")
            
            # Move data to the same device as model
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            celltypes_labels = batch_data["celltypes_labels"].to(device)
            
            # Print value ranges
            print(f"Input values range: [{input_values.min():.3f}, {input_values.max():.3f}]")
            print(f"Unique cell type labels in batch: {torch.unique(celltypes_labels).cpu().numpy()}")
            
            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            
            output_dict = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=None,
                CLS=CLS,
                CCE=False,
                MVC=False,
                ECS=False,
                do_sample=do_sample_in_train,
            )
            
            output_values = output_dict["cls_output"]
            
            # Analyze output probabilities
            probs = torch.softmax(output_values, dim=1)
            
            # Print detailed analysis for first 5 samples
            for i in range(5):
                print(f"\nSample {i}:")
                print(f"True label: {celltypes_labels[i].item()}")
                pred_label = output_values[i].argmax().item()
                print(f"Predicted label: {pred_label}")
                
                # Get top 3 predictions with probabilities
                top_probs, top_idx = probs[i].topk(3)
                print("Top 3 predictions with probabilities:")
                for prob, idx in zip(top_probs, top_idx):
                    print(f"  Class {idx.item()}: {prob.item():.3f}")
            
            # Print overall batch statistics
            predictions = output_values.argmax(1)
            accuracy = (predictions == celltypes_labels).float().mean()
            print(f"\nBatch Statistics:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Unique predictions in batch: {torch.unique(predictions).cpu().numpy()}")
            
            # Check if outputs look reasonable
            print("\nOutput Statistics:")
            print(f"Output range: [{output_values.min():.3f}, {output_values.max():.3f}]")
            print(f"Mean output value: {output_values.mean():.3f}")
            print(f"Output standard deviation: {output_values.std():.3f}")
    
    print("\nDiagnostic evaluation completed!")

# %%
# Run diagnostic evaluation on just 3 batches
evaluate_diagnostic(model, test_loader, num_batches=3)

# %%
def evaluate_quick(model: nn.Module, loader: DataLoader, num_batches: int = 5):
    """
    Quick evaluation with more informative metrics.
    """
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    print("Starting quick evaluation...")
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(loader):
            if batch_idx >= num_batches:
                break
                
            # Get the data
            input_gene_ids = batch_data["gene_ids"]
            input_values = batch_data["values"]
            celltypes_labels = batch_data["celltypes_labels"]
            
            # Forward pass
            output_dict = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=input_gene_ids.eq(vocab[pad_token]),
                batch_labels=None,
                CLS=CLS,
                CCE=False,
                MVC=False,
                ECS=False,
                do_sample=do_sample_in_train,
            )
            
            outputs = output_dict["cls_output"]
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(probs, 1)
            
            # Collect predictions and labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(celltypes_labels.cpu().numpy())
            
            # Calculate batch accuracy
            correct += (predicted == celltypes_labels).sum().item()
            total += celltypes_labels.size(0)
            
            # Print batch results
            print(f"\nBatch {batch_idx}:")
            for i in range(min(5, len(predicted))):
                confidence = probs[i][predicted[i]].item()
                print(f"Sample {i}: True={celltypes_labels[i].item()}, "
                      f"Pred={predicted[i].item()}, "
                      f"Confidence={confidence:.3f}")
            
            print(f"Batch Accuracy: {(predicted == celltypes_labels).sum().item() / len(celltypes_labels):.3f}")
    
    # Print overall results
    print("\nOverall Results:")
    print(f"Total Accuracy: {correct/total:.3f}")
    
    # Print distribution of predictions
    from collections import Counter
    pred_counter = Counter(all_preds)
    print("\nPrediction Distribution:")
    for label, count in sorted(pred_counter.items()):
        print(f"Class {label}: {count} predictions ({count/len(all_preds):.3f})")

    return correct/total

# %%
accuracy = evaluate_quick(model, test_loader)

# %%
# Check if weights are loaded
def check_model_weights(model):
    # Check a few layers for non-zero weights
    print("\nChecking model weights:")
    count_zero = 0
    count_total = 0
    for name, param in model.named_parameters():
        zeros = (param.data == 0).float().mean().item()
        print(f"{name}: {param.shape}")
        print(f"  Mean: {param.data.mean():.5f}")
        print(f"  Std: {param.data.std():.5f}")
        print(f"  % zeros: {zeros*100:.2f}%")
        count_zero += (param.data == 0).sum().item()
        count_total += param.data.numel()
    
    print(f"\nTotal % of zero weights: {count_zero/count_total*100:.2f}%")

# Use it
check_model_weights(model)

# %% [markdown]
# # Inference 

# %%
from scipy.sparse import issparse


all_counts = (adata.X.A if issparse(adata.X) else adata.X)
print (all_counts)

celltypes_labels = adata.obs["celltype_id"].tolist()
celltypes_labels = np.array(celltypes_labels)

tokenized_inference = tokenize_and_pad_batch(
    all_counts,
    gene_ids,
    max_len=max_seq_len,
    vocab=vocab,
    pad_token= pad_token,
    pad_value=pad_value,
    append_cls=True,
    include_zero_gene=include_zero_gene,
)
    

# %%
#print (tokenized_inference)

input_values_inference = random_mask_value(
    tokenized_inference["values"],
    mask_ratio=mask_ratio,
    mask_value=mask_value,
    pad_value=pad_value,
)

# %%
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader

inference_data_pt = {
    "gene_ids": tokenized_inference["genes"],
    "values": input_values_inference,
    "target_values": tokenized_inference["values"],
    "celltypes_labels": torch.from_numpy(celltypes_labels).long(),  # Changed from celltypes_labels to celltype_labels
}

test_loader = DataLoader(
    dataset = SeqDataset(inference_data_pt),
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=False, 
)

# %%
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix  

#put into evaluation mode
model.eval()

# %%
predictions = evaluate(model, test_loader, return_raw=True)

# %%
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

#accuracy = accuracy_score(celltypes_labels, predictions)
precision = precision_score(celltypes_labels, predictions, average="macro")
recall = recall_score(celltypes_labels, predictions, average="macro")
macro_f1 = f1_score(celltypes_labels, predictions, average="macro")

#print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")   
print(criterion_cls)


# %%
# Add this after loading the model
loaded_params = len(pretrained_dict.keys())
model_params = len(model.state_dict().keys())
print(f"Loaded parameters: {loaded_params}/{model_params}")
print(f"Model config:", model_configs)

# %%
print("Data shape:", adata.shape)
print("Number of unique cell types:", len(np.unique(celltypes_labels)))
print("Cell type distribution:", np.unique(celltypes_labels, return_counts=True))

# Add after preprocessing
print("Input value range:", input_values_inference.min(), input_values_inference.max())
print("Gene IDs range:", tokenized_inference["genes"].min(), tokenized_inference["genes"].max())

# Check if labels match the model's expected range
print("Label range:", celltypes_labels.min(), celltypes_labels.max())
#print("Number of classes in model:", model.n_cls)

# %%
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
celltypes = list(celltypes)
for i in set([id2type[p] for p in predictions]):
    if i not in celltypes:
        celltypes.remove(i)
cm = confusion_matrix(celltypes_labels, predictions)
cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
cm = pd.DataFrame(cm, index=celltypes[:cm.shape[0]], columns=celltypes[:cm.shape[1]])
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt=".1f", cmap="Blues")
#plt.savefig(save_dir / "confusion_matrix.png", dpi=300)



# %%
# Convert numeric predictions to cell type names
predicted_celltypes = [id2type[pred] for pred in predictions]
true_celltypes = [id2type[label] for label in celltypes_labels]

# Show first 10 predictions vs true labels
for i in range(10):
    print(f"Predicted: {predicted_celltypes[i]} | True: {true_celltypes[i]}")

# %%
# Look at raw prediction probabilities for one batch
with torch.no_grad():
    batch = next(iter(test_loader))
    output_dict = model(
        batch["gene_ids"].to(device),
        batch["values"].to(device),
        src_key_padding_mask=batch["gene_ids"].eq(vocab[pad_token]).to(device),
        CLS=True,
        CCE=False,
        MVC=False,
        ECS=False,
        do_sample=False
    )
    probs = F.softmax(output_dict["cls_output"], dim=1)
    
print("Prediction probabilities distribution:")
print(probs[0])  # Look at first sample's probabilities across classes

# Check the mapping between indices and cell types
print("\nCell type mapping:")
for idx, cell_type in id2type.items():
    print(f"{idx}: {cell_type}")

# %%
inference_data_pt ={
    "gene_ids": tokenized_inference["genes"],
    "values": input_values_inference,
    "target_values": tokenized_inference["values"],
    "celltype_labels": torch.from_numpy(celltypes_labels).long(),
}

# Add diagnostic code HERE
print("Checking label distributions:")
print(f"Unique celltype labels: {np.unique(celltypes_labels)}")
print(f"Min label: {np.min(celltypes_labels)}")
print(f"Max label: {np.max(celltypes_labels)}")

# Let's also check the model's output dimension
print("\nChecking model output dimension:")
for batch_data in DataLoader(SeqDataset(inference_data_pt), batch_size=1):
    with torch.no_grad():
        output_dict = model(
            batch_data["gene_ids"].to(device),
            batch_data["values"].to(device),
            src_key_padding_mask=batch_data["gene_ids"].eq(vocab[pad_token]).to(device),
            CLS=True,  # Add this parameter
            CCE=False,
            MVC=False,
            ECS=False,
            do_sample=False
        )
        print(f"Model output shape: {output_dict['cls_output'].shape}")
        print(f"Number of classes in model output: {output_dict['cls_output'].shape[1]}")
        break

# Print the celltypes mapping
print("\nCell type mapping:")
for idx, cell_type in id2type.items():
    print(f"ID {idx}: {cell_type}")

test_loader = DataLoader(
    dataset = SeqDataset(inference_data_pt),
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
)


