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
    nhead=4,  # number of heads in nn.MultiheadAttention
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
    nhead: int = 4
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
#print(config)
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
nhead = config.nhead  # number of heads in nn.MultiheadAttention
dropout = config.dropout  # dropout probabilit


# %%
from pathlib import Path
import scanpy as sc

#Input
dataset = Path("scGPT_data/ms/c_data.h5ad")
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

preprocessor(adata) #apply the preprocessor to the data

# %%
#in case there is no config of the model, we will get them from the genes we found in the anndata 
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
    nhead,
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
criterion = masked_mse_loss
criterion_cls = nn.CrossEntropyLoss()

# %%
def evaluate(model: nn.Module, loader: DataLoader, return_raw: bool = False) -> float:
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
    "celltype_labels": torch.from_numpy(celltypes_labels).long(),  # Changed from celltypes_labels to celltype_labels
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


