# SCGPT-annotator

Use scGPT for single cell annotation in a bioinformatics pipeline:
- Analysis of data and preparation
- Data embedding using scGPT
- Classification and evaluation of data to annotate cells

## Overview

SCGPT-annotator is a comprehensive toolkit for automated cell type annotation using pre-trained scGPT models. This pipeline enables you to process single-cell RNA sequencing data, generate embeddings using scGPT, and apply machine learning classification to annotate cell types.

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- CUDA (recommended for GPU acceleration)

### Install dependencies

pip install -r requirements.txt

### Download models

Download pre-trained scGPT models and place them in the `models/scGPT_human/` directory. Models can be downloaded from the scGPT releases page.

## Usage

### Quick Start

Run the complete pipeline with a single command:

python scgpt_annotate.py --query_file data/your_data.h5ad --all

This will:
1. Analyze the data to detect metadata
2. Generate scGPT embeddings
3. Train a classifier and predict cell types
4. Evaluate the results

### Step-by-Step Usage

You can also run specific steps of the pipeline:

# 1. Run only analysis step
python scgpt_annotate.py --query_file data/your_data.h5ad --analysis

# 2. Generate embeddings
python scgpt_annotate.py --query_file data/your_data.h5ad --embed

# 3. Classify cells
python scgpt_annotate.py --query_file data/your_data.h5ad --classify

# 4. Evaluate results
python scgpt_annotate.py --query_file data/your_data.h5ad --evaluate

### Using a Reference Dataset

For annotating cells with a reference dataset:

python scgpt_annotate.py --query_file data/query.h5ad --ref_file data/reference.h5ad --classify

### Advanced Options

# Specify gene column, cell type column, and batch key
python scgpt_annotate.py --query_file data/your_data.h5ad --all \
 --gene_col "feature_name" \
 --cell_type_col "cell_type" \
 --batch_key "sample_id"

# Use a different classifier
python scgpt_annotate.py --query_file data/your_data.h5ad --all \
 --classifier_type "knn"  # Options: randomforest, knn, svm, lightgbm

# Show top predictions
python scgpt_annotate.py --query_file data/your_data.h5ad --all \
 --n_top_predictions 5

## Input Data Format

The pipeline accepts AnnData (h5ad) files with:
- Gene expression matrix in .X
- Cell metadata in .obs
- Gene metadata in .var

## Output

The pipeline generates:
1. Analysis of input data metadata
2. scGPT embeddings stored in .obsm['X_scGPT']
3. Cell type predictions in .obs['pred_cell_type']
4. Prediction probabilities in .obs['pred_cell_type_prob_*']
5. Top N predictions for each cell
6. Saved results in a timestamped h5ad file

## Configuration

You can use a configuration file to set parameters:

python scgpt_annotate.py --query_file data/your_data.h5ad --config_file your_config.json

Example config file:
{
 "model_dir": "models/scGPT_human",
 "gene_col": "feature_name",
 "batch_size": 64,
 "cell_type_col": "cell_type",
 "batch_key": "batch",
 "classifier_type": "randomforest",
 "n_top_predictions": 5
}

  

![image](https://github.com/user-attachments/assets/fab1286a-2387-49e1-8d28-c0e125d17cbc)

# Cell Type Classification Performance Summary

## Model Comparison Across Approaches

### Overall Accuracy by Model and Donor

| Model | Donor SD34 | Donor SD35 | Donor SD36 | Average |
|-------|------------|------------|------------|---------|
| Fine-tuned scGPT | 77-78% | 74-76% | 72-76% | 75.5% |
| Standard scGPT | 49-56% | 46-52% | 43-48% | 49.0% |
| GenePT/xGPT | 34-55% | 32-49% | 26-43% | 39.8% |

### F1 Score Metrics by Model Type

| Model | Macro Avg F1 | Weighted Avg F1 |
|-------|--------------|-----------------|
| Fine-tuned scGPT | 0.659-0.686 | 0.750-0.788 |
| Standard scGPT | 0.374-0.446 | 0.465-0.530 |
| GenePT/xGPT | 0.234-0.411 | 0.308-0.505 |

### Algorithm Performance with Fine-tuned scGPT (Weighted Avg F1)

| Algorithm | Donor SD34 | Donor SD35 | Donor SD36 | Average |
|-----------|------------|------------|------------|---------|
| KNN | 0.777 | 0.750 | 0.732 | 0.753 |
| Random Forest | 0.788 | 0.764 | 0.763 | 0.772 |
| LightGBM | 0.784 | 0.760 | 0.759 | 0.768 |

## Cell Type-Specific Performance (F1 Scores for Random Forest on SD34)

| Cell Type | Fine-tuned scGPT | Standard scGPT | GenePT/xGPT |
|-----------|------------------|----------------|-------------|
| Endothelial cell | 0.97 | 0.93 | 0.77 |
| Blood cell | 0.93 | 0.80 | 0.69 |
| Endodermal cell | 0.88 | 0.80 | 0.79 |
| Lateral mesodermal cell | 0.81 | 0.62 | 0.58 |
| Splanchnic mesodermal cell | 0.84 | 0.54 | 0.43 |
| Primordial germ cell | 0.74 | 0.40 | 0.29 |
| Schwann cell | 0.35 | 0.12 | 0.03 |
| Neuron | 0.28 | 0.21 | 0.13 |
| Neural progenitor cell | 0.69 | 0.30 | 0.21 |




