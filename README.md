# Cell Type Classification Performance Summary

## Model Comparison Across Approaches

### Overall Accuracy by Model and Donor

| Model | Donor SD34 | Donor SD35 | Donor SD36 | Average |
|-------|------------|------------|------------|---------|
| Fine-tuned scGPT | 77-78% | 74-76% | 72-76% | 75.5% |
| Standard scGPT | 49-56% | 46-52% | 43-48% | 49.0% |
| GenePT/xGPT | 34-55% | 32-49% | 26-43% | 39.8% |
| Ensemble Approach | 55-56% | 53-54% | 51-52% | 53.5% |

### F1 Score Metrics by Model Type

| Model | Macro Avg F1 | Weighted Avg F1 |
|-------|--------------|-----------------|
| Fine-tuned scGPT | 0.659-0.686 | 0.750-0.788 |
| Standard scGPT | 0.374-0.446 | 0.465-0.530 |
| GenePT/xGPT | 0.234-0.411 | 0.308-0.505 |
| Ensemble Approach | 0.448-0.460 | 0.536-0.550 |

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

## Batch Effect Reduction Metrics

### Adjusted Rand Index (ARI) between Patient Clusters and Patient Labels
*Lower values indicate better batch effect removal*

| Dataset | Original Data | Fine-tuned scGPT | Standard scGPT | GenePT/xGPT |
|---------|--------------|------------------|----------------|-------------|
| Cardiomyocyte | 0.33 | 0.07 | 0.01 | 0.09 |
| Aorta | 0.24 | 0.10 | 0.18 | 0.11 |

### Disease Phenotype Classification Accuracy
*Higher values indicate better preservation of biological signal*

| Dataset | Fine-tuned scGPT | Standard scGPT | GenePT/xGPT |
|---------|------------------|----------------|-------------|
| Cardiomyocyte | 88% | 86% | 71% |
| Aorta | 73% | 75% | 69% |

## Model Properties and Resources

| Model | Embedding Dimensions | Training Data Size | Parameter Count | Training Time (GPU days) |
|-------|---------------------|-------------------|----------------|------------------------|
| Fine-tuned scGPT | 512 | 17,000 cells | 125M | 3-7 |
| Standard scGPT | 512 | 33M cells | 125M | 100+ |
| GenePT/xGPT | 3,072 | N/A (uses LLM) | N/A | N/A |
| Proposed Transformer² | 512 | Uses fine-tuned model | 125M + ~5k parameters | 2-3 days additional |

## Performance on Complex Cell States

| Cell State Type | Fine-tuned scGPT | Standard scGPT | GenePT/xGPT |
|----------------|------------------|----------------|-------------|
| Transitional states | High | Moderate | Low |
| Rare cell populations | Moderate | Low | Very low |
| Disease-specific states | High | Moderate | Moderate |
| Novel cell types | Moderate | Low | Moderate |

## Implementation Timeline for Transformer² Integration

| Stage | Timeline | Deliverables |
|-------|----------|-------------|
| GenePT Embedding Extraction | Days 1-2 | Processed embeddings for genes in dataset |
| SVD Analysis | Day 3 | Identified principal components in model weights |
| Alignment Calculation | Day 4 | Computed relationships between embeddings and components |
| Implementation | Days 5-6 | Working code for Transformer² integration |
| Parameter Tuning | Days 7-8 | Optimized scaling factors for components |
| Benchmarking | Days 9-10 | Performance metrics across test datasets |

This comprehensive overview demonstrates the superior performance of our fine-tuned scGPT approach across multiple metrics and datasets, while also highlighting the path forward with the proposed Transformer² integration.
