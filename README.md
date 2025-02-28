

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




