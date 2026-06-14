# Optimizing Artificial Neural Networks for Customer Churn Prediction at Otomoto

## Project Overview
This repository contains the complete code, saved models, visualisations, 
and dataset for the BAN6440 Module 6 assignment. The project develops and 
optimizes an Artificial Neural Network (ANN) to predict customer churn using 
the Teleconnect customer dataset (7,043 records, 21 features).

---

## Repository Contents

### Code
| File | Description |
|---|---|
| `Otomoto Marketing Segmentation Model Optimization.ipynb` | Main Jupyter Notebook — full pipeline from EDA to model evaluation |
| `Otomoto Marketing Segmentation Model Optimization 1.py` | Python script version of the same pipeline |

### Dataset
| File | Description |
|---|---|
| `teleconnect.csv` | Raw Teleconnect customer dataset (7,043 records) |

### Saved Models
| File | Description |
|---|---|
| `baseline_sgd_model.keras` | Baseline ANN trained with SGD optimizer |
| `adam_model.keras` | Optimized ANN trained with Adam optimizer |
| `rmsprop_model.keras` | Optimized ANN trained with RMSprop optimizer |
| `tuned_sgd_model.keras` | Optimized ANN trained with SGD + Momentum |

### Visualisations
| File | Description |
|---|---|
| `01_churn_distribution.png` | Churn class distribution bar chart |
| `02_numerical_vs_churn.png` | Tenure and Monthly Charges vs Churn boxplots |
| `03_contract_vs_churn.png` | Contract type vs Churn |
| `04_internet_vs_churn.png` | Internet service type vs Churn |
| `05_payment_vs_churn.png` | Payment method vs Churn |
| `06_baseline_learning_curves.png` | Baseline SGD training and validation curves |
| `07_adam_learning_curves.png` | Adam training and validation curves |
| `08_rmsprop_learning_curves.png` | RMSprop training and validation curves |
| `09_tuned_sgd_learning_curves.png` | Tuned SGD training and validation curves |
| `10_confusion_matrices.png` | All four confusion matrices side by side |
| `11_model_comparison.png` | Final performance comparison — all optimizers |

---

## Models Compared
| Model | Optimizer | Learning Rate | Momentum |
|---|---|---|---|
| Baseline | SGD | 0.001 | None |
| Optimizer 1 | Adam | 0.001 | Adaptive |
| Optimizer 2 | RMSprop | 0.001 | None |
| Optimizer 3 | Tuned SGD | 0.01 | 0.9 |

---

## Performance Summary
| Model | Accuracy | Churn F1-Score | Churn Recall | Churn Precision |
|---|---|---|---|---|
| Baseline SGD | 73% | 0.61 | 0.80 | 0.50 |
| Adam | 73% | 0.62 | 0.82 | 0.50 |
| RMSprop | 74% | 0.62 | 0.81 | 0.51 |
| Tuned SGD | 71% | 0.60 | 0.83 | 0.47 |

---

## Best Performing Model
**RMSprop** — 74% accuracy, F1-score 0.62, fewest false positives (297).  
Recommended for deployment as Otomoto's primary churn prediction tool.

---

## Pipeline Summary
1. Exploratory Data Analysis (EDA)
2. Data Preprocessing — encoding, scaling, imputation
3. Train/Test Split — 80/20 stratified
4. Baseline ANN — SGD optimizer
5. Optimization — Adam, RMSprop, Tuned SGD
6. Evaluation — accuracy, precision, recall, F1-score, loss
7. Model saving

---

## Tools and Libraries
- Python 3
- TensorFlow / Keras
- scikit-learn
- pandas, numpy
- matplotlib, seaborn

---

