# Optimizing-Artificial-Neural-Networks-for-Customer-Churn-Prediction-at-Otomoto
# Otomoto ANN Optimization — BAN6440 Module 6

This repository contains the code, models, and visualisations for the ANN optimization assignment using the Teleconnect 
customer churn dataset.

## Contents
- Jupyter Notebook: Full pipeline from EDA to model evaluation
- /models: Saved Keras models for all four optimizers
- /charts: All EDA and performance visualisation outputs

## Models Compared
- Baseline SGD
- Adam (lr=0.001)
- RMSprop (lr=0.001)
- Tuned SGD (lr=0.01, momentum=0.9)

## Best Performing Model
RMSprop — 74% accuracy, F1-score 0.62 on churned class
