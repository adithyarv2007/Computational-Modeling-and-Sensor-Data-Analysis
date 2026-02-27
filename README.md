# Computational-Modeling-and-Sensor-Data-Analysis
Mathematical modeling and classification of synthetic time-series signals using gradient-based optimization implemented from scratch.

## Overview

This repository presents a structured computational pipeline for modeling and classifying synthetic sensor-like time-series signals. The project is designed to demonstrate mathematical reasoning in data-driven systems through explicit implementation of signal simulation, feature extraction, and gradient-based optimization.

Rather than relying on high-level machine learning libraries, the core learning algorithm is implemented from first principles to emphasize clarity in optimization dynamics and model behavior.

---

## Objectives

- Simulate synthetic time-series signals with controlled noise
- Extract statistical and frequency-domain features
- Implement logistic regression from scratch
- Train using explicit gradient descent
- Analyze convergence behavior and model performance
- Evaluate classification robustness

---

## Methodology

### 1. Signal Generation
Synthetic signals are generated using sinusoidal functions with varying frequencies to simulate distinct sensor patterns. Gaussian noise is added to model real-world signal disturbances.

### 2. Feature Extraction
Each signal is transformed into a structured feature vector including:
- Mean
- Standard deviation
- Maximum value
- Minimum value
- Dominant frequency component (via FFT)

This allows transformation of raw time-series data into a compact representation suitable for classification.

### 3. Model Implementation
Binary logistic regression is implemented manually:

- Sigmoid activation
- Cross-entropy loss
- Analytical gradient derivation
- Batch gradient descent optimization

No external ML frameworks are used for training.

### 4. Evaluation
Model performance is assessed using:
- Classification accuracy
- Confusion matrix
- Training loss convergence visualization

---
## Repository Structure

```
Computational-Modeling-and-Sensor-Data-Analysis/
│
├── data_generation.py        # Synthetic time-series signal simulation
├── feature_extraction.py     # Time-domain and frequency-domain features
├── logistic_regression.py    # Model definition and loss formulation
├── optimization.py           # Gradient descent training procedure
├── evaluation.py             # Accuracy and confusion matrix metrics
├── visualization.py          # Training loss visualization
├── main.py                   # End-to-end execution pipeline
│
├── requirements.txt
└── README.md
```

## Mathematical Foundations

The model is based on binary logistic regression:

σ(z) = 1 / (1 + e^(−z))

Loss function:

J(w, b) = −(1/m) Σ [ y_i log(ŷ_i) + (1 − y_i) log(1 − ŷ_i) ]

Gradients:

∂J/∂w = (1/m) Xᵀ(ŷ − y)
∂J/∂b = (1/m) Σ(ŷ − y)

These are implemented explicitly to maintain full transparency of the optimization process.

---

## Key Learning Outcomes

This project demonstrates:

- Translation of mathematical models into working computational systems
- Structured handling of time-series data
- Feature engineering in both time and frequency domains
- Understanding of optimization convergence
- Sensitivity to noise in classification systems
- Modular software design for analytical workflows

---

## Requirements

Install dependencies:
pip install -r requirements.txt

Dependencies:
- numpy
- matplotlib

---

## How to Run

Execute:
python main.py


This will:
1. Generate synthetic signals
2. Extract features
3. Train the model
4. Print accuracy and confusion matrix
5. Display convergence plot

---

## Notes

This project is intended as a foundational exploration of data-driven modeling and sensor-style signal classification. The emphasis is on interpretability, structured reasoning, and mathematical clarity rather than reliance on high-level automation frameworks.
