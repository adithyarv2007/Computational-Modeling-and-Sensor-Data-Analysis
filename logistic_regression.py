import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def predict_proba(X, weights, bias):
    return sigmoid(np.dot(X, weights) + bias)


def predict(X, weights, bias):
    probs = predict_proba(X, weights, bias)
    return (probs >= 0.5).astype(int)


def compute_loss(X, y, weights, bias):
    m = len(y)
    probs = predict_proba(X, weights, bias)
    loss = - (1/m) * np.sum(
        y * np.log(probs + 1e-9) +
        (1 - y) * np.log(1 - probs + 1e-9)
    )
    return loss