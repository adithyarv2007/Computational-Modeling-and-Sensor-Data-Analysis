import numpy as np
from logistic_regression import predict_proba


def train_logistic_regression(X, y, lr=0.01, epochs=1000):
    """
    Train logistic regression using batch gradient descent.
    """
    n_features = X.shape[1]
    weights = np.zeros(n_features)
    bias = 0
    m = len(y)

    loss_history = []

    for _ in range(epochs):
        probs = predict_proba(X, weights, bias)

        dw = (1 / m) * np.dot(X.T, (probs - y))
        db = (1 / m) * np.sum(probs - y)

        weights -= lr * dw
        bias -= lr * db

        loss = - (1 / m) * np.sum(
            y * np.log(probs + 1e-9) +
            (1 - y) * np.log(1 - probs + 1e-9)
        )

        loss_history.append(loss)

    return weights, bias, loss_history