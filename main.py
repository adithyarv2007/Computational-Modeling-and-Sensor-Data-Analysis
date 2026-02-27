from data_generation import generate_dataset
from feature_extraction import transform_dataset
from optimization import train_logistic_regression
from logistic_regression import predict
from evaluation import accuracy, confusion_matrix
from visualization import plot_loss


def main():
    X_raw, y = generate_dataset()
    X = transform_dataset(X_raw)

    weights, bias, loss_history = train_logistic_regression(
        X, y, lr=0.05, epochs=1000
    )

    y_pred = predict(X, weights, bias)

    print("Accuracy:", accuracy(y, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y, y_pred))

    plot_loss(loss_history)


if __name__ == "__main__":
    main()