import matplotlib.pyplot as plt


def plot_loss(loss_history):
    plt.figure()
    plt.plot(loss_history)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Convergence")
    plt.grid(True)
    plt.show()