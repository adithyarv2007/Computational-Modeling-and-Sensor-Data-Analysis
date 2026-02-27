import numpy as np


def generate_sine_signal(freq, amplitude, length, noise_level=0.1):
    """
    Generate a noisy sinusoidal signal.
    """
    t = np.linspace(0, 1, length)
    signal = amplitude * np.sin(2 * np.pi * freq * t)
    noise = noise_level * np.random.randn(length)
    return signal + noise


def generate_dataset(n_samples=200, length=100):
    """
    Generate synthetic dataset with two classes:
    Class 0 → Low frequency
    Class 1 → High frequency
    """
    X = []
    y = []

    for _ in range(n_samples):
        # Class 0
        X.append(generate_sine_signal(freq=2, amplitude=1, length=length))
        y.append(0)

        # Class 1
        X.append(generate_sine_signal(freq=8, amplitude=1, length=length))
        y.append(1)

    return np.array(X), np.array(y)