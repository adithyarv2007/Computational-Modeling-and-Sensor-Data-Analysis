import numpy as np


def extract_features(signal):
    mean = np.mean(signal)
    std = np.std(signal)
    energy = np.sum(signal ** 2)

    fft = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal))

    magnitude = np.abs(fft)

    dominant_frequency = freqs[np.argmax(magnitude)]

    spectral_energy = np.sum(magnitude ** 2)

    return np.array([
        mean,
        std,
        energy,
        dominant_frequency,
        spectral_energy
    ])


def transform_dataset(X):
    features = np.array([extract_features(signal) for signal in X])

    # Standardization (critical for gradient descent)
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0) + 1e-9
    features = (features - mean) / std

    return features