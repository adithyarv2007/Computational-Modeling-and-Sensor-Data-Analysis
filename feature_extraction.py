import numpy as np


def extract_features(signal):
    """
    Extract statistical and frequency-domain features from signal.
    """
    mean = np.mean(signal)
    std = np.std(signal)
    max_val = np.max(signal)
    min_val = np.min(signal)

    fft = np.fft.fft(signal)
    dominant_freq = np.argmax(np.abs(fft))

    return np.array([mean, std, max_val, min_val, dominant_freq])


def transform_dataset(X):
    """
    Transform raw signal dataset into feature matrix.
    """
    features = [extract_features(signal) for signal in X]
    return np.array(features)