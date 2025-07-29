import numpy as np

# Apply log10(x + ε) transform to outputs
epsilon = 1e-16


def log10_transform(x):
    return np.log10(x + epsilon)


def inverse_log10_transform(x):
    return 10**x - epsilon