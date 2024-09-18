from typing import Tuple

import numpy as np
from sklearn.datasets import make_blobs


def blob(
    samples: int, centers: int, features: int, random_state: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates a blob dataset with labels"""

    X, Y = make_blobs(
        n_samples=samples,
        centers=centers,
        n_features=features,
        random_state=random_state,
    )
    return X, Y


def aniso(
    samples: int, centers: int, features: int, random_state: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates a aniso dataset with labels"""

    X, Y = blob(samples, centers, features, random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X = np.dot(X, transformation)
    return X, Y
