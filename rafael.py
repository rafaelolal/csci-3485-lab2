import warnings
from time import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm

from data_generators import aniso, blob
from read_and_write import write_to_file

# Ignoring convergence warnings to not mess up the progress bar
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Default values
DATA_SIZE = 1000
NUMBER_OF_FEATURES = 2
RANDOM_STATE = 1

# Dataset ranges
DATA_TYPES = [blob, aniso]
MIN_NUMBER_OF_CLASSES = 2
MAX_NUMBER_OF_CLASSES = 7

# Network architecture ranges
MIN_NUMBER_OF_LAYERS = 1
MAX_NUMBER_OF_LAYERS = 4
MIN_NUMBER_OF_UNITS = 1
MAX_NUMBER_OF_UNITS = 4

# Operations counts
total_datasets = len(DATA_TYPES) * (
    MAX_NUMBER_OF_CLASSES - MIN_NUMBER_OF_CLASSES + 1
)
total_networks = (MAX_NUMBER_OF_LAYERS - MIN_NUMBER_OF_LAYERS + 1) * (
    MAX_NUMBER_OF_UNITS - MIN_NUMBER_OF_UNITS + 1
)

# Initializing timer
master_start_time = time()

# Building datasets
data_pbar = tqdm(total=total_datasets, desc="Building Datasets")

datasets = []
for data_type in DATA_TYPES:
    for number_of_classes in range(
        MIN_NUMBER_OF_CLASSES, MAX_NUMBER_OF_CLASSES + 1
    ):
        X, y = data_type(
            DATA_SIZE,
            number_of_classes,
            NUMBER_OF_FEATURES,
            RANDOM_STATE,
        )

        X_train, X_test, y_train, y_test = train_test_split(X, y)

        datasets.append(
            (
                data_type.__name__,
                number_of_classes,
                X_train,
                X_test,
                y_train,
                y_test,
            )
        )
        data_pbar.update(1)

data_pbar.close()

# Running experiments
experiment_pbar = tqdm(
    total=total_networks * total_datasets, desc="Running Experiments"
)

results = []
for layers in range(MIN_NUMBER_OF_LAYERS, MAX_NUMBER_OF_LAYERS + 1):
    for units in range(MIN_NUMBER_OF_UNITS, MAX_NUMBER_OF_UNITS + 1):
        network = MLPClassifier(hidden_layer_sizes=(units,) * layers)

        for dataset in datasets:
            start_time = time()

            X_train, X_test, y_train, y_test = (
                dataset[2],
                dataset[3],
                dataset[4],
                dataset[5],
            )

            network.fit(X_train, y_train)
            score = network.score(X_test, y_test)

            results.append(
                (
                    dataset[0],
                    dataset[1],
                    layers,
                    units,
                    score,
                    time() - start_time,
                )
            )

            experiment_pbar.update(1)

experiment_pbar.close()
print(f"Total time: {time() - master_start_time}")

# Writing results to file in case of somehow losing the data
write_to_file(results, "rafael_results.txt")

# Converting results to a np array to easily access the columns
results = np.array(
    results,
    dtype=[
        ("type_of_data", "U50"),
        ("classes", int),
        ("layers", int),
        ("units", int),
        ("score", float),
        ("total_time", float),
    ],
)


# Plotting heatmaps
for data_type in DATA_TYPES:
    for number_of_classes in range(
        MIN_NUMBER_OF_CLASSES, MAX_NUMBER_OF_CLASSES + 1
    ):
        mask = (results["type_of_data"] == data_type.__name__) & (
            results["classes"] == number_of_classes
        )
        filtered_results = results[mask]

        layers = filtered_results["layers"]
        units = filtered_results["units"]
        score = filtered_results["score"]

        score_2d = np.zeros(
            (
                MAX_NUMBER_OF_LAYERS - MIN_NUMBER_OF_LAYERS + 1,
                MAX_NUMBER_OF_UNITS - MIN_NUMBER_OF_UNITS + 1,
            )
        )

        for l, u, s in zip(layers, units, score):
            score_2d[l - MIN_NUMBER_OF_LAYERS, u - MIN_NUMBER_OF_UNITS] = s

        plt.figure(figsize=(10, 10))
        plt.imshow(
            score_2d,
            cmap="magma",
            extent=(
                MIN_NUMBER_OF_LAYERS,
                MAX_NUMBER_OF_LAYERS,
                MIN_NUMBER_OF_UNITS,
                MAX_NUMBER_OF_UNITS,
            ),
            origin="lower",
            aspect="auto",
            vmin=0,
            vmax=1,
        )
        plt.colorbar(label="Score")
        plt.xlabel("Number of Layers")
        plt.ylabel("Number of Units")
        plt.title(
            f"Score vs. Number of Layers and Units ({data_type.__name__}, {number_of_classes} Classes)"
        )
        plt.show()
