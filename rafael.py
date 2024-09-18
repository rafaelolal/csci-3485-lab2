import warnings

from sklearn.exceptions import ConvergenceWarning

# Ignoring convergence warnings to not interrupt the progress bar
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from time import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm

from data_generators import aniso, blob
from read_and_write import write_to_file

# Default values
DATA_SIZE = 500
NUMBER_OF_FEATURES = 2
RANDOM_STATE = 1

# Value ranges
TYPES_OF_DATA = [blob, aniso]
MIN_NUMBER_OF_CLASSES = 4
MAX_NUMBER_OF_CLASSES = 4

MIN_NUMBER_OF_LAYERS = 1
MAX_NUMBER_OF_LAYERS = 10
MIN_NUMBER_OF_UNITS = 1
MAX_NUMBER_OF_UNITS = 10

total_ops = (
    len(TYPES_OF_DATA)
    * (MAX_NUMBER_OF_CLASSES - MIN_NUMBER_OF_CLASSES + 1)
    * (MAX_NUMBER_OF_LAYERS - MIN_NUMBER_OF_LAYERS + 1)
    * (MAX_NUMBER_OF_UNITS - MIN_NUMBER_OF_UNITS + 1)
)

# Initializing progress bar and timer
pbar = tqdm(total=total_ops)
master_start_time = time()

# Running experiments
results = []
for data_type in TYPES_OF_DATA:
    for classes in range(MIN_NUMBER_OF_CLASSES, MAX_NUMBER_OF_CLASSES + 1):
        X_train, Y_train = data_type(
            DATA_SIZE, classes, NUMBER_OF_FEATURES, RANDOM_STATE
        )
        X_test, Y_test = data_type(
            DATA_SIZE, classes, NUMBER_OF_FEATURES, RANDOM_STATE
        )

        for layers in range(MIN_NUMBER_OF_LAYERS, MAX_NUMBER_OF_LAYERS + 1):
            for units in range(MIN_NUMBER_OF_UNITS, MAX_NUMBER_OF_UNITS + 1):
                start_time = time()

                network = MLPClassifier(hidden_layer_sizes=(units,) * layers)
                network.fit(X_train, Y_train)
                score = network.score(X_test, Y_test)

                total_time = time() - start_time

                results.append(
                    (
                        data_type.__name__,
                        classes,
                        layers,
                        units,
                        score,
                        total_time,
                    )
                )
                pbar.update(1)

# Converting results to a np array to easily access the columns
print("Processing results array")
results_array = np.array(
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

master_end_time = time()
pbar.close()
print(f"Total time: {master_end_time - master_start_time}")

write_to_file(results, "rafael_results.txt")

# Plotting each heatmap
for data_type in TYPES_OF_DATA:
    for classes in range(MIN_NUMBER_OF_CLASSES, MAX_NUMBER_OF_CLASSES + 1):
        mask = (results_array["type_of_data"] == data_type.__name__) & (
            results_array["classes"] == classes
        )
        filtered_results = results_array[mask]

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
            cmap="viridis",
            extent=(
                MIN_NUMBER_OF_LAYERS,
                MAX_NUMBER_OF_LAYERS,
                MIN_NUMBER_OF_UNITS,
                MAX_NUMBER_OF_UNITS,
            ),
            origin="lower",
            aspect="auto",
        )
        plt.colorbar(label="Score")
        plt.xlabel("Number of Layers")
        plt.ylabel("Number of Units")
        plt.title(
            f"Score vs. Number of Layers and Units ({data_type.__name__}, {classes} Classes)"
        )
        plt.show()
