import warnings
from time import time

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from data_generators import aniso, blob
from read_and_write import write_to_file

# Default values
DATA_SIZE = 5000
NUMBER_OF_FEATURES = 2
RANDOM_STATE = 1

# Value ranges
TYPES_OF_DATA = [blob, aniso]
MIN_NUMBER_OF_CLASSES = 7
MAX_NUMBER_OF_CLASSES = 7

MIN_NUMBER_OF_LAYERS = 1
MAX_NUMBER_OF_LAYERS = 40
MIN_NUMBER_OF_UNITS = 1
MAX_NUMBER_OF_UNITS = 40

total_ops = (
    len(TYPES_OF_DATA)
    * (MAX_NUMBER_OF_CLASSES - MIN_NUMBER_OF_CLASSES + 1)
    * (MAX_NUMBER_OF_LAYERS - MIN_NUMBER_OF_LAYERS + 1)
    * (MAX_NUMBER_OF_UNITS - MIN_NUMBER_OF_UNITS + 1)
)


def run_experiment(type_of_data, classes, layers, units):
    start_time = time()

    X, y = type_of_data(
        DATA_SIZE, classes, NUMBER_OF_FEATURES, RANDOM_STATE
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    network = MLPClassifier(hidden_layer_sizes=(units,) * layers, solver="lbfgs")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        network.fit(X_train, y_train)
    score = network.score(X_test, y_test)

    return (
        type_of_data.__name__,
        classes,
        layers,
        units,
        score,
        time() - start_time,
    )


master_start_time = time()

# Running experiments in parallel
results = Parallel(n_jobs=-1)(
    delayed(run_experiment)(type_of_data, classes, layers, units)
    for type_of_data, classes, layers, units in tqdm(
        (
            (type_of_data, classes, layers, units)
            for type_of_data in TYPES_OF_DATA
            for classes in range(
                MIN_NUMBER_OF_CLASSES, MAX_NUMBER_OF_CLASSES + 1
            )
            for layers in range(MIN_NUMBER_OF_LAYERS, MAX_NUMBER_OF_LAYERS + 1)
            for units in range(MIN_NUMBER_OF_UNITS, MAX_NUMBER_OF_UNITS + 1)
        ),
        total=total_ops,
    )
)


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