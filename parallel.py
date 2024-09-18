"""
Executes experiments in parallel to analyze the effects of different number of layers and units per layer on the score of a network.
"""

from os import makedirs, path
from time import time
from warnings import catch_warnings, filterwarnings

import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from numpy import array as nparray
from numpy import zeros as npzeros
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm

from data_generators import aniso, blob
from read_and_write import write_to_file

# default values
FOLDER_NAME = "layers_units"
DATA_SIZE = 500
NUMBER_OF_FEATURES = 2
RANDOM_STATE = 1

print("Press enter for default values")

# dataset ranges
DATA_TYPES = [blob, aniso]
MIN_NUMBER_OF_CLASSES = int(input("MIN_NUMBER_OF_CLASSES or 2: ") or 2)
MAX_NUMBER_OF_CLASSES = int(input("MAX_NUMBER_OF_CLASSES or 7: ") or 7)

# network architecture ranges
MIN_NUMBER_OF_LAYERS = int(input("MIN_NUMBER_OF_LAYERS or 1: ") or 1)
MAX_NUMBER_OF_LAYERS = int(input("MAX_NUMBER_OF_LAYERS or 40: ") or 40)
MIN_NUMBER_OF_UNITS = int(input("MIN_NUMBER_OF_UNITS or 1: ") or 1)
MAX_NUMBER_OF_UNITS = int(input("MAX_NUMBER_OF_UNITS or 40: ") or 40)

# operations counts
TOTAL_DATASETS = len(DATA_TYPES) * (
    MAX_NUMBER_OF_CLASSES - MIN_NUMBER_OF_CLASSES + 1
)
TOTAL_NETWORKS = (MAX_NUMBER_OF_LAYERS - MIN_NUMBER_OF_LAYERS + 1) * (
    MAX_NUMBER_OF_UNITS - MIN_NUMBER_OF_UNITS + 1
)


def generate_datasets() -> list[dict]:
    pbar = tqdm(total=TOTAL_DATASETS, desc="Generate Datasets")

    datasets = []
    for data_type in DATA_TYPES:
        for number_of_classes in range(
            MIN_NUMBER_OF_CLASSES, MAX_NUMBER_OF_CLASSES + 1
        ):
            X, y = data_type(
                DATA_SIZE, number_of_classes, NUMBER_OF_FEATURES, RANDOM_STATE
            )
            X_train, X_test, y_train, y_test = train_test_split(X, y)

            datasets.append(
                {
                    "data_type": data_type.__name__,
                    "number_of_classes": number_of_classes,
                    "X_train": X_train,
                    "X_test": X_test,
                    "y_train": y_train,
                    "y_test": y_test,
                }
            )
            pbar.update(1)

    pbar.close()
    return datasets


def run_experiment(dataset: dict, layers: int, units: int) -> tuple:
    start_time = time()

    network = MLPClassifier(hidden_layer_sizes=(units,) * layers)
    with catch_warnings():
        filterwarnings("ignore", category=ConvergenceWarning)
        network.fit(dataset["X_train"], dataset["y_train"])

    score = network.score(dataset["X_test"], dataset["y_test"])

    return (
        dataset["data_type"],
        dataset["number_of_classes"],
        layers,
        units,
        score,
        time() - start_time,
    )


def run_experiments(datasets: list[dict]) -> list[tuple]:
    experiments = [
        (dataset, layers, units)
        for dataset in datasets
        for layers in range(MIN_NUMBER_OF_LAYERS, MAX_NUMBER_OF_LAYERS + 1)
        for units in range(MIN_NUMBER_OF_UNITS, MAX_NUMBER_OF_UNITS + 1)
    ]

    results = Parallel(n_jobs=-1)(
        delayed(run_experiment)(dataset, layers, units)
        for dataset, layers, units in tqdm(
            experiments, total=len(experiments), desc="Run Experiments"
        )
    )

    return results


def download_heat_maps(results: list[tuple]) -> None:
    results = nparray(
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

    for data_type in DATA_TYPES:
        for number_of_classes in range(
            MIN_NUMBER_OF_CLASSES, MAX_NUMBER_OF_CLASSES + 1
        ):
            data_type_name = data_type.__name__
            mask = (results["type_of_data"] == data_type_name) & (
                results["classes"] == number_of_classes
            )
            filtered_results = results[mask]

            layers = filtered_results["layers"]
            units = filtered_results["units"]
            score = filtered_results["score"]

            score_2d = npzeros(
                (
                    MAX_NUMBER_OF_LAYERS - MIN_NUMBER_OF_LAYERS + 1,
                    MAX_NUMBER_OF_UNITS - MIN_NUMBER_OF_UNITS + 1,
                )
            )

            for l, u, s in zip(layers, units, score):
                score_2d[l - MIN_NUMBER_OF_LAYERS, u - MIN_NUMBER_OF_UNITS] = s

            aspect_ratio = score_2d.shape[1] / score_2d.shape[0]
            fig_width = 10
            fig_height = fig_width / aspect_ratio

            plt.figure(figsize=(fig_width, fig_height), dpi=300)

            im = plt.imshow(
                score_2d,
                cmap="magma",
                extent=(
                    MIN_NUMBER_OF_LAYERS - 0.5,
                    MAX_NUMBER_OF_LAYERS + 0.5,
                    MIN_NUMBER_OF_UNITS - 0.5,
                    MAX_NUMBER_OF_UNITS + 0.5,
                ),
                origin="lower",
                aspect="auto",
                vmin=0,
                vmax=1,
            )

            plt.colorbar(im, label="Score")
            plt.xlabel("Number of Layers")
            plt.ylabel("Number of Units")
            plt.title(
                f"Score vs. Number of Layers and Units ({data_type_name}, {number_of_classes} Classes)"
            )

            plt.xticks(range(MIN_NUMBER_OF_LAYERS, MAX_NUMBER_OF_LAYERS + 1))
            plt.yticks(range(MIN_NUMBER_OF_UNITS, MAX_NUMBER_OF_UNITS + 1))

            plt.tight_layout()

            makedirs(FOLDER_NAME, exist_ok=True)
            filename = f"size{DATA_SIZE}_features{NUMBER_OF_FEATURES}_random{RANDOM_STATE}_type{data_type_name}_classes{number_of_classes}.png"
            filepath = path.join(FOLDER_NAME, filename)

            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            plt.clf()
            plt.close()

            print(f"Heatmap saved as {filepath}")


# main

master_start_time = time()

datasets = generate_datasets()
results = run_experiments(datasets)

print(f"Total time: {time() - master_start_time}")

write_to_file(results, "rafael_results.txt")
download_heat_maps(results)
