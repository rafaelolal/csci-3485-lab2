import warnings
from time import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
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
MAX_NUMBER_OF_CLASSES = 3

# Network architecture fixed values
NUMBER_OF_LAYERS = 10    # 10+ layers based on results of network architecture tests
NUMBER_OF_UNITS = 5     # Lower units(1-5) based on results of network architecture tests

# Learning rate ranges and solvers
MIN_LEARNING_RATE_INIT = 0.001
MAX_LEARNING_RATE_INIT = 0.991
LEARNING_RATE_STEP_SIZE = 0.01
SOLVERS = ['sgd', 'adam']
LEARNING_RATE_STRATEGIES = ['constant', 'invscaling', 'adaptive']

# Operations counts
total_datasets = len(DATA_TYPES) * (
    MAX_NUMBER_OF_CLASSES - MIN_NUMBER_OF_CLASSES + 1
)
total_networks = len(SOLVERS) * len(LEARNING_RATE_STRATEGIES) * (
    int((MAX_LEARNING_RATE_INIT - MIN_LEARNING_RATE_INIT) / LEARNING_RATE_STEP_SIZE) + 1
)

# Initializing timer
master_start_time = time()

# Building datasets
data_pbar = tqdm(total=total_datasets, desc="Building Datasets")

datasets = []
for data_type in DATA_TYPES:
    for number_of_classes in range(MIN_NUMBER_OF_CLASSES, MAX_NUMBER_OF_CLASSES + 1):
        X, y = data_type(DATA_SIZE, number_of_classes, NUMBER_OF_FEATURES, RANDOM_STATE)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        datasets.append((data_type.__name__, number_of_classes, X_train, X_test, y_train, y_test))
        data_pbar.update(1)

data_pbar.close()

# Running experiments
experiment_pbar = tqdm(total=total_networks * total_datasets, desc="Running Experiments")

results = []
for solver in SOLVERS:
    for learning_rate_strategy in LEARNING_RATE_STRATEGIES:
        for learning_rate_init in np.arange(MIN_LEARNING_RATE_INIT, MAX_LEARNING_RATE_INIT, LEARNING_RATE_STEP_SIZE):
            network = MLPClassifier(
                hidden_layer_sizes=(NUMBER_OF_UNITS,) * NUMBER_OF_LAYERS,
                solver=solver,
                learning_rate=learning_rate_strategy,
                learning_rate_init=learning_rate_init
            )

            for dataset in datasets:
                start_time = time()

                X_train, X_test, y_train, y_test = dataset[2], dataset[3], dataset[4], dataset[5]
                network.fit(X_train, y_train)
                score = network.score(X_test, y_test)

                results.append((
                    dataset[0], dataset[1], solver, learning_rate_strategy, 
                    learning_rate_init, score, time() - start_time
                ))

                experiment_pbar.update(1)

experiment_pbar.close()
print(f"Total time: {time() - master_start_time}")

# Writing results to file in case of somehow losing the data
write_to_file(results, "solver_learning_rate_results.txt")

# Converting results to a numpy array for easier processing
results = np.array(results, dtype=[
    ("type_of_data", "U50"),
    ("classes", int),
    ("solver", "U10"),
    ("learning_rate_strategy", "U10"),
    ("learning_rate_init", float),
    ("score", float),
    ("total_time", float),
])

# Aggregate results for blobs and aniso
def aggregate_results(results):
    aggregated_results = []
    for solver in SOLVERS:
        for lr_strategy in LEARNING_RATE_STRATEGIES:
            for learning_rate_init in np.arange(MIN_LEARNING_RATE_INIT, MAX_LEARNING_RATE_INIT, LEARNING_RATE_STEP_SIZE):
                mask = (results["solver"] == solver) & (results["learning_rate_strategy"] == lr_strategy)
                scores_for_lr_init = results[mask & (results["learning_rate_init"] == learning_rate_init)]["score"]
                mean_score = np.mean(scores_for_lr_init)
                aggregated_results.append((solver, lr_strategy, learning_rate_init, mean_score))
    return np.array(aggregated_results, dtype=[("solver", "U10"), ("learning_rate_strategy", "U10"), ("learning_rate_init", float), ("score", float)])

aggregated_results = aggregate_results(results)

# Heatmap function
def plot_heatmap(data, row_labels, col_labels, title, xlabel, ylabel):
    plt.figure(figsize=(10, 6))
    sns.heatmap(data, xticklabels=col_labels, yticklabels=row_labels, cmap="viridis", cbar=True, vmin=0, vmax=1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, fontsize=3)
    plt.tight_layout()
    plt.show()

# Heatmap for sgd vs adam for each learning rate strategy
def plot_solver_comparison(aggregated_results, lr_strategy):
    heatmap_data = np.zeros((2, len(np.arange(MIN_LEARNING_RATE_INIT, MAX_LEARNING_RATE_INIT, LEARNING_RATE_STEP_SIZE))))
    solvers = ['sgd', 'adam']
    
    for i, solver in enumerate(solvers):
        for j, learning_rate_init in enumerate(np.arange(MIN_LEARNING_RATE_INIT, MAX_LEARNING_RATE_INIT, LEARNING_RATE_STEP_SIZE)):
            mask = (aggregated_results["solver"] == solver) & (aggregated_results["learning_rate_strategy"] == lr_strategy)
            score = aggregated_results[mask & (aggregated_results["learning_rate_init"] == learning_rate_init)]["score"]
            heatmap_data[i, j] = np.mean(score)
    
    plot_heatmap(heatmap_data, solvers, np.round(np.arange(MIN_LEARNING_RATE_INIT, MAX_LEARNING_RATE_INIT, LEARNING_RATE_STEP_SIZE), 3), 
                 f"SGD vs Adam ({lr_strategy.capitalize()} Learning Rate)", "Initial Learning Rate", "Solver")

# Heatmap for comparing learning rate strategies for each solver
def plot_lr_strategy_comparison(aggregated_results, solver):
    heatmap_data = np.zeros((3, len(np.arange(MIN_LEARNING_RATE_INIT, MAX_LEARNING_RATE_INIT, LEARNING_RATE_STEP_SIZE))))
    lr_strategies = ['constant', 'invscaling', 'adaptive']
    
    for i, lr_strategy in enumerate(lr_strategies):
        for j, learning_rate_init in enumerate(np.arange(MIN_LEARNING_RATE_INIT, MAX_LEARNING_RATE_INIT, LEARNING_RATE_STEP_SIZE)):
            mask = (aggregated_results["solver"] == solver) & (aggregated_results["learning_rate_strategy"] == lr_strategy)
            score = aggregated_results[mask & (aggregated_results["learning_rate_init"] == learning_rate_init)]["score"]
            heatmap_data[i, j] = np.mean(score)
    
    plot_heatmap(heatmap_data, lr_strategies, np.round(np.arange(MIN_LEARNING_RATE_INIT, MAX_LEARNING_RATE_INIT, LEARNING_RATE_STEP_SIZE), 3),
                 f"Learning Rate Strategies ({solver.capitalize()} Solver)", "Initial Learning Rate", "Learning Rate Strategy")

# Plot visualizations
plot_solver_comparison(aggregated_results, 'constant')
plot_solver_comparison(aggregated_results, 'invscaling')
plot_solver_comparison(aggregated_results, 'adaptive')

plot_lr_strategy_comparison(aggregated_results, 'sgd')
plot_lr_strategy_comparison(aggregated_results, 'adam')
