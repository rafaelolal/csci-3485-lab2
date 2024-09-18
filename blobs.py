import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

C = 5
N = 500
D = 2

X, Y = make_blobs(n_samples=N, centers=C, n_features=D, random_state=1)

# Use a colormap to assign different colors to each cluster
colors = plt.cm.rainbow(np.linspace(0, 1, C))

# Plot each cluster
for i in range(C):
    cluster = X[Y == i]
    plt.scatter(
        cluster[:, 0], cluster[:, 1], c=[colors[i]], s=C, label=f"Cluster {i}"
    )

plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.show()
