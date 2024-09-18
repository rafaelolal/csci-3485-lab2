import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

C = 5
N = 500
D = 2

# Generate anisotropically distributed data
random_state = 170
X, Y = datasets.make_blobs(
    n_samples=N, centers=C, n_features=D, random_state=random_state
)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)

print(X)
print("-----")
print(Y)


# Use a colormap to assign different colors to each cluster
colors = plt.cm.rainbow(np.linspace(0, 1, C))

# Plot each cluster
for i in range(C):
    cluster = X_aniso[Y == i]
    plt.scatter(
        cluster[:, 0], cluster[:, 1], c=[colors[i]], s=C, label=f"Cluster {i}"
    )

plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.show()
