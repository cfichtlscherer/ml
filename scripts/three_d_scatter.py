# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility

grid = np.load("/home/cpf/Desktop/grid_plot/grid_val.npy")

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

for i in range(5,14):
    for j in range(5,15):
        if grid[i-5, j-5] < 0.0017:
            ax.scatter(i, j, grid[i-5, j-5], c = "green", marker='o')
        elif grid[i-5, j-5] < 0.0020:
            ax.scatter(i, j, grid[i-5, j-5], c = "orange", marker='o')
        elif grid[i-5, j-5] < 0.0025:
            ax.scatter(i, j, grid[i-5, j-5], c = "red", marker='o')
        #        else:
        #    ax.scatter(i, j, grid[i-5, j-5], c = "black", marker='o')


ax.set_xlabel('kernels first convolution')
ax.set_ylabel('kernels second convolution')
ax.set_zlabel('Loss on test set')

plt.show()
#plt.savefig("3d_plot.png")
