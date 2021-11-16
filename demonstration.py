import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np

import sca

# Generate five 3D vectors randomly
X_rand = np.random.rand(5, 3)

# Run SCA
sca_rand = sca.SCA(
    X_rand
)

# Now let's plot them
fig = plt.figure()
ax = fig.gca(projection='3d')

# Plot the random vectors
ax.quiver(
    np.zeros(5), 
    np.zeros(5), 
    np.zeros(5), 
    X_rand[:,0], 
    X_rand[:,1], 
    X_rand[:,2], 
    length=1, 
    normalize=True
)

# Plot the SCA loading vectors
ax.quiver(
    np.zeros(3), 
    np.zeros(3), 
    np.zeros(3), 
    sca_rand[:,0], 
    sca_rand[:,1], 
    sca_rand[:,2], 
    color='red', 
    length=1, 
    normalize=True
)

ax.set_xlim((0,2))
ax.set_ylim((0,2))
ax.set_zlim((0,2))
plt.show()

