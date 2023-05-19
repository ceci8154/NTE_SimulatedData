import numpy as np
import os
import matplotlib.pyplot as plt

# Get all .png files in this folder
pngs = os.listdir()
pngs = [png for png in pngs if '.png' in png]

# Make the cmap
colors = ['#ebe8e8', '#000000']
cmap = plt.cm.colors.LinearSegmentedColormap.from_list("", colors)

# Loop over all pngs
for png in pngs:
    # Get the data
    data = plt.imread(png)
    # Convert to grayscale
    data = np.mean(data, axis=2)
    # Remove pixels from each side
    nr_pix_rm = 6
    data = data[nr_pix_rm:-nr_pix_rm-2, nr_pix_rm+2:-nr_pix_rm-1]
    # Set outer most pixels to max
    nr_max_pix = 2
    data[:nr_max_pix, :] = np.max(data)
    data[-nr_max_pix:, :] = np.max(data)
    data[:, :nr_max_pix] = np.max(data)
    data[:, -nr_max_pix:] = np.max(data)


    # Plot with new cmap
    plt.figure()
    plt.imshow(data, cmap=cmap)
    plt.axis('off')
    plt.savefig('i'+png, bbox_inches='tight', dpi=500)