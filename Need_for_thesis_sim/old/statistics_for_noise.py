import numpy as np
import matplotlib.pyplot as plt
import os
from astropy.io import fits

# load the fits files
dir = 'data/noise_ir/noise-mer-64/'
files = os.listdir(dir)
fits_data = []
for file in files:
    f = fits.open(dir+file)
    # get data
    data = f[1].data
    fits_data.append(data)

# get the mean, median and std of all files
mean = np.mean(fits_data)
median = np.median(fits_data)
std = np.std(fits_data)

fits_data = np.array(fits_data)
flat = fits_data.flatten()

mi = -7
ma = 13

# remove all values outside the range
flat = flat[(flat > mi) & (flat < ma)]

# make a histogram over the pixel values. Only counting values between -10 and 10
plt.figure()
plt.hist(flat, bins=26)
plt.xlabel('Pixel value')
plt.ylabel('Number of pixels')
plt.savefig('noise_histogram.png')

# load the fits files
dir = 'data/noise_ir/noise-mer-64/'
files = os.listdir(dir)
fits_first = []
for file in files:
    f = fits.open(dir+file)
    # get data of the first pixel in each frame
    data = f[1].data
    fits_first.append(data[0,0])

# Make histogram over these pixels
plt.figure()
plt.hist(fits_first, bins=26)
plt.xlabel('Pixel value')
plt.ylabel('Number of pixels')
plt.savefig('noise_first_pixel_histogram.png')