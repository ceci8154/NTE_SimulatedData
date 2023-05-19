'''
Written by Cecilie and Mads
This script is used to make the normalized flat files.
It would be nice to have additional files for the flat fields, but for now
we just have a single one.
We only have a flatfield for the IR detector,
so the other ones are just from a normal distribution.
'''

from astropy.io import fits
import numpy as np
import os
import matplotlib.pyplot as plt

# For ir
dir = 'data/flat_ir'

data = []
for filename in os.listdir(dir):
    if '300' in filename:
        data.append(fits.getdata(dir + '/' + filename))

# sum of data
sum_data = np.sum(data, axis=0)
# normalize data
norm_data = (sum_data - np.min(sum_data)) / (np.max(sum_data) - np.min(sum_data))
# write to file
fits.writeto('data/flat_ir/norm_flat.fits', norm_data, overwrite=True)

# TODO: add correct for both vis and uvb

# For vis
dir = 'data/flat_vis'

# load file for shape TODO change to real file later
temp = fits.getdata(dir + '/vis_bias_schem.fits')
norm_data = np.random.normal(1, 0.01, temp.shape)
# write to file
fits.writeto('data/flat_vis/norm_flat.fits', norm_data, overwrite=True)

# For uvb
dir = 'data/flat_uvb'

# load file for shape TODO change to real file later
temp = fits.getdata(dir + '/uvb_bias_schem.fits')
norm_data = np.random.normal(1, 0.01, temp.shape)
# write to file
fits.writeto('data/flat_uvb/norm_flat.fits', norm_data, overwrite=True)
