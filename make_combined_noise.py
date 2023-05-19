'''
Written by Cecilie and Mads
This code generates the combined noise files used to add noise and bias.

In its final form, this code will take in a directory of bias frames and
calculate the median and standard deviation for each pixel. It will then
save these values to a fits file, to be used later.

At the moment, this only does it for the IR detector, as data for the other
detectors is not available. This will be updated when/if we get the data.
'''


import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os

# For ir
dir = 'data/noise_ir/noise-mer-64'
data = []
# Load all the bias frames
for filename in os.listdir(dir):
    data.append(fits.getdata(dir + '/' + filename))

# get the median for a schematic bias frame
med_data = np.median(data, axis=0)
# get the standard deviation for a schematic bias frame
std_data = np.std(data, axis=0)

fits.writeto('data/noise_ir/comb_noise.fits', med_data, overwrite=True)
# add the std to the fits file
fits.append('data/noise_ir/comb_noise.fits', std_data, overwrite=True)

# For vis
dir = 'data/noise_vis'
# TODO We have to get real files for these. This will be placeholder
# Load a bias frame simply to get the shape:
temp = fits.getdata(dir + '/vis_bias_schem.fits')
data = []
for i in range(10):
    data.append(np.random.normal(4,3,size=temp.shape))

med_data = np.median(data, axis=0)
std_data = np.std(data, axis=0)

fits.writeto('data/noise_vis/comb_noise.fits', med_data, overwrite=True)
fits.append('data/noise_vis/comb_noise.fits', std_data, overwrite=True)

# For uvb
dir = 'data/noise_uvb'
# TODO We have to get real files for these. This will be placeholder
# Load a bias frame simply to get the shape:
temp = fits.getdata(dir + '/uvb_bias_schem.fits')
data = []
for i in range(10):
    data.append(np.random.normal(4,3,size=temp.shape))
med_data = np.median(data, axis=0)
std_data = np.std(data, axis=0)

fits.writeto('data/noise_uvb/comb_noise.fits', med_data, overwrite=True)
fits.append('data/noise_uvb/comb_noise.fits', std_data, overwrite=True)