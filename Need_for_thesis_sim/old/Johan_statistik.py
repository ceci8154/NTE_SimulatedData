import numpy as np
import matplotlib.pyplot as plt
import glob
from astropy.io import fits
from astropy import stats
import astropy.visualization as asviz
from astropy.utils.data import get_pkg_data_filename

plt.close('all')

#image_file = get_pkg_data_filename('bias.fits')
#fits.info(image_file)
#image_data = fits.getdata(image_file, ext=0)
# Load fits file
f = fits.open('data/noise_ir/noise-mer-64/noise_mer-64_0001.fits')
# get data from file
image_data = f[1].data
#print(image_data)

plt.figure()
plt.imshow(image_data, cmap='gray_r', vmin=-10, vmax=15, origin='lower')
plt.colorbar()
print('Click on lower left and upper right corner. End with q')
tpoints = plt.ginput(n=2, timeout=30, show_clicks=True, mouse_add=1, mouse_stop=2)
xstart = int(tpoints[0][0])
ystart = int(tpoints[0][1])
xend = int(tpoints[1][0])
yend = int(tpoints[1][1])
plt.show()

#Derive mean, median and std.dev.
print('mean, median and std.dev.')
print(stats.sigma_clipped_stats(image_data[ystart:yend,xstart:xend], sigma=5, maxiters=5))

#Make histogram
numbers = image_data[ystart:yend,xstart:xend].flatten()

plt.figure()
plt.hist(numbers, bins=20000)
plt.xlim(-20,20)
plt.show()