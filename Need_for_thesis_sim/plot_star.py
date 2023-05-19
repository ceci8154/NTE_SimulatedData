import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

# Read in the data
files = ['ir_slitwidth1.0_10sec_bd17d4708_0_0.fits',
         'uv_slitwidth1.0_10sec_bd17d4708_0_0.fits',
         'vis_slitwidth1.0_10sec_bd17d4708_0_0.fits']

for file in files:
    hdulist = fits.open(file)
    data = hdulist[0].data
    hdulist.close()

    cmin = 1000
    cmax = 1100
    limi = data.shape[0]*data.shape[1]*0.01
    # get how many pixels are above cmax
    while np.sum(data > cmax) > limi:
        cmax += 100
    data = data.clip(cmin, cmax)
    data = data - cmin

    # Plot the data
    plt.figure()
    plt.imshow(np.log(data+1), origin='lower', cmap='gray')
    plt.axis('off')
    plt.savefig(file[:-5]+'.png', bbox_inches='tight', pad_inches=0.1, dpi=300)

#plt.show()