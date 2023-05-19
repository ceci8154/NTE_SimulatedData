import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np

data_list = ['uvb_grb25_fullslit_0500.fits', 'uvb_grb25_point_0500.fits', 'uvb_grb25_point_0200.fits', 'uvb_grb25_point_5_0500.fits']

for file_name in data_list:
    data = fits.getdata(file_name)
    #data = data[400:550,500:650]
    plt.figure()
    plt.imshow(data[500:650, 400:550], cmap='gray', origin='lower')
    plt.axis('off')
    plt.savefig(file_name[:-5] + '.png', bbox_inches='tight')#, pad_inches=0)

    plt.show()
    