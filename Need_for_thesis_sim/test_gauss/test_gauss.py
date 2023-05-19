import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

with fits.open('nir_dark_0000.fits') as f:
    data = f[0].data
    min = np.min(data)
    max = np.max(data)
    plt.figure()
    plt.title('Histogram of NIR dark')
    plt.hist(data.flatten(), bins=37, range=(min-0.5, max+0.5))
    plt.xlabel('Pixel value')
    plt.ylabel('Number of pixels')
    plt.tight_layout()
    plt.savefig('nir_dark_hist.pdf')