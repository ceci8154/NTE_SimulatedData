from NTEpyechelle.simulator import Simulator
from NTEpyechelle.sources import Phoenix, Constant, ThAr, ThNe, Etalon, CSV
from NTEpyechelle.spectrograph import ZEMAX
import os
from astropy.io import fits
import numpy as np
import h5py
import matplotlib.pyplot as plt

output_dir = 'Output/'

gpu = False

sky_source = CSV(filepath="data/line_files/skycalc_radiance.dat", \
        name="", list_like=True, wavelength_unit='nm', flux_in_photons=False, delimiter=' ')

lamp_source = CSV(filepath="data/line_files/HgAr.csv", name="", list_like=True, wavelength_unit='micron', flux_in_photons=True, delimiter=' ')


sim = Simulator(ZEMAX('NTE'))
sim.set_ccd(2)
sim.set_fibers(3)
sim.set_sources(lamp_source)
sim.set_exposure_time(10)
name = output_dir + 'jump_test.fits'
sim.set_output(name, overwrite=True)
#sim.read_noise = 3
sim.set_cuda(gpu)
sim.run()

# x_temp, y_temp = 0, 0
# # load the fits file
# with fits.open(name) as hdul:
#     # get data
#     data = hdul[0].data
#     # load hdf file
#     plt.figure()
#     with h5py.File('NTEpyechelle/models/NTE.hdf', 'r') as f:
#         # load data from CCD1
#         ccdgrp = f['CCD_2']
#         # load data from fiber 3
#         fbr_grp = ccdgrp['fiber_3']
#         orders = [8,9,10,11,12,13,14,15]
#         wavelenghts_to_check = [0.366,0.405,0.694,0.853]
#         x_temp = 0
#         y_temp = 0
#         for wl in wavelenghts_to_check:
#             print(wl)
#             for order in orders:
#                 #print(order)
#                 # load data from order
#                 order_grp = fbr_grp['order' + str(order)]
#                 # get the wavelength
#                 wavelength = order_grp['wavelength']
#                 # get scale_x and scale_y
#                 scale_x = order_grp['scale_x']
#                 scale_y = order_grp['scale_y']
#                 # save x and y for the wavelength closest to wl
#                 wl_temp = 999999999999
#                 for i,l in enumerate(wavelength):
#                     if abs(l - wl) < abs(wl_temp - wl):
#                         wl_temp = l
#                         x_temp = scale_x[i] *10.5
#                         y_temp = scale_y[i] *10.5
                    
#                     print(x_temp, y_temp)
#                     plt.plot(x_temp, y_temp, 'ro', label=str(wl))

#     plt.imshow(data, origin='lower')
#     plt.legend()
#     plt.show()
