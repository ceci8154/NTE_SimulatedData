from NTEpyechelle.simulator import Simulator
from NTEpyechelle.sources import Phoenix, Constant, ThAr, ThNe, Etalon, CSV
from NTEpyechelle.spectrograph import ZEMAX
import os
from astropy.io import fits
import numpy as np
import h5py
import matplotlib.pyplot as plt

output_dir = 'Output/'

gpu = True

sky_source = CSV(filepath="data/line_files/skycalc_radiance.dat", \
        name="", list_like=True, wavelength_unit='nm', flux_in_photons=False, delimiter=' ')

sky_source_bad = CSV(filepath="data/line_files/skycalc_radiance_bad_resolution.dat", \
        name="", list_like=True, wavelength_unit='nm', flux_in_photons=False, delimiter=' ')

lamp_source = CSV(filepath="data/line_files/HgAr.csv", name="", list_like=True, wavelength_unit='micron', flux_in_photons=True, delimiter=' ')


sim = Simulator(ZEMAX('NTE'))
sim.set_ccd(3)
sim.set_fibers(1)
sim.set_sources(sky_source)
sim.set_exposure_time(100)
name = output_dir + 'good.fits'
sim.set_output(name, overwrite=True)
sim.set_cuda(gpu)
sim.run()

sim = Simulator(ZEMAX('NTE'))
sim.set_ccd(3)
sim.set_fibers(1)
sim.set_sources(sky_source_bad)
sim.set_exposure_time(100)
name = output_dir + 'bad.fits'
sim.set_output(name, overwrite=True)
sim.set_cuda(gpu)
sim.run()