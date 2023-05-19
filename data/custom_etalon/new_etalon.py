'''
File to make the final etalon file through the mosaic filter.
'''

import pandas as pd
from scipy.interpolate import UnivariateSpline

dat = pd.read_csv('etalon_data/mosaic_filter_relthrough.dat', skiprows=3, header=None, delimiter=' ', dtype=float)
wl1 = dat.iloc[:,0]
flux1 = dat.iloc[:,1]

dat = pd.read_csv('etalon_data/etalon_sim_300-2500nm.dat', skiprows=3, header=None, delimiter='  ', dtype=float)
wl2 = dat.iloc[:,0]
flux2 = dat.iloc[:,1]

y2 = UnivariateSpline(wl1, flux1, s=0, k=2)
flux1new = y2(wl2)

flux = flux1new * flux2

output_df = pd.DataFrame({'wavelength':wl2, 'flux':flux})
output_df.to_csv('etalon_mosaic.dat', header=False, sep=' ', index=False)



