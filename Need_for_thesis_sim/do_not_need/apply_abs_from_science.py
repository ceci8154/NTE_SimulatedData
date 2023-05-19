import numpy as np

dat = np.loadtxt("data/scienceframe_data/GRB/extended_GRBspec_2.5.dat")
abs = np.loadtxt("data/skycalc_abs.dat")
abs[:,0] *= 10

#interpolate dat
from scipy.interpolate import interp1d
f = interp1d(abs[:,0], abs[:,1], kind='cubic')

xnew = dat[:,0]
ynew = f(xnew)

dat[:,1] = dat[:,1] * ynew

np.savetxt("data/scienceframe_data/GRB/extended_GRBspec_2.5_withabs.dat", dat, fmt='%1.6f')





