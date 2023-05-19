'''
File for extending the spectrum of a GRB to the range of the detectors.
'''

import numpy as np
import matplotlib.pyplot as plt

input_file = 'GRBspec_2.5.dat'

dat = np.loadtxt(input_file)

wl_min = 3000
wl_max = 25000

#interpolate dat
from scipy.interpolate import interp1d
f = interp1d(dat[:,0], dat[:,1], kind='cubic')

xnew = np.linspace(wl_min, wl_max, num=100000, endpoint=True)
ynew = xnew.copy()
for i in range(len(xnew)):
    if xnew[i] < dat[0,0]:
        ynew[i] = dat[0,1]
    elif xnew[i] > dat[-1,0]:
        ynew[i] = dat[-1,1]
    elif xnew[i] >= dat[0,0] and xnew[i] <= dat[-1,0]:
        ynew[i] = f(xnew[i])


# write to dat file
np.savetxt('extended_'+input_file, np.transpose([xnew, ynew]), fmt='%1.6f')
