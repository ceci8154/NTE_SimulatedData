import numpy as np
import matplotlib.pyplot as plt

# make wavelengths in nm
wls = np.linspace(400,700, 7)
# convert to mm
wls = wls*1e-6

def calc_diffraction_angle(wl, n, d):
    return np.arcsin(n*wl/d)

diffraction_angles = calc_diffraction_angle(wls, 1, 1/150)
# convert to degrees
diffraction_angles = np.rad2deg(diffraction_angles)
print(diffraction_angles)