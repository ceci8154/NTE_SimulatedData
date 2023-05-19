import numpy as np
import h5py
from astropy.io import fits
import matplotlib.pyplot as plt


# wl 853

width = 0.0125 # * 1.5

print('wl 853')
wl_low = 0.853 - width/2
wl_high = 0.853 + width/2
print(wl_low, wl_high)

filename = "../NTEpyechelle/models/NTE.hdf"
with h5py.File(filename, "r") as f:
    tab = f['CCD_2']['fiber_3']['order8']
    wls = tab['wavelength']
    x = tab['translation_x']
    y = tab['translation_y']
    ang = tab['rotation']

# Find the index of the wavelength closest to wl_low
idx = np.argmin(np.abs(wls - wl_low))
if wls[idx] < wl_low:
    idx_lowhigh = idx + 1
    idx_lowlow = idx
else:
    idx_lowhigh = idx
    idx_lowlow = idx - 1

# Find right x value for wl_low
d = (x[idx_lowhigh] - x[idx_lowlow]) / (wls[idx_lowhigh] - wls[idx_lowlow])
x_low = x[idx_lowlow] + (wl_low - wls[idx_lowlow]) * d

# Find right y value for wl_low
d = (y[idx_lowhigh] - y[idx_lowlow]) / (wls[idx_lowhigh] - wls[idx_lowlow])
y_low = y[idx_lowlow] + (wl_low - wls[idx_lowlow]) * d

# Find rotation for wl_low
d = (ang[idx_lowhigh] - ang[idx_lowlow]) / (wls[idx_lowhigh] - wls[idx_lowlow])
ang_low = ang[idx_lowlow] + (wl_low - wls[idx_lowlow]) * d

# Find the index of the wavelength closest to wl_high
idx = np.argmin(np.abs(wls - wl_high))
if wls[idx] < wl_high:
    idx_highhigh = idx + 1
    idx_highlow = idx
else:
    idx_highhigh = idx
    idx_highlow = idx - 1

# Find right x value for wl_high
d = (x[idx_highhigh] - x[idx_highlow]) / (wls[idx_highhigh] - wls[idx_highlow])
x_high = x[idx_highlow] + (wl_high - wls[idx_highlow]) * d

# Find right y value for wl_high
d = (y[idx_highhigh] - y[idx_highlow]) / (wls[idx_highhigh] - wls[idx_highlow])
y_high = y[idx_highlow] + (wl_high - wls[idx_highlow]) * d

# Find rotation for wl_high
d = (ang[idx_highhigh] - ang[idx_highlow]) / (wls[idx_highhigh] - wls[idx_highlow])
ang_high = ang[idx_highlow] + (wl_high - wls[idx_highlow]) * d



file = '/Users/ceciliehenneberg/Dropbox/Cecilies Dropbox/KU/Speciale/nte-simulated-data/Output/nuuuutest/vis_HgAr_schem_both_test.fits'
data = fits.getdata(file)

selection = data[int(np.round(y_low))-5:int(np.round(y_low))+90, int(np.round(x_low)):int(np.round(x_high))]
plt.figure()
plt.imshow(selection, origin='lower')
s_853 = np.sum(selection)
print(6.5e8/s_853)
print(s_853)


# x = np.linspace(-12,70,10000)
# x_ind = x_in + x*np.sin(-angle)
# y_ind = y_in + x*np.cos(-angle)
# # plt.plot(x_ind, y_ind, 'gx')

# x_ind = np.array([math.floor(x) for x in x_ind])
# y_ind = np.array([math.floor(y) for y in y_ind])
# y = data[y_ind, x_ind]





# wl 694

width = 0.011 # * 1.5

print('wl 694')
wl_low = 0.694 - width/2
wl_high = 0.694 + width/2
print(wl_low, wl_high)

filename = "../NTEpyechelle/models/NTE.hdf"
with h5py.File(filename, "r") as f:
    tab = f['CCD_2']['fiber_3']['order9']
    wls = tab['wavelength']
    x = tab['translation_x']
    y = tab['translation_y']
    ang = tab['rotation']

# Find the index of the wavelength closest to wl_low
idx = np.argmin(np.abs(wls - wl_low))
if wls[idx] < wl_low:
    idx_lowhigh = idx + 1
    idx_lowlow = idx
else:
    idx_lowhigh = idx
    idx_lowlow = idx - 1

# Find right x value for wl_low
d = (x[idx_lowhigh] - x[idx_lowlow]) / (wls[idx_lowhigh] - wls[idx_lowlow])
x_low = x[idx_lowlow] + (wl_low - wls[idx_lowlow]) * d

# Find right y value for wl_low
d = (y[idx_lowhigh] - y[idx_lowlow]) / (wls[idx_lowhigh] - wls[idx_lowlow])
y_low = y[idx_lowlow] + (wl_low - wls[idx_lowlow]) * d

# Find rotation for wl_low
d = (ang[idx_lowhigh] - ang[idx_lowlow]) / (wls[idx_lowhigh] - wls[idx_lowlow])
ang_low = ang[idx_lowlow] + (wl_low - wls[idx_lowlow]) * d

# Find the index of the wavelength closest to wl_high
idx = np.argmin(np.abs(wls - wl_high))
if wls[idx] < wl_high:
    idx_highhigh = idx + 1
    idx_highlow = idx
else:
    idx_highhigh = idx
    idx_highlow = idx - 1

# Find right x value for wl_high
d = (x[idx_highhigh] - x[idx_highlow]) / (wls[idx_highhigh] - wls[idx_highlow])
x_high = x[idx_highlow] + (wl_high - wls[idx_highlow]) * d

# Find right y value for wl_high
d = (y[idx_highhigh] - y[idx_highlow]) / (wls[idx_highhigh] - wls[idx_highlow])
y_high = y[idx_highlow] + (wl_high - wls[idx_highlow]) * d

# Find rotation for wl_high
d = (ang[idx_highhigh] - ang[idx_highlow]) / (wls[idx_highhigh] - wls[idx_highlow])
ang_high = ang[idx_highlow] + (wl_high - wls[idx_highlow]) * d



file = '/Users/ceciliehenneberg/Dropbox/Cecilies Dropbox/KU/Speciale/nte-simulated-data/Output/nuuuutest/vis_HgAr_schem_both_test.fits'
data = fits.getdata(file)

selection = data[int(np.round(y_low))-5:int(np.round(y_low))+90, int(np.round(x_low)):int(np.round(x_high))]
plt.figure()
plt.imshow(selection, origin='lower')
s_694_1 = np.sum(selection)



# wl 694

width = 0.011 # * 1.5

#print('wl 694')
wl_low = 0.694 - width/2
wl_high = 0.694 + width/2

filename = "../NTEpyechelle/models/NTE.hdf"
with h5py.File(filename, "r") as f:
    tab = f['CCD_2']['fiber_3']['order10']
    wls = tab['wavelength']
    x = tab['translation_x']
    y = tab['translation_y']
    ang = tab['rotation']

# Find the index of the wavelength closest to wl_low
idx = np.argmin(np.abs(wls - wl_low))
if wls[idx] < wl_low:
    idx_lowhigh = idx + 1
    idx_lowlow = idx
else:
    idx_lowhigh = idx
    idx_lowlow = idx - 1

# Find right x value for wl_low
d = (x[idx_lowhigh] - x[idx_lowlow]) / (wls[idx_lowhigh] - wls[idx_lowlow])
x_low = x[idx_lowlow] + (wl_low - wls[idx_lowlow]) * d

# Find right y value for wl_low
d = (y[idx_lowhigh] - y[idx_lowlow]) / (wls[idx_lowhigh] - wls[idx_lowlow])
y_low = y[idx_lowlow] + (wl_low - wls[idx_lowlow]) * d

# Find rotation for wl_low
d = (ang[idx_lowhigh] - ang[idx_lowlow]) / (wls[idx_lowhigh] - wls[idx_lowlow])
ang_low = ang[idx_lowlow] + (wl_low - wls[idx_lowlow]) * d

# Find the index of the wavelength closest to wl_high
idx = np.argmin(np.abs(wls - wl_high))
if wls[idx] < wl_high:
    idx_highhigh = idx + 1
    idx_highlow = idx
else:
    idx_highhigh = idx
    idx_highlow = idx - 1

# Find right x value for wl_high
d = (x[idx_highhigh] - x[idx_highlow]) / (wls[idx_highhigh] - wls[idx_highlow])
x_high = x[idx_highlow] + (wl_high - wls[idx_highlow]) * d

# Find right y value for wl_high
d = (y[idx_highhigh] - y[idx_highlow]) / (wls[idx_highhigh] - wls[idx_highlow])
y_high = y[idx_highlow] + (wl_high - wls[idx_highlow]) * d

# Find rotation for wl_high
d = (ang[idx_highhigh] - ang[idx_highlow]) / (wls[idx_highhigh] - wls[idx_highlow])
ang_high = ang[idx_highlow] + (wl_high - wls[idx_highlow]) * d



file = '/Users/ceciliehenneberg/Dropbox/Cecilies Dropbox/KU/Speciale/nte-simulated-data/Output/nuuuutest/vis_HgAr_schem_both_test.fits'
data = fits.getdata(file)

selection = data[int(np.round(y_low))-5:int(np.round(y_low))+90, int(np.round(x_low)):int(np.round(x_high))]
plt.figure()
plt.imshow(selection, origin='lower')
s_694_2 = np.sum(selection)
s_694 = s_694_1 + s_694_2
# print(1.2e8/s_694_1)
# print(1.2e8/s_694_2)
print(1.2e8/s_694_1)
print(s_694_1)




# wl 405

width = 0.014 # * 1.5

print('wl 405')
wl_low = 0.405 - width/2
wl_high = 0.405 + width/2

filename = "../NTEpyechelle/models/NTE.hdf"
with h5py.File(filename, "r") as f:
    tab = f['CCD_3']['fiber_3']['order16']
    wls = tab['wavelength']
    x = tab['translation_x']
    y = tab['translation_y']
    ang = tab['rotation']

# Find the index of the wavelength closest to wl_low
idx = np.argmin(np.abs(wls - wl_low))
if wls[idx] < wl_low:
    idx_lowhigh = idx + 1
    idx_lowlow = idx
else:
    idx_lowhigh = idx
    idx_lowlow = idx - 1

# Find right x value for wl_low
d = (x[idx_lowhigh] - x[idx_lowlow]) / (wls[idx_lowhigh] - wls[idx_lowlow])
x_low = x[idx_lowlow] + (wl_low - wls[idx_lowlow]) * d

# Find right y value for wl_low
d = (y[idx_lowhigh] - y[idx_lowlow]) / (wls[idx_lowhigh] - wls[idx_lowlow])
y_low = y[idx_lowlow] + (wl_low - wls[idx_lowlow]) * d

# Find rotation for wl_low
d = (ang[idx_lowhigh] - ang[idx_lowlow]) / (wls[idx_lowhigh] - wls[idx_lowlow])
ang_low = ang[idx_lowlow] + (wl_low - wls[idx_lowlow]) * d

# Find the index of the wavelength closest to wl_high
idx = np.argmin(np.abs(wls - wl_high))
if wls[idx] < wl_high:
    idx_highhigh = idx + 1
    idx_highlow = idx
else:
    idx_highhigh = idx
    idx_highlow = idx - 1

# Find right x value for wl_high
d = (x[idx_highhigh] - x[idx_highlow]) / (wls[idx_highhigh] - wls[idx_highlow])
x_high = x[idx_highlow] + (wl_high - wls[idx_highlow]) * d

# Find right y value for wl_high
d = (y[idx_highhigh] - y[idx_highlow]) / (wls[idx_highhigh] - wls[idx_highlow])
y_high = y[idx_highlow] + (wl_high - wls[idx_highlow]) * d

# Find rotation for wl_high
d = (ang[idx_highhigh] - ang[idx_highlow]) / (wls[idx_highhigh] - wls[idx_highlow])
ang_high = ang[idx_highlow] + (wl_high - wls[idx_highlow]) * d



file = '/Users/ceciliehenneberg/Dropbox/Cecilies Dropbox/KU/Speciale/nte-simulated-data/Output/nuuuutest/uvb_HgAr_schem_both_test.fits'
data = fits.getdata(file)

selection = data[int(np.round(y_low))-5:int(np.round(y_low))+110, int(np.round(x_low)):int(np.round(x_high))]
plt.figure()
plt.imshow(selection, origin='lower')
s_405 = np.sum(selection)
print(4.3e9/s_405)
print(s_405)




# wl 366

width = 0.007 # * 1.5

print('wl 366')
wl_low = 0.366 - width/2
wl_high = 0.366 + width/2

print(wl_low, wl_high)

filename = "../NTEpyechelle/models/NTE.hdf"
with h5py.File(filename, "r") as f:
    tab = f['CCD_3']['fiber_3']['order18']
    wls = tab['wavelength']
    x = tab['translation_x']
    y = tab['translation_y']
    ang = tab['rotation']

# Find the index of the wavelength closest to wl_low
idx = np.argmin(np.abs(wls - wl_low))
if wls[idx] < wl_low:
    idx_lowhigh = idx + 1
    idx_lowlow = idx
else:
    idx_lowhigh = idx
    idx_lowlow = idx - 1

# Find right x value for wl_low
d = (x[idx_lowhigh] - x[idx_lowlow]) / (wls[idx_lowhigh] - wls[idx_lowlow])
x_low = x[idx_lowlow] + (wl_low - wls[idx_lowlow]) * d

# Find right y value for wl_low
d = (y[idx_lowhigh] - y[idx_lowlow]) / (wls[idx_lowhigh] - wls[idx_lowlow])
y_low = y[idx_lowlow] + (wl_low - wls[idx_lowlow]) * d

# Find rotation for wl_low
d = (ang[idx_lowhigh] - ang[idx_lowlow]) / (wls[idx_lowhigh] - wls[idx_lowlow])
ang_low = ang[idx_lowlow] + (wl_low - wls[idx_lowlow]) * d

# Find the index of the wavelength closest to wl_high
idx = np.argmin(np.abs(wls - wl_high))
if wls[idx] < wl_high:
    idx_highhigh = idx + 1
    idx_highlow = idx
else:
    idx_highhigh = idx
    idx_highlow = idx - 1

# Find right x value for wl_high
d = (x[idx_highhigh] - x[idx_highlow]) / (wls[idx_highhigh] - wls[idx_highlow])
x_high = x[idx_highlow] + (wl_high - wls[idx_highlow]) * d

# Find right y value for wl_high
d = (y[idx_highhigh] - y[idx_highlow]) / (wls[idx_highhigh] - wls[idx_highlow])
y_high = y[idx_highlow] + (wl_high - wls[idx_highlow]) * d

# Find rotation for wl_high
d = (ang[idx_highhigh] - ang[idx_highlow]) / (wls[idx_highhigh] - wls[idx_highlow])
ang_high = ang[idx_highlow] + (wl_high - wls[idx_highlow]) * d



file = '/Users/ceciliehenneberg/Dropbox/Cecilies Dropbox/KU/Speciale/nte-simulated-data/Output/nuuuutest/uvb_HgAr_schem_both_test.fits'
data = fits.getdata(file)

selection = data[int(np.round(y_low))-5:int(np.round(y_low))+100, int(np.round(x_low)):int(np.round(x_high))]
plt.figure()
plt.imshow(selection, origin='lower')
s_366 = np.sum(selection)
print((3.3e9/s_366))
print(s_366)





# Hg

# 844.56 5.0
# 849.326 4.0
# 850.748 6.0
# 851.34 5.0
# 851.59 4.0
# 854.109 3.0
# 855.96 5.0
# 858.49 6.0
# 861.534 6.0

# Ar

# 850.3327 2.0
# 852.3782 15000.0
# 854.937 4.0
# 856.4901 2.0
# 858.7616 3.0
# 859.49854 3.0
# 860.63806 7.0
# 860.814 7.0
# 860.9977 2.0



# 6.5e8 = s * ((5 + 4 + 6 + 5 + 4 + 3 + 5 + 6 + 6) + ar * (2 + 15000 + 4 + 2 + 3 + 3 + 7 + 7 + 2))





# Hg

# 365.1198 9000.0
# 365.5884 3000.0
# 366.393 500.0
# 366.43265 2000.0
# 370.2501 50.0
# 370.3391 1.0
# 370.52194 9.0

# Ar

# 360.755 7.0
# 360.84326 2.0
# 361.23965 1.0
# 361.28433 14.0
# 362.18417 3.0
# 362.20465 3.0
# 362.317 30.0
# 362.44827 2.0
# 363.58484 9.0
# 363.6674 3.0
# 363.80676 19.0
# 363.891 78.0
# 364.08698 42.0
# 365.19293 8.0
# 365.22342 0.5
# 365.6319 37.0
# 365.7091 27.0
# 365.82587 3.0
# 366.14792 28.0
# 367.0647 26.0
# 367.139 7.0
# 367.20496 19.0
# 367.431 22.0
# 367.93167 17.0
# 368.11084 20.0
# 368.35928 9.0
# 368.45062 6.0
# 368.66406 5.0
# 368.76447 6.0
# 368.92764 7.0
# 369.02817 4.0
# 369.31747 1.0
# 369.56937 2.0
# 369.7138 9.0
# 370.30548 2.0
# 370.4604 1.0
# 370.79843 7.0
# 371.0964 11.0

# 3.3e9 = s * ((9000 + 3000 + 500 + 2000 + 50 + 1 + 9) + ar * (7 + 2 + 1 + 14 + 3 + 3 + 30 + 2 + 9 + 3 + 19 + 78 + 42 + 8 + 0.5 + 37 + 27 + 3 + 28 + 26 + 7 + 19 + 22 + 17 + 20 + 9 + 6 + 5 + 6 + 7 + 4 + 1 + 2 + 9 + 2 + 1 + 7 + 11))


# 6.5e8 = a * (44 + b * 15030)

# 3.3e9 = a * (14560 + b * 497.5)

# a = 6.5e8 / (44 + b * 15030)

# 3.3e9 = 6.5e8 / (44 + b * 15030) * (14560 + b * 497.5)

# 3.3e9 * 44 + b * 3.3e9 * 15030 = 6.5e8 * 14560 + b * 6.5e8 * 497.5

# b * 3.3e9 * 15030 - b * 6.5e8 * 497.5 = 6.5e8 * 14560 - 3.3e9 * 44

# b * (3.3e9 * 15030 - 6.5e8 * 497.5) = 6.5e8 * 14560 - 3.3e9 * 44

# b = (6.5e8 * 14560 - 3.3e9 * 44) / (3.3e9 * 15030 - 6.5e8 * 497.5)

# b = 0.1891158153752489

# a = 225193.17810655592









# Hg

# # 844.56 5.0
# 849.326 4.0
# 850.748 6.0
# 851.34 5.0
# 851.59 4.0
# 854.109 3.0
# 855.96 5.0
# 858.49 6.0
# # 861.534 6.0

# Ar

# 852.3782 15000.0
# # 860.814 7.0


# 6.5e8 = s * ((5 + 4 + 6 + 5 + 4 + 3 + 5 + 6 + 6) + ar * (15000 + 7))

# 6.5e8 = s * ((4 + 6 + 5 + 4 + 3 + 5 + 6) + ar * (15000))




# Hg

# 365.1198 9000.0
# 365.5884 3000.0
# 366.393 500.0
# 366.43265 2000.0
# # 370.2501 50.0
# # 370.3391 1.0
# # 370.52194 9.0

# Ar

# # 360.755 7.0


# 3.3e9 = s * ((9000 + 3000 + 500 + 2000 + 50 + 1 + 9) + ar * (7))

# 3.3e9 = s * ((9000 + 3000 + 500 + 2000) + ar * (0))



# 6.5e8 = a * (44 + b * 15007)

# 3.3e9 = a * (14560 + b * 7)

# a = 6.5e8 / (44 + b * 15007)

# 3.3e9 = 6.5e8 / (44 + b * 15007) * (14560 + b * 7)

# 3.3e9 * 44 + b * 3.3e9 * 15007 = 6.5e8 * 14560 + b * 6.5e8 * 7

# b * 3.3e9 * 15007 - b * 6.5e8 * 7 = 6.5e8 * 14560 - 3.3e9 * 44

# b * (3.3e9 * 15007 - 6.5e8 * 7) = 6.5e8 * 14560 - 3.3e9 * 44

# b = (6.5e8 * 14560 - 3.3e9 * 44) / (3.3e9 * 15007 - 6.5e8 * 7)

# b = 0.18818806285725248

# a = 226627.84748700162






# 6.5e8 = s * ((33) + ar * (15000))
# 3.3e9 = s * ((14500) + ar * (0))


# s = 227586.2068965517
# ar = (6.5e8 - 7510344.827586207)/3413793103.4482756

# ar = 0.18820404040404043














# 686.235 679883.5
# 688.271 906511.4
# 690.937 226627840.0
# 692.604 679883.5
# 695.276 1133139.2
# 701.102 906511.4
# 687.3185 6397298.0
# 688.1481 213243.28
# 689.0075 426486.56
# 693.9578 2132432.8
# 695.3395 298540.6
# 696.217 298540.6
# 696.7351 426486560.0

a = 679883.5 + 906511.4 + 226627840.0 + 679883.5 + 1133139.2 + 906511.4 + 6397298.0 + 213243.28 + 426486.56 + 2132432.8 + 298540.6 + 298540.6 + 426486560.0
print(a)


# 844.56 1133139.2
# 849.326 906511.4
# 850.748 1359767.0
# 851.34 1133139.2
# 851.59 906511.4
# 854.109 679883.5
# 855.96 1133139.2
# 858.49 1359767.0
# 861.534 1359767.0
# 852.3782 639729800.0
# 860.814 298540.6

a = 1133139.2 + 906511.4 + 1359767.0 + 1133139.2 + 906511.4 + 679883.5 + 1133139.2 + 1359767.0 + 1359767.0 + 639729800.0 + 298540.6
print(a)