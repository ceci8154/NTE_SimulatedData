import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import math
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
import h5py
import os

# data = fits.open('/Users/ceciliehenneberg/Downloads/Output/Actual_Files/point_test_const_testposwide.fits')[0].data
# plt.imshow(data, origin='lower')

# # x_in = 591.707
# # y_in = 	935.533
# # angle = 0.0478803

# # x_in = 2065.29
# # y_in = 		919.908
# # angle = 0.0434972

# x_in = 509.382
# y_in =	792.608
# angle = 0.104622

# x = np.linspace(-10,60,10000)
# x_ind = x_in + x*np.sin(-angle)
# y_ind = y_in + x*np.cos(-angle)
# plt.plot(x_ind[-1], y_ind[-1], 'gx')
# plt.plot(x_in, y_in, 'gx')

# x_ind = np.array([math.floor(x) for x in x_ind])
# y_ind = np.array([math.floor(y) for y in y_ind])

# y = data[y_ind, x_ind]

# plt.figure()
# plt.plot(x,y)

# # 0.38 arcsec/pix (60 pixel, 22.8 arcsec?)

# # IR 50 pix, 0.456 arcsec/pix
# # VIS ~77 pix, 0.296 arcsec/pix

# run through all files in folder
# for file in os.listdir('pics_for_point_source_section'):


fwhm_pics = False
if fwhm_pics == True:
    # run through all files in folder
    labels = ['exp time 70', 'exp time 90', 'exp time 200', 'exp time 8000']
    dir = 'pics_for_point_source_section/many/'
    files = ['uvb_flat_schem70.fits', 'uvb_flat_schem90.fits', 'uvb_flat_schem200.fits', 'uvb_flat_schem8000.fits']
    for j in range(len(files)):
        file = files[j]
        print(file)
        data = fits.open(dir + file)[0].data


        #data = fits.open('pics_for_point_source_section/uvb_grb25_point_5_0500.fits')[0].data
        type = 'UVB'
        # plt.imshow(data, origin='lower')

        filename = "NTEpyechelle/models/NTE.hdf"
        with h5py.File(filename, "r") as f:
            d = f['CCD_3']
            e = d['fiber_3']
            table = e['order16']

            for i in range(len(table)):
                tab = table[i]

                x_in = tab[4]
                y_in = tab[5]
                angle = tab[0]

                if x_in < 470 or x_in > 500:
                    continue

                # plt.figure()
                # plt.imshow(data, origin='lower')
                # plt.plot(x_in, y_in, 'rx')
                # plt.plot(x_in + 100*np.sin(-angle), y_in + 100*np.cos(-angle), 'rx')
                x = np.linspace(-12,70,10000)
                x_ind = x_in + x*np.sin(-angle)
                y_ind = y_in + x*np.cos(-angle)
                # plt.plot(x_ind, y_ind, 'gx')

                x_ind = np.array([math.floor(x) for x in x_ind])
                y_ind = np.array([math.floor(y) for y in y_ind])
                y = data[y_ind, x_ind]

                # plt.figure()
                plt.plot(x,y,label=labels[j])
    plt.xlabel('pix')
    plt.ylabel('counts')
    plt.legend()
    plt.savefig('pics_for_point_source_section/fwhm_pics/70902008000.png', bbox_inches='tight')



fwhm_stuff = True
if fwhm_stuff == True:
    data = fits.open('/Users/ceciliehenneberg/Dropbox/Cecilies Dropbox/KU/Speciale/nte-simulated-data/pics_for_point_source_section/many_seeings/nir_flat_schem0s3.2.fits')[0].data
    type = 'IR'
    # plt.imshow(data, origin='lower')

    filename = "NTEpyechelle/models/NTE.hdf"
    with h5py.File(filename, "r") as f:
        if type == 'IR':
            d = f['CCD_1']
        elif type == 'VIS':
            d = f['CCD_2']
        elif type == 'UV':
            d = f['CCD_3']
        #pix_size = d.attrs['pixelsize']
        e = d['fiber_3']
        orders = e.keys()
        vals_list = []

        for order in orders:
            if order[:5] != 'order':
                continue
            print(order)
            vals = []
            table = e[order]

            # if key == 'order16':
            #     range = np.arange(6,35)
            # elif key == 'order17':
            #     range = np.arange(6,49)
            # else:
            #     range = range(len(table))


            for i in range(len(table)):
                tab = table[i]

                x_in = tab[4]
                y_in = tab[5]
                angle = tab[0]

                if type == 'IR':
                    if order == 'order3' or order == 'order4':
                        if x_in < 500 or x_in > 1600:
                            continue
                    if order == 'order5':
                        if x_in < 700 or x_in > 1300:
                            continue
                    elif order == 'order6':
                        if x_in < 800 or x_in > 1200:
                            continue
                    elif order == 'order7':
                        if x_in < 900 or x_in > 1100:
                            continue
                elif type == 'VIS':
                    if order == 'order8':
                        if x_in < 1550 or x_in > 2650:
                            continue
                    elif order == 'order9':
                        if x_in < 1600 or x_in > 2550:
                            continue
                    elif order == 'order10':
                        if x_in < 1650 or x_in > 2500:
                            continue
                    elif order == 'order11':
                        if x_in < 1700 or x_in > 2400:
                            continue
                    elif order == 'order12':
                        if x_in < 1800 or x_in > 2300:
                            continue
                    elif order == 'order13':
                        if x_in < 1850 or x_in > 2200:
                            continue
                    elif order == 'order14':
                        if x_in < 1900 or x_in > 2200:
                            continue
                    elif order == 'order15':
                        if x_in < 1900 or x_in > 2200:
                            continue
                elif type == 'UV':
                    # if x_in < 150 or x_in > 850:
                    if x_in < 450 or x_in > 500:
                        continue
                    if order == 'order18':
                        if x_in < 180 or x_in > 800:
                            continue
                    elif order == 'order19':
                        if x_in < 200 or x_in > 750:
                            continue
                    elif order == 'order20':
                        if x_in < 200 or x_in > 750:
                            continue
                    
                

                plt.figure()
                plt.imshow(data, origin='lower')
                plt.plot(x_in, y_in, 'rx')
                plt.plot(x_in + 100*np.sin(-angle), y_in + 100*np.cos(-angle), 'rx')
                # plt.plot(509.382,	792.608, 'rx')

                if type == 'IR':
                    x = np.linspace(-10,60,10000)
                elif type == 'VIS':
                    x = np.linspace(-10,90,10000)
                elif type == 'UV':
                    x = np.linspace(-10,70,10000)
                x_ind = x_in + x*np.sin(-angle)
                y_ind = y_in + x*np.cos(-angle)
                plt.plot(x_ind, y_ind, 'gx')

                x_ind = np.array([math.floor(x) for x in x_ind])
                y_ind = np.array([math.floor(y) for y in y_ind])

                y = data[y_ind, x_ind]

                def Gauss(x, A, B, C):
                    y = A*np.exp(-(x-B)**2/2*C**2)
                    return y
                if type == 'IR':
                    parameters, covariance = curve_fit(Gauss, x, y, p0=[20000, 25, 10])
                elif type == 'VIS':
                    parameters, covariance = curve_fit(Gauss, x, y, p0=[7000, 40, 20])
                elif type == 'UV':
                    parameters, covariance = curve_fit(Gauss, x, y, p0=[65000, 25, 5])


                fit_A = parameters[0]
                fit_B = parameters[1]
                fit_C = parameters[2]
                
                fit_y = Gauss(x, fit_A, fit_B, fit_C)

                plt.figure()
                plt.plot(x,y)
                plt.plot(x, fit_y, 'r-')

                spline = UnivariateSpline(x, fit_y-np.max(fit_y)/2, s=0)
                r1, r2 = spline.roots() # find the roots
                
                # if i >= 0:
                #     plt.figure()
                #     plt.plot(x,y)
                #     plt.plot(x, fit_y, 'r-')
                #     plt.axvspan(r1, r2, facecolor='g', alpha=0.5)

                if type == 'IR':
                    vals.append((r2-r1) * 0.465) # IR
                elif type == 'VIS':
                    vals.append((r2-r1) * 0.27) # VIS
                elif type == 'UV':
                    vals.append((r2-r1) * 0.345) # UV
                # print(str(i) + ':  ' + str((r2-r1)/pix_size))
            m = np.mean(vals)
            print(np.max(vals))
            # plt.figure()
            # plt.plot(vals)
            vals_list.append(m)
        print('MEAN: ', np.mean(vals_list))



    # 60 pixel, 22.8 arcsec
    # 0.38 arcsec/pix











fwhm_stuff = False
if fwhm_stuff == True:
    data = fits.open('/Users/ceciliehenneberg/Dropbox/Cecilies Dropbox/KU/Speciale/nte-simulated-data/pics_for_point_source_section/many_seeings/vis_flat_schem0s1.8.fits')[0].data
    type = 'VIS'
    # plt.imshow(data, origin='lower')

    filename = "NTEpyechelle/models/NTE.hdf"
    with h5py.File(filename, "r") as f:
        if type == 'IR':
            d = f['CCD_1']
        elif type == 'VIS':
            d = f['CCD_2']
        elif type == 'UV':
            d = f['CCD_3']
        #pix_size = d.attrs['pixelsize']
        e = d['fiber_3']
        orders = e.keys()
        vals_list = []
        if type == 'IR':
            order = 'order3'
        elif type == 'VIS':
            order = 'order8'
        elif type == 'UV':
            order = 'order16'

        vals = []
        table = e[order]

        # if key == 'order16':
        #     range = np.arange(6,35)
        # elif key == 'order17':
        #     range = np.arange(6,49)
        # else:
        #     range = range(len(table))


        for i in range(len(table)):
            tab = table[i]

            x_in = tab[4]
            y_in = tab[5]
            angle = tab[0]

            if type == 'IR':
                if x_in < 400 or x_in > 1980:
                    continue
            elif type == 'VIS':
                if x_in < 2100 or x_in > 2200:
                    continue
            elif type == 'UV':
                if x_in < 470 or x_in > 500:
                    continue
                
            

            plt.figure()
            plt.imshow(data, origin='lower')
            plt.plot(x_in, y_in, 'rx')
            plt.savefig('fwhm1.png', bbox_inches='tight')
            # plt.plot(x_in + 100*np.sin(-angle), y_in + 100*np.cos(-angle), 'rx')
            # plt.plot(509.382,	792.608, 'rx')

            if type == 'IR':
                x = np.linspace(-10,60,10000)
            elif type == 'VIS':
                x = np.linspace(-10,90,10000)
            elif type == 'UV':
                x = np.linspace(-10,70,10000)
            x_ind = x_in + x*np.sin(-angle)
            y_ind = y_in + x*np.cos(-angle)

            plt.figure()
            plt.imshow(data, origin='lower')
            plt.plot(x_ind, y_ind, 'gx')
            plt.savefig('fwhm2.png', bbox_inches='tight')

            x_ind = np.array([math.floor(x) for x in x_ind])
            y_ind = np.array([math.floor(y) for y in y_ind])

            y = data[y_ind, x_ind]

            def Gauss(x, A, B, C):
                y = A*np.exp(-(x-B)**2/2*C**2)
                return y
            if type == 'IR':
                parameters, covariance = curve_fit(Gauss, x, y, p0=[65000, 25, 5])
            elif type == 'VIS':
                parameters, covariance = curve_fit(Gauss, x, y, p0=[7000, 40, 20])
            elif type == 'UV':
                parameters, covariance = curve_fit(Gauss, x, y, p0=[65000, 25, 5])


            fit_A = parameters[0]
            fit_B = parameters[1]
            fit_C = parameters[2]
            
            fit_y = Gauss(x, fit_A, fit_B, fit_C)

            plt.figure()
            plt.plot(x,y,label='data')
            plt.plot(x, fit_y, 'r-',label='Gauss fit')
            plt.xlabel('pix')
            plt.ylabel('counts')
            plt.legend()
            plt.savefig('fwhm3.png', bbox_inches='tight')

            spline = UnivariateSpline(x, fit_y-np.max(fit_y)/2, s=0)
            r1, r2 = spline.roots() # find the roots
            
            # if i >= 0:
            #     plt.figure()
            #     plt.plot(x,y)
            #     plt.plot(x, fit_y, 'r-')
            #     plt.axvspan(r1, r2, facecolor='g', alpha=0.5)

            if type == 'IR':
                vals.append((r2-r1) * 0.465) # IR
            elif type == 'VIS':
                vals.append((r2-r1) * 0.27) # VIS
            elif type == 'UV':
                vals.append((r2-r1) * 0.345) # UV
            # print(str(i) + ':  ' + str((r2-r1)/pix_size))
        m = np.mean(vals)
        print(np.max(vals))
        # plt.figure()
        # plt.plot(vals)
        vals_list.append(m)
    print('MEAN: ', np.mean(vals_list))



# 60 pixel, 22.8 arcsec
# 0.38 arcsec/pix











