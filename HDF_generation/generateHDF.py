'''
Also written by Joonas
'''

import h5py
import numpy as np
from sys import exit
import os
from Process_files import read_file, find_affine, read_psfs

# Copied from stackoverflow: https://stackoverflow.com/questions/7687679/how-to-generate-2d-gaussian-with-python
# def makeGaussian(size, fwhm = 3, center=None):
#     """
#     Make a square gaussian kernel.
#     size is the length of a side of the square
#     fwhm is full-width-half-maximum, which
#     can be thought of as an effective radius.
#     """

#     x = np.arange(0, size, 1, float)
#     y = x[:,np.newaxis]

#     if center is None:
#         x0 = y0 = size // 2
#     else:
#         x0 = center[0]
#         y0 = center[1]

#     return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)


# Following https://docs.h5py.org/en/latest/index.html

# Try to create a new file and complain if a file with same name exists
outdir = '../NTEpyechelle/models/'
outfile_name = 'NTE.hdf'
outfile = outdir + outfile_name

# Datalen is how many points we use for each order
file_dir = 'Zemax_data'
files = os.listdir(file_dir)
keyword = 'nte'
NTE_files = [f for f in files if keyword == f[:len(keyword)]]
order_num, WL_range, order_WL, order_points = read_file(file_dir + '/' + NTE_files[0])
datalen = order_WL.shape[1]

if os.path.isfile(outfile):
    print('File already exists. Do you want to overwrite? (y/n)')
    answer = input()
    if answer == 'n':
        exit()
    elif answer == 'y':
        pass
    else:
        print('Please answer y or n')
        exit()

f = h5py.File(outfile, 'w')

# Detector specifications
detector_1 = {
    'Nx'         : 2048,
    'Ny'         : 2048,
    'pixelsize'  : 15,
    'name'       : 'Hawaii 2RG',
    'order_min'  : 3,
    'order_max'  : 7,
    'offset_x'   : 0,
    'offset_y'   : 0
}

detector_2 = {
    'Nx'         : 4096,
    'Ny'         : 1024,
    'pixelsize'  : 10.5,
    'name'       : 'Skipper CCD',
    'order_min'  : 8,
    'order_max'  : 15,
    'offset_x'   : 0,
    'offset_y'   : 4.0
}

detector_3 = {
    'Nx'         : 1024,
    'Ny'         : 1024,
    'pixelsize'  : 13,
    'name'       : 'EMCCD',
    'order_min'  : 16,
    'order_max'  : 21,
    'offset_x'   : 0,
    'offset_y'   : 1.5
}

# Spectrograph grating specifications
spec = {
    'blaze': 17.5,
    'gpmm' : 91.2,
    'name' : 'nte_spec_220929_final'
}

# Matrices and sampling the number of point per order?
# Field info important or no? Field width typoed 'field_with in MaroonX.hdf
# All input here is guesses, norm_field binary?
slit_1_param = {
    'MatricesPerOrder': datalen,
    'sampling_input_x': datalen,
    'field_height'    : 0,
    'field_width'     : 0,
    'field_shape'     : 'rectangular',
    'norm_field'      : 0,
    'name'            : '0.5 arcsec slit',
    'slit_width'      : 0.5,
    'slit_height'     : 22.8
}

slit_2_param = {
    'MatricesPerOrder': datalen,
    'sampling_input_x': datalen,
    'field_height'    : 0,
    'field_width'     : 0,
    'field_shape'     : 'rectangular',
    'norm_field'      : 0,
    'name'            : '0.8 arcsec slit',
    'slit_width'      : 0.8,
    'slit_height'     : 22.8
}

slit_3_param = {
    'MatricesPerOrder': datalen,
    'sampling_input_x': datalen,
    'field_height'    : 0,
    'field_width'     : 0,
    'field_shape'     : 'rectangular',
    'norm_field'      : 0,
    'name'            : '1.0 arcsec slit',
    'slit_width'      : 1.0,
    'slit_height'     : 22.8
}

slit_4_param = {
    'MatricesPerOrder': datalen,
    'sampling_input_x': datalen,
    'field_height'    : 0,
    'field_width'     : 0,
    'field_shape'     : 'rectangular',
    'norm_field'      : 0,
    'name'            : '1.2 arcsec slit',
    'slit_width'      : 1.2,
    'slit_height'     : 22.8
}

slit_5_param = {
    'MatricesPerOrder': datalen,
    'sampling_input_x': datalen,
    'field_height'    : 0,
    'field_width'     : 0,
    'field_shape'     : 'rectangular',
    'norm_field'      : 0,
    'name'            : '1,5 arcsec slit',
    'slit_width'      : 1.5,
    'slit_height'     : 22.8
}

slit_6_param = {
    'MatricesPerOrder': datalen,
    'sampling_input_x': datalen,
    'field_height'    : 0,
    'field_width'     : 0,
    'field_shape'     : 'rectangular',
    'norm_field'      : 0,
    'name'            : '1.7 arcsec slit',
    'slit_width'      : 1.7,
    'slit_height'     : 22.8
}

slit_7_param = {
    'MatricesPerOrder': datalen,
    'sampling_input_x': datalen,
    'field_height'    : 0,
    'field_width'     : 0,
    'field_shape'     : 'rectangular',
    'norm_field'      : 0,
    'name'            : '2.0 arcsec slit',
    'slit_width'      : 2.0,
    'slit_height'     : 22.8
}

slit_8_param = {
    'MatricesPerOrder': datalen,
    'sampling_input_x': datalen,
    'field_height'    : 0,
    'field_width'     : 0,
    'field_shape'     : 'rectangular',
    'norm_field'      : 0,
    'name'            : '5.0 arcsec slit',
    'slit_width'      : 5.0,
    'slit_height'     : 22.8
}

slit_9_param = {
    'MatricesPerOrder': datalen,
    'sampling_input_x': datalen,
    'field_height'    : 0,
    'field_width'     : 0,
    'field_shape'     : 'circular',
    'norm_field'      : 0,
    'name'            : 'pinhole',
    'slit_diameter'   : 0.5
}

order = {
    'FIELD_0_NAME' : 'rotation',
    'FIELD_1_NAME' : 'scale_x',
    'FIELD_2_NAME' : 'scale_y',
    'FIELD_3_NAME' : 'shear',
    'FIELD_4_NAME' : 'translation_x',
    'FIELD_5_NAME' : 'translation_y',
    'FIELD_6_NAME' : 'wavelength',
    'FIELD_0_FILL' : 0.0,
    'FIELD_1_FILL' : 0.0,
    'FIELD_2_FILL' : 0.0,
    'FIELD_3_FILL' : 0.0,
    'FIELD_4_FILL' : 0.0,
    'FIELD_5_FILL' : 0.0,
    'FIELD_6_FILL' : 0.0,
    'NROWS'        : datalen,
    'order'        : 0
}

psf = {}


# Create groups following the organization and naming of MaroonX file
ccd1 = f.create_group('CCD_1')
ccd2 = f.create_group('CCD_2')
ccd3 = f.create_group('CCD_3')

spectrograph1 = ccd1.create_group('Spectrograph')
spectrograph2 = ccd2.create_group('Spectrograph')
spectrograph3 = ccd3.create_group('Spectrograph')

# Assign attributes to groups
for item in detector_1:
    ccd1.attrs.create( item, detector_1[item] )

for item in detector_2:
    ccd2.attrs.create( item, detector_2[item] )

for item in detector_3:
    ccd3.attrs.create( item, detector_3[item] )

for item in spec:
    spectrograph1.attrs.create( item, spec[item] )
    spectrograph2.attrs.create( item, spec[item] )
    spectrograph3.attrs.create( item, spec[item] )


detectors = [ccd1, ccd2, ccd3]
aperturenames = ['fiber_1', 'fiber_2', 'fiber_3', 'fiber_4', 'fiber_5', 'fiber_6', 'fiber_7', 'fiber_8', 'fiber_9']
slitnames = ['0.5arcsec', '0.8arcsec', '1.0arcsec', '1.2arcsec', '1.5arcsec', '1.7arcsec', '2.0arcsec', '5.0arcsec', 'pinhole']
detector_area = ['NIR', 'VIS', 'UV']

slit_nr_param = [slit_1_param, slit_2_param, slit_3_param, slit_4_param, slit_5_param, slit_6_param, slit_7_param, slit_8_param, slit_9_param]

# Create groups for each and every slit
for detector in detectors:
    for slitno in range(1,9+1):
        # Name as fibers for now since that is only thing pyechelle understands
        slitname = 'fiber_' + str(slitno)
        slit = detector.create_group(slitname)

        for item in slit_nr_param[slitno-1]:
            slit.attrs.create( item, slit_nr_param[slitno-1][item] )


# Create a datatype to store the table data
order_type = np.dtype({'names':['rotation', 'scale_x', 'scale_y', 'shear', 'translation_x', 'translation_y', 'wavelength'], 'formats': [('<f4'), ('<f4'), ('<f4'), ('<f4'), ('<f4'), ('<f4'), ('<f4')]})

# Loop over a list of order definition files
for i, detector in enumerate(detectors):
    # Fill the table with the data
    px_per_mm = 1e3 / detector.attrs['pixelsize']
    for j, aperturename in enumerate(aperturenames):
        detector_type = detector_area[i]
        slit = slitnames[j]
        filename = file_dir + '/' + 'nte_spec_'+detector_type+'_'+slit+'.txt'
        order_num, WL_range, order_WL, order_points = read_file(filename)
        for order in range(
            int(detector.attrs['order_min']),
            int(detector.attrs['order_max']+1)
            ):
            points_for_order = order_points[order_num==order][0]
            # Now do get the affine transformations
            scale_x = np.zeros(50)
            scale_y = np.zeros(50)
            shear = np.zeros(50)
            rotation = np.zeros(50)
            translation_x = np.zeros(50)
            translation_y = np.zeros(50)
            for k, point in enumerate(points_for_order):
                tx, ty, theta, s, sx, sy = find_affine(point.reshape(5,2))
                point = point.reshape(5,2)

                # Pyechelle for some reason sets the translation to be the bottom left corner. So we need to
                # get half of the width and height to make sure the center is in the center.
                p1 = point[2]
                p2 = point[4]
                diffs = (p2 - p1)/2
                # Then also the x diffs from the other 2 points for the thickness
                p1 = point[1]
                p2 = point[3]
                x_diffs = (p2[0] - p1[0])/2

                # And append
                scale_x[k] = sx
                scale_y[k] = sy
                shear[k] = s
                rotation[k] = theta
                translation_x[k] = tx + diffs[0] + x_diffs
                translation_y[k] = ty + diffs[1]

                # Pyechelle actually needs the negative of the angle we use in the find_affine function, 
                # so we multiply with negative 1.
                rotation[k] *= -1

            wavelength = order_WL[order_num==order][0]
            WL = order_WL[order_num==order][0]

            # Prepare empty table, with the name of order_XX as Pyechelle wants
            ordername = 'order' + str(order) 
            order_dataset = detector[aperturename].create_dataset(ordername, (datalen,), dtype=order_type)

            order_dataset['rotation'] = rotation
            order_dataset['scale_x'] = scale_x * px_per_mm
            order_dataset['scale_y'] = scale_y * px_per_mm
            order_dataset['shear'] = shear
            order_dataset['translation_x'] = (-translation_x + detector.attrs['offset_x']) * px_per_mm + detector.attrs['Nx']/2
            order_dataset['translation_y'] = (-translation_y + detector.attrs['offset_y']) * px_per_mm + detector.attrs['Ny']/2
            order_dataset['wavelength'] = wavelength


# Loop over wavelenght
for i, detector in enumerate(detectors):
    if i+1 == 1:
        CCD_name = 'NIR'
        input_dir = 'Zemax_data/NTE_spec_'+CCD_name+'_psfs'
        psfs, wls, orders = read_psfs(input_dir)
        for aperturename in aperturenames:
            for order in range( int(detector.attrs['order_min']),
                                int(detector.attrs['order_max'])+1):
                psfname = 'psf_order_' + str(int(order))

                wls_temp = wls[orders==order]

                psf = detector[aperturename].create_group( psfname )

                for wl in wls_temp:
                    
                    psfdata = psf.create_dataset(
                        'wavelength_'+str(wl), (64,64), dtype=np.float64 )
                    psfdata.attrs.create('wavelength', wl)
                    psfdata.attrs.create('order', order)
                    psfdata.attrs.create('dataSpacing', 1) # This probably means that element in PSF array has physical size of 1um
                    #gauss_psf = makeGaussian( 64, fwhm=2.5*float(detector.attrs['pixelsize']) )
                    # find psf for order and wl
                    zemax_psf = psfs[orders==order][wls_temp==wl][0]
                    psfdata.write_direct( zemax_psf )
    elif i+1 == 2:
        CCD_name = 'VIS'
        input_dir = 'Zemax_data/NTE_spec_'+CCD_name+'_psfs'
        psfs, wls, orders = read_psfs(input_dir)
        for aperturename in aperturenames:
            for order in range( int(detector.attrs['order_min']),
                                int(detector.attrs['order_max'])+1):
                psfname = 'psf_order_' + str(int(order))

                wls_temp = wls[orders==order]

                psf = detector[aperturename].create_group( psfname )

                for wl in wls_temp:
                    
                    psfdata = psf.create_dataset(
                        'wavelength_'+str(wl), (64,64), dtype=np.float64 )
                    psfdata.attrs.create('wavelength', wl)
                    psfdata.attrs.create('order', order)
                    psfdata.attrs.create('dataSpacing', 1) # This probably means that element in PSF array has physical size of 1um
                    #gauss_psf = makeGaussian( 64, fwhm=2.5*float(detector.attrs['pixelsize']) )
                    # find psf for order and wl
                    zemax_psf = psfs[orders==order][wls_temp==wl][0]
                    psfdata.write_direct( zemax_psf )
    elif i+1 == 3:
        CCD_name = 'UVB'
        input_dir = 'Zemax_data/NTE_spec_'+CCD_name+'_psfs'
        psfs, wls, orders = read_psfs(input_dir)
        for aperturename in aperturenames:
            for order in range( int(detector.attrs['order_min']),
                                int(detector.attrs['order_max'])+1):
                psfname = 'psf_order_' + str(int(order))

                wls_temp = wls[orders==order]

                psf = detector[aperturename].create_group( psfname )

                for wl in wls_temp:
                    psfdata = psf.create_dataset(
                        'wavelength_'+str(wl), (64,64), dtype=np.float64 )
                    psfdata.attrs.create('wavelength', wl)
                    psfdata.attrs.create('order', order)
                    psfdata.attrs.create('dataSpacing', 1) # This probably means that element in PSF array has physical size of 1um
                    psfdata.write_direct( zemax_psf )
    

f.close(); exit#()
