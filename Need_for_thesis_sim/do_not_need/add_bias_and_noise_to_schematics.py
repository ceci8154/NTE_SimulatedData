'''
Written by Cecilie and Mads
This scribt is used to produce the final simulated data.
It reads in schematics from pyechelle.
    The schematics are from the simulation and includes effeciency.
Then we take the combined bias and noise files and add them to the schematic files.

As the noise files simply contain the median and standard deviation for each pixel,
we add the noise by drawing a random number from a normal distribution with the
median and standard deviation as mean and standard deviation.

Due to the huge file sizes, the schematics are stores in a zip file.
Therefore the code first unzips the files, then adds the noise and bias,
and then deletes all the unzipped files.
'''

import numpy as np
import os
from astropy.io import fits
import zipfile

schematic_dir = 'schematics/'
output_dir = 'Output/'

# Get all files with .zip extension in schematic_dir
zip_files_names = [f for f in os.listdir(schematic_dir) if f.endswith('.zip')]

# Unzip all schematic zip files
for zip_file_name in zip_files_names:
    with zipfile.ZipFile(schematic_dir + zip_file_name, 'r') as zip_ref:
        zip_ref.extractall(schematic_dir)

# Get all files with .fits extension in schematic_dir
fits_files_names = [f for f in os.listdir(schematic_dir) if f.endswith('.fits')]
ir_names = [f for f in fits_files_names if 'nir' in f]
vis_names = [f for f in fits_files_names if 'vis' in f]
uvb_names = [f for f in fits_files_names if 'uvb' in f]

# Define how many fits images to generate for each schematic
name_list = ['bias', 'flat', 'HgAr', 'etalon', 'trace', 'skylines', 'grb25']
bias_nr = 10
flat_nr = 5
hgar_nr = 5
etalon_nr = 1
trace_nr = 5
skylines_nr = 1
grb25_nr = 1
nr_list = [bias_nr, flat_nr, hgar_nr, etalon_nr, trace_nr, skylines_nr, grb25_nr]

# Define functions for adding bias and noise to fits images for each arm
def bias_noise_ir(file_names):
    name_list_ir = name_list[0:7]
    name_list_ir[0] = 'dark'
    # First load the bias and std file
    dir = 'data/noise_ir'
    with fits.open(dir + '/comb_noise.fits') as file:
        bias = file[0].data # TODO Problem with this file cause no bias
        std = np.array(file[1].data, dtype=np.float32)

    # Load in the normalized flat
    dir = 'data/flat_ir'
    flat = fits.getdata(dir + '/norm_flat.fits')

    for i, name in enumerate(name_list_ir):
        print('ir ' + name)
        # Get the schematic files with name in file_names
        schematic_files = [f for f in file_names if name in f]
        for file in schematic_files:
            with fits.open(schematic_dir + file) as hdul:
                for j in range(nr_list[i]):
                    data = hdul[0].data
                    data = data.astype(np.float32)
                    # Add bias and noise
                    if 'grb25' in name:
                        skylines_fits = fits.open('schematics/nir_skylines_schem.fits')
                        skylines_data = skylines_fits[0].data
                        skylines_data = skylines_data.astype(np.float32)*2 /1 *1 /30
                        data = data + skylines_data
                    data = data * flat + bias + 1000 # TODO bias stuff?
                    data = np.random.normal(data, std)
                    data = np.clip(data, 0, None)
                    # Save the fits file
                    if 'grb25' in name:
                        posy = file[-9:-5]
                        if ('A' in posy) or ('B' in posy):
                            posy = file[-18:-5]
                        fits.writeto(output_dir + 'ir_' + name + '_' + posy + '_' + str(j).zfill(2) + '.fits', data, overwrite=True)
                    else:
                        fits.writeto(output_dir + 'ir_' + name + '_' + str(j).zfill(2) + '.fits', data, overwrite=True)

def bias_noise_vis(file_names):
    # First load the bias and std file
    dir = 'data/noise_vis'
    with fits.open(dir + '/comb_noise.fits') as file:
        bias = file[0].data
        std = np.array(file[1].data, dtype=np.float32)

    # Load in the normalized flat
    dir = 'data/flat_vis'
    flat = fits.getdata(dir + '/norm_flat.fits')

    for i, name in enumerate(name_list):
        print('vis ' + name)
        # Get the schematic files with name in file_names
        schematic_files = [f for f in file_names if name in f]
        for file in schematic_files:
            with fits.open(schematic_dir + file) as hdul:
                for j in range(nr_list[i]):
                    data = hdul[0].data
                    data = data.astype(np.float32)
                    if 'grb25' in name:
                        skylines_fits = fits.open('schematics/vis_skylines_schem.fits')
                        skylines_data = skylines_fits[0].data
                        skylines_data = skylines_data.astype(np.float32)*2 /6 *3 /30 *10
                        data = data + skylines_data
                    # Add bias and noise
                    data = data * flat + bias + 1000 # TODO bias stuff?
                    data = np.random.normal(data, std)
                    data = np.clip(data, 0, None)
                    # Save the fits file
                    if 'grb25' in name:
                        posy = file[-9:-5]
                        if ('A' in posy) or ('B' in posy):
                            posy = file[-18:-5]
                        fits.writeto(output_dir + 'vis_' + name + '_' + posy + '_' + str(j).zfill(2) + '.fits', data, overwrite=True)
                    else:
                        fits.writeto(output_dir + 'vis_' + name + '_' + str(j).zfill(2) + '.fits', data, overwrite=True)

def bias_noise_uvb(file_names):
    # First load the bias and std file
    dir = 'data/noise_uvb'
    with fits.open(dir + '/comb_noise.fits') as file:
        bias = file[0].data
        std = np.array(file[1].data, dtype=np.float32)

    # Load in the normalized flat
    dir = 'data/flat_uvb'
    flat = fits.getdata(dir + '/norm_flat.fits')

    for i, name in enumerate(name_list):
        print('uvb ' + name)
        # Get the schematic files with name in file_names
        schematic_files = [f for f in file_names if name in f]
        for file in schematic_files:
            with fits.open(schematic_dir + file) as hdul:
                for j in range(nr_list[i]):
                    data = hdul[0].data
                    data = data.astype(np.float32)
                    if 'grb25' in name:
                        skylines_fits = fits.open('schematics/uvb_skylines_schem.fits')
                        skylines_data = skylines_fits[0].data
                        skylines_data = skylines_data.astype(np.float32)*2 /466 *5 /30 *20
                        data = data + skylines_data
                    # Add bias and noise
                    data = data * flat + bias + 1000 # TODO bias stuff?
                    data = np.random.normal(data, std)
                    data = np.clip(data, 0, None)
                    # Save the fits file
                    if 'grb25' in name:
                        posy = file[-9:-5]
                        if ('A' in posy) or ('B' in posy):
                            posy = file[-18:-5]
                        fits.writeto(output_dir + 'uvb_' + name + '_' + posy + '_' + str(j).zfill(2) + '.fits', data, overwrite=True)
                    else:
                        fits.writeto(output_dir + 'uvb_' + name + '_' + str(j).zfill(2) + '.fits', data, overwrite=True)

bias_noise_ir(ir_names)
bias_noise_vis(vis_names)
bias_noise_uvb(uvb_names)

# Delete all schematic fits files
for fits_file_name in fits_files_names:
    os.remove(schematic_dir + fits_file_name)