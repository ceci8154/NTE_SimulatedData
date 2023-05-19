import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import os
from NTEsim.make_fits import MakeFits

# Get all file names in folder
fold = 'schematics/shorts'
file_names = os.listdir(fold)

def remove_duplicates(filenames, arm):
    types = ['bias', 'dark', 'flat', 'sky', 'grb25', 'HgAr', 'etalon', 'trace', 'bd']

    # First only get images for one arm
    filenames_arm = []
    for filename in filenames:
        if arm in filename:
            filenames_arm.append(filename)
    
    # Now remove multiple instances of same type
    files = []
    for type in types:
        filenames_one_each = []
        for filename in filenames_arm:
            if type in filename:
                filenames_one_each.append(filename)
        
        if len(filenames_one_each) > 0:
            fn = filenames_one_each[0]
        else:
            fn = 0
        
        if fn != 0:
            files.append(fn)
    
    return files

files_ir = remove_duplicates(file_names, 'ir')
files_vis = remove_duplicates(file_names, 'vis')
files_uv = remove_duplicates(file_names, 'uv')

def get_data_and_plot(files, dir):
    for file in files:
        print(file)
        data = fits.getdata(dir+file)
        print(np.max(data))
        # plot of logscale
        plt.figure()
        #plt.tight_layout()
        #plt.show()
        #data = data - np.min(data)
        if 'schem' in file:
            data = data + 1000
        cmin = 1000
        cmax = 1100
        limi = data.shape[0]*data.shape[1]*0.01
        # get how many pixels are above cmax
        while np.sum(data > cmax) > limi:
            cmax += 100
        print(np.max(data))
        data = data.clip(cmin, cmax) #TODO find out how to do this
        print(np.max(data))
        data = data - cmin
        print(np.max(data))
        colors = ['#ebe8e8', '#000000']
        #cmap = plt.cm.colors.ListedColormap(colors)
        cmap = plt.cm.colors.LinearSegmentedColormap.from_list("", colors)
        if np.min(data) < 1:
            dat = np.log(data+1)
            ma = np.max(dat)
            dat = np.pad(dat, pad_width=6, mode='constant', constant_values=ma)
            plt.imshow(dat, origin='lower', cmap=cmap)#'gray_r')#cmap)
        else:
            dat = np.log(data)
            ma = np.max(dat)
            dat = np.pad(dat, pad_width=6, mode='constant', constant_values=ma)
            plt.imshow(dat, origin='lower', cmap=cmap)#'gray_r')#cmap)
        plt.axis('off')
        print(np.max(data))
        print('---')
        name = file[:-5]
        if 'schem' not in file:
            name += '_noise'
        plt.savefig(name+'.png', bbox_inches='tight', pad_inches=0.1, dpi=500)#

get_data_and_plot(files_ir, fold+'/')

MF = MakeFits()
MF.unzip_schematics()
MF.add_bias_and_noise_for_all_preset()
#MF.add_bias_and_noise_for_keyword('bd', nr=1, add_sky=True)
MF.delete_schematics()

# Get all file names in Output
file_names = os.listdir('Output')
# Check if they are .fits files
file_names = [file for file in file_names if '.fits' in file]

files_ir = remove_duplicates(file_names, 'ir')
files_vis = remove_duplicates(file_names, 'vis')
files_uv = remove_duplicates(file_names, 'uv')

files_ir = np.append(files_ir, 'ir_slitwidth1.0_10sec_standardstarbd17d4708_0.fits')

get_data_and_plot(files_ir, 'Output/')
get_data_and_plot(files_vis, 'Output/')
get_data_and_plot(files_uv, 'Output/')

# plt.show()