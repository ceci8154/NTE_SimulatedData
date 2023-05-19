import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import h5py
# Read in hdf file from ../NTEpyechelle/models/NTE.hdf
location = '../NTEpyechelle/models/NTE.hdf'
with h5py.File(location,'r') as file:
    # Print the keys
    print(list(file.keys()))
    # Get key name
    key = list(file.keys())[2]
    # Make group for that key
    group = file[key]
    # Print the keys in that group
    print(list(group.keys()))
    # Take group with 2nd key in that group
    group2 = group[list(group.keys())[1]]
    # Print the keys in that group
    print(list(group2.keys()))
    plt.figure()
    for i in range(2):
        # Take i key in that group
        data = group2[list(group2.keys())[i]]
        # Get coloumn names of data
        colnames = data.dtype.names
        # Print coloumn names
        data = data[()]
        colnames = list(colnames)
        print(colnames)
        rotation, scale_x, scale_y, shear, translation_x, translation_y, wavelength = 0,0,0,0,0,0,0
        # Put the data into variable names equal to the colnames
        for i in range(len(colnames)):
            exec(colnames[i] + '= data[colnames[i]]')
        plt.plot(translation_x, translation_y)
    
    # Import the fits file
    with fits.open('uvb_flat_0000.fits') as f:
        # Get the data 
        data = f[0].data
        # Plot the data
        plt.imshow(data, origin='lower')
    
    
    for j in range(len(colnames)):
        colname = colnames[j]
        plt.figure()
        for i in range(2):
            # Take i key in that group
            data = group2[list(group2.keys())[i]]
            # Print coloumn names
            data = data[()]
            rotation, scale_x, scale_y, shear, translation_x, translation_y, wavelength = 0,0,0,0,0,0,0
            # Put the data into variable names equal to the colnames
            for i in range(len(colnames)):
                exec(colnames[i] + '= data[colnames[i]]')
            plt.plot(wavelength, data[colname])
            plt.title(colname)

    for j in range(len(colnames)):
        colname = colnames[j]
        plt.figure()
        for i in range(2):
            # Take i key in that group
            data = group2[list(group2.keys())[i]]
            # Print coloumn names
            data = data[()]
            rotation, scale_x, scale_y, shear, translation_x, translation_y, wavelength = 0,0,0,0,0,0,0
            # Put the data into variable names equal to the colnames
            for i in range(len(colnames)):
                exec(colnames[i] + '= data[colnames[i]]')
            plt.plot(wavelength, np.gradient(data[colname]))
            plt.title('gradient of ' + str(colname))
    

    plt.figure()
    for i in range(2):
        # Take i key in that group
        data = group2[list(group2.keys())[i]]
        # Print coloumn names
        data = data[()]
        rotation, scale_x, scale_y, shear, translation_x, translation_y, wavelength = 0,0,0,0,0,0,0
        # Put the data into variable names equal to the colnames
        for i in range(len(colnames)):
            exec(colnames[i] + '= data[colnames[i]]')
        
        # find index where wavelength is closest to 0.4
        index = np.argmin(np.abs(wavelength - 0.4))
        print(np.abs(wavelength - 0.4)[index])
        # Plot translation_x and translation_y at that index as a big circle
        plt.plot(translation_x[index], translation_y[index], 'o', markersize=5)
    
    # Import the fits file
    with fits.open('uvb_flat_0000.fits') as f:
        # Get the data 
        data = f[0].data
        # Plot the data
        plt.imshow(data, origin='lower')
    plt.ylim(650,850)
    
    
    
    
    plt.show()

