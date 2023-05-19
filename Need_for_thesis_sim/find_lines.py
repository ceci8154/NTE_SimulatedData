import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import h5py
# Read in hdf file from ../NTEpyechelle/models/NTE.hdf
location = 'NTEpyechelle/models/NTE.hdf'
with h5py.File(location,'r') as file:
    # Print the keys
    print(list(file.keys()))
    # Get key name
    key = list(file.keys())[1]
    print(key)
    # Make group for that key
    group = file[key]
    # Print the keys in that group
    print(list(group.keys()))
    # Take group with 2nd key in that group
    group2 = group[list(group.keys())[3]]
    # Print the keys in that group
    print(list(group2.keys()))
    wls_to_check = [0.366,0.405,0.6967351,0.8523782]#[0.8523782]#[0.366,0.405,0.6967351,0.8523782] #[0.366,0.405,0.694,0.853]
    plt.figure()
    for i in range(len(list(group2.keys()))//2):
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
        #plt.plot(translation_x, translation_y)
        for wl in wls_to_check:
            wl_temp = 999999999999
            wl_temp2 = 999999999999
            for i,l in enumerate(wavelength):
                if i != 0 & i != len(wavelength)-1:
                    if abs(l - wl) < abs(wl_temp - wl):
                        wl_temp = l
                        x_temp = translation_x[i]
                        y_temp = translation_y[i]
                        if wl_temp < wl:
                            if len(wavelength) > i+1:
                                wl_temp2 = wavelength[i+1]
                                x_temp2 = translation_x[i+1]
                                y_temp2 = translation_y[i+1]
                        else:# wl_temp > wl:
                            wl_temp2 = wavelength[i-1]
                            x_temp2 = translation_x[i-1]
                            y_temp2 = translation_y[i-1]
                    # plt.plot(x_temp,y_temp,'.',color='g') # TODO confused why the lines stop
                    # if x_temp2 > x_temp:
                    #     plt.plot(x_temp2,y_temp2,'.',color='r')
                    # else:
                    #     plt.plot(x_temp2,y_temp2,'.',color='y')
                    
                    # distance between wl and closest
            if 'x_temp2' in locals():
                d1 = abs(wl_temp - wl)
                if d1 < 0.01:
                    # distance between wl and next closest
                    d2 = abs(wl_temp2 - wl)
                    # distance between closest and next closest
                    d3 = abs(wl_temp2 - wl_temp)/d1
                    if d3 != 0:
                        #print(d3)
                        # Get direction from cloest to next closest
                        if x_temp2 > x_temp:
                            direction = 1
                        else:
                            direction = -1
                        Dx = (x_temp2 - x_temp)
                        Dy = (y_temp2 - y_temp)
                        dx = Dx/d3
                        dy = Dy/d3
                        point_x = x_temp + dx
                        point_y = y_temp + dy
                        print(point_x, point_y)
                        plt.plot(point_x,point_y+38,'x',color='r')

                        #plt.plot(x_temp,y_temp,'.',color='y')
                        #plt.plot(x_temp2,y_temp2,'.',color='g')
                        # Estimate the point for wl by using the two closest points
                        #x_est = x_temp + direction * d1/d3 * (x_temp2 - x_temp)
                        #y_est = y_temp + direction * d1/d3 * (y_temp2 - y_temp)
                        #plt.plot(x_est,y_est,'.',color='g')
    
    # Import the fits file
    with fits.open('Output_testing/jump_test.fits') as f:
        # Get the data 
        data = f[0].data +10
        # Plot the data on a log scale
        plt.imshow(np.log(data), origin='lower')
        plt.savefig('jump_test_0.8523782.png')
    
    
    # for j in range(len(colnames)):
    #     colname = colnames[j]
    #     plt.figure()
    #     for i in range(2):
    #         # Take i key in that group
    #         data = group2[list(group2.keys())[i]]
    #         # Print coloumn names
    #         data = data[()]
    #         rotation, scale_x, scale_y, shear, translation_x, translation_y, wavelength = 0,0,0,0,0,0,0
    #         # Put the data into variable names equal to the colnames
    #         for i in range(len(colnames)):
    #             exec(colnames[i] + '= data[colnames[i]]')
    #         plt.plot(wavelength, data[colname])
    #         plt.title(colname)

    # for j in range(len(colnames)):
    #     colname = colnames[j]
    #     plt.figure()
    #     for i in range(2):
    #         # Take i key in that group
    #         data = group2[list(group2.keys())[i]]
    #         # Print coloumn names
    #         data = data[()]
    #         rotation, scale_x, scale_y, shear, translation_x, translation_y, wavelength = 0,0,0,0,0,0,0
    #         # Put the data into variable names equal to the colnames
    #         for i in range(len(colnames)):
    #             exec(colnames[i] + '= data[colnames[i]]')
    #         plt.plot(wavelength, np.gradient(data[colname]))
    #         plt.title('gradient of ' + str(colname))
    

    # plt.figure()
    # for i in range(2):
    #     # Take i key in that group
    #     data = group2[list(group2.keys())[i]]
    #     # Print coloumn names
    #     data = data[()]
    #     rotation, scale_x, scale_y, shear, translation_x, translation_y, wavelength = 0,0,0,0,0,0,0
    #     # Put the data into variable names equal to the colnames
    #     for i in range(len(colnames)):
    #         exec(colnames[i] + '= data[colnames[i]]')
        
    #     # find index where wavelength is closest to 0.4
    #     index = np.argmin(np.abs(wavelength - 0.4))
    #     print(np.abs(wavelength - 0.4)[index])
    #     # Plot translation_x and translation_y at that index as a big circle
    #     plt.plot(translation_x[index], translation_y[index], 'o', markersize=5)
    
    # # Import the fits file
    # with fits.open('Output/jump_test.fits') as f:
    #     # Get the data 
    #     data = f[0].data
    #     # Plot the data
    #     plt.imshow(data, origin='lower')
    # plt.ylim(650,850)
    
    
    
    
    plt.show()








find_relation = True
if find_relation == True:
    # find relation between counts and intensity

    from astropy.io import fits
    import matplotlib.pyplot as plt
    import numpy as np

    #file = '/Users/ceciliehenneberg/Downloads/vis_HgAr_schem.fits'
    #file = '/Users/ceciliehenneberg/Dropbox/Cecilies Dropbox/KU/Speciale/nte-simulated-data/Output/vis_HgAr_schem.fits'
    file = '/Users/ceciliehenneberg/Dropbox/Cecilies Dropbox/KU/Speciale/nte-simulated-data/Output/uvb_HgAr_schem_both_nyny/vis_HgAr_schem_both.fits'
    data = fits.getdata(file)
    # print(data[721-1,2931-1])


    # Low
    # 759 2917
    # 759 2947
    # 657 2947
    # 657 2917

    selection = data[657-1:759-1,2917-1:2947-1]
    plt.figure()
    plt.imshow(selection, origin='lower')
    plt.figure()
    plt.imshow(np.log(data+0.001), origin='lower')
    plt.plot(2917,657,'x',color='r')
    s = np.sum(selection)
    print(s)
    print((1.2*10**8)/(s/60))
    s1 = s

    # wl 696



    # MID
    # 739 1391
    # 739 1351
    # 846 1391
    # 846 1351

    selection = data[739-1:846-1,1351-1:1391-1]
    plt.figure()
    plt.imshow(selection, origin='lower')
    plt.figure()
    plt.imshow(np.log(data+0.001), origin='lower')
    plt.plot(1351,739,'x',color='r')
    s = np.sum(selection)
    print(s)
    print((1.2*10**8)/(s/60))
    s2 = s

    # wl 696

    s3 = s1+s2
    print('sum: ', (1.2*10**8)/(s3/60))

    # HI
    # 950 2582
    # 950 2623
    # 832 2623
    # 832 2582

    selection = data[832-1:950-1,2582-1:2623-1]
    plt.figure()
    plt.imshow(selection, origin='lower')
    plt.figure()
    plt.imshow(np.log(data+0.001), origin='lower')
    plt.plot(2602,892,'x',color='r')
    s = np.sum(selection)
    print(s)
    print('this',(6.5*10**8)/(s))

    # wl 852




#print((6.5*10**8) / (150000))






find_relation = True
if find_relation == True:
    # find relation between counts and intensity

    from astropy.io import fits
    import matplotlib.pyplot as plt
    import numpy as np

    file = '/Users/ceciliehenneberg/Dropbox/Cecilies Dropbox/KU/Speciale/nte-simulated-data/Output/uvb_HgAr_schem_both_nyny/uvb_HgAr_schem_both.fits'
    #file = '/Users/ceciliehenneberg/Dropbox/Cecilies Dropbox/KU/Speciale/nte-simulated-data/schematics/all_schem/uvb_HgAr_schem.fits'
    data = fits.getdata(file)
    # print(data[721-1,2931-1])


    # Low
    # 759 2917
    # 759 2947
    # 657 2947
    # 657 2917

    selection = data[771-1:856-1,327-1:446-1,]
    plt.figure()
    plt.imshow(selection, origin='lower')
    plt.figure()
    plt.imshow(np.log(data+0.001), origin='lower')
    plt.plot(370.08,	785.105,'x',color='r')
    
    s = np.sum(selection)
    print(s)
    print((1.2*10**8)/(s))
    s1 = s

    # 405


    selection = data[534-1:608-1,511-1:534-1]
    plt.figure()
    plt.imshow(selection, origin='lower')
    plt.figure()
    plt.imshow(np.log(data+0.001), origin='lower')
    plt.plot(533.618,	542.513,'x',color='r')
    
    s = np.sum(selection)
    print(s)
    print((1.2*10**8)/(s))
    s1 = s

    # 366







