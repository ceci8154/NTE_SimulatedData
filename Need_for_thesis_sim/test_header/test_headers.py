from astropy.io import fits

# load the fits file
with fits.open('uvb_flat_0000.fits') as file:
    # get the header
    header = file[0].header

    # add "instrument", "night", "bias" to the header
    header['instrument'] = 'NTE'
    header['night'] = '2019-01-01'
    header['bias'] = 'yes'

    # print the header
    print(header)

    # save the header to the file
    file.writeto('uvb_flat_0000.fits', overwrite=True)

# load same file again
with fits.open('uvb_flat_0000.fits') as file:
    # print the header
    print(file[0].header)