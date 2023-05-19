# NTE Simulated Data

Hello there!

Welcome to this repository. This repository contains the code and data for the data simulation for the NTE.

## Termonology
Before we start, I will explain some of the terms used in this repository.

### CCD
CCD here are the different detectors. There are 3 CCDs in the NTE, one for the UV, one for the visible, and one for the IR. 

The UV CCD is CCD 3, the visible CCD is CCD 2, and the IR CCD is CCD 1.

### Schematic
A schematic is the data from the CCD before it has been processed. It is simply the raw flux from the object. There is no bias or noise added to it.

### FITS
FITS stands for Flexible Image Transport System. It is a file format that is used to store data. It is usually used to store astronomical data. It is a very flexible format, and can store a lot of different data types. It can also store a lot of different data in one file.

### HDF
HDF stands for Hierarchical Data Format. It is a file format that is used to store data. It is usually used to store astronomical data. It is a very flexible format, and can store a lot of different data types. It can also store a lot of different data in one file.

### Fiber
Here fiber is the same as slit. The reason that fiber and slit are used interchangeably is because Pyechelle was originally made for a different spectrograph, and they used the term fiber instead of slit.

There are 9 different slits in the NTE, and they are called:

    slits = {
        'fiber_1': 0.5arcsec,
        'fiber_2': 0.8arcsec,
        'fiber_3': 1.0arcsec,
        'fiber_4': 1.2arcsec,
        'fiber_5': 1.5arcsec,
        'fiber_6': 1.7arcsec,
        'fiber_7': 2.0arcsec,
        'fiber_8': 5.0arcsec,
        'fiber_9': pinhole 0.5arcsec diameter
    }

And so if you set fiber=1 then you will get the 0.5 arcsec slit.

## What is in this repository
This repository contains the code and data for the data simulation for the NTE.

### NTEpyechelle
NTEpyechelle is our modified version of Pyechelle. It is used to simulate the spectra of targets. Keep in mind that this is not interchangeable with Pyechelle. It is a modified version of it. 

### NTEsim
NTEsim is a package that we have made to make the simulation of data easier. It is used to make schematics and to add bias and noise to schematics. 

It has two main classes: 

#### MakeFits
This can be called with:

    from NTEsim.make_fits import MakeFits
    MF = MakeFits()

This class is meant to use schematics to generate FITS files that include bias and noise. 

It is also used to add schematics together in case you want to e.g. add a sky spectrum on top of an object spectrum.

##### Packages needed to run MakeFits
To run MakeFits you need to have the following packages installed:

    numpy
    astropy
    pyxel (if cosmic rays are to be added)


#### MakeSchematics
This can be called with:

    from NTEsim.make_schematics import MakeSchematics
    MS = MakeSchematics()

This class is meant to make schematics from a source.

It is much more involved to get working than MakeFits.

You really should run the code on a GPU, as it is very slow to run on a CPU. However, if you want to disable this, then simply write:

    MS.gpu = False

Right after calling the class.

To make the code work on a CPU, you need to have the following packages installed:

    python==3.10.9
    skycalc_ipy==0.1.3
    joblib==1.1.1
    scipy==1.10.0
    numba==0.56.4
    pandas==1.5.3
    synphot==1.1.1
    numpy==1.23.5
    chardet
    h5py==3.7.0

On the CPU, the code should work no matter the package versions, but this is what we tested with.

And for the running it on a GPU you will also need:

    cudatoolkit==11.3.1

Can be installed through anaconda with:

    conda install cudatoolkit==11.3.1

##### Complete guide to make it run on a WSL2 system:

From a totally clean WSL2 system you must first make sure GGC is installed. To install it you can run:

    sudo apt update

    sudo apt install build-essential

Then you must install anaconda on the system. This is easiest done on WSL2 by downloading the linux file from the anaconda website, and then installing it with bash. 

This can be done by going to https://www.anaconda.com/products/distribution, clicking the penguin under the download button, and then downloading the 64-Bit installer for Linux. Then navigate to the download folder through the terminal and write:

    bash name_of_installer.sh

Then restart your terminal.

Then you run the following commands:

    conda create -n NTEsimCUDA python=3.10.9

    conda activate NTEsimCUDA

    conda install pip

    conda install cudatoolkit==11.3.1

    pip3 install skycalc_ipy==0.1.3
    
    conda install joblib==1.1.1
    
    conda install scipy==1.10.0
    
    conda install numba==0.56.4
    
    conda install pandas==1.5.3

    pip3 install synphot==1.1.1

    pip3 install chardet==5.1.0
    
    conda install h5py==3.7.0

Then you have an environment that should work for the simulation with the GPU. Type:

    conda activate NTEsimCUDA

to enter the environment before you run the code.

This is the method that we tested to get a working environment. If you decide to install it in a different way, then make sure that your version of cudatoolkits and numba is compatible.

### Pyechelle
Pyechelle is a spectrograph simulator. It is used to simulate the spectra of targets. It is a python package that can be found here: https://gitlab.com/Stuermer/pyechelle

But we use a modified version of it.

## How to use: simple
If you simply want to get a full set of simulated data based on some presets we have put, then the code is very easy to use.

The package we have made for this is the "NTEsim" package. It has to be in the same folder as NTEpyechelle, data, and schematics.

Then to get a full set of data you can simply make a python script and run the following code:

    from NTEsim.make_fits import MakeFits 

    MF = MakeFits()
    MF.unzip_schematics()
    MF.add_bias_and_noise_for_all_preset()
    MF.delete_schematics()

This will make a full set of simulated data, with detector noise, bias, and efficiency already applied.

The output data will be put in the "Output" folder. If this folder does not exist upon downloading the repository, it simply creates it.

## How to use: more in-depth
For full knowledge of the code, I will ask you to go through the documentation htmls instead. But here I will show you how to use the code to make your own data.

### Firstly to make the schematic of your own data
To make the schematic of your own data, you could make a script as so:
 
    from NTEsim.make_schematics import MakeSchematics
    from NTEpyechelle.sources import CSV

    MS = MakeSchematics()

    source = CSV(filepath="data/scienceframe_data/GRB/extended_GRBspec_2.5_withabs.dat", \
                 name="", list_like=True, wavelength_unit='a', flux_in_photons=False, delimiter=' ')
    MS.make_custom('uv', 5, source, 'custom.fits', point=True)

This will make a schematic of a GRB spectrum with CCD 3 (UV CCD), exposure time of 5 seconds, and the spectrum is taken from the file "data/scienceframe_data/GRB/extended_GRBspec_2.5_withabs.dat".

To change what CCD you want to use, simply change the first argument of the function to either a number between 1 and 3, or the wavelength area of the CCD. These are the options:

    'ir' = 1
    'vis' = 2
    'uv' = 3

Keep in mind, if flux_in_photons=False, then the flux is in ergs/s/cm^2/cm. If flux_in_photons=True, then the flux is in photons/s.

Now keep in mind that this code assumes that you have a NVIDEA GPU to run the code, as it is otherwise very slow to run. However, if you want to disable this, then simply write “MS.gpu = False”. 

It is also possible to change the slit used (in our code it is called fiber because pyechelle was made with a different spectrograph in mind) by writing “MS.fiber = x”. Where x would be an integer which relates to the fiber number of the slit in the HDF file. As as example:

    from NTEsim.make_schematics import MakeSchematics
    from NTEpyechelle.sources import CSV

    MS = MakeSchematics()
    MS.fiber = 2
    MS.overw = True

    source = CSV(filepath="data/scienceframe_data/GRB/extended_GRBspec_2.5_withabs.dat", \
                 name="", list_like=True, wavelength_unit='a', flux_in_photons=False, delimiter=' ')
    MS.make_custom(3, 5, source, 'custom.fits', point=True)

Would then simulate the 0.8arcsecond slit. What fiber number relates to what slit is written at the top of this file.

It is also possible to change the “output_dir”, “overw” (if you want to overwrite existing schematics), h5file (what HDF file is used), and the “efficiency” files used. The efficiency should be a list of Pyechelle CSVEfficiency objects. 

Also keep in mind that if you are simulating a continuum, the resolution of the data has to be quite fine. If you notice weird artifacts in the output then try increasing the resolution of the input with an interpolation.

#### Point source with different seeing and position in the slit
If you want to simulate a point source with different seeing and position in slit, then you can do so like this:

    from NTEsim.make_schematics import MakeSchematics
    from NTEpyechelle.sources import CSV

    MS = MakeSchematics()

    source = CSV(filepath="data/scienceframe_data/GRB/extended_GRBspec_2.5_withabs.dat", \
                 name="", list_like=True, wavelength_unit='a', flux_in_photons=False, delimiter=' ')
    MS.make_custom(3, 5, source, 'custom.fits', point=True, posx=0.5, posy=0.3, seeing=0.8)
Here the positions are based on the size of the slits. The number you give should be between 0 and 1, with 0.5 being the middle of the slit.

### Secondly to add bias and noise to the schematic
So imagine that you made a lot of simulations of a stellar target, and all schematics have "stellar" in the name. Then put them in a zip folder in the schematics folder. Then you could run the following code:

    from NTEsim.make_fits import MakeFits 

    MF = MakeFits()
    MF.unzip_schematics()
    MF.add_bias_and_noise_for_keyword('stellar', nr=1, add_sky=True)
    MF.delete_schematics()

This will unpack all zip files, and here it will create files for all schematics with "stellar" in the name. Then it will add bias and noise to all of them, and then it will delete the schematics. The “add_sky” keywords determines if you want to add a sky spectrum on top, and the sky is automaticcally scaled to the same exposuretime. If you do not wish this scaling of the sky, then use the method "add_schematics_and_add_noise" as explained further down.

The reason for the zip files is due to the large files of the schematics. If you do not want to use zip files, then you can simply put the schematics in the schematics folder and then run the code, without the unzip and delete schematics lines.

Also, it is required for the names of the schematic files to start with uv, vis, or ir, as this is how the code knows what bias and noise to add.

It is also possible to give a list of keywords as shown here:

    from NTEsim.make_fits import MakeFits 

    MF = MakeFits()
    MF.unzip_schematics()
    MF.add_bias_and_noise_for_keyword(['stellar', 'grb25'], nr=1, add_sky=True)
    MF.delete_schematics()

There is also an equivalent function that instead excludes files containing a keyword. Could be used as so:

from NTEsim.make_fits import MakeFits 

    MF = MakeFits()
    MF.unzip_schematics()
    MF.add_bias_and_noise_for_not_keyword(['stellar', 'grb25'], nr=1, add_sky=False)
    MF.delete_schematics()

### Adding schematics together.
If you want to add multiple schematics together on your own, then you can do so like this:

    from NTEsim.make_fits import MakeFits 

    MF = MakeFits()
    files_to_add = ['schematic1.fits', 'schematic2.fits',
                    'schematic3.fits']
    factors = [1, 1, 1]
    output_name = 'combined.fits'
    MF.add_schematics_and_add_noise(files_to_add, factors, output_name)

The 'factors' input will multiply the data from the schematic with the factor. This is mostly in case the files you want to add do not have the same exposure time.

## Simulation with cosmic rays
We have only made an implementation for getting cosmic rays on the output.

This implementation also required an additional package to be installed. Installing Pyxel depends on your so we will reference to their website: https://esa.gitlab.io/pyxel/doc/stable/tutorials/install.html

The version of Pyxel we tested with was 1.7

It is in the 'make_fits' part of the package and all you have to do is write:

    from NTEsim.make_fits import MakeFits 

    MF = MakeFits()
    MF.cosmic_rays = True

And then proceed to make the fits files and explained above.

For making cosmic rays, the filename must be included the exposure time with 'sec' afterwards. Here is an example: 'ir_0.5sec_sky.fits'

Keep in mind that this is not a great implementation. The value of the cosmic ray hits are simply set to the maximum value of the inputted image. There is no real meaning behind these cosmic rays, it is purely for testing purposes.

## Now we will go through some of the scripts that will rarely be used, but are still important to know about if new data from the NTE is added to make better simulations.

### "make_flat.py"
This script is used to make the normalized flat files.

It takes all the flat files in the 'data/flat_ir'/vis/uvb folders and makes a normalized flat file for each of them.

At the current moment, we only have data for the IR detector, so the other detectors are not implemented yet. 

The other detectors are simply taken from a normal distribution around 1 with standard deviation of 0.01.

To include new data, for the VIS and UVB detector, in this there will have to be made some changes to the code. But it should simply be a matter of adding the new data to an appropriate folder and copying the code for the IR detector.

### "make_combined_noise.py"
This script is used to make the combined noise files.

It takes all the noise files in the 'data/noise_ir'/vis/uvb folders and makes a combined noise file for each of them.

In its final form, this code will take in a directory of bias frames and
calculate the median and standard deviation for each pixel. It will then
save these values to a fits file, to be used later.

At the moment, this only does it for the IR detector, as data for the other
detectors is not available. This will be updated when/if we get the data.

To include new data, for the VIS and UVB detector, in this there will have to be made some changes to the code. But it should simply be a matter of adding the new data to an appropriate folder and copying the code for the IR detector.

## New ZEMAX data
If a new set of data is simulated from ZEMAX, then a new hdf model file for pyechelle must be made. 

In the 'HDF_generation' folder, you must put in all ZEMAX files, while the PSFs must be put into the subfolders. The code for reading these files is written in a way where they must be formatted in the same manner, so make sure the ZEMAX files are formatted as the existing ones are.

After replacing the files, run the 'generateHDF.py' file from within that folder, and it will put a new HDF model in NTEpyechelle folder. If you do not wish it to replace the old one, then change the 'outfile' variable in 'generateHDF.py'. 

### Changing what HDF file is used for simulation
To change what HDF file is used for the generation of schematics you can do as follows:

    from NTEsim.make_schematics import MakeSchematics
    import h5py

    MS = MakeSchematics()
    MS.h5file = h5py.File('NTEpyechelle/models/NTE_2.hdf', 'r')

## New efficiency data
When new efficiency data is added, then the files in the 'data/efficiency' folder must be updated. 

If the new files are in the same units as the existing ones, then you should be able to simply replace the existing files with the new ones, and run the 'make_effs.py' file from within the 'data/efficiency' folder.


