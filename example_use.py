from NTEsim.make_schematics import MakeSchematics
from NTEsim.make_fits import MakeFits
from NTEpyechelle.sources import CSV
import h5py

'''
First an example to use the premade schematics to generate new simulated data.

This example would unzip the schematics insise the schematics folder,
add bias and noise to all the preset data,
and then delete the schematics.

All the output fits files will be saved in the 'Output' folder.
'''

MF = MakeFits()
MF.unzip_schematics()
MF.add_bias_and_noise_for_all_preset()
MF.delete_schematics()


'''
This example will generate a full set of simulated data
based on the presets we have used.

This would be usefull if new efficiency data is added, new Zemax 
data is included, or if you wnat the data for different slits.

All the output fits files will be saved in the 'Output' folder.

Keep in mind you need a NVIDIA GPU to run this.
'''

MS = MakeSchematics()
MS.make_all(with_grb25=True)


'''
This example would be how you could generate your own custom schematics.

This example would generate a custom spectrum for the UV detector,
with a exposuretime of 5 seconds.

Keep in mind here that if flux_in_photons is set to False,
then the fluxes should be ergs/s/cm^2/cm.
'''

MS = MakeSchematics()
source = CSV(filepath="data/scienceframe_data/GRB/extended_GRBspec_2.5_withabs.dat", \
                name="", list_like=True, wavelength_unit='a', flux_in_photons=False, delimiter=' ')
MS.make_custom(3, 5, source, 'uv_testing_custom.fits')


'''
This small example will be how to change the slit.

In our code, we use fiber to refer to the slit, because 
the pyechelle code uses fiber.
'''

MS = MakeSchematics()
MS.fiber = 2


'''
If you, for example, have made a lot of schematics, and you want to add
bias and noise to them, you can use the MakeFits class.

Say if all your new schematics have 'stellar' in the name, then you can
use the following code to add bias and noise to them.

But keep in mind that add_sky=True will add the sky to the data,
but there will have to also be sky files of the same slit and ccd,
in the schematics folder.
'''

MF = MakeFits()
MF.add_bias_and_noise_for_keyword('stellar', nr=1, add_sky=True)

'''
There is another function that also takes a keyword, 
but in this case it would make a fits for all files that 
do not contain the keyword. This can be used together with the other one. 

Imagine if you have a GRB spectrum where you want to add the sky, 
but you do not wish to add the sky to the other files. 
Then you could do something like this:
'''

MF = MakeFits()
MF.add_bias_and_noise_for_keyword('grb25', nr=1, add_sky=True)
MF.add_bias_and_noise_for_not_keyword('grb25', nr=1)


'''
Another way to add files together in case this is too premade
and doesn't fit your needs, is to use the add_schems_and_add_noise
function.

You will make a list of schematics you want to add together,
make a list of scales that will be applied to the files
(in case they have different exposuretimes),
and then you can add bias and noise to the data.

The output will be saved in the 'Output' folder.
'''

MF = MakeFits()
files_to_add = ['schematic1.fits', 'schematic2.fits',
                'schematic3.fits']
factors = [1, 1, 1]
output_name = 'combined.fits'
MF.add_schematics_and_add_noise(files_to_add, factors, output_name)


'''
If you want to use a different setup from zemax to simulate the data,
then you can change what HDF file is used.

That would look like this:
'''

MS = MakeSchematics()
MS.h5file = h5py.File('NTEpyechelle/models/NTE_BU.hdf', 'r')