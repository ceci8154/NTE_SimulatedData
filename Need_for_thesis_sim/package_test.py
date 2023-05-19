from NTEsim.make_schematics import MakeSchematics
from NTEsim.make_fits import MakeFits
from NTEpyechelle.sources import CSV
import h5py
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import os
import zipfile

# s = 'bd_17d4708_stisnic_002'
# ss_fits = fits.open('data/scienceframe_data/standard_star/'+s+'.fits')
# # extract wavelengths and fluxes
# wavelengths = ss_fits[1].data['WAVELENGTH']
# fluxes = ss_fits[1].data['FLUX']
# wavelengths = np.append(wavelengths, 26000)
# fluxes = np.append(fluxes, fluxes[-1])
# # extrace the units
# print('wl unit: ', ss_fits[1].header['TUNIT1'])
# print('flux unit: ', ss_fits[1].header['TUNIT2'] + ' erg/s/cm2/Å')
# # convert flux from erg/s/cm2/Å to erg/s/cm2/cm
# fluxes = fluxes * 1e-8
# # make interpolation to get more points
# f = interp1d(wavelengths, fluxes, kind='slinear')
# wavelengths = np.linspace(wavelengths[0], wavelengths[-1], 30000)
# fluxes = f(wavelengths)
# # write wl and flux to csv
# np.savetxt('data/scienceframe_data/standard_star/'+s+'.csv', np.transpose([wavelengths, fluxes]), delimiter=',')

# # Get all fits files from data/scienceframe_data/standard_star
# files = os.listdir('data/scienceframe_data/standard_star')
# # Then load the data from each file
# for f in files:
#     if f.endswith('.fits'):
#         ss_fits = fits.open('data/scienceframe_data/standard_star/'+f)
#         star_name = f.split('_')[0]
#         # extract wavelengths and fluxes
#         wavelengths = ss_fits[1].data['WAVELENGTH']
#         fluxes = ss_fits[1].data['FLUX']
#         maxwl = 26000
#         minwl = 3000
#         if np.max(wavelengths) < maxwl:
#             wavelengths = np.append(wavelengths, 26000)
#             fluxes = np.append(fluxes, fluxes[-1])
#         if np.min(wavelengths) > minwl:
#             wavelengths = np.append(3000, wavelengths)
#             fluxes = np.append(fluxes[0], fluxes)
#         # convert flux from erg/s/cm2/Å to erg/s/cm2/cm
#         fluxes = fluxes * 1e-8
#         # make interpolation to get more points
#         f = interp1d(wavelengths, fluxes, kind='slinear')
#         wavelengths = np.linspace(wavelengths[0], wavelengths[-1], 30000)
#         fluxes = f(wavelengths)
#         # write wl and flux to csv
#         np.savetxt('data/scienceframe_data/standard_star/'+star_name+'.csv', np.transpose([wavelengths, fluxes]), delimiter=',')


# MS = MakeSchematics()
# MS.gpu = True
# MS.overw = True
# # Make all preset frames
# MS.make_all()
# # Make all star frames
# # Get all csv files from data/scienceframe_data/standard_star
# files = os.listdir('data/scienceframe_data/standard_star')
# for f in files:
#     if '.csv' not in f:
#         files.remove(f)

# for f in files:
#     print(f)
#     source = CSV(filepath="data/scienceframe_data/standard_star/"+f, \
#                     name="", list_like=True, wavelength_unit='a', flux_in_photons=False, delimiter=',')
    
#     fn = 'standardstar'+f.split('.')[0]
#     et = 10
#     MS.make_custom('ir', et, source, 'ir_slitwidth1.0_'+str(et)+'sec_'+fn+'_schem.fits', point=True)

#     et = 20
#     MS.make_custom('vis', et, source, 'vis_slitwidth1.0_'+str(et)+'sec_'+fn+'_schem.fits', point=True)

#     et = 30
#     MS.make_custom('uv', et, source, 'uv_slitwidth1.0_'+str(et)+'sec_'+fn+'_schem.fits', point=True)

# Make gd71.dat into a csv file
# datdat = np.loadtxt('data/scienceframe_data/standard_star/gd71.dat')
# np.savetxt('data/scienceframe_data/standard_star/gd71.csv', datdat, delimiter=',')

# source = CSV(filepath="data/scienceframe_data/standard_star/gd71.csv", \
#                     name="", list_like=True, wavelength_unit='a', flux_in_photons=False, delimiter=',')

# fn = 'standardstargd71'
# et = 10*3
# MS.make_custom('ir', et, source, 'ir_slitwidth1.0_'+str(et)+'sec_'+fn+'_schem.fits', point=True)

# et = 20
# MS.make_custom('vis', et, source, 'vis_slitwidth1.0_'+str(et)+'sec_'+fn+'_schem.fits', point=True)

# et = 30
# MS.make_custom('uv', et, source, 'uv_slitwidth1.0_'+str(et)+'sec_'+fn+'_schem.fits', point=True)



# # Now make the fits files
# MF = MakeFits()
# MF.cosmic_rays = True
# MF.unzip_schematics()
# # MF.add_bias_and_noise_for_all_preset()
# MF.add_bias_and_noise_for_keyword('gd71', nr=1, add_sky=True)
# # # MF.add_bias_and_noise_for_not_keyword('grb25', nr=1)
# MF.delete_schematics()

# # zip all files in schematics/new_full
# zipf = zipfile.ZipFile('new_full.zip', 'w')

# l = len(os.listdir('schematics/new_full/'))
# count = 0
# for file in os.listdir('schematics/new_full/'):
#     count += 1
#     print(str(count)+'/'+str(l))
#     zipf.write('schematics/new_full/'+file)

# zipf.close()

# MS = MakeSchematics()
# MS.gpu = True
# MS.overw = True

# # Make ir dark
# MS.ir_make_dark()
# # Make ir flat
# et = 0.025
# MS.ir_make_flat(exposure_time=et)
# # Make ir etalon
# et = 2.25
# MS.ir_make_etalon(exposure_time=et)
# # Make ir HgAr
# et = 0.00625
# MS.ir_make_HgAr(exposure_time=et)
# # Make ir sky
# et = 0.0125
# MS.ir_make_sky(exposure_time=et)
# # Make ir trace
# et = 0.00125
# MS.ir_make_trace(exposure_time=et)
# # Make ir grb25
# et = 0.05
# MS.ir_make_grb25(exposure_time=et)
# # Make standard star
# et = 1/3 *2*10
# source = CSV(filepath="data/scienceframe_data/standard_star/bd17d4708.csv", \
#                     name="", list_like=True, wavelength_unit='a', flux_in_photons=False, delimiter=',')
# MS.make_custom('ir', et, source, 'ir_slitwidth1.0_0.66667sec_standardstarbd17d4708_schem.fits', point=True)
# # Make standard star also for ir, vis and uv with et = 10
# et = 10
# MS.make_custom('ir', et, source, 'ir_slitwidth1.0_10sec_standardstarbd17d4708_schem.fits', point=True)
# MS.make_custom('vis', et, source, 'vis_slitwidth1.0_10sec_standardstarbd17d4708_schem.fits', point=True)
# MS.make_custom('uv', et, source, 'uv_slitwidth1.0_10sec_standardstarbd17d4708_schem.fits', point=True)


# MF = MakeFits()
# MF.cosmic_rays = False
# MF.unzip_schematics()
# MF.add_bias_and_noise_for_all_preset()
# MF.delete_schematics()


MS = MakeSchematics()
MS.gpu = False
MS.overw = True

# Make ir dark
MS.ir_make_dark()
# Make ir flat
et = 0.025
MS.ir_make_flat(exposure_time=et)































# print('uv bias')
# MS.uv_make_bias()
# print('vis bias')
# MS.vis_make_bias()
# print('ir bias')
# MS.ir_make_dark()
# print('uv flat')
# MS.uv_make_flat(exposure_time=0.1)
# print('vis flat')
# MS.vis_make_flat(exposure_time=0.1/1.5)
# print('ir flat')
# MS.ir_make_flat(exposure_time=0.1/40)
# print('uv HgAr')
# MS.uv_make_HgAr(exposure_time=0.1/40/30)
# print('vis HgAr')
# MS.vis_make_HgAr(exposure_time=0.1/40/50)
# print('ir HgAr')
# MS.ir_make_HgAr(exposure_time=0.1/80/2)
# print('uv etalon')
# MS.uv_make_etalon(exposure_time=0.1*10)
# print('vis etalon')
# MS.vis_make_etalon(exposure_time=0.1*10)
# print('ir etalon')
# MS.ir_make_etalon(exposure_time=0.1/40*5*10)
# print('uv trace')
# MS.uv_make_trace(exposure_time=0.1/20)
# print('vis trace')
# MS.vis_make_trace(exposure_time=0.1/100)
# print('ir trace')
# MS.ir_make_trace(exposure_time=0.1/40/20)
# print('uv sky')
# MS.uv_make_sky(exposure_time=0.1*100/2)
# print('vis sky')
# MS.vis_make_sky(exposure_time=0.1)
# print('ir sky')
# MS.ir_make_sky(exposure_time=0.1/40/2)
# print('uv grb25')
# MS.uv_make_grb25(exposure_time=0.1*10)
# print('vis grb25')
# MS.vis_make_grb25(exposure_time=0.1)
# print('ir grb25')
# MS.ir_make_grb25(exposure_time=0.1/40*2)
# print('ss')
# source = CSV(filepath="data/scienceframe_data/standard_star/bd_17d4708_stisnic_002.csv", \
#                 name="", list_like=True, wavelength_unit='a', flux_in_photons=False, delimiter=',')
# et = 0.1/1.5
# MS.make_custom('ir', et, source, 'ir_slitwidth1.0_'+str(et)+'sec_bd17d4708'+'_schem.fits', point=True)


# #MS.gpu = False
# MS.overw = False

# source = CSV(filepath="data/scienceframe_data/standard_star/bd_75d325_stis_006.csv", \
#                 name="", list_like=True, wavelength_unit='a', flux_in_photons=False, delimiter=',')

# # Make 10 frames of this source
# for i in range(10):
#     et = 10
#     MS.make_custom('uv', et, source, 'uv_slitwidth1.0_'+str(et)+'sec_bd75d325_'+str(i)+'_schem.fits', point=True)
#     MS.make_custom('vis', et, source, 'vis_slitwidth1.0_'+str(et)+'sec_bd75d325_'+str(i)+'_schem.fits', point=True)
#     MS.make_custom('ir', et, source, 'ir_slitwidth1.0_'+str(et)+'sec_bd75d325_'+str(i)+'_schem.fits', point=True)



# print(wavelengths)
# print(fluxes)
# plt.plot(wavelengths, fluxes)
# plt.show()
# source = CSV(filepath="data/scienceframe_data/GRB/extended_GRBspec_2.5_withabs.dat", \
#                 name="", list_like=True, wavelength_unit='a', flux_in_photons=False, delimiter=' ')
# MS.make_custom('uv', 5, source, 'uv_testing_custom.fits')
# MS.make_custom('uv', 5, source, 'uv_testing_custom_point.fits', point=True)
# MS.make_custom('vis', 3, source, 'vis_testing_custom.fits')
# MS.make_custom('vis', 3, source, 'vis_testing_custom_point.fits', point=True)
# MS.make_custom('ir', 1, source, 'ir_testing_custom.fits')
# MS.make_custom('ir', 1, source, 'ir_testing_custom_point.fits', point=True)
# for i in range(10):
#     MS.uv_make_flat(exposure_time=0.001, name=str(i)+'.fits')
# # MS.make_all()
# MS.h5file = h5py.File('NTEpyechelle/models/NTE_testing.hdf', 'r')
# MS.gpu = False
# MS.uv_make_ab()
# MS.vis_make_ab()
# MS.ir_make_ab()
# MS.uv_make_bias()
# MS.vis_make_bias()
# MS.ir_make_dark()
# MS.uv_make_flat()
# MS.vis_make_flat()
# MS.ir_make_flat()
# for i in range(9):
#     MS.fiber = i+1
#     MS.uv_make_HgAr()
#     MS.vis_make_HgAr()
#     MS.ir_make_HgAr()
# MS.fiber = 1
# MS.vis_make_HgAr()
# MS.fiber = 3
# MS.uv_make_HgAr()
# MS.vis_make_HgAr()
# MS.ir_make_HgAr()
# MS.uv_make_etalon()
# MS.vis_make_etalon()
# MS.ir_make_etalon()
# MS.uv_make_trace()
# MS.vis_make_trace()
# MS.ir_make_trace()
# MS.uv_make_sky()
# MS.vis_make_sky()
# MS.ir_make_sky()
# MS.uv_make_grb25()
# MS.vis_make_grb25()
# MS.ir_make_grb25()

# source = CSV(filepath="data/scienceframe_data/GRB/extended_GRBspec_2.5_withabs.dat", \
#                 name="", list_like=True, wavelength_unit='a', flux_in_photons=False, delimiter=' ')
# MS.make_custom('uv', 5, source, 'uv_testing_custom.fits')
# MS.make_custom('uv', 5, source, 'uv_testing_custom_point.fits', point=True)
# MS.make_custom('vis', 3, source, 'vis_testing_custom.fits')
# MS.make_custom('vis', 3, source, 'vis_testing_custom_point.fits', point=True)
# MS.make_custom('ir', 1, source, 'ir_testing_custom.fits')
# MS.make_custom('ir', 1, source, 'ir_testing_custom_point.fits', point=True)

# MF = MakeFits()
# # FM.cosmic_rays = True
# MF.unzip_schematics()
# # FM.add_bias_and_noise_for_all_preset()
# # MF.add_bias_and_noise_for_keyword('bd75d325', nr=1, add_sky=True)
# MF.delete_schematics()

# FM.add_sky('vis_slitwidth1.0_1.001sec_HgAr_schem.fits',2,3)
# FM.add_bias_and_noise_for_not_keyword('grb25', nr=1)
# FM.add_bias_and_noise_for_keyword('grb25', nr=1, add_sky=True)
# files_to_add = ['ir_slitwidth1.0_0.5sec_3_sky_schem.fits', 'ir_slitwidth1.0_1sec_grb25_schem.fits', 'ir_slitwidth1.0_1sec_HgAr_schem.fits']
# scales = [0, 0, 1]
# FM.add_schems_and_add_noise(files_to_add, scales, 'ir_testing.fits')



