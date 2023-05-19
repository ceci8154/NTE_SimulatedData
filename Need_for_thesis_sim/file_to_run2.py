from NTEpyechelle.simulator import Simulator
from NTEpyechelle.sources import Phoenix, Constant, ThAr, ThNe, Etalon, CSV, LineList
from NTEpyechelle.spectrograph import ZEMAX
from NTEpyechelle.efficiency import CSVEfficiency
import os
from astropy.io import fits
import numpy as np
from astropy import units as u

gpu = True

output_dir = 'Output/'
overw = True

make_bias = False
make_flat = False
make_hgar = True
make_etalon = False
make_trace = False
make_skylines = False
make_science = False
make_AB = False
nr_array = []
for i in range(20):
    nr_array.append(str(i).zfill(4))

efficiency = CSVEfficiency('NTE_eff','data/nte_eff.csv')

files_in_output = os.listdir(output_dir)

# BIAS
print('bias')
bias_source = Constant()
bias_exposure_time = 0.00000001

name_nir_bias = output_dir + 'nir_dark_schem'+'.fits'
name_vis_bias = output_dir + 'vis_bias_schem'+'.fits'
name_uvb_bias = output_dir + 'uvb_bias_schem'+'.fits'

def make_bias_f():
    sim = Simulator(ZEMAX('NTE'))
    sim.set_ccd(1)
    sim.set_fibers(3)
    sim.set_sources(bias_source)
    sim.set_efficiency(efficiency)
    sim.set_exposure_time(bias_exposure_time)
    sim.set_output(name_nir_bias, overwrite=True)
    sim.run()

    sim = Simulator(ZEMAX('NTE'))
    sim.set_ccd(2)
    sim.set_fibers(3)
    sim.set_sources(bias_source)
    sim.set_efficiency(efficiency)
    sim.set_exposure_time(bias_exposure_time)
    sim.set_output(name_vis_bias, overwrite=True)
    sim.run()

    sim = Simulator(ZEMAX('NTE'))
    sim.set_ccd(3)
    sim.set_fibers(3)
    sim.set_sources(bias_source)
    sim.set_efficiency(efficiency)
    sim.set_exposure_time(bias_exposure_time)
    sim.set_output(name_uvb_bias, overwrite=True)
    sim.run()

if make_bias:
    if overw:
        make_bias_f()
    else:
        nir = name_nir_bias[len(output_dir):] not in files_in_output
        vis = name_vis_bias[len(output_dir):] not in files_in_output
        uvb = name_uvb_bias[len(output_dir):] not in files_in_output
        if nir or vis or uvb:
            make_bias_f()


# Flat
print('flat')
flat_source = Constant()
flat_exposure_time = 0.5 #0.5 # Background should be 1000 more than noise.  Count at 20000 in center.

name_nir_flat = output_dir + 'nir_flat_schem'+'.fits'
name_vis_flat = output_dir + 'vis_flat_schem'+'.fits'
name_uvb_flat = output_dir + 'uvb_flat_schem'+'.fits'

def make_flat_f():
    sim = Simulator(ZEMAX('NTE'))
    sim.set_ccd(1)
    sim.set_fibers(3)
    sim.set_sources(flat_source)
    sim.set_efficiency(efficiency)
    sim.set_exposure_time(flat_exposure_time)
    sim.set_output(name_nir_flat, overwrite=True)
    sim.set_cuda(gpu)
    sim.run()

    sim = Simulator(ZEMAX('NTE'))
    sim.set_ccd(2)
    sim.set_fibers(3)
    sim.set_sources(flat_source)
    sim.set_efficiency(efficiency)
    sim.set_exposure_time(flat_exposure_time*10)
    sim.set_output(name_vis_flat, overwrite=True)
    sim.set_cuda(gpu)
    sim.run()

    sim = Simulator(ZEMAX('NTE'))
    sim.set_ccd(3)
    sim.set_fibers(3)
    sim.set_sources(flat_source)
    sim.set_efficiency(efficiency)
    sim.set_exposure_time(flat_exposure_time*20)
    sim.set_output(name_uvb_flat, overwrite=True)
    sim.set_cuda(gpu)
    sim.run()

if make_flat:
    if overw:
        make_flat_f()
    else:
        nir = name_nir_flat[len(output_dir):] not in files_in_output
        vis = name_vis_flat[len(output_dir):] not in files_in_output
        uvb = name_uvb_flat[len(output_dir):] not in files_in_output
        if nir or vis or uvb:
            make_flat_f()


# Mercury Lamp
print('mercury')
dat = np.loadtxt("data/line_files/HgArNew_onlyAr.csv")
wl = dat[:,0] * u.nm
wl_micron = wl.to(u.micron)
wl_micron = wl_micron.value
flux = dat[:,1]
mercury_source_Ar = LineList(wavelengths=wl_micron, intensities=flux)

dat = np.loadtxt("data/line_files/HgArNew_onlyHg.csv")
wl = dat[:,0] * u.nm
wl_micron = wl.to(u.micron)
wl_micron = wl_micron.value
flux = dat[:,1]
mercury_source_Hg = LineList(wavelengths=wl_micron, intensities=flux)

mercury_exposure_time = 1

name_nir_hgar = output_dir + 'nir_HgAr_schem'+'.fits'
name_vis_hgar = output_dir + 'vis_HgAr_schem_onlyAr'+'.fits'
name_uvb_hgar = output_dir + 'uvb_HgAr_schem_onlyHg'+'.fits'

def make_hgar_f():
    # sim = Simulator(ZEMAX('NTE'))
    # sim.set_ccd(1)
    # sim.set_fibers(3)
    # sim.set_sources(mercury_source)
    # sim.set_efficiency(efficiency)
    # sim.set_exposure_time(mercury_exposure_time)
    # sim.set_output(name_nir_hgar, overwrite=True)
    # sim.set_cuda(gpu)
    # sim.run()

    sim = Simulator(ZEMAX('NTE'))
    sim.set_ccd(2)
    sim.set_fibers(3)
    sim.set_sources(mercury_source_Ar)
    #sim.set_efficiency(efficiency)
    sim.set_exposure_time(mercury_exposure_time)
    sim.set_output(name_vis_hgar, overwrite=True)
    sim.set_cuda(gpu)
    sim.run()

    sim = Simulator(ZEMAX('NTE'))
    sim.set_ccd(3)
    sim.set_fibers(3)
    sim.set_sources(mercury_source_Hg)
    #sim.set_efficiency(efficiency)
    sim.set_exposure_time(mercury_exposure_time)
    sim.set_output(name_uvb_hgar, overwrite=True)
    sim.set_cuda(gpu)
    sim.run()

if make_hgar:
    if overw:
        make_hgar_f()
    else:
        nir = name_nir_hgar[len(output_dir):] not in files_in_output
        vis = name_vis_hgar[len(output_dir):] not in files_in_output
        uvb = name_uvb_hgar[len(output_dir):] not in files_in_output
        if nir or vis or uvb:
            make_hgar_f()


# Etalon
print('etalon')
etalon_source = CSV(filepath="data/custom_etalon/etalon_mosaic.dat", \
        name="", list_like=True, wavelength_unit='nm', flux_in_photons=False, delimiter=' ')
etalon_exposure_time = 6

name_nir_etalon = output_dir + 'nir_etalon_schem'+'.fits'
name_vis_etalon = output_dir + 'vis_etalon_schem'+'.fits'
name_uvb_etalon = output_dir + 'uvb_etalon_schem'+'.fits'

def make_etalon_f():
    sim = Simulator(ZEMAX('NTE'))
    sim.set_ccd(1)
    sim.set_fibers(3)
    sim.set_sources(etalon_source)
    sim.set_efficiency(efficiency)
    sim.set_exposure_time(etalon_exposure_time)
    sim.set_output(name_nir_etalon, overwrite=True)
    sim.set_cuda(gpu)
    sim.run()

    sim = Simulator(ZEMAX('NTE'))
    sim.set_ccd(2)
    sim.set_fibers(3)
    sim.set_sources(etalon_source)
    sim.set_efficiency(efficiency)
    sim.set_exposure_time(etalon_exposure_time*11)
    sim.set_output(name_vis_etalon, overwrite=True)
    sim.set_cuda(gpu)
    sim.run()

    sim = Simulator(ZEMAX('NTE'))
    sim.set_ccd(3)
    sim.set_fibers(3)
    sim.set_sources(etalon_source)
    sim.set_efficiency(efficiency)
    sim.set_exposure_time(etalon_exposure_time*3)
    sim.set_output(name_uvb_etalon, overwrite=True)
    sim.set_cuda(gpu)
    sim.run()

if make_etalon:
    if overw:
        make_etalon_f()
    else:
        nir = name_nir_etalon[len(output_dir):] not in files_in_output
        vis = name_vis_etalon[len(output_dir):] not in files_in_output
        uvb = name_uvb_etalon[len(output_dir):] not in files_in_output
        if nir or vis or uvb:
            make_etalon_f()


# Trace
print('trace')
trace_source = Constant()
trace_exposure_time = 1/3

name_nir_trace = output_dir + 'nir_trace_schem'+'.fits'
name_vis_trace = output_dir + 'vis_trace_schem'+'.fits'
name_uvb_trace = output_dir + 'uvb_trace_schem'+'.fits'

def make_trace_f():
    sim = Simulator(ZEMAX('NTE'))
    sim.set_ccd(1)
    sim.set_fibers(9)
    sim.set_sources(trace_source)
    sim.set_efficiency(efficiency)
    sim.set_exposure_time(trace_exposure_time)
    sim.set_output(name_nir_trace, overwrite=True)
    sim.set_cuda(gpu)
    sim.run()

    sim = Simulator(ZEMAX('NTE'))
    sim.set_ccd(2)
    sim.set_fibers(9)
    sim.set_sources(trace_source)
    sim.set_efficiency(efficiency)
    sim.set_exposure_time(trace_exposure_time)
    sim.set_output(name_vis_trace, overwrite=True)
    sim.set_cuda(gpu)
    sim.run()

    sim = Simulator(ZEMAX('NTE'))
    sim.set_ccd(3)
    sim.set_fibers(9)
    sim.set_sources(trace_source)
    sim.set_efficiency(efficiency)
    sim.set_exposure_time(trace_exposure_time)
    sim.set_output(name_uvb_trace, overwrite=True)
    sim.set_cuda(gpu)
    sim.run()

if make_trace:
    if overw:
        make_trace_f()
    else:
        nir = name_nir_trace[len(output_dir):] not in files_in_output
        vis = name_vis_trace[len(output_dir):] not in files_in_output
        uvb = name_uvb_trace[len(output_dir):] not in files_in_output
        if nir or vis or uvb:
            make_trace_f()


# Skylines
print('skylines')
skyline_source = CSV(filepath="data/line_files/skycalc_radiance.dat", \
        name="", list_like=True, wavelength_unit='nm', flux_in_photons=False, delimiter=' ')#TODO flux in photons?
skyline_exposure_time = 1/2

name_nir_skylines = output_dir + 'nir_skylines_schem'+'.fits'
name_vis_skylines = output_dir + 'vis_skylines_schem'+'.fits'
name_uvb_skylines = output_dir + 'uvb_skylines_schem'+'.fits'

def make_skylines_f():
    sim = Simulator(ZEMAX('NTE'))
    sim.set_ccd(1)
    sim.set_fibers(3)
    sim.set_sources(skyline_source)
    sim.set_efficiency(efficiency)
    sim.set_exposure_time(skyline_exposure_time)
    sim.set_output(name_nir_skylines, overwrite=True)
    sim.set_cuda(gpu)
    sim.run()

    sim = Simulator(ZEMAX('NTE'))
    sim.set_ccd(2)
    sim.set_fibers(3)
    sim.set_sources(skyline_source)
    sim.set_efficiency(efficiency)
    sim.set_exposure_time(skyline_exposure_time*6)
    sim.set_output(name_vis_skylines, overwrite=True)
    sim.set_cuda(gpu)
    sim.run()

    sim = Simulator(ZEMAX('NTE'))
    sim.set_ccd(3)
    sim.set_fibers(3)
    sim.set_sources(skyline_source)
    sim.set_efficiency(efficiency)
    sim.set_exposure_time(skyline_exposure_time*466)
    sim.set_output(name_uvb_skylines, overwrite=True)
    sim.set_cuda(gpu)
    sim.run()

if make_skylines:
    if overw:
        make_skylines_f()
    else:
        nir = name_nir_skylines[len(output_dir):] not in files_in_output
        vis = name_vis_skylines[len(output_dir):] not in files_in_output
        uvb = name_uvb_skylines[len(output_dir):] not in files_in_output
        if nir or vis or uvb:
            make_skylines_f()


# Science frames
# GRB 2.5
print('GRB 2.5')
grb25_source = CSV(filepath="data/scienceframe_data/GRB/extended_GRBspec_2.5_withabs.dat", \
        name="", list_like=True, wavelength_unit='a', flux_in_photons=False, delimiter=' ')
grb25_exposure_time = 1 # count på 300-400 over baggrunden.

name_nir_grb25 = output_dir + 'nir_grb25_schem'#+'.fits'
name_vis_grb25 = output_dir + 'vis_grb25_schem'#+'.fits'
name_uvb_grb25 = output_dir + 'uvb_grb25_schem'#+'.fits'

def make_grb25_f(posy, posx=0.5, see=0.8, extra_name='', x_name=False):
    sim = Simulator(ZEMAX('NTE'))
    sim.set_ccd(1)
    sim.set_fibers(3)
    sim.set_sources(grb25_source)
    sim.set_efficiency(efficiency)
    sim.set_exposure_time(grb25_exposure_time)
    sim.set_point(seeing=see, pos=[posx,posy])
    if x_name:
        name = name_nir_grb25 +'_'+ "{:.3f}".format(posy).replace('.','') + "_{:.3f}".format(posx).replace('.','') + extra_name + '.fits'
    else:
        name = name_nir_grb25 +'_'+ "{:.3f}".format(posy).replace('.','') + extra_name + '.fits'
    sim.set_output(name, overwrite=True)
    sim.set_cuda(gpu)
    sim.run()

    sim = Simulator(ZEMAX('NTE'))
    sim.set_ccd(2)
    sim.set_fibers(3)
    sim.set_sources(grb25_source)
    sim.set_efficiency(efficiency)
    sim.set_exposure_time(grb25_exposure_time*3)
    sim.set_point(seeing=see, pos=[posx,posy])
    if x_name:
        name = name_vis_grb25 +'_'+ "{:.3f}".format(posy).replace('.','') + "_{:.3f}".format(posx).replace('.','') + extra_name + '.fits'
    else:
        name = name_vis_grb25 +'_'+ "{:.3f}".format(posy).replace('.','') + extra_name + '.fits'
    sim.set_output(name, overwrite=True)
    sim.set_cuda(gpu)
    sim.run()

    sim = Simulator(ZEMAX('NTE'))
    sim.set_ccd(3)
    sim.set_fibers(3)
    sim.set_sources(grb25_source)
    sim.set_efficiency(efficiency)
    sim.set_exposure_time(grb25_exposure_time*5)
    sim.set_point(seeing=see, pos=[posx,posy])
    if x_name:
        name = name_uvb_grb25 +'_'+ "{:.3f}".format(posy).replace('.','') + "_{:.3f}".format(posx).replace('.','') + extra_name + '.fits'
    else:
        name = name_uvb_grb25 +'_'+ "{:.3f}".format(posy).replace('.','') + extra_name + '.fits'
    sim.set_output(name, overwrite=True)
    sim.set_cuda(gpu)
    sim.run()



poss = [0.2, 0.35, 0.5, 0.65, 0.8]
seeing_grb = 2
if make_science:
    for posy in poss:
        if overw:
            make_grb25_f(posy, see=seeing_grb)
        else:
            nir_name = name_nir_grb25 +'_'+ "{:.3f}".format(posy).replace('.','') + '.fits'
            nir = nir_name[len(output_dir):] not in files_in_output
            vis_name = name_vis_grb25 +'_'+ "{:.3f}".format(posy).replace('.','') + '.fits'
            vis = vis_name[len(output_dir):] not in files_in_output
            uvb_name = name_uvb_grb25 +'_'+ "{:.3f}".format(posy).replace('.','') + '.fits'
            uvb = uvb_name[len(output_dir):] not in files_in_output
            if nir or vis or uvb:
                make_grb25_f(posy, see=seeing_grb)

# Make AB setup where the position in slit changes little bit each time.
np.random.seed(42)
possx_A = np.random.uniform(0.45, 0.55, 6)
possy_A = np.random.uniform(0.30, 0.40, 6)
possx_B = np.random.uniform(0.45, 0.55, 6)
possy_B = np.random.uniform(0.60, 0.70, 6)
seeing_AB = 2
if make_AB:
    for i in range(len(possx_A)):
        print(i)
        if overw:
            make_grb25_f(possy_A[i], posx=possx_A[i], see=seeing_AB, extra_name='_A_{}'.format(i), x_name=True)
            make_grb25_f(possy_B[i], posx=possx_B[i], see=seeing_AB, extra_name='_B_{}'.format(i), x_name=True)
        else:
            nir_name = name_nir_grb25 +'_'+ "{:.3f}".format(possy_A[i]).replace('.','') + "_{:.3f}".format(possx_A[i]).replace('.','') + '_A_{}'.format(i) + '.fits'
            nir = nir_name[len(output_dir):] not in files_in_output
            vis_name = name_vis_grb25 +'_'+ "{:.3f}".format(possy_A[i]).replace('.','') + "_{:.3f}".format(possx_A[i]).replace('.','') + '_A_{}'.format(i) + '.fits'
            vis = vis_name[len(output_dir):] not in files_in_output
            uvb_name = name_uvb_grb25 +'_'+ "{:.3f}".format(possy_A[i]).replace('.','') + "_{:.3f}".format(possx_A[i]).replace('.','') + '_A_{}'.format(i) + '.fits'
            uvb = uvb_name[len(output_dir):] not in files_in_output
            if nir or vis or uvb:
                print('i am here___________________________________')
                print(nir, vis, uvb)
                print(nir_name)
                make_grb25_f(possy_A[i], posx=possx_A[i], see=seeing_AB, extra_name='_A_{}'.format(i), x_name=True)

            nir_name = name_nir_grb25 +'_'+ "{:.3f}".format(possy_B[i]).replace('.','') + "_{:.3f}".format(possx_B[i]).replace('.','') + '_B_{}'.format(i) + '.fits'
            nir = nir_name[len(output_dir):] not in files_in_output
            vis_name = name_vis_grb25 +'_'+ "{:.3f}".format(possy_B[i]).replace('.','') + "_{:.3f}".format(possx_B[i]).replace('.','') + '_B_{}'.format(i) + '.fits'
            vis = vis_name[len(output_dir):] not in files_in_output
            uvb_name = name_uvb_grb25 +'_'+ "{:.3f}".format(possy_B[i]).replace('.','') + "_{:.3f}".format(possx_B[i]).replace('.','') + '_B_{}'.format(i) + '.fits'
            uvb = uvb_name[len(output_dir):] not in files_in_output
            if nir or vis or uvb:
                make_grb25_f(possy_B[i], posx=possx_B[i], see=seeing_AB, extra_name='_B_{}'.format(i), x_name=True)