from NTEpyechelle.simulator import Simulator
from NTEpyechelle.sources import Phoenix, Constant, ThAr, ThNe, Etalon, CSV
from NTEpyechelle.spectrograph import ZEMAX
from NTEpyechelle.efficiency import CSVEfficiency
import os
from astropy.io import fits

gpu = True

output_dir = 'Output/'
overw = True

uvb_ros = 3
vis_ros = 3
nir_ros = 3.5

nr_array = []
for i in range(20):
    nr_array.append(str(i).zfill(4))

efficiency = CSVEfficiency('NTE_eff','data/nte_eff.csv')

# BIAS
print('bias')
bias_source = Constant()
bias_exposure_time = 0.00000001

for i in range(10):
    nr = nr_array[i]
    sim = Simulator(ZEMAX('NTE'))
    sim.set_ccd(1)
    sim.set_fibers(3)
    sim.set_sources(bias_source)
    sim.set_efficiency(efficiency)
    sim.set_exposure_time(bias_exposure_time)
    name = output_dir + 'nir_dark_'+str(nr)+'.fits'
    sim.set_output(name, overwrite=overw)
    sim.read_noise = nir_ros
    sim.set_cuda(gpu)
    sim.run()

    sim = Simulator(ZEMAX('NTE'))
    sim.set_ccd(2)
    sim.set_fibers(3)
    sim.set_sources(bias_source)
    sim.set_efficiency(efficiency)
    sim.set_exposure_time(bias_exposure_time)
    name = output_dir + 'vis_bias_'+str(nr)+'.fits'
    sim.set_output(name, overwrite=overw)
    sim.read_noise = vis_ros
    sim.set_cuda(gpu)
    sim.run()

    sim = Simulator(ZEMAX('NTE'))
    sim.set_ccd(3)
    sim.set_fibers(3)
    sim.set_sources(bias_source)
    sim.set_efficiency(efficiency)
    sim.set_exposure_time(bias_exposure_time)
    name = output_dir + 'uvb_bias_'+str(nr)+'.fits'
    sim.set_output(name, overwrite=overw)
    sim.read_noise = uvb_ros
    sim.set_cuda(gpu)
    sim.run()

# FLAT
print('flat')
flat_source = Constant()
flat_exposure_time = 0.5 # baggrund kunne være 1000 mere end noise.  Count på 20000 i centrum.

for i in range(5):
    nr = nr_array[i]
    sim = Simulator(ZEMAX('NTE'))
    sim.set_ccd(1)
    sim.set_fibers(3)
    sim.set_sources(flat_source)
    sim.set_efficiency(efficiency)
    sim.set_exposure_time(flat_exposure_time)
    name = output_dir + 'nir_flat_'+str(nr)+'.fits'
    sim.set_output(name, overwrite=overw)
    sim.read_noise = nir_ros
    sim.set_cuda(gpu)
    sim.run()

    sim = Simulator(ZEMAX('NTE'))
    sim.set_ccd(2)
    sim.set_fibers(3)
    sim.set_sources(flat_source)
    sim.set_efficiency(efficiency)
    sim.set_exposure_time(flat_exposure_time*10)
    name = output_dir + 'vis_flat_'+str(nr)+'.fits'
    sim.set_output(name, overwrite=overw)
    sim.read_noise = vis_ros
    sim.set_cuda(gpu)
    sim.run()

    sim = Simulator(ZEMAX('NTE'))
    sim.set_ccd(3)
    sim.set_fibers(3)
    sim.set_sources(flat_source)
    sim.set_efficiency(efficiency)
    sim.set_exposure_time(flat_exposure_time*20)
    name = output_dir + 'uvb_flat_'+str(nr)+'.fits'
    sim.set_output(name, overwrite=overw)
    sim.read_noise = uvb_ros
    sim.set_cuda(gpu)
    sim.run()


# Mercury lamp
print('mercury')
mercury_source = CSV(filepath="data/line_files/HgAr.csv", \
        name="", list_like=True, wavelength_unit='micron', flux_in_photons=True, delimiter=' ')
mercury_exposure_time = 60

for i in range(5):
    nr = nr_array[i]
    sim = Simulator(ZEMAX('NTE'))
    sim.set_ccd(1)
    sim.set_fibers(3)
    sim.set_sources(mercury_source)
    sim.set_efficiency(efficiency)
    sim.set_exposure_time(mercury_exposure_time)
    name = output_dir + 'nir_HgAr_'+str(nr)+'.fits'
    sim.set_output(name, overwrite=overw)
    sim.read_noise = nir_ros
    sim.set_cuda(gpu)
    sim.run()

    sim = Simulator(ZEMAX('NTE'))
    sim.set_ccd(2)
    sim.set_fibers(3)
    sim.set_sources(mercury_source)
    sim.set_efficiency(efficiency)
    sim.set_exposure_time(mercury_exposure_time)
    name = output_dir + 'vis_HgAr_'+str(nr)+'.fits'
    sim.set_output(name, overwrite=overw)
    sim.read_noise = vis_ros
    sim.set_cuda(gpu)
    sim.run()

    sim = Simulator(ZEMAX('NTE'))
    sim.set_ccd(3)
    sim.set_fibers(3)
    sim.set_sources(mercury_source)
    sim.set_efficiency(efficiency)
    sim.set_exposure_time(mercury_exposure_time)
    name = output_dir + 'uvb_HgAr_'+str(nr)+'.fits'
    sim.set_output(name, overwrite=overw)
    sim.read_noise = uvb_ros
    sim.set_cuda(gpu)
    sim.run()
    

# Etalon
print('etalon')
etalon_source = CSV(filepath="data/custom_etalon/etalon_mosaic.dat", \
        name="", list_like=True, wavelength_unit='nm', flux_in_photons=False, delimiter=' ')
etalon_exposure_time = 6

nr = '0000'
sim = Simulator(ZEMAX('NTE'))
sim.set_ccd(1)
sim.set_fibers(3)
sim.set_sources(etalon_source)
sim.set_efficiency(efficiency)
sim.set_exposure_time(etalon_exposure_time)
name = output_dir + 'nir_etalon_'+str(nr)+'.fits'
sim.set_output(name, overwrite=overw)
sim.read_noise = nir_ros
sim.set_cuda(gpu)
sim.run()

sim = Simulator(ZEMAX('NTE'))
sim.set_ccd(2)
sim.set_fibers(3)
sim.set_sources(etalon_source)
sim.set_efficiency(efficiency)
sim.set_exposure_time(etalon_exposure_time*11)
name = output_dir + 'vis_etalon_'+str(nr)+'.fits'
sim.set_output(name, overwrite=overw)
sim.read_noise = vis_ros
sim.set_cuda(gpu)
sim.run()

sim = Simulator(ZEMAX('NTE'))
sim.set_ccd(3)
sim.set_fibers(3)
sim.set_sources(etalon_source)
sim.set_efficiency(efficiency)
sim.set_exposure_time(etalon_exposure_time*3)
name = output_dir + 'uvb_etalon_'+str(nr)+'.fits'
sim.set_output(name, overwrite=overw)
sim.read_noise = uvb_ros
sim.set_cuda(gpu)
sim.run()

#trace
trace_source = Constant()
trace_exposure_time = 1/3

for i in range(5):
    nr = nr_array[i]
    sim = Simulator(ZEMAX('NTE'))
    sim.set_ccd(1)
    sim.set_fibers(9)
    sim.set_sources(trace_source)
    sim.set_efficiency(efficiency)
    sim.set_exposure_time(trace_exposure_time)
    name = output_dir + 'nir_trace_'+str(nr)+'.fits'
    sim.set_output(name, overwrite=overw)
    sim.read_noise = nir_ros
    sim.set_cuda(gpu)
    sim.run()

    sim = Simulator(ZEMAX('NTE'))
    sim.set_ccd(2)
    sim.set_fibers(9)
    sim.set_sources(trace_source)
    sim.set_efficiency(efficiency)
    sim.set_exposure_time(trace_exposure_time)
    name = output_dir + 'vis_trace_'+str(nr)+'.fits'
    sim.set_output(name, overwrite=overw)
    sim.read_noise = vis_ros
    sim.set_cuda(gpu)
    sim.run()

    sim = Simulator(ZEMAX('NTE'))
    sim.set_ccd(3)
    sim.set_fibers(9)
    sim.set_sources(trace_source)
    sim.set_efficiency(efficiency)
    sim.set_exposure_time(trace_exposure_time)
    name = output_dir + 'uvb_trace_'+str(nr)+'.fits'
    sim.set_output(name, overwrite=overw)
    sim.read_noise = uvb_ros
    sim.set_cuda(gpu)
    sim.run()


# skylines
skyline_source = CSV(filepath="data/line_files/skycalc_radiance.dat", \
        name="", list_like=True, wavelength_unit='nm', flux_in_photons=False, delimiter=' ')#TODO flux in photons?
skyline_exposure_time = 1/2

nr = '0000'
sim = Simulator(ZEMAX('NTE'))
sim.set_ccd(1)
sim.set_fibers(3)
sim.set_sources(skyline_source)
sim.set_efficiency(efficiency)
sim.set_exposure_time(skyline_exposure_time)
name = output_dir + 'nir_skylines_'+str(nr)+'.fits'
sim.set_output(name, overwrite=overw)
sim.read_noise = nir_ros
sim.set_cuda(gpu)
sim.run()

sim = Simulator(ZEMAX('NTE'))
sim.set_ccd(2)
sim.set_fibers(3)
sim.set_sources(skyline_source)
sim.set_efficiency(efficiency)
sim.set_exposure_time(skyline_exposure_time*6)
name = output_dir + 'vis_skylines_'+str(nr)+'.fits'
sim.set_output(name, overwrite=overw)
sim.read_noise = vis_ros
sim.set_cuda(gpu)
sim.run()

sim = Simulator(ZEMAX('NTE'))
sim.set_ccd(3)
sim.set_fibers(3)
sim.set_sources(skyline_source)
sim.set_efficiency(efficiency)
sim.set_exposure_time(skyline_exposure_time*466)
name = output_dir + 'uvb_skylines_'+str(nr)+'.fits'
sim.set_output(name, overwrite=overw)
sim.read_noise = uvb_ros
sim.set_cuda(gpu)
sim.run()

# Science frames
# GRB 2.5
print('GRB 2.5')
grb25_source = CSV(filepath="data/scienceframe_data/GRB/extended_GRBspec_2.5.dat", \
        name="", list_like=True, wavelength_unit='a', flux_in_photons=False, delimiter=' ')
grb25_exposure_time = 1 # count på 300-400 over baggrunden.


poss = [0.2, 0.35, 0.5, 0.65, 0.8]
for posy in poss:
    sim = Simulator(ZEMAX('NTE'))
    sim.set_ccd(1)
    sim.set_fibers(3)
    sim.set_sources(grb25_source)
    sim.set_efficiency(efficiency)
    sim.set_exposure_time(grb25_exposure_time)
    sim.set_point(seeing=0.8, pos=[0.5,posy])
    name = output_dir + 'nir_grb25_' + "{:.3f}".format(posy).replace('.','') +'.fits'
    sim.set_output(name, overwrite=overw)
    sim.read_noise = nir_ros
    sim.set_cuda(gpu)
    sim.run()

    sim = Simulator(ZEMAX('NTE'))
    sim.set_ccd(2)
    sim.set_fibers(3)
    sim.set_sources(grb25_source)
    sim.set_efficiency(efficiency)
    sim.set_exposure_time(grb25_exposure_time*3)
    sim.set_point(seeing=0.8, pos=[0.5,posy])
    name = output_dir + 'vis_grb25_' + "{:.3f}".format(posy).replace('.','') +'.fits'
    sim.set_output(name, overwrite=overw)
    sim.read_noise = vis_ros
    sim.set_cuda(gpu)
    sim.run()

    sim = Simulator(ZEMAX('NTE'))
    sim.set_ccd(3)
    sim.set_fibers(3)
    sim.set_sources(grb25_source)
    sim.set_efficiency(efficiency)
    sim.set_exposure_time(grb25_exposure_time*5)
    sim.set_point(seeing=0.8, pos=[0.5,posy])
    name = output_dir + 'uvb_grb25_' + "{:.3f}".format(posy).replace('.','') +'.fits'
    sim.set_output(name, overwrite=overw)
    sim.read_noise = uvb_ros
    sim.set_cuda(gpu)
    sim.run()


# all_files = os.listdir(output_dir)
# for file_name in all_files:
#     with fits.open(output_dir+file_name) as file:
#         data = file[0].data
#         data += 1000 
#         file.writeto(output_dir+file_name, overwrite=overw)


#slits  1 buesekund spalte
#bias - 10   For NIR ikke bias. Kun dark.
#flat - 5
#lampe = arc - 5
#dark - tom fil? Kun relavant for IR. For UV meget lav.   - samme exposuretime som science
#skylines - 2?
#trace pinhole flat    Værdier på 0 eller 1
#object some point source - 2 med forskellig exposuretime     - 5 filer. Forskellige steder i slit. Ikke for tæt på kanten. 8000 counts?
#etalon
# bias level + 1000 på alle pixels

#pinhole
#flat - 5
#lampe?? dont need