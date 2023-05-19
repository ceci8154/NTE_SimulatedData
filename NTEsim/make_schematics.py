'''
Written by Cecilie and Mads.
This package makes all the schematics for the NTEpyechelle simulator.

The package can make the following schematics:
    
    - Bias

    - Flat

    - Mercury lamp

    - Etalon

    - Trace

    - Sky lines

    - Science - at the moment it is only one GRB spectrum

    - AB - uses the same GRB spectrum and does an AB setup


Keep in mind that this code will be incredibly slow without a GPU.
Additionaly it only works with CUDA, as pyechelle uses Numba.
'''

from NTEpyechelle.efficiency import CSVEfficiency
from NTEpyechelle.spectrograph import ZEMAX
from NTEpyechelle.simulator import Simulator
from NTEpyechelle.sources import Constant, LineList, CSV
from NTEpyechelle.telescope import Telescope
import os
import numpy as np
from astropy import units as u
import h5py


class MakeSchematics:
    '''
    Class to contain all functions and variables needed to make the simulations.
    '''
    def __init__(self):
        self.gpu = True
        self.fiber = 3
        em_eff = CSVEfficiency('EM_eff', 'data/efficiency/em_eff.csv')
        skipper_eff = CSVEfficiency('Skipper_eff', 'data/efficiency/skipper_eff.csv')
        h2rg_eff = CSVEfficiency('H2RG_eff', 'data/efficiency/h2rg_eff.csv')
        self.efficiency = [h2rg_eff, skipper_eff, em_eff]
        self.output_dir = 'Output/'
        self.overw = False
        self.check_output_dir()
        self.files_in_output = os.listdir(self.output_dir)
        self.h5file = h5py.File('NTEpyechelle/models/NTE.hdf', 'r')

        # For AB default setup
        self.Axs = np.random.uniform(0.45, 0.55, 6)
        self.Ays = np.random.uniform(0.30, 0.40, 6)
        self.Bxs = np.random.uniform(0.45, 0.55, 6)
        self.Bys = np.random.uniform(0.60, 0.70, 6)


    def find_slit(self,ccd):
        '''
        Used to find the slit size for a ccd and fiber.
        Just used to naming the files.
        '''
        fib = self.h5file['CCD_'+str(ccd)]['fiber_'+str(self.fiber)]
        if fib.attrs['field_shape'] == 'circular':
            slit = 'pinhole' + str(fib.attrs['slit_diameter'])
        elif fib.attrs['field_shape'] == 'rectangular':
            slit = 'slitwidth' + str(fib.attrs['slit_width'])
        return slit

    
    def check_output_dir(self):
        '''
        Check if the output directiory exists, if not it makes it.
        '''
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)


    def simulator(self, ccd, source, name, exposure_time):
        '''
        Makes the simulation given the input parameters.

        Parameters
        ----------
        ccd : int
            1, 2 or 3. 1 for IR, 2 for VIS and 3 for UV.

        source : NTEpyechelle.sources
            The source to be simulated.

        name : str
            The name of the file to be saved.

        exposure_time : float
            The exposure time of the simulation.

        Returns
        -------
        It makes a fits file in the output directory.
        '''
        if ccd == 'ir':
            ccd = 1
        elif ccd == 'vis':
            ccd = 2
        elif ccd == 'uv':
            ccd = 3
        self.check_output_dir()
        self.h5file_name = self.h5file.filename.split('/')[-1][:-4]
        sim = Simulator(ZEMAX(self.h5file_name))
        #sim.set_telescope(Telescope(2.56, 0.51))
        sim.set_ccd(ccd)
        sim.set_fibers(self.fiber)
        sim.set_sources(source)
        sim.set_efficiency(self.efficiency[ccd-1])
        sim.set_exposure_time(exposure_time)
        sim.set_output(self.output_dir + name, overwrite=True)
        sim.set_cuda(self.gpu)
        sim.run()


    def point_simulator(self, ccd, source, name, exposure_time, posx, posy, seeing):
        '''
        Makes the simulation given the input parameters. Now as a point source.

        Parameters
        ----------
        ccd : int
            1, 2 or 3. 1 for IR, 2 for VIS and 3 for UV.

        source : NTEpyechelle.sources
            The source to be simulated.

        name : str
            The name of the file to be saved.

        exposure_time : float
            The exposure time of the simulation.

        posx : float
            The x position of the point source.

        posy : float
            The y position of the point source.

        seeing : float
            The seeing of the point source.

        Returns
        -------
        It makes a fits file in the output directory.
        '''
        if ccd == 'ir':
            ccd = 1
        elif ccd == 'vis':
            ccd = 2
        elif ccd == 'uv':
            ccd = 3
        self.check_output_dir()
        self.h5file_name = self.h5file.filename.split('/')[-1][:-4]
        sim = Simulator(ZEMAX(self.h5file_name))
        sim.set_ccd(ccd)
        sim.set_fibers(self.fiber)
        sim.set_sources(source)
        sim.set_efficiency(self.efficiency[ccd-1])
        sim.set_exposure_time(exposure_time)
        sim.set_point(seeing=seeing, pos=[posx,posy])
        sim.set_output(self.output_dir + name, overwrite=True)
        sim.set_cuda(self.gpu)
        sim.run()


    '''
    Bias functions
    '''


    def uv_make_bias(self, exposure_time=0.000000001, name=False):
        '''
        Makes a bias for the UV ccd.

        Parameters
        ----------
        exposure_time : float
            The exposure time of the simulation.

        name : str
            The name of the file to be saved.

        Returns
        -------
        It makes a fits file in the output directory.
        '''
        source = Constant()
        if not name:
            slit = self.find_slit(ccd=3)
            name = 'uv_'+str(slit)+'_bias_schem.fits'
        if self.overw:
            self.simulator(ccd=3, source=source, name=name, exposure_time=exposure_time)
        else:
            if name not in self.files_in_output:
                self.simulator(ccd=3, source=source, name=name, exposure_time=exposure_time)


    def vis_make_bias(self, exposure_time=0.000000001, name=False):
        '''
        Makes a bias for the VIS ccd.
        
        Parameters
        ----------
        exposure_time : float
            The exposure time of the simulation.

        name : str
            The name of the file to be saved.
        
        Returns
        -------
        It makes a fits file in the output directory.
        '''
        source = Constant()
        if not name:
            slit = self.find_slit(ccd=2)
            name = 'vis_'+str(slit)+'_bias_schem.fits'
        if self.overw:
            self.simulator(ccd=2, source=source, name=name, exposure_time=exposure_time)
        else:
            if name not in self.files_in_output:
                self.simulator(ccd=2, source=source, name=name, exposure_time=exposure_time)


    def ir_make_dark(self, exposure_time=0.000000001, name=False):
        '''
        Makes a dark for the IR ccd.

        Parameters
        ----------
        exposure_time : float
            The exposure time of the simulation.

        name : str
            The name of the file to be saved.

        Returns
        -------
        It makes a fits file in the output directory.
        '''
        source = Constant()
        if not name:
            slit = self.find_slit(ccd=1)
            name = 'ir_'+str(slit)+'_dark_schem.fits'
        if self.overw:
            self.simulator(ccd=1, source=source, name=name, exposure_time=exposure_time)
        else:
            if name not in self.files_in_output:
                self.simulator(ccd=1, source=source, name=name, exposure_time=exposure_time)


    '''
    Flat functions
    '''


    def uv_make_flat(self, exposure_time=10, name=False):
        '''
        Makes a flat for the UV ccd.

        Parameters
        ----------
        exposure_time : float
            The exposure time of the simulation.

        name : str
            The name of the file to be saved.

        Returns
        -------
        It makes a fits file in the output directory.
        '''
        source = Constant()
        if not name:
            slit = self.find_slit(ccd=3)
            name = 'uv_'+str(slit)+'_'+str(exposure_time)+'sec'+'_flat_schem.fits'
        if self.overw:
            self.simulator(ccd=3, source=source, name=name, exposure_time=exposure_time)
        else:
            if name not in self.files_in_output:
                self.simulator(ccd=3, source=source, name=name, exposure_time=exposure_time)


    def vis_make_flat(self, exposure_time=5, name=False):
        '''
        Makes a flat for the VIS ccd.

        Parameters
        ----------
        exposure_time : float
            The exposure time of the simulation.

        name : str
            The name of the file to be saved.
        
        Returns
        -------
        It makes a fits file in the output directory.
        '''
        source = Constant()
        if not name:
            slit = self.find_slit(ccd=2)
            name = 'vis_'+str(slit)+'_'+str(exposure_time)+'sec'+'_flat_schem.fits'
        if self.overw:
            self.simulator(ccd=2, source=source, name=name, exposure_time=exposure_time)
        else:
            if name not in self.files_in_output:
                self.simulator(ccd=2, source=source, name=name, exposure_time=exposure_time)


    def ir_make_flat(self, exposure_time=0.5, name=False):
        '''
        Makes a flat for the IR ccd.

        Parameters
        ----------
        exposure_time : float
            The exposure time of the simulation.

        name : str
            The name of the file to be saved.

        Returns
        -------
        It makes a fits file in the output directory.
        '''
        source = Constant()
        if not name:
            slit = self.find_slit(ccd=1)
            name = 'ir_'+str(slit)+'_'+str(exposure_time)+'sec'+'_flat_schem.fits'
        if self.overw:
            self.simulator(ccd=1, source=source, name=name, exposure_time=exposure_time)
        else:
            if name not in self.files_in_output:
                self.simulator(ccd=1, source=source, name=name, exposure_time=exposure_time)


    '''
    HgAr functions
    '''


    def get_HgAr_source(self):
        '''
        Gets the HgAr source from the data/line_files/HgArFinal.csv file.

        Parameters
        ----------
        None

        Returns
        -------
        source : LineList
            The HgAr source.
        '''
        dat = np.loadtxt("data/line_files/HgArFinal.csv")
        wl = dat[:,0] * u.nm
        wl_micron = wl.to(u.micron)
        wl_micron = wl_micron.value
        flux = dat[:,1]
        source = LineList(wavelengths=wl_micron, intensities=flux)
        return source


    def uv_make_HgAr(self, exposure_time=1, name=False):
        '''
        Makes a HgAr for the UV ccd.
        
        Parameters
        ----------
        exposure_time : float
            The exposure time of the simulation.

        name : str
            The name of the file to be saved.
        
        Returns
        -------
        It makes a fits file in the output directory.
        '''
        source = self.get_HgAr_source()
        if not name:
            slit = self.find_slit(ccd=3)
            name = 'uv_'+str(slit)+'_'+str(exposure_time)+'sec'+'_HgAr_schem.fits'
        if self.overw:
            self.simulator(ccd=3, source=source, name=name, exposure_time=exposure_time)
        else:
            if name not in self.files_in_output:
                self.simulator(ccd=3, source=source, name=name, exposure_time=exposure_time)


    def vis_make_HgAr(self, exposure_time=1, name=False):
        '''
        Makes a HgAr for the VIS ccd.
        
        Parameters
        ----------
        exposure_time : float
            The exposure time of the simulation.

        name : str
            The name of the file to be saved.
        
        Returns
        -------
        It makes a fits file in the output directory.
        '''
        source = self.get_HgAr_source()
        if not name:
            slit = self.find_slit(ccd=2)
            name = 'vis_'+str(slit)+'_'+str(exposure_time)+'sec'+'_HgAr_schem.fits'
        if self.overw:
            self.simulator(ccd=2, source=source, name=name, exposure_time=exposure_time)
        else:
            if name not in self.files_in_output:
                self.simulator(ccd=2, source=source, name=name, exposure_time=exposure_time)


    def ir_make_HgAr(self, exposure_time=1, name=False):
        '''
        Makes a HgAr for the IR ccd.
        
        Parameters
        ----------
        exposure_time : float
            The exposure time of the simulation.

        name : str
            The name of the file to be saved.
        
        Returns
        -------
        It makes a fits file in the output directory.
        '''
        source = self.get_HgAr_source()
        if not name:
            slit = self.find_slit(ccd=1)
            name = 'ir_'+str(slit)+'_'+str(exposure_time)+'sec'+'_HgAr_schem.fits'
        if self.overw:
            self.simulator(ccd=1, source=source, name=name, exposure_time=exposure_time)
        else:
            if name not in self.files_in_output:
                self.simulator(ccd=1, source=source, name=name, exposure_time=exposure_time)


    '''
    Etalon functions
    '''


    def get_etalon_source(self):
        '''
        Gets the etalon source from the data/custom_etalon/etalon_mosaic.dat file.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        etalon_source : CSV
            The etalon source.
        '''
        etalon_source = CSV(filepath="data/custom_etalon/etalon_mosaic.dat", \
                name="", list_like=True, wavelength_unit='nm', flux_in_photons=False, delimiter=' ')
        return etalon_source

    
    def uv_make_etalon(self, exposure_time=18, name=False):
        '''
        Makes a etalon for the UV ccd.
        
        Parameters
        ----------
        exposure_time : float
            The exposure time of the simulation.

        name : str
            The name of the file to be saved.
        
        Returns
        -------
        It makes a fits file in the output directory.
        '''
        source = self.get_etalon_source()
        if not name:
            slit = self.find_slit(ccd=3)
            name = 'uv_'+str(slit)+'_'+str(exposure_time)+'sec'+'_etalon_schem.fits'
        if self.overw:
            self.simulator(ccd=3, source=source, name=name, exposure_time=exposure_time)
        else:
            if name not in self.files_in_output:
                self.simulator(ccd=3, source=source, name=name, exposure_time=exposure_time)

    
    def vis_make_etalon(self, exposure_time=66, name=False):
        '''
        Makes a etalon for the VIS ccd.
        
        Parameters
        ----------
        exposure_time : float
            The exposure time of the simulation.

        name : str
            The name of the file to be saved.
        
        Returns
        -------
        It makes a fits file in the output directory.
        '''
        source = self.get_etalon_source()
        if not name:
            slit = self.find_slit(ccd=2)
            name = 'vis_'+str(slit)+'_'+str(exposure_time)+'sec'+'_etalon_schem.fits'
        if self.overw:
            self.simulator(ccd=2, source=source, name=name, exposure_time=exposure_time)
        else:
            if name not in self.files_in_output:
                self.simulator(ccd=2, source=source, name=name, exposure_time=exposure_time)

        
    def ir_make_etalon(self, exposure_time=6, name=False):
        '''
        Makes a etalon for the IR ccd.
        
        Parameters
        ----------
        exposure_time : float
            The exposure time of the simulation.

        name : str
            The name of the file to be saved.
        
        Returns
        -------
        It makes a fits file in the output directory.
        '''
        source = self.get_etalon_source()
        if not name:
            slit = self.find_slit(ccd=1)
            name = 'ir_'+str(slit)+'_'+str(exposure_time)+'sec'+'_etalon_schem.fits'
        if self.overw:
            self.simulator(ccd=1, source=source, name=name, exposure_time=exposure_time)
        else:
            if name not in self.files_in_output:
                self.simulator(ccd=1, source=source, name=name, exposure_time=exposure_time)


    '''
    Trace functions
    '''

    def uv_make_trace(self, exposure_time=0.33, name=False):
        '''
        Makes a trace for the UV ccd.
        
        Parameters
        ----------
        exposure_time : float
            The exposure time of the simulation.

        name : str
            The name of the file to be saved.
        
        Returns
        -------
        It makes a fits file in the output directory.
        '''
        source = Constant()
        save_old_fiber = self.fiber
        self.fiber = 9
        if not name:
            slit = self.find_slit(ccd=3)
            name = 'uv_'+str(slit)+'_'+str(exposure_time)+'sec'+'_trace_schem.fits'
        if self.overw:
            self.simulator(ccd=3, source=source, name=name, exposure_time=exposure_time)
        else:
            if name not in self.files_in_output:
                self.simulator(ccd=3, source=source, name=name, exposure_time=exposure_time)
        self.fiber = save_old_fiber


    def vis_make_trace(self, exposure_time=0.33, name=False):
        '''
        Makes a trace for the VIS ccd.
        
        Parameters
        ----------
        exposure_time : float
            The exposure time of the simulation.

        name : str
            The name of the file to be saved.
        
        Returns
        -------
        It makes a fits file in the output directory.
        '''
        source = Constant()
        save_old_fiber = self.fiber
        self.fiber = 9
        if not name:
            slit = self.find_slit(ccd=2)
            name = 'vis_'+str(slit)+'_'+str(exposure_time)+'sec'+'_trace_schem.fits'
        if self.overw:
            self.simulator(ccd=2, source=source, name=name, exposure_time=exposure_time)
        else:
            if name not in self.files_in_output:
                self.simulator(ccd=2, source=source, name=name, exposure_time=exposure_time)
        self.fiber = save_old_fiber

    
    def ir_make_trace(self, exposure_time=0.33, name=False):
        '''
        Makes a trace for the IR ccd.
        
        Parameters
        ----------
        exposure_time : float
            The exposure time of the simulation.

        name : str
            The name of the file to be saved.
        
        Returns
        -------
        It makes a fits file in the output directory.
        '''
        source = Constant()
        save_old_fiber = self.fiber
        self.fiber = 9
        if not name:
            slit = self.find_slit(ccd=1)
            name = 'ir_'+str(slit)+'_'+str(exposure_time)+'sec'+'_trace_schem.fits'
        if self.overw:
            self.simulator(ccd=1, source=source, name=name, exposure_time=exposure_time)
        else:
            if name not in self.files_in_output:
                self.simulator(ccd=1, source=source, name=name, exposure_time=exposure_time)
        self.fiber = save_old_fiber

    
    '''
    Sky functions
    '''


    def get_sky_source(self):
        '''
        Gets the sky source.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        skyline_source: CSV
            The sky source.
        '''
        skyline_source = CSV(filepath="data/line_files/skycalc_radiance.dat", \
                name="", list_like=True, wavelength_unit='nm', flux_in_photons=False, delimiter=' ',\
                    scale_now=False)
        # Right now the input file is in ph/s/m^2/micron/arcsec^2 so we have to convert it to
        # erg/s/cm^2/cm
        # First we find the area of the slit.
        slit = self.find_slit(ccd=3)
        if slit[:3] == 'pin':
            slit_w = float(slit[7:])
            slit_area = np.pi*(slit_w/2)**2
        elif slit[:3] == 'sli':
            slit_w = float(slit[9:])
            slit_l = 22.8
            slit_area = slit_w*slit_l

        skyline_source.flux_data *= slit_area
        # Now the unit is ph/s/m^2/micron so we have to convert it to erg/s/cm^2/cm
        # Firstly convert the photons to ergs
        wl = skyline_source.wl_data * u.micron # in microns
        wl = wl.to(u.cm).value
        # Energy for photon at wl in ergs
        planck_cgs = 6.6260755e-27
        c_cgs = 2.99792458e10
        energy = planck_cgs*c_cgs/wl
        # Now convert the photons to ergs
        skyline_source.flux_data *= energy
        # Now the unit is erg/s/m^2/micron so we have to convert it to erg/s/cm^2/cm
        f = skyline_source.flux_data * u.erg/u.s/u.m**2/u.micron
        f = f.to(u.erg/u.s/u.cm**2/u.cm).value
        skyline_source.flux_data = f

        skyline_source.flux_scale()

        return skyline_source

    
    def uv_make_sky(self, exposure_time=233, name=False, nr=1):
        '''
        Makes a sky for the UV ccd.
        
        Parameters
        ----------
        exposure_time : float
            The exposure time of the simulation.

        name : str
            The name of the file to be saved.
        
        Returns
        -------
        It makes a fits file in the output directory.
        '''
        source = self.get_sky_source()
        for i in range(nr):
            if not name or nr > 1:
                slit = self.find_slit(ccd=3)
                name = 'uv_'+str(slit)+'_'+str(exposure_time)+'sec'+'_'+str(i)+'_sky_schem.fits'
            if self.overw:
                self.simulator(ccd=3, source=source, name=name, exposure_time=exposure_time)
            else:
                if name not in self.files_in_output:
                    self.simulator(ccd=3, source=source, name=name, exposure_time=exposure_time)    

        
    def vis_make_sky(self, exposure_time=3, name=False, nr=1):
        '''
        Makes a sky for the VIS ccd.
        
        Parameters
        ----------
        exposure_time : float
            The exposure time of the simulation.

        name : str
            The name of the file to be saved.
        
        Returns
        -------
        It makes a fits file in the output directory.
        '''
        source = self.get_sky_source()
        for i in range(nr):
            if not name or nr > 1:
                slit = self.find_slit(ccd=2)
                name = 'vis_'+str(slit)+'_'+str(exposure_time)+'sec'+'_'+str(i)+'_sky_schem.fits'
            if self.overw:
                self.simulator(ccd=2, source=source, name=name, exposure_time=exposure_time)
            else:
                if name not in self.files_in_output:
                    self.simulator(ccd=2, source=source, name=name, exposure_time=exposure_time)


    def ir_make_sky(self, exposure_time=0.5, name=False, nr=1):
        '''
        Makes a sky for the IR ccd.
        
        Parameters
        ----------
        exposure_time : float
            The exposure time of the simulation.

        name : str
            The name of the file to be saved.
        
        Returns
        -------
        It makes a fits file in the output directory.
        '''
        source = self.get_sky_source()
        for i in range(nr):
            if not name or nr > 1:
                slit = self.find_slit(ccd=1)
                name = 'ir_'+str(slit)+'_'+str(exposure_time)+'sec'+'_'+str(i)+'_sky_schem.fits'
            if self.overw:
                self.simulator(ccd=1, source=source, name=name, exposure_time=exposure_time)
            else:
                if name not in self.files_in_output:
                    self.simulator(ccd=1, source=source, name=name, exposure_time=exposure_time)


    '''
    GRB25 functions
    '''


    def get_grb25_source(self):
        '''
        Gets the GRB25 source.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        grb25_source: CSV
            The GRB25 source.
        '''
        grb25_source = CSV(filepath="data/scienceframe_data/GRB/extended_GRBspec_2.5_withabs.dat", \
                name="", list_like=True, wavelength_unit='a', flux_in_photons=False, delimiter=' ')
        return grb25_source


    def uv_make_grb25(self, exposure_time=5, name=False, seeing=0.8, posx=0.5, posy=0.5):
        '''
        Makes a GRB25 for the UV ccd.
        
        Parameters
        ----------
        exposure_time : float
            The exposure time of the simulation.

        name : str
            The name of the file to be saved.

        seeing : float
            The seeing of the simulation.

        posx : float
            The x position of the source.

        posy : float
            The y position of the source.
        
        Returns
        -------
        It makes a fits file in the output directory.
        '''
        source = self.get_grb25_source()
        if not name:
            slit = self.find_slit(ccd=3)
            name = 'uv_'+str(slit)+'_'+str(exposure_time)+'sec'+'_grb25_schem.fits'
        if self.overw:
            self.point_simulator(ccd=3, source=source, name=name, exposure_time=exposure_time, posx=posx, posy=posy, seeing=seeing)
        else:
            if name not in self.files_in_output:
                self.point_simulator(ccd=3, source=source, name=name, exposure_time=exposure_time, posx=posx, posy=posy, seeing=seeing)


    def vis_make_grb25(self, exposure_time=3, name=False, seeing=0.8, posx=0.5, posy=0.5):
        '''
        Makes a GRB25 for the VIS ccd.
        
        Parameters
        ----------
        exposure_time : float
            The exposure time of the simulation.

        name : str
            The name of the file to be saved.

        seeing : float
            The seeing of the simulation.

        posx : float
            The x position of the source.

        posy : float
            The y position of the source.
        
        Returns
        -------
        It makes a fits file in the output directory.
        '''
        source = self.get_grb25_source()
        if not name:
            slit = self.find_slit(ccd=2)
            name = 'vis_'+str(slit)+'_'+str(exposure_time)+'sec'+'_grb25_schem.fits'
        if self.overw:
            self.point_simulator(ccd=2, source=source, name=name, exposure_time=exposure_time, posx=posx, posy=posy, seeing=seeing)
        else:
            if name not in self.files_in_output:
                self.point_simulator(ccd=2, source=source, name=name, exposure_time=exposure_time, posx=posx, posy=posy, seeing=seeing)


    def ir_make_grb25(self, exposure_time=1, name=False, seeing=0.8, posx=0.5, posy=0.5):
        '''
        Makes a GRB25 for the IR ccd.
        
        Parameters
        ----------
        exposure_time : float
            The exposure time of the simulation.

        name : str
            The name of the file to be saved.

        seeing : float
            The seeing of the simulation.

        posx : float
            The x position of the source.

        posy : float
            The y position of the source.
        
        Returns
        -------
        It makes a fits file in the output directory.
        '''
        source = self.get_grb25_source()
        if not name:
            slit = self.find_slit(ccd=1)
            name = 'ir_'+str(slit)+'_'+str(exposure_time)+'sec'+'_grb25_schem.fits'
        if self.overw:
            self.point_simulator(ccd=1, source=source, name=name, exposure_time=exposure_time, posx=posx, posy=posy, seeing=seeing)
        else:
            if name not in self.files_in_output:
                self.point_simulator(ccd=1, source=source, name=name, exposure_time=exposure_time, posx=posx, posy=posy, seeing=seeing)

    
    '''
    Custom functions
    '''


    def make_custom(self, ccd, exposure_time, source, name, point=False, posx=0.5, posy=0.5, seeing=0.8):
        '''
        Makes a simulation for a custom source.

        Parameters
        ----------
        ccd : int
            The ccd to simulate - 1 for IR, 2 for VIS, 3 for UV.

        exposure_time : float
            The exposure time of the simulation.

        source : CSV
            The source to simulate.

        name : str
            The name of the file to be saved.

        point : bool
            If the source is a point source or not.
        
        posx : float
            The x position of the source. Only used if point=True.

        posy : float
            The y position of the source. Only used if point=True.

        seeing : float
            The seeing of the simulation. Only used if point=True.

        Returns
        -------
        It makes a fits file in the output directory.
        '''
        if self.overw:
            if point:
                self.point_simulator(ccd, source, name, exposure_time, posx, posy, seeing)
            else:
                self.simulator(ccd, source, name, exposure_time)
        else:
            if name not in self.files_in_output:
                if point:
                    self.point_simulator(ccd, source, name, exposure_time, posx, posy, seeing)
                else:
                    self.simulator(ccd, source, name, exposure_time)


    '''
    AB function for grb25
    '''


    def uv_make_ab(self, exposure_time=5, name=False, source='grb25', possx_A=False, possy_A=False, possx_B=False, possy_B=False, seeing=0.8):
        '''
        Makes a AB setup for the UV ccd.

        Parameters
        ----------
        exposure_time : float
            The exposure time of the simulation.

        name : str
            The name of the file to be saved. 'A' and 'B' will be added to the name. Also a number to indicate number of simulation.

        source : NTEpyechelle source
            The source to simulate.

        possx_A : list
            The x positions of the sources in the A slit.

        possy_A : list
            The y positions of the sources in the A slit.

        possx_B : list
            The x positions of the sources in the B slit.

        possy_B : list
            The y positions of the sources in the B slit.

        seeing : float
            The seeing of the simulation.

        Returns
        -------
        It makes a fits file in the output directory.
        '''
        if source == 'grb25':
            source = self.get_grb25_source()
        if not np.sum([possx_A, possy_A, possx_B, possy_B]):
            possx_A = self.Axs
            possy_A = self.Ays
            possx_B = self.Bxs
            possy_B = self.Bys
        for i in range(len(possx_A)):
            if not name:
                slit = self.find_slit(ccd=3)
                A_name = 'uv_'+str(slit)+'_A_'+str(i)+'_'+str(exposure_time)+'sec'+'_grb25_schem.fits'
                B_name = 'uv_'+str(slit)+'_B_'+str(i)+'_'+str(exposure_time)+'sec'+'_grb25_schem.fits'
            else:
                A_name = name.replace('.fits', '')+'_A_'+str(i)+'.fits'
                B_name = name.replace('.fits', '')+'_B_'+str(i)+'.fits'
            if self.overw:
                Ax = possx_A[i]
                Ay = possy_A[i]
                Bx = possx_B[i]
                By = possy_B[i]
                self.point_simulator(ccd=3, source=source, name=A_name, exposure_time=exposure_time, posx=Ax, posy=Ay, seeing=seeing)
                self.point_simulator(ccd=3, source=source, name=B_name, exposure_time=exposure_time, posx=Bx, posy=By, seeing=seeing)
            else:
                if A_name not in self.files_in_output:
                    Ax = possx_A[i]
                    Ay = possy_A[i]
                    self.point_simulator(ccd=3, source=source, name=A_name, exposure_time=exposure_time, posx=Ax, posy=Ay, seeing=seeing)
                if B_name not in self.files_in_output:
                    Bx = possx_B[i]
                    By = possy_B[i]
                    self.point_simulator(ccd=3, source=source, name=B_name, exposure_time=exposure_time, posx=Bx, posy=By, seeing=seeing)

    
    def vis_make_ab(self, exposure_time=5, name=False, source='grb25', possx_A=False, possy_A=False, possx_B=False, possy_B=False, seeing=0.8):
        '''
        Makes a AB setup for the VIS ccd.

        Parameters
        ----------
        exposure_time : float
            The exposure time of the simulation.

        name : str
            The name of the file to be saved. 'A' and 'B' will be added to the name. Also a number to indicate number of simulation.

        source : NTEpyechelle source
            The source to simulate.

        possx_A : list
            The x positions of the sources in the A slit.

        possy_A : list
            The y positions of the sources in the A slit.

        possx_B : list
            The x positions of the sources in the B slit.

        possy_B : list
            The y positions of the sources in the B slit.

        Returns
        -------
        It makes a fits file in the output directory.
        '''
        if source == 'grb25':
            source = self.get_grb25_source()
        if not np.sum([possx_A, possy_A, possx_B, possy_B]):
            possx_A = self.Axs
            possy_A = self.Ays
            possx_B = self.Bxs
            possy_B = self.Bys
        for i in range(len(possx_A)):
            if not name:
                slit = self.find_slit(ccd=2)
                A_name = 'vis_'+str(slit)+'_A_'+str(i)+'_'+str(exposure_time)+'sec'+'_grb25_schem.fits'
                B_name = 'vis_'+str(slit)+'_B_'+str(i)+'_'+str(exposure_time)+'sec'+'_grb25_schem.fits'
            else:
                A_name = name.replace('.fits', '')+'_A_'+str(i)+'.fits'
                B_name = name.replace('.fits', '')+'_B_'+str(i)+'.fits'
            if self.overw:
                Ax = possx_A[i]
                Ay = possy_A[i]
                Bx = possx_B[i]
                By = possy_B[i]
                self.point_simulator(ccd=2, source=source, name=A_name, exposure_time=exposure_time, posx=Ax, posy=Ay, seeing=seeing)
                self.point_simulator(ccd=2, source=source, name=B_name, exposure_time=exposure_time, posx=Bx, posy=By, seeing=seeing)
            else:
                if A_name not in self.files_in_output:
                    Ax = possx_A[i]
                    Ay = possy_A[i]
                    self.point_simulator(ccd=2, source=source, name=A_name, exposure_time=exposure_time, posx=Ax, posy=Ay, seeing=seeing)
                if B_name not in self.files_in_output:
                    Bx = possx_B[i]
                    By = possy_B[i]
                    self.point_simulator(ccd=2, source=source, name=B_name, exposure_time=exposure_time, posx=Bx, posy=By, seeing=seeing)


    def ir_make_ab(self, exposure_time=5, name=False, source='grb25', possx_A=False, possy_A=False, possx_B=False, possy_B=False, seeing=0.8):
        '''
        Makes a AB setup for the IR ccd.
        
        Parameters
        ----------
        exposure_time : float
            The exposure time of the simulation.

        name : str
            The name of the file to be saved. 'A' and 'B' will be added to the name. Also a number to indicate number of simulation.

        source : NTEpyechelle source
            The source to simulate.

        possx_A : list
            The x positions of the sources in the A slit.

        possy_A : list
            The y positions of the sources in the A slit.

        possx_B : list
            The x positions of the sources in the B slit.

        possy_B : list
            The y positions of the sources in the B slit.

        Returns
        -------
        It makes a fits file in the output directory.
        '''
        if source == 'grb25':
            source = self.get_grb25_source()
        if not np.sum([possx_A, possy_A, possx_B, possy_B]):
            possx_A = self.Axs
            possy_A = self.Ays
            possx_B = self.Bxs
            possy_B = self.Bys
        for i in range(len(possx_A)):
            if not name:
                slit = self.find_slit(ccd=1)
                A_name = 'ir_'+str(slit)+'_A_'+str(i)+'_'+str(exposure_time)+'sec'+'_grb25_schem.fits'
                B_name = 'ir_'+str(slit)+'_B_'+str(i)+'_'+str(exposure_time)+'sec'+'_grb25_schem.fits'
            else:
                A_name = name.replace('.fits', '')+'_A_'+str(i)+'.fits'
                B_name = name.replace('.fits', '')+'_B_'+str(i)+'.fits'
            if self.overw:
                Ax = possx_A[i]
                Ay = possy_A[i]
                Bx = possx_B[i]
                By = possy_B[i]
                self.point_simulator(ccd=1, source=source, name=A_name, exposure_time=exposure_time, posx=Ax, posy=Ay, seeing=seeing)
                self.point_simulator(ccd=1, source=source, name=B_name, exposure_time=exposure_time, posx=Bx, posy=By, seeing=seeing)
            else:
                if A_name not in self.files_in_output:
                    Ax = possx_A[i]
                    Ay = possy_A[i]
                    self.point_simulator(ccd=1, source=source, name=A_name, exposure_time=exposure_time, posx=Ax, posy=Ay, seeing=seeing)
                if B_name not in self.files_in_output:
                    Bx = possx_B[i]
                    By = possy_B[i]
                    self.point_simulator(ccd=1, source=source, name=B_name, exposure_time=exposure_time, posx=Bx, posy=By, seeing=seeing)
            
    
    '''
    All functions
    '''


    def make_all(self, with_grb25 = True):
        '''
        Makes all possible default simulations

        Parameters
        ----------
        with_grb25 : bool
            If True, it will make the grb25 simulations.

        Returns
        -------
        It makes fits files in the output directory.
        '''
        self.uv_make_bias()
        self.vis_make_bias()
        self.ir_make_dark()
        self.uv_make_flat()
        self.vis_make_flat()
        self.ir_make_flat()
        self.uv_make_HgAr()
        self.vis_make_HgAr()
        self.ir_make_HgAr()
        self.uv_make_etalon()
        self.vis_make_etalon()
        self.ir_make_etalon()
        self.uv_make_trace()
        self.vis_make_trace()
        self.ir_make_trace()
        self.uv_make_sky(nr=5)
        self.vis_make_sky(nr=5)
        self.ir_make_sky(nr=5)
        if with_grb25:
            self.uv_make_grb25()
            self.vis_make_grb25()
            self.ir_make_grb25()
            self.uv_make_ab()
            self.vis_make_ab()
            self.ir_make_ab()


    def make_all_uv(self, with_grb25 = True):
        '''
        Makes all possible default simulations for the UV
        
        Parameters
        ----------
        with_grb25 : bool
            If True, it will make the grb25 simulations.
        
        Returns
        -------
        It makes fits files in the output directory.
        '''
        self.uv_make_bias()
        self.uv_make_flat()
        self.uv_make_HgAr()
        self.uv_make_etalon()
        self.uv_make_trace()
        self.uv_make_sky(nr=5)
        if with_grb25:
            self.uv_make_grb25()
            self.uv_make_ab()
    

    def make_all_vis(self, with_grb25 = True):
        '''
        Makes all possible default simulations for the VIS
        
        Parameters
        ----------
        with_grb25 : bool
            If True, it will make the grb25 simulations.
        
        Returns
        -------
        It makes fits files in the output directory.
        '''
        self.vis_make_bias()
        self.vis_make_flat()
        self.vis_make_HgAr()
        self.vis_make_etalon()
        self.vis_make_trace()
        self.vis_make_sky(nr=5)
        if with_grb25:
            self.vis_make_grb25()
            self.vis_make_ab()
    

    def make_all_ir(self, with_grb25 = True):
        '''
        Makes all possible default simulations for the IR
        
        Parameters
        ----------
        with_grb25 : bool
            If True, it will make the grb25 simulations.
        
        Returns
        -------
        It makes fits files in the output directory.
        '''
        self.ir_make_dark()
        self.ir_make_flat()
        self.ir_make_HgAr()
        self.ir_make_etalon()
        self.ir_make_trace()
        self.ir_make_sky(nr=5)
        if with_grb25:
            self.ir_make_grb25()
            self.ir_make_ab()