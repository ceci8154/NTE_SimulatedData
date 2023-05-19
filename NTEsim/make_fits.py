'''
Written by Cecilie and Mads.
This script is used to make fits files from the schematics.

You can choose the files you want to make based on keywords, or use one of the preset defaults.
'''
import numpy as np
import os
from astropy.io import fits
import zipfile

class MakeFits:
    '''
    This class is used to make fits files from the schematics.
    It can be used to make fits files for a specific keyword, or use one of the preset defaults.
    '''
    def __init__(self):
        self.schematic_dir = 'schematics/'
        self.output_dir = 'Output/'
        self.zip_files_names = [f for f in os.listdir(self.schematic_dir) if f.endswith('.zip')]
        ir_bias_std_dir = 'data/noise_ir'
        vis_bias_std_dir = 'data/noise_vis'
        uv_bias_std_dir = 'data/noise_uvb'
        with fits.open(ir_bias_std_dir + '/comb_noise.fits') as file:
            self.ir_bias = file[0].data
            self.ir_std = np.array(file[1].data, dtype=np.float32)
        with fits.open(vis_bias_std_dir + '/comb_noise.fits') as file:
            self.vis_bias = file[0].data
            self.vis_std = np.array(file[1].data, dtype=np.float32)
        with fits.open(uv_bias_std_dir + '/comb_noise.fits') as file:
            self.uv_bias = file[0].data
            self.uv_std = np.array(file[1].data, dtype=np.float32)
        ir_flat_dir = 'data/flat_ir'
        vis_flat_dir = 'data/flat_vis'
        uv_flat_dir = 'data/flat_uvb'
        self.ir_flat = fits.getdata(ir_flat_dir + '/norm_flat.fits')
        self.vis_flat = fits.getdata(vis_flat_dir + '/norm_flat.fits')
        self.uv_flat = fits.getdata(uv_flat_dir + '/norm_flat.fits')

        self.cosmic_rays = False


    def check_output_dir(self):
        '''
        Check if the output directiory exists, if not it makes it.
        '''
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)


    def unzip_schematics(self):
        '''
        Unzips all the schematic zip files.
        '''
        for zip_file_name in self.zip_files_names:
            with zipfile.ZipFile(self.schematic_dir + zip_file_name, 'r') as zip_ref:
                zip_ref.extractall(self.schematic_dir)


    def delete_schematics(self):
        '''
        Deletes all fits files in the schematic folder.
        '''
        self.files_in_schematics = [f for f in os.listdir(self.schematic_dir) if f.endswith('.fits')]
        for file in self.files_in_schematics:
            os.remove(self.schematic_dir + file)


    def add_sky_template(self, filename, data):
        '''
        Adds the sky to the data.
        
        Parameters
        ----------
        filename : str
            The name of the file.
            
        data : numpy.ndarray
            The data to add the sky to.

        Returns
        -------
        data : numpy.ndarray
            The data with the sky added.
        '''

        # Find all schematics with sky in the name
        self.files_in_schematics = [f for f in os.listdir(self.schematic_dir) if f.endswith('.fits')]
        sky_files = [f for f in self.files_in_schematics if 'sky' in f]
        # Split the filename into a list by '_'
        filename_split = filename.split('_')
        ccd = filename_split[0]
        slit = filename_split[1]
        sec_split = [s for s in filename_split if 'sec' in s][0]
        exposure = float(sec_split[:-3])
        # Find all sky files that match ccd and slit
        sky_files = [f for f in sky_files if ccd in f and slit in f]
        # Choose random sky file
        sky_file = np.random.choice(sky_files)
        # Open sky file
        with fits.open(self.schematic_dir + sky_file) as hdul:
            sky_data = hdul[0].data
            sky_data = sky_data.astype(np.float32)
            # Get the exposure time of the sky file
            sky_exposure = float(sky_file.split('_')[2][:-3])
            # Then normalize the sky data to have the same exposure time as the data
            sky_data = sky_data/sky_exposure*exposure
            data += sky_data
        return data


    def add_bias_and_noise(self, filename, add_sky, total_scale):
        '''
        Adds the bias and noise to the data from the filename
        
        Parameters
        ----------
        filename : str
            The name of the file.
   
        add_sky : bool
            If the sky should be added.
            
        Returns
        -------
        data : numpy.ndarray
            The data with the bias and noise added. (and sky if True)
        '''
        with fits.open(self.schematic_dir + filename) as hdul:
            data = hdul[0].data
            data = data.astype(np.float32)
            if 'ir' in filename:
                if add_sky:
                    data = self.add_sky_template(filename, data)
                data *= total_scale
                data = data * self.ir_flat + self.ir_bias + 1000 # TODO this 1000 needs to be removed with final bias files
                data = np.random.normal(data, self.ir_std)
                data = np.clip(data, 0, None)
                if self.cosmic_rays and 'bias' not in filename and 'dark' not in filename:
                    print("Doing cosmic rays for "+ filename)
                    import pyxel
                    config = pyxel.load("NTEsim/h2rg_cosmic_only.yaml")
                    filename_split = filename.split('_')
                    sec_split = [s for s in filename_split if 'sec' in s][0]
                    exp = float(sec_split[:-3])
                    
                    config.exposure.readout.times = [exp]

                    exposure = config.exposure
                    detector = config.detector
                    pipeline = config.pipeline


                    result = pyxel.exposure_mode(
                        exposure=exposure,
                        detector=detector,
                        pipeline=pipeline,
                    )
                    result

                    n = result["image"].sel().to_numpy()
                    n[n<1000] = 0

                    if np.max(n) > 0:
                        n = n/np.max(n)*np.max(data)
                        
                    data += n[0,:,:]

            elif 'vis' in filename:
                if add_sky:
                    data = self.add_sky_template(filename, data)
                data *= total_scale
                data = data * self.vis_flat + self.vis_bias + 1000 # TODO this 1000 needs to be removed with final bias files
                data = np.random.normal(data, self.vis_std)
                data = np.clip(data, 0, None)
                if self.cosmic_rays and 'bias' not in filename and 'dark' not in filename:
                    print("Doing cosmic rays for "+ filename)
                    import pyxel
                    config = pyxel.load("NTEsim/skip_ccd_cosmic_only.yaml")
                    filename_split = filename.split('_')
                    sec_split = [s for s in filename_split if 'sec' in s][0]
                    exp = float(sec_split[:-3])
                    
                    config.exposure.readout.times = [exp]

                    exposure = config.exposure
                    detector = config.detector
                    pipeline = config.pipeline


                    result = pyxel.exposure_mode(
                        exposure=exposure,
                        detector=detector,
                        pipeline=pipeline,
                    )
                    result

                    n = result["image"].sel().to_numpy()
                    n[n<3278] = 0

                    if np.max(n) > 0:
                        n = n/np.max(n)*np.max(data)

                    data += n[0,:,:]

            elif 'uv' in filename:
                if add_sky:
                    data = self.add_sky_template(filename, data)
                data *= total_scale
                data = data * self.uv_flat + self.uv_bias + 1000 # TODO this 1000 needs to be removed with final bias files
                data = np.random.normal(data, self.uv_std)
                data = np.clip(data, 0, None)
                if self.cosmic_rays and 'bias' not in filename and 'dark' not in filename:
                    print("Doing cosmic rays for "+ filename)
                    import pyxel
                    config = pyxel.load("NTEsim/em_ccd_cosmic_only.yaml")
                    filename_split = filename.split('_')
                    sec_split = [s for s in filename_split if 'sec' in s][0]
                    exp = float(sec_split[:-3])
                    
                    config.exposure.readout.times = [exp]

                    exposure = config.exposure
                    detector = config.detector
                    pipeline = config.pipeline


                    result = pyxel.exposure_mode(
                        exposure=exposure,
                        detector=detector,
                        pipeline=pipeline,
                    )
                    result

                    n = result["image"].sel().to_numpy()
                    n[n<3278] = 0

                    if np.max(n) > 0:
                        n = n/np.max(n)*np.max(data)

                    data += n[0,:,:]

            return data

    def add_bias_and_noise_from_data(self, data, filename):
        '''
        Adds the bias and noise to the data from the data
        
        Parameters
        ----------
        data : numpy.ndarray
            The data to add the bias and noise to.

        filename : str
            The name of the file.
            
        Returns
        -------
        data : numpy.ndarray
            The data with the bias and noise added.
        '''
        if 'ir' in filename:
            data = data * self.ir_flat + self.ir_bias + 1000 # TODO this 1000 needs to be removed with final bias files
            data = np.random.normal(data, self.ir_std)
            data = np.clip(data, 0, None)
        elif 'vis' in filename:
            data = data * self.vis_flat + self.vis_bias + 1000 # TODO this 1000 needs to be removed with final bias files
            data = np.random.normal(data, self.vis_std)
            data = np.clip(data, 0, None)
        elif 'uv' in filename:
            data = data * self.uv_flat + self.uv_bias + 1000 # TODO this 1000 needs to be removed with final bias files
            data = np.random.normal(data, self.uv_std)
            data = np.clip(data, 0, None)
        
        return data
    
    def add_schems_and_add_noise(self, list_of_names, list_of_scales, output_name):
        '''
        Adds the fits together for a combined fits file, and adds bias and noise.

        Parameters
        ----------
        list_of_names : list
            A list of the names of the fits files to add together.
        
        list_of_scales : list
            A list of the scales to multiply the fits files with. (In case the exposuretimes are different)

        Returns
        -------
        It makes a fits file in the output directory.
        '''
        self.check_output_dir()
        # Open the first file
        with fits.open(self.schematic_dir + list_of_names[0]) as hdul:
            data = hdul[0].data 
            data = data.astype(np.float32)
            data *= list_of_scales[0]
            # Add the rest of the files
            for i in range(1, len(list_of_names)):
                with fits.open(self.schematic_dir + list_of_names[i]) as hdul:
                    data += hdul[0].data * list_of_scales[i]
        # Add bias and noise
        data = self.add_bias_and_noise_from_data(data, list_of_names[0])
        # Save the fits file
        fits.writeto(self.output_dir + output_name, data, overwrite=True)


    def add_bias_and_noise_for_keyword(self, keywords, nr=1, add_sky=False, total_scale=1):
        '''
        Adds the bias and noise to all data with the keyword in the filename.
        
        Parameters
        ----------
        keyword : str
            The keyword to look for in the filename.
        
        nr : int
            The number of files to create.
            
        add_sky : bool

        Returns
        -------
        It makes fits files in the output directory.
        '''
        self.check_output_dir()
        self.files_in_schematics = [f for f in os.listdir(self.schematic_dir) if f.endswith('.fits')]
        # Now only the files that contain any of the strings in the keywords list
        if isinstance(keywords, str):
            keywords = [keywords]

        self.keyword_files = []
        for file in self.files_in_schematics:
            if any(keyword in file for keyword in keywords):
                self.keyword_files.append(file)

        for i in range(nr):
            for file in self.keyword_files:
                data = self.add_bias_and_noise(file, add_sky, total_scale)
                filename = file.replace('schem', str(i))
                fits.writeto(self.output_dir + filename, data, overwrite=True)

    
    def add_bias_and_noise_for_not_keyword(self, keywords, nr=1, add_sky=False, total_scale=1):
        '''
        Adds the bias and noise to all data without the keyword in the filename.

        Parameters
        ----------
        keyword : str
            The keyword to look for in the filename.
        
        nr : int
            The number of files to create.
        
        add_sky : bool
            If the sky should be added.

        Returns
        -------
        It makes fits files in the output directory.
        '''
        self.check_output_dir()
        self.files_in_schematics = [f for f in os.listdir(self.schematic_dir) if f.endswith('.fits')]
        # Now only the files that contain any of the strings in the keywords list
        if isinstance(keywords, str):
            keywords = [keywords]

        no_keyword_files = []
        for file in self.files_in_schematics:
            if any(keyword in file for keyword in keywords):
                no_keyword_files.append(file)

        self.keyword_files = [file for file in self.files_in_schematics if file not in no_keyword_files]

        for i in range(nr):
            for file in self.keyword_files:
                data = self.add_bias_and_noise(file, add_sky, total_scale)
                filename = file.replace('schem', str(i))
                fits.writeto(self.output_dir + filename, data, overwrite=True)

           
    def add_bias_and_noise_for_all_preset(self, nr=1):
        '''
        Uses all preset values to make a full set of data.

        Parameters
        ----------
        nr : int
            The number of files to create from each schematic.

        Returns
        -------
        It makes fits files in the output directory.
        '''
        self.check_output_dir()
        self.add_bias_and_noise_for_keyword(['standardstar', 'grb25'], nr, add_sky=True)
        self.add_bias_and_noise_for_not_keyword(['standardstar', 'grb25'], nr)