import math
import random

import numba.cuda
import numpy as np

from NTEpyechelle.CCD import CCD
from NTEpyechelle.optics import convert_matrix
from NTEpyechelle.randomgen import make_alias_sampling_arrays, unravel_index
from NTEpyechelle.sources import Source
from NTEpyechelle.spectrograph import Spectrograph
from NTEpyechelle.telescope import Telescope

#import matplotlib.pyplot as plt


@numba.njit(cache=True, parallel=False, nogil=True)
def raytrace(spectrum_wl, spectrum_q, spectrum_j, transformations, trans_wl, trans_wld, transf_deriv,
             psfs_q, psfs_j, psf_wl, psf_wld, psf_shape, psf_sampling, ccd, pixelsize, slitfun, nphotons, sigma_x, sigma_y, pos_x, pos_y):
    max_y, max_x = ccd.shape

    # x_ints=[]
    # y_ints=[]
    # x_ints_after=[]
    # y_ints_after=[]
    # psfx=[]
    # psfy=[]
    # wls=[]
    for _ in range(nphotons):
        # sample from spectrum
        k = int(math.floor(random.random() * len(spectrum_j)))
        wl = spectrum_wl[k] if random.random() < spectrum_q[k] else spectrum_wl[spectrum_j[k]]

        # find index for transformation
        idx_trans_float = (wl - trans_wl[0]) / trans_wld
        idx_trans = math.floor(idx_trans_float)
        r = (idx_trans_float - idx_trans)

        # do linear interpolation of transformation matrices
        m0, m1, m2, m3, m4, m5 = transformations[:, idx_trans] + r * transf_deriv[:, idx_trans]

        # random start points in slit
        if sigma_x==-9:
            x, y = slitfun(random.random(), random.random())
        else:
            x_gauss, y_gauss = random.gauss(pos_x,sigma_x), random.gauss(pos_y,sigma_y)
            if x_gauss < 0 or x_gauss > 1 or y_gauss < 0 or y_gauss > 1:
                continue
            x, y = slitfun(x_gauss, y_gauss) # for point source
        # transform
        xt = m0 * x + m1 * y + m2
        yt = m3 * x + m4 * y + m5
        # xt0 = m0 * x + m1 * y + m2
        # yt0 = m3 * x + m4 * y + m5

        # apply PSF
        idx_psf = int((wl - psf_wl[0]) / psf_wld)  # find psf index
        # next 3 lines implement drawing random number via alias sampling
        k = int(math.floor(random.random() * len(psfs_j[idx_psf])))
        if not random.random() < psfs_q[idx_psf][k]:
            k = psfs_j[idx_psf][k]

        dx, dy = unravel_index(k, psf_shape)
        xt += (dx - psf_shape[1] / 2.) * psf_sampling / pixelsize
        yt += (dy - psf_shape[0] / 2.) * psf_sampling / pixelsize
            
        x_int = math.floor(xt)
        y_int = math.floor(yt)


        if (0 <= x_int < max_x) and (0 <= y_int < max_y):
            ccd[y_int, x_int] += 1
    #         x_ints.append(xt0)
    #         y_ints.append(yt0)
    #         x_ints_after.append(xt)
    #         y_ints_after.append(yt)
    #         psfx.append(dx)
    #         psfy.append(dy)
    #         wls.append(wl)

    # return x_ints, y_ints, wls, psfx, psfy, x_ints_after, y_ints_after


def raytrace_order_cpu(o, spec: Spectrograph, source: Source, slit_fun: callable,
                       telescope: Telescope, rv: float, t, ccd: CCD, fiber: int, ccd_index: int, sigma, pos, efficiency=None,
                       n_cpu=1):
    wavelength = np.linspace(*spec.get_wavelength_range(o, fiber, ccd_index), num=100000)
    sigma_x, sigma_y = sigma[0], sigma[1]
    pos_x, pos_y = pos[0], pos[1]

    # get spectral density per order
    spectral_density = source.get_spectral_density_rv(wavelength, rv)
    # if source returns own wavelength vector, use that for further calculations instead of default grid
    if isinstance(spectral_density, tuple):
        wavelength, spectral_density = spectral_density

    # for stellar targets calculate collected flux by telescope area
    if source.stellar_target:
        spectral_density *= telescope.area

    # get efficiency per order
    if efficiency is not None:
        eff = efficiency.get_efficiency_per_order(wavelength=wavelength, order=o)
        effective_density = eff * spectral_density
    else:
        effective_density = spectral_density

    # calculate photon flux
    if source.flux_in_photons:
        flux = effective_density
    else:
        ch_factor = 5.03E12  # convert microwatts / micrometer to photons / s per wavelength interval
        wl_diffs = np.ediff1d(wavelength, wavelength[-1] - wavelength[-2])
        flux = effective_density * wavelength * wl_diffs * ch_factor

    #plt.plot(flux)
    flux_photons = flux * t
    total_photons = int(np.sum(flux_photons))
    print(f'Order {o:3d}:    {(np.min(wavelength) * 1000.):7.1f} - {(np.max(wavelength) * 1000.):7.1f} nm.     '
          f'Number of photons: {total_photons}')

    minwl, maxwl = spec.get_wavelength_range(o, fiber, ccd_index)
    trans_wl, trans_wld = np.linspace(minwl, maxwl, 10000, retstep=True)
    transformations = convert_matrix(np.array(spec.get_transformation(trans_wl, o, fiber, ccd_index)))

    # transformations = np.array(spec.transformations[f'order{o}'].get_matrices_spline(trans_wl))
    # derivatives for simple linear interpolation
    trans_deriv = np.array([np.ediff1d(t, t[-1] - t[-2]) for t in transformations])

    psf_sampler_qj = np.array(
        [make_alias_sampling_arrays(p.data.T.ravel()) for p in spec.get_psf(None, o, fiber, ccd_index)])

    psfs_wl = np.array([p.wavelength for p in spec.get_psf(None, o, fiber, ccd_index)])
    psfs_wld = np.ediff1d(psfs_wl, psfs_wl[-1] - psfs_wl[-2])
    psf_shape = spec.get_psf(None, o, fiber, ccd_index)[0].data.shape

    #plt.figure()
    #plt.plot(np.asarray(flux_photons / np.sum(flux_photons),dtype=np.float32))
    spectrum_sampler_q, spectrum_sampler_j = make_alias_sampling_arrays(np.asarray(flux_photons / np.sum(flux_photons),
                                                                                   dtype=np.float32))
    # plt.figure()
    # plt.plot(spectrum_sampler_q)
    # plt.figure()
    # plt.plot(spectrum_sampler_j)
    # plt.show()
    

    psf_sampling = spec.get_psf(None, o, fiber, ccd_index)[0].sampling
    if n_cpu > 1:
        ccd_new = np.zeros_like(ccd.data, dtype=np.uint32)
        raytrace(wavelength, spectrum_sampler_q, spectrum_sampler_j,
                 transformations, trans_wl, trans_wld, trans_deriv,
                 psf_sampler_qj[:, 0], psf_sampler_qj[:, 1], psfs_wl, psfs_wld[0], psf_shape, psf_sampling,
                 ccd_new, float(ccd.pixelsize), slit_fun, total_photons, sigma_x, sigma_y, pos_x, pos_y)
        return ccd_new, total_photons
    else:
        raytrace(wavelength, spectrum_sampler_q, spectrum_sampler_j,
                 transformations, trans_wl, trans_wld, trans_deriv,
                 psf_sampler_qj[:, 0], psf_sampler_qj[:, 1], psfs_wl, psfs_wld[0], psf_shape, psf_sampling,
                 ccd.data, float(ccd.pixelsize), slit_fun, total_photons, sigma_x, sigma_y, pos_x, pos_y)

        # plt.figure()
        # plt.title('x_ints')
        # xbins = int(max(x_ints) - min(x_ints))
        # plt.hist2d(wls, x_ints, bins=xbins)
        # plt.figure()
        # plt.title('y_ints')
        # ybins = int(max(y_ints) - min(y_ints))
        # plt.hist2d(wls, y_ints, bins=ybins)
        # plt.figure()
        # plt.title('x_ints_after')
        # xbins = int(max(x_ints_after) - min(x_ints_after))
        # plt.hist2d(wls, x_ints_after, bins=xbins)
        # plt.figure()
        # plt.title('y_ints_after')
        # ybins = int(max(y_ints_after) - min(y_ints_after))
        # plt.hist2d(wls, y_ints_after, bins=ybins)
        # plt.figure()
        # plt.title('psfx')
        # psfxbins = int(max(psfx) - min(psfx))
        # plt.hist2d(wls, psfx, bins=psfxbins)
        # plt.figure()
        # psfybins = int(max(psfy) - min(psfy))
        # plt.hist2d(wls, psfy, bins=psfybins)
        
        
        # plt.show()
        return total_photons
    
