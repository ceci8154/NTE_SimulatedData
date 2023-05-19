""" Spectral sources

Implementing various spectral sources that can be used in pyechelle.

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    import pyechelle.sources as sources
    from pyechelle.simulator import available_sources

    available_sources.remove('CSV')

    fig, ax = plt.subplots(len(available_sources), 1, figsize=(9, len(available_sources) * 2.5), sharex=True)
    fig.suptitle('Supported source functions')
    for i, source_name in enumerate(available_sources):
        wavelength = np.linspace(0.4999, 0.501, 1000, dtype=float)
        source = getattr(sources, source_name)()
        sd = source.get_spectral_density(wavelength)
        if source.list_like:
            if isinstance(sd, tuple):
                ax[i].vlines(sd[0], [0]*len(sd[1]), sd[1])
            else:
                ax[i].vlines(wavelength, [0]*len(sd), sd)
        else:
            if isinstance(sd, tuple):
                ax[i].plot(*sd)
            else:
                ax[i].plot(wavelength, sd)
        ax[i].set_title(source_name)
        ax[i].set_ylabel("")
        ax[i].set_yticks([])
    ax[-1].set_xlabel("Wavelength [microns]")
    plt.tight_layout()
    plt.show()
"""
from __future__ import annotations

import io
import pathlib
import urllib.request

import astropy.io.fits as fits
import astropy.units as u
import numpy as np
import pandas as pd
import scipy.interpolate
from joblib import Memory
from synphot import SourceSpectrum, SpectralElement, BlackBodyNorm1D, units

try:
    from astroquery.nist import Nist
except ImportError:
    Nist = None

path = pathlib.Path(__file__).parent.resolve()
cache_path = path.joinpath('.cache')
# create data directory if it doesn't exist:
pathlib.Path(cache_path).mkdir(parents=False, exist_ok=True)
memory = Memory(cache_path, verbose=0)


@memory.cache
def pull_catalogue_lines(min_wl: float, max_wl: float, catalogue: str = 'Th', wavelength_type: str = 'vacuum'):
    """
    Reads NIST catalogue lines between min_wl and max_wl of catalogue.

    Args:
        min_wl: minimum wavelength bound [micron]
        max_wl: maximum wavelength bound [micron]
        catalogue: catalogue appreciation label, e.g. 'Th', 'Ar', etc.
        wavelength_type: either 'var+air' or 'vacuum'

    Returns:
        (tuple) line catalogue wavelength and relative intensities. wavelength is in [angstrom]
    """
    try:
        table_lines = Nist.query(min_wl * u.micron, max_wl * u.micron, linename=catalogue, output_order='wavelength',
                                 wavelength_type=wavelength_type)[['Ritz', 'Rel.']]
        df = table_lines.filled(0).to_pandas()
        df['Rel.'] = pd.to_numeric(df['Rel.'], downcast='float', errors='coerce')
        df['Ritz'] = pd.to_numeric(df['Ritz'], downcast='float', errors='coerce')
        df.dropna(inplace=True)
        idx = np.logical_and(df['Rel.'] > 0, df['Ritz'] > 0)
        return df['Ritz'].values[idx], df['Rel.'].values[idx]
    except Exception as e:
        print(e)
        print(f"Warning: Couldn't retrieve {catalogue} catalogue data between {min_wl} and {max_wl} micron")
        return np.array([]), np.array([])


def calc_flux_scale(source_wavelength, source_spectral_density, mag):
    # V - band-filter
    v_filter_wl = [0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6,
                   0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7]
    v_filter_tp = [0, 0.03, 0.163, 0.458, 0.78, 0.967, 1, 0.973, 0.898, 0.792, 0.684, 0.574, 0.461,
                   0.359, 0.27, 0.197, 0.135, 0.081, 0.045, 0.025, 0.017, 0.013, 0.009, 0]

    # Reference flux obtained from integration of vega over bessel filter (units are microwatts/m^2*micrometer)
    v_zp = 3.68E-02

    v_filter_interp = scipy.interpolate.interp1d(v_filter_wl, v_filter_tp)

    # get total flux in filter/source range
    lower_wl_limit = max(np.min(source_wavelength), np.min(v_filter_wl))
    upper_wl_limit = min(np.max(source_wavelength), np.max(v_filter_wl))

    idx = np.logical_and(source_wavelength > lower_wl_limit, source_wavelength < upper_wl_limit)

    step = np.ediff1d(source_wavelength[idx], source_wavelength[idx][-1] - source_wavelength[idx][-2])
    total_flux = np.sum(source_spectral_density[idx] * v_filter_interp(source_wavelength[idx]) * step)

    return pow(10, mag / (-2.5)) * v_zp / total_flux


class Source:
    """ A spectral source.

    This class should be subclassed to implement different spectral sources.

    Attributes:
        name (str): name of the source. This will end up in the .fits header.
        min_wl (float): lower wavelength limit [nm] (for normalization purposes)
        max_wl (float): upper wavelength limit [nm] (for normalization purposes)
        list_like (bool): if True, the Source has a bunch of discrete wavelength, rather than a continuous spectral
        density.
        flux_in_photons (bool): if True, get_spectral_density() returns flux in photons rather than micro watts

    """

    def __init__(self, min_wl=599.8, max_wl=600.42, name="", list_like=False, flux_in_photons=False,
                 stellar_target=False):
        self.name = name
        self.min_wl = min_wl
        self.max_wl = max_wl
        self.stellar_target = stellar_target
        self.flux_in_photons = flux_in_photons
        self.list_like = list_like

    def get_spectral_density(self, wavelength):
        raise NotImplementedError()

    def get_spectral_density_rv(self, wavelength, rv=0.):
        c = 299792458.  # m/s
        rv_shifted = wavelength * ((c - rv) / c)

        spec_density = self.get_spectral_density(rv_shifted)
        # first case: source returns own wavelength vector:
        if isinstance(spec_density, tuple):
            wl, sd = spec_density
            return wl * ((c - rv) / c), sd
        else:
            return spec_density


class Constant(Source):
    """ Constant spectral density.

    Implements a constant spectral density with given intensity [microW / microns*s]

    """

    def __init__(self, intensity=0.001, **kwargs):
        super().__init__(**kwargs, name="Constant", list_like=False)
        self.intensity = intensity

    def get_spectral_density(self, wavelength):
        return np.ones_like(wavelength) * self.intensity


class ConstantPhotons(Source):
    """ Constant spectral density.

    Implements a constant photon flux density with given intensity [photons / microns*s]

    """

    def __init__(self, intensity=1000, **kwargs):
        super().__init__(**kwargs, name="ConstantPhotons", list_like=False)
        self.intensity = intensity
        self.flux_in_photons = True
        self.stellar_target = False

    def get_spectral_density(self, wavelength):
        return np.ones_like(wavelength) * self.intensity


class ThAr(Source):
    """ Thorium-Argon lamp

    Implements a Thorium Argon arc-lamp.
    Uses NIST vacuum catalogue wavelength as source.

    Attributes:
         scale (float): relative intensity scaling factor between the Thorium and the Argon lines.

    """

    def __init__(self, argon_to_thorium_factor=10):
        super().__init__(name='ThAr', list_like=True)
        self.flux_in_photons = True
        self.scale = argon_to_thorium_factor

    def get_spectral_density(self, wavelength):
        minwl = np.min(wavelength)
        maxwl = np.max(wavelength)
        thwl, thint = pull_catalogue_lines(minwl, maxwl, 'Th')
        arwl, arint = pull_catalogue_lines(minwl, maxwl, 'Ar')
        arint *= self.scale
        return np.hstack((thwl / 10000., arwl / 10000.)), np.hstack((thint, arint))


class ThNe(Source):
    """ Thorium-Neon lamp

    Implements a Thorium Neon arc-lamp.
    Uses NIST vacuum catalogue wavelength as source.

    Attributes:
         scale (float): relative intensity scaling factor between the Thorium and the Neon lines.

    """

    def __init__(self, neon_to_thorium_factor=10):
        super().__init__(name='ThNe', list_like=True)
        self.flux_in_photons = True
        self.scale = neon_to_thorium_factor

    def get_spectral_density(self, wavelength):
        minwl = np.min(wavelength)
        maxwl = np.max(wavelength)
        thwl, thint = pull_catalogue_lines(minwl, maxwl, 'Th')
        newl, neint = pull_catalogue_lines(minwl, maxwl, 'Ne')
        neint *= self.scale

        return np.hstack((thwl / 10000., newl / 10000.)), np.hstack((thint, neint))


class Etalon(Source):
    r""" Fabry-Perot etalon.

    Implements spectrum of an ideal (i.e. dispersion-free) Fabry-Perot etalon.
    This means, the peak wavelength are at:

    .. math::
        \lambda_{peak} = \frac{d \cdot n \cdot \cos{(\theta)}}{m}

    Attributes:
        d (float): mirror distance [mm]
        n (float): refractive index between mirrors
        theta (float): angle of incidence onto mirrors
        min_m (int): minimum peak interference number
        max_m (int): maximum peak interference number
        n_photons (int): number of photons per peak per second

    """

    def __init__(self, d=5.0, n=1.0, theta=0.0, n_photons=1000, **kwargs):
        super().__init__(**kwargs, name="Etalon", list_like=True)
        self.d = d
        self.n = n
        self.theta = theta
        self.min_m = np.ceil(2e3 * d * np.cos(theta) / self.max_wl)
        self.max_m = np.floor(2e3 * d * np.cos(theta) / self.min_wl)
        self.n_photons = n_photons
        self.flux_in_photons = True

    @staticmethod
    def peak_wavelength_etalon(m, d=10.0, n=1.0, theta=0.0):
        return 2e3 * d * n * np.cos(theta) / m

    def get_spectral_density(self, wavelength):
        self.min_m = np.ceil(2e3 * self.d * np.cos(self.theta) / np.max(wavelength))
        self.max_m = np.floor(2e3 * self.d * np.cos(self.theta) / np.min(wavelength))
        intensity = np.ones_like(np.arange(self.min_m, self.max_m), dtype=float) * float(self.n_photons)
        return self.peak_wavelength_etalon(
            np.arange(self.min_m, self.max_m), self.d, self.n, self.theta
        ), np.asarray(intensity, dtype=int)


class Phoenix(Source):
    """ Phoenix M-dwarf spectra.

    This class provides a convenient handling of PHOENIX M-dwarf spectra.
    For a given set of effective Temperature, log g, metalicity and alpha, it downloads the spectrum from PHOENIX ftp
    server.

    See the `original paper <http://dx.doi.org/10.1051/0004-6361/201219058>`_ for more details.


    Attributes:
        t_eff (float): effective Temperature [K]
        log_g (float): surface gravity
        z (float): metalicity [Fe/H]
        alpha (float): abundance of alpha elements [α/Fe]

    """
    valid_t = [*list(range(2300, 7000, 100)), *list((range(7000, 12200, 200)))]
    valid_g = [*list(np.arange(0, 6, 0.5))]
    valid_z = [*list(np.arange(-4, -2, 1)), *list(np.arange(-2.0, 1.5, 0.5))]
    valid_a = [-0.2, 0., 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]

    def __init__(
            self, t_eff=3600, log_g=5.0, z=0, alpha=0.0, magnitude=10, **kwargs
    ):
        assert t_eff in self.valid_t, f'Not a valid effective Temperature {t_eff}'
        assert log_g in self.valid_g, f'Not a valid log g value {log_g}'
        assert alpha in self.valid_a, f'Not a valid alpha value {alpha}'
        assert z in self.valid_z, f'Not a valid metalicity value {z}'
        if not np.isclose(alpha, 0.):
            assert 3500. <= t_eff <= 8000. and -3. <= z <= 0., 'PHOENIX parameters are not valid. Please check them ' \
                                                               'again. '
        self.t_eff = t_eff
        self.log_g = log_g
        self.z = z
        self.alpha = alpha
        self.magnitude = magnitude
        super().__init__(**kwargs, name="phoenix")
        self.stellar_target = True

        wavelength_path = cache_path.joinpath('WAVE_PHOENIX-ACES-AGSS-COND-2011.fits')

        if not wavelength_path.is_file():
            print("Download Phoenix wavelength file...")
            with urllib.request.urlopen(self.get_wavelength_url()) as response, open(wavelength_path,
                                                                                     "wb") as out_file:
                data = response.read()
                out_file.write(data)

        self.wl_data = fits.getdata(wavelength_path) / 10000.0
        url = self.get_spectrum_url(t_eff, alpha, log_g, z)
        spectrum_path = cache_path.joinpath(url.split("/")[-1])

        if not spectrum_path.is_file():
            print(f"Download Phoenix spectrum from {url}...")
            with urllib.request.urlopen(url) as response, open(spectrum_path, "wb") as out_file:
                print("Trying to download:" + url)
                data = response.read()
                out_file.write(data)

        self.spectrum_data = 0.1 * fits.getdata(spectrum_path)  # convert ergs/s/cm^2/cm to uW/m^2/um
        self.spectrum_data *= calc_flux_scale(self.wl_data, self.spectrum_data, self.magnitude)
        self.ip_spectra = scipy.interpolate.interp1d(self.wl_data, self.spectrum_data)

    @staticmethod
    def get_wavelength_url():
        return "ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"

    @staticmethod
    def get_spectrum_url(t_eff, alpha, log_g, z):
        zstring = f"{'+' if z > 0 else '-'}{abs(z):2.1f}"
        alphastring = f"" if np.isclose(alpha, 0.) else f".Alpha={alpha:+2.2f}"

        url = (
            f"ftp://phoenix.astro.physik.uni-goettingen.de/"
            f"HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z{zstring}{alphastring}/lte{t_eff:05}-{log_g:2.2f}{zstring}"
            f"{alphastring}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
        )
        return url

    def get_spectral_density(self, wavelength):
        idx = np.logical_and(self.wl_data > np.min(wavelength), self.wl_data < np.max(wavelength))
        return self.wl_data[idx], self.ip_spectra(self.wl_data[idx])


class CSV(Source):
    """ Spectral source based on custom .csv file

    The .csv file needs to have two columns, a wavelength column and a flux column. The wavelength must be given
    in either angstroms, nanometers, microns or meters (specified via wavelength_unit), while the flux must either be in
    ergs/s/cm^2/cm (like Phoenix spectra) for stellar targets, or in photons/s.

    Attributes:
         name: name of the spectrum
         list_like: when True, no wavelength interpolation is active (useful for non-continuous spectra e.g.
         custom line lists)
         flux_in_photons: if True, flux column treated as photons/s otherwise ergs/s/cm^2/cm
         stellar_target: if True, flux is expected to be in ergs/s/cm^2/cm and will scale with telescope size
         magnitude: V magnitude in case of stellar_target
    """
    wavelength_scaling = {'a': 1E-4, 'nm': 1E-3, 'micron': 1, 'm': 1E-6}

    def __init__(self, filepath: str | pathlib.Path, name: str | None = None, list_like: bool = False,
                 wavelength_unit: str = 'a', flux_in_photons: bool = False, stellar_target: bool = False,
                 magnitude: float = 10., delimiter: str = ',', scale_now: bool = True):
        """ Constructor

        Args:
         filepath: path to .csv file
         name: name of the spectrum
         list_like: when True, no wavelength interpolation is active (useful for non-continuous spectra e.g.
         custom line lists)
         wavelength_unit: either 'a', 'nm', 'micron', or 'm' specifying the unit of the wavelength column
         flux_in_photons: if True, flux column treated as photons/s otherwise ergs/s/cm^2/cm
         stellar_target: if True, flux is expected to be in ergs/s/cm^2/cm and will scale with telescope size
         magnitude: V magnitude in case of stellar_target
         delimiter: delimiter character of .csv file
        """
        assert wavelength_unit in self.wavelength_scaling.keys(), f'Supported wavelength units are ' \
                                                                  f'{self.wavelength_scaling.keys()}'
        if isinstance(filepath, io.TextIOWrapper):
            filepath = filepath.name
        if isinstance(filepath, str):
            filepath = pathlib.Path(filepath)
        if name is None:
            name = filepath.name
        super().__init__(name=name, list_like=list_like, flux_in_photons=flux_in_photons, stellar_target=stellar_target)
        self.magnitude = magnitude
        print(f'{filepath=}, {type(filepath)}')
        data = pd.read_csv(filepath, delimiter=delimiter)

        self.wl_data = data.iloc[:, 0].values * self.wavelength_scaling[wavelength_unit]
        self.flux_data = data.iloc[:, 1].values
        
        if scale_now:
            self.flux_scale()

        self.flux_in_photons = flux_in_photons

    def flux_scale(self):
        if not self.flux_in_photons:
            self.flux_data *= 0.1  # convert ergs/s/cm^2/cm to uW/m^2/um
            self.flux_data *= calc_flux_scale(self.wl_data, self.flux_data, self.magnitude)

    def get_spectral_density(self, wavelength):
        idx = np.logical_and(self.wl_data > np.min(wavelength), self.wl_data < np.max(wavelength))
        return self.wl_data[idx], self.flux_data[idx]


class LineList(Source):
    """ Line-list spectrum

    Attributes:
        wavelengths: wavelengths [micron] of line(s)
        intensities: intensities [photons] of line(s)
    """

    def __init__(self, wavelengths: list[float] | np.ndarray | float = 0.5,
                 intensities: list[float] | np.ndarray | float = 1000.):
        super().__init__(name='LineList', list_like=True)
        # convert always to numpy array
        self.wavelengths = np.array([wavelengths] if isinstance(wavelengths, float) else wavelengths)

        self.intensities = np.ones_like(self.wavelengths) * intensities if isinstance(intensities,
                                                                                      float) else intensities
        assert len(self.wavelengths) == len(self.intensities), 'wavelengths and intensities do not have the same length'
        self.flux_in_photons = True

    def get_spectral_density(self, wavelength):
        idx = np.logical_and(self.wavelengths > np.min(wavelength), self.wavelengths < np.max(wavelength))
        return self.wavelengths[idx], self.intensities[idx]


class Blackbody(Source):
    """ Blackbody spectrum

    Implements a (stellar) blackbody spectrum of given temperature and V-magnitude.
    """

    def __init__(self, temperature: float = 6000, magnitude: float = 15., name='blackbody'):
        """

        Args:
            temperature: effective temperature of blackbody [K]
            magnitude: Johnson V magnitude of blackbody [mag]
            name: name of the source (default: blackbody)
        """
        super().__init__(name=name, stellar_target=True, flux_in_photons=False)
        self._sp = SourceSpectrum(BlackBodyNorm1D(temperature=temperature))
        self._bp = SpectralElement.from_filter('johnson_v')
        self._vega = SourceSpectrum.from_vega()  # For unit conversion
        self._sp_norm = self._sp.normalize(magnitude * units.VEGAMAG, self._bp, vegaspec=self._vega)

    def get_spectral_density(self, wavelength):
        # convert input wavelength from micron to angstrom
        # convert output from FLAM (erg/s/cm^2/A) to (microW/m^2/microns)
        return wavelength, self._sp_norm(wavelength * 10000., flux_unit=units.FLAM).value * 1E8
