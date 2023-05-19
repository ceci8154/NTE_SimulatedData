from __future__ import annotations

import pathlib

import numpy as np
import skycalc_ipy
from joblib import Memory
from numpy import deg2rad
from numpy import sin, cos, tan, arcsin
from scipy.interpolate import interp1d

path = pathlib.Path(__file__).parent.resolve()
cache_path = path.joinpath('.cache')
# create data directory if it doesn't exist:
pathlib.Path(cache_path).mkdir(parents=False, exist_ok=True)
memory = Memory(cache_path, verbose=0)


class Efficiency:
    def __init__(self, name):
        self.name = name

    def get_efficiency(self, wavelength):
        raise NotImplementedError

    def get_efficiency_per_order(self, wavelength, order):
        raise NotImplementedError


class ConstantEfficiency(Efficiency):
    def __init__(self, name, eff=1.):
        super().__init__(name)
        self.eff = eff

    def get_efficiency(self, wavelength):
        return np.ones_like(wavelength) * self.eff

    def get_efficiency_per_order(self, wavelength, order):
        return self.get_efficiency(wavelength)


class SystemEfficiency(Efficiency):
    def __init__(self, efficiencies: list[Efficiency], name: str):
        super().__init__(name)
        self.efficiencies = efficiencies

    def get_efficiency(self, wavelength):
        e = np.ones_like(wavelength)
        for ef in self.efficiencies:
            e *= ef.get_efficiency(wavelength)
        return e

    def get_efficiency_per_order(self, wavelength, order):
        e = np.ones_like(wavelength)
        for ef in self.efficiencies:
            e *= ef.get_efficiency_per_order(wavelength, order)
        return e

    def add_efficiency(self, efficiency):
        self.efficiencies.append(efficiency)


class GratingEfficiency(Efficiency):
    def __init__(self, alpha, blaze, gpmm, peak_efficiency=1.0, name="Grating"):
        super().__init__(name)
        self.alpha = deg2rad(alpha)
        self.blaze = deg2rad(blaze)
        self.gpmm = gpmm
        self.peak_efficiency = peak_efficiency

    def calc_efficiency(self, order, wl):
        bb = np.nan_to_num(
            arcsin(-sin(self.alpha) + order * wl * 1e-6 / (1.0 / self.gpmm / 1000.0))
        )
        # blaze_wavelength = 2.*self.gpmm * sin(self.blaze) / order
        # fsr = blaze_wavelength / order

        x = (
                order
                * (cos(self.alpha) / cos(self.alpha - self.blaze))
                * (cos(self.blaze) - sin(self.blaze) / tan((self.alpha + bb) / 2.0))
        )
        sinc = np.sinc(x)

        return self.peak_efficiency * sinc * sinc

    def get_efficiency(self, wavelength):
        e = np.zeros_like(wavelength)
        for o in range(50, 150):
            e += self.calc_efficiency(o, wavelength)
        return e

    def get_efficiency_per_order(self, wavelength, order):
        return self.calc_efficiency(order, wavelength)


class TabulatedEfficiency(Efficiency):
    def __init__(self, name, wavelength, efficiency, orders=None):
        super().__init__(name)
        wavelength = np.atleast_1d(wavelength)
        efficiency = np.atleast_1d(efficiency)
        assert len(wavelength) == len(efficiency)
        if orders is None:
            # do constant extrapolation if only 1 point
            if len(wavelength) == 1:
                self.ip = lambda x: efficiency
            # do linear interpolation if only 2 points
            elif len(wavelength) == 2:
                self.ip = interp1d(wavelength, efficiency, kind="linear", fill_value=0., bounds_error=False)
            # do linear interpolation if only 2 points
            elif len(wavelength) == 3:
                self.ip = interp1d(wavelength, efficiency, kind="quadratic", fill_value=0., bounds_error=False)
            # do cubic interpolation if >= 4 points
            else:
                self.ip = interp1d(wavelength, efficiency, kind="cubic", fill_value=0., bounds_error=False)
            self.ip_per_order = self.ip
        else:
            self.ip_per_order = {}

            for o in np.unique(orders):
                idx = orders == o
                self.ip_per_order[o] = interp1d(wavelength[idx], efficiency[idx], kind="cubic", fill_value=0.0,
                                                bounds_error=False)

            y = np.zeros_like(wavelength)
            for o in np.unique(orders):
                y += self.ip_per_order[o](wavelength)

            self.ip = interp1d(wavelength, y)

    def get_efficiency(self, wavelength):
        return self.ip(wavelength)

    def get_efficiency_per_order(self, wavelength, order):
        if isinstance(self.ip_per_order, dict):
            return self.ip_per_order[order](wavelength)
        else:
            return self.ip_per_order(wavelength)


class CSVEfficiency(TabulatedEfficiency):
    def __init__(self, name, path, delimiter=","):
        data = np.genfromtxt(path, delimiter=delimiter)
        if data.shape[1] == 2:  # file contains wavelength, efficiency
            super().__init__(name, data[:, 0], data[:, 1])
        if data.shape[1] == 3:  # file contains order, wavelength, efficiency
            super().__init__(name, data[:, 1], data[:, 2], data[:, 0])


class Atmosphere(Efficiency):
    def __init__(self, name, sky_calc_kwargs=None):
        super().__init__(name)
        # set default atmosphere arguments to high-resolution
        self.kwargs = {"wres": 1E6, "wgrid_mode": "fixed_spectral_resolution"}
        if sky_calc_kwargs is not None:
            self.kwargs.update(sky_calc_kwargs)

    def get_efficiency(self, wavelength):
        return self.get_atmosphere_data(wavelength, self.kwargs)

    def get_efficiency_per_order(self, wavelength, order):
        return self.get_atmosphere_data(wavelength, self.kwargs)

    @staticmethod
    @memory.cache
    def get_atmosphere_data(wavelength, sky_calc_kwargs=None):
        kwargs = {"wres": 1E6, "wgrid_mode": "fixed_spectral_resolution"}
        if sky_calc_kwargs is not None:
            kwargs.update(sky_calc_kwargs)
        sky = skycalc_ipy.SkyCalc()
        sky.update(kwargs)  # set sky arguments
        wmin = np.min(wavelength) * 1000.
        wmax = np.max(wavelength) * 1000.
        sky.update({'wmin': wmin, 'wmax': wmax})

        tbl = sky.get_sky_spectrum()
        # extrapolate is needed because tbl['lam'].data might not contain exact wavelength limits,
        # since we are in constant resolution mode ("wgrid_mode": "fixed_spectral_resolution")
        ip = interp1d(tbl['lam'].data, tbl['trans'].data, assume_sorted=True, fill_value="extrapolate")

        return ip(wavelength)
