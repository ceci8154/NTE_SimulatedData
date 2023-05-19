from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError

import h5py
import numpy as np

from NTEpyechelle.CCD import CCD
from NTEpyechelle.efficiency import SystemEfficiency, GratingEfficiency, TabulatedEfficiency, ConstantEfficiency
from NTEpyechelle.optics import AffineTransformation, PSF, TransformationSet, convert_matrix, apply_matrix


def check_url_exists(url: str) -> bool:
    """
    Check if URL exists.
    Args:
        url: url to be tested

    Returns:
        if URL exists
    """
    try:
        with urllib.request.urlopen(url) as response:
            return float(response.headers['Content-length']) > 0
    except URLError:
        return False


def check_for_spectrograph_model(model_name: str, download=True):
    """
    Check if spectrograph model exists locally. Otherwise: Download if download is true (default) or check if URL to
    spectrograph model is valid (this is mainly for testing purpose).

    Args:
        model_name: name of spectrograph model. See models/available_models.txt for valid names
        download: download flag

    Returns:

    """
    file_path = Path(__file__).resolve().parent.joinpath("models").joinpath(f"{model_name}.hdf")
    if not file_path.is_file():
        url = f"https://stuermer.science/nextcloud/index.php/s/ps5Pk379LgcpLwN/download?path=/&files={model_name}.hdf"
        if download:
            print(f"Spectrograph model {model_name} not found locally. Trying to download from {url}...")
            Path(Path(__file__).resolve().parent.joinpath("models")).mkdir(parents=False, exist_ok=True)
            with urllib.request.urlopen(url) as response, open(file_path, "wb") as out_file:
                data = response.read()
                out_file.write(data)
        else:
            check_url_exists(url)
    return file_path


@dataclass
class Spectrograph:
    """ Abstract spectrograph model

    Describes all methods that a spectrograph model must have to be used in a simulation. \n
    When subclassing, all methods need to be implemented in the subclass.

    A spectrograph model as at least one CCD (with CCD_index 1), at least one field/fiber (with fiber index 1),
    and at least one diffraction order.
    """
    name: str = 'Spectrograph'

    def get_fibers(self, ccd_index: int = 1) -> list[int]:
        """ Fields/fiber indices

        Args:
            ccd_index: CCD index

        Returns:
            available fields/fiber indices
        """
        raise NotImplementedError

    def get_orders(self, fiber: int = 1, ccd_index: int = 1) -> list[int]:
        """ Diffraction orders

        Args:
            fiber: fiber/field index
            ccd_index: CCD index

        Returns:
            available diffraction order(s) for given indices
        """
        raise NotImplementedError

    def get_transformation(self, wavelength: float | np.ndarray, order: int, fiber: int = 1,
                           ccd_index: int = 1) -> AffineTransformation | np.ndarray:
        """ Transformation matrix/matrices

        Args:
            wavelength: wavelength(s) [micron]
            order: diffraction order
            fiber: fiber index
            ccd_index: CCD index

        Returns:
            transformation matrix/matrices
        """
        raise NotImplementedError

    def get_psf(self, wavelength: float | None, order: int, fiber: int = 1, ccd_index: int = 1) -> PSF | list[PSF]:
        """ PSF

        PSFs are tabulated. When wavelength is provided, the closest available PSF of the model is returned.

        When wavelength is None, all PSFs for that particular order (and fiber and CCD index) are returned.

        Args:
            wavelength: wavelength [micron] or None
            order: diffraction order
            fiber: fiber index
            ccd_index: ccd index

        Returns:
            PSF(s)
        """
        raise NotImplementedError

    def get_wavelength_range(self, order: int | None = None, fiber: int | None = None, ccd_index: int | None = None) \
            -> tuple[float, float]:
        """ Wavelength range

        Returns minimum and maximum wavelength of the entire spectrograph unit or an individual order if specified.

        Args:
            ccd_index: CCD index
            fiber: fiber index
            order: diffraction order

        Returns:
            minimum and maximum wavelength [microns]
        """
        raise NotImplementedError

    def get_ccd(self, ccd_index: int | None = None) -> CCD | dict[int, CCD]:
        """ Get CCD object(s)

        When index is provided the corresponding CCD object is returned.\n
        If no index is provided, all available CCDs are return as a dict with the index as key.

        Args:
            ccd_index: CCD index

        Returns:
            CCD object(s)
        """
        raise NotImplementedError

    def get_field_shape(self, fiber: int, ccd_index: int) -> str:
        """ Shape of field/fiber

        Returning the field/fiber shape for the given indices as a string.
        See slit.py for currently implemented shapes.

        Args:
            fiber: fiber index
            ccd_index: ccd index

        Returns:
            field/fiber shape as string (e.g. rectangular, octagonal)
        """
        raise NotImplementedError

    def get_efficiency(self, fiber: int, ccd_index: int) -> SystemEfficiency:
        """ Spectrograph efficiency

        Args:
            fiber: fiber/field index
            ccd_index: CCD index

        Returns:
            System efficiency for given indices
        """
        raise NotImplementedError


class SimpleSpectrograph(Spectrograph):
    def __init__(self):
        self._ccd = {1: CCD()}
        self._fibers = {}
        self._orders = {}
        self._transformations = {}
        for c in self._ccd.keys():
            self._fibers[c] = [1]
            self._orders[c] = {}
            for f in self._fibers.keys():
                self._orders[c][f] = [1]

    def get_fibers(self, ccd_index: int = 1) -> list[int]:
        return self._fibers[ccd_index]

    def get_orders(self, fiber: int = 1, ccd_index: int = 1) -> list[int]:
        return self._orders[ccd_index][fiber]

    def get_transformation(self, wavelength: float | np.ndarray, order: int, fiber: int = 1,
                           ccd_index: int = 1) -> AffineTransformation | list[AffineTransformation]:
        if isinstance(wavelength, float):
            return AffineTransformation(0.0, 1.0, 10., 0., (wavelength - 0.5) * wavelength * 100000. + 2000.,
                                        fiber * 10. + 2000., wavelength)
        else:
            ts = TransformationSet([AffineTransformation(0.0, 1.0, 10., 0., (w - 0.5) * w * 100000. + 2000.,
                                                         fiber * 10. + 2000., w) for w in wavelength])
            return ts.get_affine_transformations(wavelength)

    @staticmethod
    def gauss_map(size_x, size_y=None, sigma_x=5., sigma_y=None):
        if size_y is None:
            size_y = size_x
        if sigma_y is None:
            sigma_y = sigma_x

        assert isinstance(size_x, int)
        assert isinstance(size_y, int)

        x0 = size_x // 2
        y0 = size_y // 2

        x = np.arange(0, size_x, dtype=float)
        y = np.arange(0, size_y, dtype=float)[:, np.newaxis]

        x -= x0
        y -= y0

        exp_part = x ** 2 / (2 * sigma_x ** 2) + y ** 2 / (2 * sigma_y ** 2)
        return 1 / (2 * np.pi * sigma_x * sigma_y) * np.exp(-exp_part)

    def get_psf(self, wavelength: float | None, order: int, fiber: int = 1, ccd_index: int = 1) -> PSF | list[PSF]:
        if wavelength is None:
            wl = np.linspace(*self.get_wavelength_range(order, fiber, ccd_index), 20)
            return [PSF(w, self.gauss_map(11, sigma_x=3., sigma_y=10.), 1.5) for w in wl]
        else:
            return PSF(wavelength, self.gauss_map(11, sigma_x=3., sigma_y=10.), 1.5)

    def get_wavelength_range(self, order: int | None = None, fiber: int | None = None, ccd_index: int | None = None) \
            -> tuple[float, float]:
        return 0.4, 0.6

    def get_ccd(self, ccd_index: int | None = None) -> CCD | dict[int, CCD]:
        if ccd_index is None:
            return self._ccd
        else:
            return self._ccd[ccd_index]

    def get_field_shape(self, fiber: int, ccd_index: int) -> str:
        return 'rectangular'

    def get_efficiency(self, fiber: int, ccd_index: int) -> SystemEfficiency:
        return SystemEfficiency([ConstantEfficiency(1.0)], 'System')


class ZEMAX(Spectrograph):
    def __init__(self, path: str | Path):
        self.path = check_for_spectrograph_model(path)
        self._CCDs = {}
        self._ccd_keys = []
        self._h5f = None

        self._transformations = {}
        self._spline_transformations = {}
        self._psfs = {}
        self._efficiency = {}

        # self.name = self.h5f[f"Spectrograph"].attrs['name']
        self._field_shape = {}

        self._orders = {}

        self.CCD = [self._read_ccd_from_hdf]

    @property
    def h5f(self):
        if self._h5f is None:
            self._h5f = h5py.File(self.path, "r")
        return self._h5f

    def _read_ccd_from_hdf(self, k) -> CCD:
        # read in CCD information
        nx = self.h5f[f"CCD_{k}"].attrs['Nx']
        ny = self.h5f[f"CCD_{k}"].attrs['Ny']
        ps = self.h5f[f"CCD_{k}"].attrs['pixelsize']
        return CCD(n_pix_x=nx, n_pix_y=ny, pixelsize=ps)

    def get_fibers(self, ccd_index: int = 1) -> list[int]:
        return [int(k[6:]) for k in self.h5f[f"CCD_{ccd_index}"].keys() if "fiber" in k]

    def get_field_shape(self, fiber: int, ccd_index: int) -> str:
        if ccd_index not in self._field_shape.keys():
            self._field_shape[ccd_index] = {}
        if fiber not in self._field_shape[ccd_index].keys():
            self._field_shape[ccd_index][fiber] = self.h5f[f"CCD_{ccd_index}/fiber_{fiber}"].attrs["field_shape"]
        return self._field_shape[ccd_index][fiber]

    def get_orders(self, fiber: int = 1, ccd_index: int = 1) -> list[int]:
        if ccd_index not in self._orders.keys():
            self._orders[ccd_index] = {}
        if fiber not in self._orders[ccd_index].keys():
            self._orders[ccd_index][fiber] = [int(k[5:]) for k
                                              in self.h5f[f"CCD_{ccd_index}/fiber_{fiber}/"].keys() if "psf" not in k]
            self._orders[ccd_index][fiber].sort()
        return self._orders[ccd_index][fiber]

    def transformations(self, order: int, fiber: int = 1, ccd_index: int = 1) -> list[AffineTransformation]:
        if ccd_index not in self._transformations.keys():
            self._transformations[ccd_index] = {}
        if fiber not in self._transformations[ccd_index].keys():
            self._transformations[ccd_index][fiber] = {}
        if order not in self._transformations[ccd_index][fiber].keys():
            try:
                self._transformations[ccd_index][fiber][order] = [AffineTransformation(*af)
                                                                  for af in
                                                                  self.h5f[
                                                                      f"CCD_{ccd_index}/fiber_{fiber}/order{order}"][
                                                                      ()]]
                self._transformations[ccd_index][fiber][order].sort()

            except KeyError:
                raise KeyError(
                    f"You asked for the affine transformation matrices in diffraction order {order}. "
                    f"But this data is not available")

        return self._transformations[ccd_index][fiber][order]

    def spline_transformations(self, order: int, fiber: int = 1, ccd_index: int = 1) -> TransformationSet:
        if ccd_index not in self._spline_transformations.keys():
            self._spline_transformations[ccd_index] = {}
        if fiber not in self._spline_transformations[ccd_index].keys():
            self._spline_transformations[ccd_index][fiber] = {}
        if order not in self._spline_transformations[ccd_index][fiber].keys():
            tfs = self.transformations(order, fiber, ccd_index)
            self._spline_transformations[ccd_index][fiber][order] = TransformationSet(tfs)
        return self._spline_transformations[ccd_index][fiber][order]

    def get_transformation(self, wavelength: float | np.ndarray, order: int, fiber: int = 1,
                           ccd_index: int = 1) -> AffineTransformation | list[AffineTransformation] | np.ndarray:
        return self.spline_transformations(order, fiber, ccd_index).get_affine_transformations(wavelength)

    def psfs(self, order: int, fiber: int = 1, ccd_index: int = 1) -> list[PSF]:
        if ccd_index not in self._psfs.keys():
            self._psfs[ccd_index] = {}
        if order not in self._psfs[ccd_index].keys():
            try:
                self._psfs[ccd_index][order] = \
                    [PSF(self.h5f[f"CCD_{ccd_index}/fiber_{fiber}/psf_order_{order}/{wl}"].attrs['wavelength'],
                         self.h5f[f"CCD_{ccd_index}/fiber_{fiber}/psf_order_{order}/{wl}"][()],
                         self.h5f[f"CCD_{ccd_index}/fiber_{fiber}/psf_order_{order}/{wl}"].attrs[
                             'dataSpacing'])
                     for wl in self.h5f[f"CCD_{ccd_index}/fiber_{fiber}/psf_order_{order}"]]

            except KeyError:
                raise KeyError(f"You asked for the PSFs in diffraction order {order}. But this data is not available")
            self._psfs[ccd_index][order].sort()
        return self._psfs[ccd_index][order]

    def get_psf(self, wavelength: float | None, order: int, fiber: int = 1, ccd_index: int = 1) -> PSF | list[PSF]:
        if wavelength is None:
            return self.psfs(order, fiber, ccd_index)
        else:
            # find the nearest PSF:
            idx = min(range(len(self.psfs(order, fiber, ccd_index))),
                      key=lambda i: abs(self.psfs(order, fiber, ccd_index)[i].wavelength - wavelength))
            return self.psfs(order, fiber, ccd_index)[idx]

    def get_wavelength_range(self, order: int | None = None, fiber: int | None = None, ccd_index: int | None = None) \
            -> tuple[float, float]:
        min_w = []
        max_w = []

        if ccd_index is None:
            new_ccd_index = self.available_ccd_keys()
        else:
            new_ccd_index = [ccd_index]

        for ci in new_ccd_index:
            if fiber is None:
                new_fiber = self.get_fibers(ci)
            else:
                new_fiber = [fiber]

            for f in new_fiber:
                if order is None:
                    new_order = self.get_orders(f, ci)
                else:
                    new_order = [order]
                for o in new_order:
                    min_w.append(self.transformations(o, f, ci)[0].wavelength)
                    max_w.append(self.transformations(o, f, ci)[-1].wavelength)
        return min(min_w), max(max_w)

    def available_ccd_keys(self) -> list[int]:
        if not self._ccd_keys:
            self._ccd_keys = [int(k[4:]) for k in self.h5f[f"/"].keys() if "CCD" in k]
        return self._ccd_keys

    def get_ccd(self, ccd_index: int | None = None) -> CCD | dict[int, CCD]:
        if ccd_index is None:
            return dict(zip(self.available_ccd_keys(), [self._read_ccd_from_hdf(k) for k in self.available_ccd_keys()]))

        if ccd_index not in self._CCDs:
            self._CCDs[ccd_index] = self._read_ccd_from_hdf(ccd_index)
        return self._CCDs[ccd_index]

    def get_efficiency(self, fiber: int, ccd_index: int) -> SystemEfficiency:
        ge = GratingEfficiency(self.h5f[f"CCD_{ccd_index}/Spectrograph"].attrs['blaze'],
                               self.h5f[f"CCD_{ccd_index}/Spectrograph"].attrs['blaze'],
                               self.h5f[f"CCD_{ccd_index}/Spectrograph"].attrs['gpmm'])

        if ccd_index not in self._efficiency.keys():
            self._efficiency[ccd_index] = {}
        if fiber not in self._efficiency[ccd_index].keys():
            try:
                self._efficiency[ccd_index][fiber] = \
                    SystemEfficiency([ge,
                                      TabulatedEfficiency('System', *self.h5f[f"CCD_{ccd_index}/fiber_{fiber}"].attrs[
                                          "efficiency"])], 'System')

            except KeyError:
                logging.warning(f'No spectrograph efficiency data found for fiber {fiber}.')
                self._efficiency[ccd_index][fiber] = SystemEfficiency([ge], 'System')
        return self._efficiency[ccd_index][fiber]

    def __exit__(self):
        if self._h5f:
            self._h5f.close()


class InteractiveZEMAX(Spectrograph):

    def get_fibers(self, ccd_index: int = 1) -> list[int]:
        pass

    def get_orders(self, fiber: int = 1, ccd_index: int = 1) -> list[int]:
        pass

    def get_transformation(self, wavelength: float | np.ndarray, order: int, fiber: int = 1,
                           ccd_index: int = 1) -> AffineTransformation | list[AffineTransformation]:
        pass

    def get_psf(self, wavelength: float | None, order: int, fiber: int = 1, ccd_index: int = 1) -> PSF | list[PSF]:
        pass

    def get_wavelength_range(self, order: int | None = None, fiber: int | None = None, ccd_index: int | None = None) -> \
            tuple[float, float]:
        pass

    def get_ccd(self, ccd_index: int | None = None) -> CCD | dict[int, CCD]:
        pass

    def get_field_shape(self, fiber: int, ccd_index: int) -> str:
        pass

    def get_efficiency(self, fiber: int, ccd_index: int) -> SystemEfficiency:
        pass


class LocalDisturber(Spectrograph):

    def __init__(self, spec: Spectrograph, d_tx=0., d_ty=0., d_rot=0., d_shear=0., d_sx=0., d_sy=0.):
        self.spec = spec
        for method in dir(Spectrograph):
            if method.startswith('get_') and method != 'get_transformation':
                setattr(self, method, getattr(self.spec, method))
        self.disturber_matrix = AffineTransformation(d_rot, d_sx, d_sy, d_shear, d_tx, d_ty, None)

    def get_transformation(self, wavelength: float | np.ndarray, order: int, fiber: int = 1,
                           ccd_index: int = 1) -> AffineTransformation | np.ndarray:
        if isinstance(wavelength, float):
            return self.spec.get_transformation(wavelength, order, fiber, ccd_index) + self.disturber_matrix
        else:
            return self.spec.get_transformation(wavelength, order, fiber, ccd_index) + \
                   np.expand_dims(self.disturber_matrix.as_matrix(), axis=-1)


class GlobalDisturber(Spectrograph):
    def __init__(self, spec: Spectrograph, tx: float = 0., ty: float = 0., rot: float = 0., shear: float = 0.,
                 sx: float = 1., sy: float = 1., reference_x: float = None, reference_y: float = None):
        self.spec = spec
        for method in dir(Spectrograph):
            if method.startswith('get_') and method != 'get_transformation':
                setattr(self, method, getattr(self.spec, method))
        self.disturber_matrix = AffineTransformation(rot, sx, sy, shear, tx, ty, None)
        self.ref_x = reference_x
        self.ref_y = reference_y

    def _get_transformation_matrix(self, dx, dy, wavelength):
        if isinstance(wavelength, float):
            return AffineTransformation(0., 1., 1., 0., dx, dy, wavelength)
        else:
            assert isinstance(wavelength, np.ndarray) or isinstance(wavelength, list)
            n_wavelength = len(wavelength)
            return np.array([[0.] * n_wavelength,
                             [1.] * n_wavelength,
                             [1.] * n_wavelength,
                             [0.] * n_wavelength,
                             [dx] * n_wavelength,
                             [dy] * n_wavelength])

    def _get_disturbance_matrix(self, wavelength):
        if isinstance(wavelength, float):
            return AffineTransformation(self.disturber_matrix.rot, self.disturber_matrix.sx,
                                        self.disturber_matrix.sy, self.disturber_matrix.shear,
                                        0., 0., wavelength)
        else:
            assert isinstance(wavelength, np.ndarray) or isinstance(wavelength, list)
            n_wavelength = len(wavelength)
            return np.array([[self.disturber_matrix.rot] * n_wavelength,
                             [self.disturber_matrix.sx] * n_wavelength,
                             [self.disturber_matrix.sy] * n_wavelength,
                             [self.disturber_matrix.shear] * n_wavelength,
                             [0.] * n_wavelength,
                             [0.] * n_wavelength])

    def get_transformation(self, wavelength: float | np.ndarray, order: int, fiber: int = 1,
                           ccd_index: int = 1) -> AffineTransformation | np.ndarray:
        # by default take center of CCD as reference point
        w, h = self.spec.get_ccd(ccd_index).data.shape
        w /= 2.
        h /= 2.

        if self.ref_x is not None:
            w = self.ref_x
        if self.ref_y is not None:
            h = self.ref_y

        if isinstance(wavelength, float):
            tm = self.spec.get_transformation(wavelength, order, fiber, ccd_index)
            xy = tm.tx, tm.ty
            # affine transformation to shift origin to center of image
            tm_trans = self._get_transformation_matrix(-w / 2., -h / 2., tm.wavelength)
            xy = tm_trans * xy
            # affine transformation to rotate/shear/scale
            tm_trans = self._get_disturbance_matrix(wavelength)
            xy = tm_trans * xy
            # affine transformation to shift origin back
            tm_trans = AffineTransformation(0., 1., 1., 0., w / 2., h / 2., tm.wavelength)
            xy = tm_trans * xy
            tm.tx = xy[0] + self.disturber_matrix.tx
            tm.ty = xy[1] + self.disturber_matrix.ty
            return tm
        else:
            tm = self.spec.get_transformation(wavelength, order, fiber, ccd_index)
            xy = tm[4:6].T
            # affine transformation to shift origin to center of image
            tm_trans = convert_matrix(self._get_transformation_matrix(-w / 2., -h / 2., wavelength))
            xy = np.array([apply_matrix(c, p) for c, p in zip(tm_trans.T, xy)])

            tm_trans = convert_matrix(self._get_disturbance_matrix(wavelength))
            xy = np.array([apply_matrix(c, p) for c, p in zip(tm_trans.T, xy)])

            # affine transformation to shift origin back
            tm_trans = convert_matrix(self._get_transformation_matrix(w / 2., h / 2., wavelength))
            xy = np.array([apply_matrix(c, p) for c, p in zip(tm_trans.T, xy)])
            tm[4:6] = xy.T
            tm[4] += self.disturber_matrix.tx
            tm[5] += self.disturber_matrix.ty
            return tm
