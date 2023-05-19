""" Optics module

PyEchelle concept is to describe the optics of an instrument by applying a wavelength dependent affine transformation
 to the input plane and applying a PSF. This module implements the two basic classes that are needed to do so.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numba
import numpy as np
import scipy.interpolate


@dataclass
class AffineTransformation:
    r""" Affine transformation matrix

    This class represents an affine transformation matrix.

    PyEchelle uses affine transformations (which are represented by affine transformation matrices) to describe the
    mapping of a monochromatic image from the input focal plane of the spectrograph to the detector plane.

    See `wikipedia <https://en.wikipedia.org/wiki/Affine_transformation>`_ for more details about affine transformations.

    In two dimensions an affine transformation matrix can be written as:

    .. math::
        M = \begin{bmatrix}
        m0 & m1 & m2 \\
        m3 & m4 & m4 \\
        0 & 0 & 1
        \end{bmatrix}

    The last row is constant and is therefore be omitted. This is the form that is returned (as a flat array) by
    :meth:`pyechelle.AffineTransformation.as_matrix`

    There is another more intuitive representation of an affine transformation matrix in terms of the parameters:
    rotation, scaling in x- and y-direction, shear and translation in x- and y-direction.
    See :meth:`pyechelle.AffineTransformation.__post_init__` for how those representations are connected.

    Instances of this class can be sorted by wavelength.

    Attributes:
        rot (float): rotation [radians]
        sx (float): scaling factor in x-direction
        sy (float): scaling factor in y-direction
        shear (float): shearing factor
        tx (float): translation in x-direction
        ty (float): translation in y-direction
        wavelength (float | None): wavelength [micron] of affine transformation matrix
    """
    rot: float
    sx: float
    sy: float
    shear: float
    tx: float
    ty: float
    wavelength: float | None

    def __le__(self, other):
        return self.wavelength <= other.wavelength

    def __lt__(self, other):
        return self.wavelength < other.wavelength

    def __add__(self, other):
        wl = None
        if other.wavelength and self.wavelength:
            assert np.isclose(other.wavelength, self.wavelength)
            wl = self.wavelength
        if other.wavelength and self.wavelength is None:
            wl = other.wavelength
        if self.wavelength and other.wavelength is None:
            wl = self.wavelength
        return AffineTransformation(self.rot + other.rot,
                                    self.sx + other.sx,
                                    self.sy + other.sy,
                                    self.shear + other.shear,
                                    self.tx + other.tx,
                                    self.ty + other.ty,
                                    wl)

    def __sub__(self, other):
        wl = None
        if other.wavelength and self.wavelength:
            assert np.isclose(other.wavelength, self.wavelength)
            wl = self.wavelength
        if other.wavelength and self.wavelength is None:
            wl = other.wavelength
        if self.wavelength and other.wavelength is None:
            wl = self.wavelength
        return AffineTransformation(self.rot - other.rot,
                                    self.sx - other.sx,
                                    self.sy - other.sy,
                                    self.shear - other.shear,
                                    self.tx - other.tx,
                                    self.ty - other.ty,
                                    wl)

    def __iadd__(self, other):
        assert np.isclose(other.wavelength, self.wavelength)
        self.sx += other.sx
        self.sy += other.sy
        self.shear += other.shear
        self.tx += other.tx
        self.ty += other.ty
        return self

    def __isub__(self, other):
        assert np.isclose(other.wavelength, self.wavelength)
        self.sx -= other.sx
        self.sy -= other.sy
        self.shear -= other.shear
        self.tx -= other.tx
        self.ty -= other.ty
        return self

    def __mul__(self, other):
        assert isinstance(other, tuple), "You can only multiply an affine matrix with a tuple of length 2 (x," \
                                         "y coordinate) "
        assert len(other) == 2, "You can only multiply an affine matrix with a tuple of length 2  (x,y coordinate)"
        x_new = self.sx * math.cos(self.rot) * other[0] - self.sy * math.sin(self.rot + self.shear) * other[1] + self.tx
        y_new = self.sx * math.sin(self.rot) * other[0] + self.sy * math.cos(self.rot + self.shear) * other[1] + self.ty
        return x_new, y_new

    def as_matrix(self) -> tuple:
        """flat affine matrix

        Returns:
            flat affine transformation matrix
        """
        return self.rot, self.sx, self.sy, self.shear, self.tx, self.ty


@dataclass
class TransformationSet:
    affine_transformations: list[AffineTransformation]
    rot: np.ndarray = field(init=False)
    sx: np.ndarray = field(init=False)
    sy: np.ndarray = field(init=False)
    shear: np.ndarray = field(init=False)
    tx: np.ndarray = field(init=False)
    ty: np.ndarray = field(init=False)
    wl: np.ndarray = field(init=False)

    _spline_affine: list = field(init=False, default=None)

    def __post_init__(self):
        self.affine_transformations.sort()

        self.rot = np.array([at.rot for at in self.affine_transformations])
        self.sx = np.array([at.sx for at in self.affine_transformations])
        self.sy = np.array([at.sy for at in self.affine_transformations])
        self.shear = np.array([at.shear for at in self.affine_transformations])
        self.tx = np.array([at.tx for at in self.affine_transformations])
        self.ty = np.array([at.ty for at in self.affine_transformations])

        # correct for possible jumps in rot and shear
        # assert max(abs(np.ediff1d(self.shear))) < 4, 'There is a jump in the shear parameter of the model file. ' \
        #                                              'Please correct the jump, by wrapping shear to e.g. to (-pi, pi) or (0, 2.*pi)'

        # self.shear = np.mod(self.shear, np.pi * 2.)

        self.wl = np.array([at.wavelength for at in self.affine_transformations])

    def get_affine_transformations(self, wl: float | np.ndarray) -> AffineTransformation | np.ndarray:
        if self._spline_affine is None:
            self._spline_affine = [scipy.interpolate.CubicSpline(self.wl, self.rot),
                                   scipy.interpolate.CubicSpline(self.wl, self.sx),
                                   scipy.interpolate.CubicSpline(self.wl, self.sy),
                                   scipy.interpolate.CubicSpline(self.wl, self.shear),
                                   scipy.interpolate.CubicSpline(self.wl, self.tx),
                                   scipy.interpolate.CubicSpline(self.wl, self.ty)
                                   ]
        if isinstance(wl, float):
            return AffineTransformation(*[af(wl) for af in self._spline_affine], wl)
        else:
            return np.array([af(wl) for af in self._spline_affine])


def convert_matrix(input: AffineTransformation | np.ndarray) -> np.ndarray:
    if isinstance(input, AffineTransformation):
        return np.array([input.sx * math.cos(input.rot),
                         -input.sy * math.sin(input.rot + input.shear),
                         input.tx,
                         input.sx * math.sin(input.rot),
                         input.sy * math.cos(input.rot + input.shear),
                         input.ty])
    else:
        assert isinstance(input, np.ndarray)
        return np.array([
            input[1] * np.cos(input[0]),
            -input[2] * np.sin(input[0] + input[3]),
            input[4],
            input[1] * np.sin(input[0]),
            input[2] * np.cos(input[0] + input[3]),
            input[5]])


@numba.njit
def apply_matrix(matrix, coords):
    return matrix[0] * coords[0] + matrix[1] * coords[1] + matrix[2], \
           matrix[3] * coords[0] + matrix[4] * coords[1] + matrix[5]


@dataclass
class PSF:
    """Point spread function

    The point spread function describes how an optical system responds to a point source.

    Attributes:
        wavelength (float): wavelength [micron]
        data (np.ndarray): PSF data as 2D array
        sampling (float): physical size of the sampling of data [micron]

    """
    wavelength: float
    data: np.ndarray
    sampling: float

    def __post_init__(self):
        self.data /= np.sum(self.data)

    def __le__(self, other):
        return self.wavelength <= other.wavelength

    def __lt__(self, other):
        return self.wavelength < other.wavelength

    def __str__(self):
        res = f'PSF@\n{self.wavelength:.4f}micron\n'
        letters = ['.', ':', 'o', 'x', '#', '@']
        norm = np.max(self.data)
        for d in self.data:
            for dd in d:
                i = int(math.floor(dd / norm / 0.2))
                res += letters[i]
            res += '\n'
        return res
