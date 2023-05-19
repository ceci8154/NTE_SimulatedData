""" Slit transformation functions

This module implements various coordinate transformations for implementing different slit shapes such as circular,
octagonal etc.
Both, numba compiled functions and cuda device functions are implemented.

In general, a slit function transforms a random coordinate x, y (both from the interval [0, 1]) to a random coordinate
x', y'

.. plot::

    import matplotlib.pyplot as plt
    import random
    import numpy
    import pyechelle.slit
    import inspect
    from numba.core.registry import CPUDispatcher

    available_slits = [m[0] for m in inspect.getmembers(pyechelle.slit) if isinstance(m[1], CPUDispatcher)]


    fig, ax = plt.subplots(1, len(available_slits), sharey=True, sharex=True,figsize=(len(available_slits)*3, 3))
    fig.suptitle('Supported slit functions')
    for i, slit in enumerate(available_slits):
        s = getattr(pyechelle.slit, slit)
        xy = [s(random.random(), random.random()) for _ in range(2000)]
        ax[i].scatter(*numpy.array(xy).T, s=1)
        ax[i].set_aspect('equal', 'box')
        ax[i].set_title(slit)
    plt.tight_layout()
    plt.show()

"""
import math
import random

import numba.cuda.random
from numba import njit, float64, cuda
from numba.types import UniTuple


@njit(UniTuple(float64, 2)(float64, float64))
def rectangular(xx, yy):
    """ Rectangular transformation
    """
    return xx, yy


@njit(UniTuple(float64, 2)(float64, float64))
def circular(xx, yy):
    """ Circular transformation
    """
    r = math.sqrt(xx) / 2.
    phi = yy * math.pi * 2
    return r * math.cos(phi) + 0.5, r * math.sin(phi) + 0.5


@njit(UniTuple(float64, 2)(float64, float64))
def octagonal(r1, r2):
    """ Octagonal transformation
    """
    phi = 0.
    s1 = math.sqrt(r1)
    phi_segment = 2. * math.pi / 8

    b = [1., 0.]
    c = [math.cos(phi_segment), math.sin(phi_segment)]
    x = b[0] * (1.0 - r2) * s1 + c[0] * r2 * s1
    y = b[1] * (1.0 - r2) * s1 + c[1] * r2 * s1

    segments = random.randint(0, 7)
    arg_values = phi_segment * segments + phi
    cos_values = math.cos(arg_values)
    sin_values = math.sin(arg_values)
    x_new = x * cos_values - y * sin_values
    y_new = x * sin_values + y * cos_values
    return x_new / 2. + 0.5, y_new / 2. + 0.5


@njit(UniTuple(float64, 2)(float64, float64))
def hexagonal(r1, r2):
    """ Hexagonal transformation
    """
    phi = 0.
    s1 = math.sqrt(r1)
    phi_segment = 2. * math.pi / 6

    b = [1., 0.]
    c = [math.cos(phi_segment), math.sin(phi_segment)]
    x = b[0] * (1.0 - r2) * s1 + c[0] * r2 * s1
    y = b[1] * (1.0 - r2) * s1 + c[1] * r2 * s1

    segments = random.randint(0, 5)
    arg_values = phi_segment * segments + phi
    cos_values = math.cos(arg_values)
    sin_values = math.sin(arg_values)
    xnew = x * cos_values - y * sin_values
    ynew = x * sin_values + y * cos_values
    return xnew / 2. + 0.5, ynew / 2. + 0.5


@cuda.jit(inline=True, device=True)
def cuda_rectangular(x, y, rng_states, thread_id):
    return x, y


@cuda.jit(inline=True, device=True)
def cuda_circular(x, y, rng_states, thread_id):
    r = math.sqrt(x) / 2.
    phi = y * math.pi * 2
    return r * math.cos(phi) + 0.5, r * math.sin(phi) + 0.5


@cuda.jit(inline=True, device=True)
def cuda_octagonal(r1, r2, rng_states, thread_id):
    phi = 0.
    s1 = math.sqrt(r1)
    phi_segment = 2. * math.pi / 8.

    b = (1., 0.)
    c = (math.cos(phi_segment), math.sin(phi_segment))
    xx = b[0] * (1.0 - r2) * s1 + c[0] * r2 * s1
    yy = b[1] * (1.0 - r2) * s1 + c[1] * r2 * s1

    segments = math.floor(numba.cuda.random.xoroshiro128p_uniform_float64(rng_states, thread_id) * 8.)
    arg_values = phi_segment * segments + phi
    cos_values = math.cos(arg_values)
    sin_values = math.sin(arg_values)
    return (xx * cos_values - yy * sin_values) / 2. + 0.5, (xx * sin_values + yy * cos_values) / 2. + 0.5


@cuda.jit(inline=True, device=True)
def cuda_hexagonal(r1, r2, rng_states, thread_id):
    phi = 0.
    s1 = math.sqrt(r1)
    phi_segment = 2. * math.pi / 6.

    b = (1., 0.)
    c = (math.cos(phi_segment), math.sin(phi_segment))
    xx = b[0] * (1.0 - r2) * s1 + c[0] * r2 * s1
    yy = b[1] * (1.0 - r2) * s1 + c[1] * r2 * s1

    segments = math.floor(numba.cuda.random.xoroshiro128p_uniform_float64(rng_states, thread_id) * 6.)
    arg_values = phi_segment * segments + phi
    cos_values = math.cos(arg_values)
    sin_values = math.sin(arg_values)
    return (xx * cos_values - yy * sin_values) / 2. + 0.5, (xx * sin_values + yy * cos_values) / 2. + 0.5
