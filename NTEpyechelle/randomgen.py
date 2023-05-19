import numba
import numpy as np
from numba import int32, float32, njit, float64

#import matplotlib.pyplot as plt


@njit()
def unravel_index(index, shape):
    """ Unravels the index of a flat array
    Converts a flat index into a tuple of coordinate arrays. More or less equivalent to numpy.unravel_index()
    Args:
        index: An integer array whose elements are indices into the flattened version of an array of dimensions shape.
        shape: The shape of the array to use for unraveling indices.

    Returns:

    """
    out = []
    for dim in shape[::-1]:
        out.append(index % dim)
        index = index // dim
    return out[::-1]


@njit([numba.types.Tuple((float32[:], int32[:]))(float32[:]),
      numba.types.Tuple((float32[:], int32[:]))(float64[:])], nogil=True, parallel=True)
def make_alias_sampling_arrays(probability: np.ndarray):
    """
    As described `here <https://www.keithschwarz.com/darts-dice-coins/>`__, the most efficient way to draw random
    numbers from a discrete probability distribution are alias sampling methods.
    Here, we use a slightly adapted implementation of the Vose sampling method from
    `here <https://gist.github.com/jph00/30cfed589a8008325eae8f36e2c5b087>`__.

    Note:
        probability needs to be normalized, so that sum(probability)==1

    Examples:
        x = np.linspace(-1, 1, 10)
        probability = np.arange(10)
        q, j = make_alias_sampling_arrays(probability/np.sum(probability))

        k = int(np.floor(np.random.rand() * len(j)))
        x_random = x[k] if np.random.rand() < q[k] else x[j[k]]

    Args:
        probability: discrete probability vector

    Returns:

    """
    n = len(probability)
    q = np.zeros(n, dtype=np.float32)
    j = np.zeros(n, dtype=np.int32)

    # plt.figure()
    # plt.title('i start here')
    # plt.plot(probability)

    smaller, larger = [], []
    for kk, prob in enumerate(probability):
        q[kk] = n * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)
        
    # plt.figure()
    # plt.plot(smaller)
    # plt.figure()
    # plt.plot(larger)

    while len(smaller) > 0 and len(larger) > 0:
        small, large = smaller.pop(), larger.pop()
        j[small] = large
        q[large] = q[large] - (1.0 - q[small])
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    # plt.figure()
    # plt.plot(q)
    # plt.figure()
    # plt.title('i end here')
    # plt.plot(j)


    return q, j
