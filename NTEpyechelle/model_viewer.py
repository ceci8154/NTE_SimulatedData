import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

from NTEpyechelle.optics import convert_matrix
from NTEpyechelle.simulator import available_models
from NTEpyechelle.spectrograph import Spectrograph, ZEMAX


def plot_transformations(spectrograph: Spectrograph, fiber: int = 1, ccd_index: int = 1):
    """ Plot affine transformation matrices

    Args:
        spectrograph: Spectrograph model

    Returns:

    """
    fig, ax = plt.subplots(2, 3, sharex=True)
    fig.suptitle(f"Affine transformations of {spectrograph.name}")
    if isinstance(spectrograph, ZEMAX):
        for o in spectrograph.get_orders(fiber, ccd_index):
            ax[0, 0].set_title("sx")
            ax[0, 0].plot([af.sx for af in spectrograph.transformations(o, fiber, ccd_index)])
            ax[0, 1].set_title("sy")
            ax[0, 1].plot([af.sy for af in spectrograph.transformations(o, fiber, ccd_index)])
            ax[0, 2].set_title("shear")
            ax[0, 2].plot([af.shear for af in spectrograph.transformations(o, fiber, ccd_index)])
            ax[1, 0].set_title("rot")
            ax[1, 0].plot([af.rot for af in spectrograph.transformations(o, fiber, ccd_index)])
            ax[1, 1].set_title("tx")
            ax[1, 1].plot([af.tx for af in spectrograph.transformations(o, fiber, ccd_index)])
            ax[1, 2].set_title("ty")
            ax[1, 2].plot([af.ty for af in spectrograph.transformations(o, fiber, ccd_index)])
    else:
        raise NotImplementedError
    return fig


def plot_transformation_matrices(spectrograph: Spectrograph, fiber: int = 1, ccd_index: int = 1):
    """ Plot affine transformation matrices

    Args:
        fiber: fiber index
        ccd_index: ccd index
        spectrograph: Spectrograph model

    Returns:

    """
    fig, ax = plt.subplots(2, 3, sharex=True)
    fig.suptitle(f"Affine transformation matrices of {spectrograph.name}")
    for o in spectrograph.get_orders(fiber, ccd_index):
        if isinstance(spectrograph, ZEMAX):
            transformations = convert_matrix(
                np.array([tm.as_matrix() for tm in spectrograph.transformations(o, fiber, ccd_index)]).T)
            ax[0, 0].set_title("m0")
            ax[0, 0].plot(transformations[0])
            ax[0, 1].set_title("m1")
            ax[0, 1].plot(transformations[1])
            ax[0, 2].set_title("m2")
            ax[0, 2].plot(transformations[2])
            ax[1, 0].set_title("m3")
            ax[1, 0].plot(transformations[3])
            ax[1, 1].set_title("m4")
            ax[1, 1].plot(transformations[4])
            ax[1, 2].set_title("m5")
            ax[1, 2].plot(transformations[5])
        else:
            raise NotImplementedError
    return fig


def plot_psfs(spectrograph: Spectrograph, fiber: int = 1, ccd_index: int = 1):
    """ Plot PSFs as one big map
    Args:
        fiber: fiber index
        ccd_index: ccd index
        spectrograph: Spectrograph model

    Returns:

    """

    fig, ax = plt.subplots()
    orders = spectrograph.get_orders(fiber, ccd_index)
    n_orders = len(orders)
    if isinstance(spectrograph, ZEMAX):
        n_psfs = len(spectrograph.psfs(orders[0], fiber, ccd_index))
    else:
        n_psfs = 10

    shape_psfs = spectrograph.get_psf(sum(spectrograph.get_wavelength_range(orders[0], fiber, ccd_index)) / 2.,
                                      orders[0], fiber, ccd_index).data.shape

    # shape_psfs = spectrograph.psfs[next(spectrograph.psfs.keys().__iter__())].psfs[0].data.shape
    img = np.empty((n_psfs * shape_psfs[0], n_orders * shape_psfs[1]))

    for oo, o in enumerate(np.sort(orders)):
        if isinstance(spectrograph, ZEMAX):
            psfs = spectrograph.psfs(o, fiber, ccd_index)
        else:
            wl = np.linspace(*spectrograph.get_wavelength_range(o, fiber, ccd_index), num=n_psfs)
            psfs = [spectrograph.get_psf(w, o, fiber, ccd_index) for w in wl]

        for i, p in enumerate(psfs):
            if p.data.shape == shape_psfs:
                img[int(i * shape_psfs[0]):int((i + 1) * shape_psfs[0]),
                int(oo * shape_psfs[1]):int((oo + 1) * shape_psfs[1])] = p.data
    ax.imshow(img, vmin=0, vmax=np.mean(img) * 10.0)
    return fig


# def plot_fields(spec: ZEMAX, show=True):
#     plt.figure()
#     with h5py.File(spec.modelpath, 'r') as h5f:
#         for k in h5f.keys():
#             if 'fiber_' in k:
#                 a = h5f[k].attrs['norm_field'].decode('utf-8').split('\n')
#
#                 for b in a:
#                     if 'aF' in b:
#                         print(b[2:])


def main(args):
    if not args:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(description='PyEchelle Simulator Model Viewer')
    parser.add_argument('-s', '--spectrograph', choices=available_models, type=str, default="MaroonX", required=True,
                        help=f"Filename of spectrograph model. Model file needs to be located in models/ folder. ")
    parser.add_argument('--fiber', type=int, default=1, required=False)
    parser.add_argument('--show', action='store_true')

    args = parser.parse_args(args)
    spec = ZEMAX(args.spectrograph)
    # if args.plot_transformations:
    plot_transformations(spec, args.fiber)
    # if args.plot_transformation_matrices:
    plot_transformation_matrices(spec, args.fiber)
    # if args.plot_field:
    # plot_fields(spec)
    # if args.plot_psfs:
    plot_psfs(spec)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
