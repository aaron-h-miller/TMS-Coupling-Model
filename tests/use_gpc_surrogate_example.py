import matplotlib.pyplot as plt
import pygpc
import numpy as np
import pathlib
import os
import h5py
import matplotlib
matplotlib.use("Qt5Agg")
from helper_functions import __file__
MODULE_BASE = pathlib.Path(__file__).parent.parent


if __name__ == '__main__':

    # time axis
    t = np.linspace(0.01, 99.81, 500)

    # scaling factor for current (gpc was done in normalized current space)
    i_scale = 5.148136e-09

    # read gpc session
    fn_session = str(MODULE_BASE.joinpath('reference_data/Miller_2025/dendritic_current/gpc.pkl'))
    session = pygpc.read_session(fname=fn_session)

    with h5py.File(os.path.splitext(fn_session)[0] + ".hdf5", "r") as f:
        coeffs = f["coeffs"][:]

    # create grid object to transform from real to normalized coordinates [-1, 1]
    theta = 0               # angle of e-field [0, 180]°
    gradient = 0            # relative gradient of e-field [-20, 20] %/mm
    intensity = 250         # intensity of e-field [100, 400] V/m
    fraction_nmda = 0.5     # fraction of nmda synapses [0.25, 0.75]
    fraction_gaba_a = 0.95  # fraction of nmda synapses [0.9, 1.0]
    fraction_ex = 0.5       # fraction of exc/ihn synapses [0.2, 0.8]

    coords = np.array([[theta, gradient, intensity, fraction_nmda, fraction_gaba_a, fraction_ex]])

    grid = pygpc.RandomGrid(parameters_random=session.parameters_random, coords=coords)

    # use gpc approximation to compute current
    current = session.gpc[0].get_approximation(coeffs, grid.coords_norm) * i_scale

    # plot current
    plt.plot(t, current.flatten())
