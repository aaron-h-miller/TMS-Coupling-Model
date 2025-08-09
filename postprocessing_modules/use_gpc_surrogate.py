import matplotlib.pyplot as plt
import pygpc
import numpy as np
import os
import h5py
import matplotlib
import pathlib
matplotlib.use("Qt5Agg")
from helper_functions import __file__
MODULE_BASE = pathlib.Path(__file__).parent.parent


if __name__ == '__main__':
    SAVE = False
    # load time axis
    fn_data = str(MODULE_BASE.joinpath('reference_data/Miller_2025/dendritic_current/merged_dendritic_delay_current_2_4000.hdf5'))
    with h5py.File(fn_data, 'r') as f:
        t = f['t'][:]  # 10001 (Units ms)

    # take every 20th t value
    t = t[1::20]

    # read session
    ########################################################################################################################
    fn_session = str(MODULE_BASE.joinpath('reference_data/Miller_2025/dendritic_current/gpc.pkl'))
    session = pygpc.read_session(fname=fn_session)

    with h5py.File(os.path.splitext(fn_session)[0] + ".hdf5", "r") as f:
        coeffs = f["coeffs"][:]

    intensity_list = [150, 160, 170, 175, 180, 190, 200, 210, 220, 225, 230, 250, 275, 300, 325, 350, 375, 400]
    results_2D_reshaped_list = []
    i_scale = 5.148136e-09

    for _intensity in intensity_list:
        # create grid to compute the results on
        ########################################################################################################################
        theta = np.linspace(0, 180, 181)
        intensity = np.linspace(_intensity, _intensity, 1)

        theta_2D, intensity_2D = np.meshgrid(theta, intensity)

        theta_1D = theta_2D.flatten()
        intensity_1D = intensity_2D.flatten()
        gradient_1D = np.zeros(theta_1D.shape[0])
        fraction_nmda_1D = 0.5 * np.ones(theta_1D.shape[0])
        fraction_gaba_a_1D = 0.95 * np.ones(theta_1D.shape[0])
        fraction_ex_1D = 0.5 * np.ones(theta_1D.shape[0])

        coords = np.hstack((theta_1D[:, np.newaxis],
                            gradient_1D[:, np.newaxis],
                            intensity_1D[:, np.newaxis],
                            fraction_nmda_1D[:, np.newaxis],
                            fraction_gaba_a_1D[:, np.newaxis],
                            fraction_ex_1D[:, np.newaxis]))

        grid = pygpc.RandomGrid(parameters_random=session.parameters_random, coords=coords)

        # compute results using gpc surrogate model
        ########################################################################################################################
        results_1D = session.gpc[0].get_approximation(coeffs, grid.coords_norm) * i_scale
        results_1D[results_1D<0] = 0
        theta_2D_reshaped = np.reshape(theta_1D, theta_2D.shape)
        intensity_2D_reshaped = np.reshape(intensity_1D, intensity_2D.shape)
        results_2D_reshaped = np.reshape(results_1D, np.hstack((theta_2D.shape, coeffs.shape[1])))[0, :, :]
        results_2D_reshaped_list.append(results_2D_reshaped)

        # plot results
        ########################################################################################################################
        i_max = 0.25 * i_scale
        clockwise = 1
        radius_array = t
        theta_array = np.radians(theta)
        r_mesh, theta_mesh = np.meshgrid(radius_array, theta_array)  # cols unique in r, rows unique in phi
        # current_matrix should have rows containing each timeseries corresponding to the values of theta

        fig_polar = plt.figure()
        ax = plt.subplot(1, 1, 1, projection='polar')
        if clockwise:
            ax.set_theta_direction(-1)
        else:
            ax.set_theta_direction(1)
        ax.set_theta_offset(np.pi / 2.0)
        levels = np.linspace(0, i_max, 200)
        contour = ax.contourf(theta_mesh, r_mesh, results_2D_reshaped, levels, cmap="jet", vmin=0, vmax=i_max)
        ax.set_thetamin(0)
        ax.set_thetamax(180)
        ax.set_rmax(30)
        ax.set_ylabel('t (ms)')
        ax.set_rticks([0, 10, 20, 30])
        ax.tick_params(axis='y', pad=5)
        ax.tick_params(axis='x', pad=10)
        ax.grid(axis='y', color='k', alpha=0.7)
        ax.grid(axis='x', color='k', alpha=0.7)
        title = '$|\mathbf{E}|$' + f' = {intensity} V/m'
        # ax.set_title(title, pad=15)
        cbar = fig_polar.colorbar(contour, ax=ax, label='Dendritic Current (nA)', location='right', orientation='vertical', ticks=np.arange(0, i_max, 0.1), pad=0.005)
        fig_polar.tight_layout()

        plt.savefig(os.path.split(fn_session)[0] + f"/current_{_intensity}.png", dpi=600)
        plt.close()

    savepath = pathlib.Path('/path/to/folder')
    if SAVE:
        with h5py.File(savepath.joinpath("/dendritic_gPC_current_results.hdf5"), "w") as f:
            f.create_dataset(name=f"r_mesh", data=r_mesh)
            f.create_dataset(name=f"theta_mesh", data=theta_mesh)

            for i in range(len(intensity_list)):
                f.create_dataset(name=f"i_{intensity_list[i]}", data=results_2D_reshaped_list[i])
