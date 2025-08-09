"""
Algorithm: Static_IO
==============================
"""
# Windows users have to encapsulate the code into a main function to avoid multiprocessing errors.
# def main():
import pygpc
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import os
import matplotlib.pyplot as plt
import pathlib
from enum import Enum
import h5py
from collections import OrderedDict
from helper_functions import __file__
MODULE_BASE = pathlib.Path(__file__).parent.parent


if __name__ == '__main__':
    fn_results = str(MODULE_BASE.joinpath('reference_data/Miller_2025/dendritic_current/gpc'))
    fn_results = '/data/pt_02872/studies/MEP_modeling/gpc/gpc_order_5_int_3_full_with_zeros_4000_L1/gpc'   # filename of output
    save_session_format = ".pkl"    # file format of saved gpc session ".hdf5" (slow) or ".pkl" (fast)
    np.random.seed(1)

    #%%
    # Setup input and output data
    # ----------------------------------------------------------------------------------------------------------------
    # load data (4000)
    fn_data = MODULE_BASE.joinpath('/reference_data/Miller_2025/dendritic_current/merged_dendritic_delay_current_2_4000.hdf5')
    with h5py.File(fn_data, 'r') as f:
        t = f['t'][:]  # 10001 (Units ms)
        coords = f['params'][:]  # 1000 x 6 (Units [degrees, degrees, %/mm, fraction, fraction, fraction])
        results = f['current'][:]  # 1000 x 10001  (Units Amps)

    # load data (validation set)
    fn_data = MODULE_BASE.joinpath('reference_data/Miller_2025/dendritic_current/merged_dendritic_delay_current_3_1000.hdf5')
    with h5py.File(fn_data, 'r') as f:
        t_val = f['t'][:]  # 10001 (Units ms)
        coords_val = f['params'][:]  # 1000 x 6 (Units [degrees, degrees, %/mm, fraction, fraction, fraction])
        results_val = f['current'][:]  # 1000 x 10001  (Units Amps)


    # filter out values < 7.5 * 1e-11
    #mask = np.logical_not(np.all(results < (7.5*1e-11), axis=1))
    #results = results[mask, :]
    #coords = coords[mask, :]

    #mask_gradient = coords[:, 1] == 0
    #results = results[mask_gradient, :]
    #coords = coords[mask_gradient, :]
    #coords = np.delete(coords, 1, 1)

    # take every 20th t value
    results = results[:, 1::20]
    results_val = results_val[:, 1::20]
    t = t[1::20]

    results_max = np.max(results)
    results = results/results_max
    results_val = results_val/results_max

    #results = results[:, t<50]
    #results_val = results_val[:, t<50]
    #t = t[t<50]

    #col_enum = Enum('Columns', [col_label for col_label in cols], start=0)
    # Parameters: theta, gradient, intensity, fraction_nmda, fraction_gaba_a, fraction_ex
    # You can also get the column index of each variable using the Enum
    #theta_column_index = col_enum['theta'].value

    # define the properties of the random variables
    parameters = OrderedDict()
    parameters["theta"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 180])
    parameters["gradient"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-20, 20])
    parameters["intensity"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[100, 400])
    parameters["fraction_nmda"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0.25, 0.75])
    parameters["fraction_gaba_a"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0.9, 1.0])
    parameters["fraction_ex"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0.2, 0.8])

    # generate a grid object from the input data
    grid = pygpc.RandomGrid(parameters_random=parameters, coords=coords)

    # create validation set
    grid_val = pygpc.RandomGrid(parameters_random=parameters, coords=coords_val)
    validation = pygpc.ValidationSet(grid=grid_val, results=results_val)

    #%%
    # Setting up the algorithm
    # ------------------------

    # gPC options
    options = dict()
    options["method"] = "reg"
    options["solver"] = "LarsLasso"  #  "LarsLasso"  "Moore-Penrose"
    options["settings"] = None
    options["order"] = [20, 6, 20, 20, 20, 20]
    options["order_max"] = 8
    options["interaction_order"] = 3
    options["error_type"] = "nrmsd"
    options["error_norm"] = "absolute" # "absolute" "relative"
    options["n_samples_validation"] = None
    options["fn_results"] = fn_results
    options["save_session_format"] = save_session_format
    options["backend"] = "omp"
    options["verbose"] = True

    # determine number of gPC coefficients (hint: compare it with the amount of output data you have)
    n_coeffs = pygpc.get_num_coeffs_sparse(order_dim_max=options["order"],
                                           order_glob_max=options["order_max"],
                                           order_inter_max=options["interaction_order"],
                                           dim=len(parameters))

    # interaction_order = 2
    # order 3: n_coeffs = 64
    # order 4: n_coeffs = 115
    # order 5: n_coeffs = 181
    # order 6: n_coeffs = 262
    # order 7: n_coeffs = 358
    # order 8: n_coeffs = 469

    # define algorithm
    algorithm = pygpc.Static_IO(parameters=parameters, options=options, grid=grid, results=results, validation=validation)

    #%%
    # Running the gpc
    # ---------------

    # initialize gPC Session
    session = pygpc.Session(algorithm=algorithm)

    # run gPC algorithm
    session, coeffs, results = session.run()

    results_test = session.gpc[0].get_approximation(coeffs, grid.coords_norm)

    plt.figure()
    plt.plot(t, results[70, :])
    plt.plot(t, results_test[70, :])
    plt.show()

    plt.plot(t, results[10, :])
    plt.plot(t, results_test[10, :])

    plt.plot(t, results[20, :])
    plt.plot(t, results_test[20, :])

    plt.plot(t, results[50, :])
    plt.plot(t, results_test[50, :])

    plt.plot(t, results[150, :])
    plt.plot(t, results_test[150, :])

    plt.plot(t, results[250, :])
    plt.plot(t, results_test[250, :])

    plt.plot(t, results[350, :])
    plt.plot(t, results_test[350, :])

    #%%
    # Postprocessing
    # --------------

    # read session
    session = pygpc.read_session(fname=fn_results + ".pkl", folder=None)
    with h5py.File(pathlib.Path(fn_results + ".hdf5"), "r") as f:
        coeffs = f["coeffs"][:]

    # Post-process gPC
    pygpc.get_sensitivities_hdf5(fn_gpc=options["fn_results"],
                                 output_idx=None,
                                 calc_sobol=True,
                                 calc_global_sens=True,
                                 calc_pdf=True,
                                 algorithm="standard")

    # get a summary of the sensitivity coefficients
    sobol, gsens = pygpc.get_sens_summary(fn_results, parameters)
    print(sobol)
    print(gsens)

    # plot gPC approximation and IO data
    pygpc.plot_gpc(session=session,
                   coeffs=coeffs,
                   random_vars=["theta", "intensity"],
                   output_idx=10,
                   n_grid=[100, 100],
                   coords=grid.coords,
                   results=results)

    pygpc.plot_gpc(session=session,
                   coeffs=coeffs,
                   random_vars=["fraction_ex", "fraction_nmda"],
                   output_idx=10,
                   n_grid=[100, 100],
                   coords=grid.coords,
                   results=results)

    # plot time course of mean together with probability density, sobol sensitivity coefficients and global derivatives
    pygpc.plot_sens_summary(session=session,
                            coeffs=coeffs,
                            sobol=sobol,
                            gsens=gsens,
                            plot_pdf_over_output_idx=False,
                            qois=t,
                            mean=pygpc.SGPC.get_mean(coeffs),
                            std=pygpc.SGPC.get_std(coeffs),
                            x_label="t in s",
                            y_label="i(t)"
                            )# zlim=[0, 0.4]

    # On Windows subprocesses will import (i.e. execute) the main module at start.
    # You need to insert an if __name__ == '__main__': guard in the main module to avoid
    # creating subprocesses recursively.
    #
    # if __name__ == '__main__':
    #     main()

    plt.errorbar(t, pygpc.SGPC.get_mean(coeffs), pygpc.SGPC.get_std(coeffs))
    plt.errorbar(t, np.mean(results, axis=0), np.std(results, axis=0), marker="o", elinewidth=0.5)

    plt.plot(t, pygpc.SGPC.get_mean(coeffs))
    plt.plot(t, pygpc.SGPC.get_mean(coeffs)+pygpc.SGPC.get_std(coeffs), "k--")
    plt.plot(t, pygpc.SGPC.get_mean(coeffs)-pygpc.SGPC.get_std(coeffs), "k--")

    # plot mean and pdf over time
    ########################################################################################################################
    mean = pygpc.SGPC.get_mean(coeffs)
    std = pygpc.SGPC.get_std(coeffs)

    output_idx = np.arange(0, coeffs.shape[1])
    pdf_x, pdf_y, _, y_gpc_samples = session.gpc[0].get_pdf(coeffs=coeffs,
                                                            n_samples=1e5,
                                                            output_idx=output_idx,
                                                            return_samples=True)
    x_interp = np.linspace(0, np.max(pdf_x), 500)
    y_interp = np.zeros((len(x_interp), np.shape(pdf_x)[1]))
    for i in range(np.shape(pdf_x)[1]):
        y_interp[:, i] = np.interp(x_interp, pdf_x[:, i], pdf_y[:, i], left=0, right=0)
    vmin = 0  # np.min(y_interp)
    vmax = 15# np.max(y_interp)
    x_axis = t
    xx, yy = np.meshgrid(x_axis, x_interp)

    # plot pdf over output_idx
    fig, (ax1) = plt.subplots(1, 1, figsize=[5.47, 4.09])

    ax1.pcolor(xx, yy, y_interp, cmap="bone_r", vmin=vmin, vmax=vmax)
    ax1.plot(x_axis, mean, "r", linewidth=1.5)
    ax1.plot(x_axis, mean+std, "b--", linewidth=1.5)
    ax1.plot(x_axis, mean-std, "b--", linewidth=1.5)
    legend_elements = [matplotlib.lines.Line2D([0], [0], color='r', lw=2, label='mean'),
                       matplotlib.patches.Patch(facecolor='grey', edgecolor='k', label='pdf'),
                       matplotlib.lines.Line2D([0], [0], color='b', lw=2, label='std', linestyle="--")]
    ax1.legend(handles=legend_elements, loc=1, fontsize=12)
    ax1.set_xlabel("$t$ in ms", fontsize=13)
    ax1.set_ylabel("Dendritic current", fontsize=13)
    ax1.grid()
    ax1.set_xlim([0, 50])
    ax1.set_ylim([0, 0.5])
    plt.tight_layout()

    plt.savefig(os.path.join(os.path.split(fn_results)[0], "mean_pdf.png"), dpi=600)

    # plot average normalized Sobol indices
    ########################################################################################################################
    sobol_mean = np.mean(np.array(sobol)[:, t<50], axis=1)
    sobol_sort_idx = np.flip(np.argsort(sobol_mean))
    sobol_names = list(sobol.index)
    gsens_names = list(gsens.index)
    sobol_names_sorted = [sobol_names[i] for i in sobol_sort_idx]
    sobol_mean_sorted = sobol_mean[sobol_sort_idx]
    sobol_mean_sorted_condensed = np.hstack((sobol_mean_sorted[:9], 1-np.sum(sobol_mean_sorted[:9])))
    sobol_names_sorted_condensed = [sobol_names_sorted[i] for i in range(9)] + ["remaining"]
    sobol_names_sorted_condensed = [s.replace("[", "") for s in sobol_names_sorted_condensed]
    sobol_names_sorted_condensed = [s.replace("]", "") for s in sobol_names_sorted_condensed]
    sobol_names_sorted_condensed = [s.replace("'", "") for s in sobol_names_sorted_condensed]

    sobol_names_sorted_condensed = [s.replace("fraction_ex", "$f_{ex}$") for s in sobol_names_sorted_condensed]
    sobol_names_sorted_condensed = [s.replace("fraction_nmda", "$f_{NMDA}$") for s in sobol_names_sorted_condensed]
    sobol_names_sorted_condensed = [s.replace("fraction_gaba", "$f_{GABAa}$") for s in sobol_names_sorted_condensed]
    sobol_names_sorted_condensed = [s.replace("intensity", "$|\\mathbf{E}|$") for s in sobol_names_sorted_condensed]
    sobol_names_sorted_condensed = [s.replace("theta", "$\\theta$") for s in sobol_names_sorted_condensed]
    sobol_names_sorted_condensed = [s.replace("gradient", "$\\Delta|\\mathbf{E}|$") for s in sobol_names_sorted_condensed]

    gsens_names = [s.replace("fraction_ex", "$f_{ex}$") for s in gsens_names]
    gsens_names = [s.replace("fraction_nmda", "$f_{NMDA}$") for s in gsens_names]
    gsens_names = [s.replace("fraction_gaba", "$f_{GABAa}$") for s in gsens_names]
    gsens_names = [s.replace("intensity", "$|\\mathbf{E}|$") for s in gsens_names]
    gsens_names = [s.replace("theta", "$\\theta$") for s in gsens_names]
    gsens_names = [s.replace("gradient", "$\\Delta|\\mathbf{E}|$") for s in gsens_names]

    fig, ax1 = plt.subplots(1, 1, figsize=(7.7, 5))
    wedgeprops = {"linewidth": 0.5, 'width': 0.5, "edgecolor": "k"}
    wedges, texts = ax1.pie(sobol_mean_sorted_condensed, wedgeprops=wedgeprops, startangle=60)
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="w", lw=0.72)
    kw = dict(arrowprops=dict(arrowstyle="-"), bbox=bbox_props, zorder=0, va="center", fontsize=12)

    last_label = False
    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1) / 2. + p.theta1

        if not last_label:
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))
            horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
            connectionstyle = "angle,angleA=0,angleB={}".format(ang)
            kw["arrowprops"].update({"connectionstyle": connectionstyle})
            ax1.annotate(sobol_names_sorted_condensed[i] + f" ({sobol_mean_sorted_condensed[i]*100:.1f}%)", xy=(x, y), xytext=(1.35 * np.sign(x), 1.4 * y),
                        horizontalalignment=horizontalalignment, **kw)
            # if ang > 310:
            #     last_label = True

    # ax1.set_title("Normalized Sobol indices")
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.split(fn_results)[0], "sobol_mean.png"), dpi=600)

    # plot normalized Sobol indices over time
    ########################################################################################################################
    plt.figure(figsize=[5.6, 5])
    sobol_time = np.array(sobol)
    sobol_time_sorted = sobol_time[sobol_sort_idx, :]
    sobol_time_sorted_condensed = np.vstack((sobol_time_sorted[:9, :], 1-np.sum(sobol_time_sorted[:9, :], axis=0)))
    plt.plot(t, sobol_time_sorted_condensed.T)
    plt.xlim([0, 50])
    plt.xlabel("$t$ in ms", fontsize=13)
    plt.ylabel("Normalized Sobol indices", fontsize=13)
    plt.grid()
    plt.legend(sobol_names_sorted_condensed, loc='center left', bbox_to_anchor=(-0.01, 1.15), ncol=4)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.split(fn_results)[0], "sobol_time.png"), dpi=600)

    # plot global derivatives over time
    ########################################################################################################################
    plt.figure(figsize=[5.652, 4.75])
    gsens_time = np.array(gsens)
    plt.plot(t, gsens_time.T)
    plt.xlim([0, 50])
    plt.xlabel("$t$ in ms", fontsize=13)
    plt.ylabel("Global Derivatives", fontsize=13)
    plt.grid()
    plt.legend(gsens_names, loc='center left', bbox_to_anchor=(-0.01, 1.115), ncol=4)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.split(fn_results)[0], "gsens_time.png"), dpi=600)
