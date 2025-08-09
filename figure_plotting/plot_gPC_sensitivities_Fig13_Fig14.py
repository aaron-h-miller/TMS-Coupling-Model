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

    fn_results = '/reference_data/Miller_2025/dendritic_current/gpc'   # filename of output

    # load time axis
    fn_data = MODULE_BASE.joinpath('reference_data/Miller_2025/dendritic_current/merged_dendritic_delay_current_2_4000.hdf5')
    with h5py.File(fn_data, 'r') as f:
        t = f['t'][:]   # 10001 (Units ms)
        t = t[1::20]    # subsample t

    # read session
    session = pygpc.read_session(fname=fn_results + ".pkl", folder=None)
    with h5py.File(pathlib.Path(fn_results + ".hdf5"), "r") as f:
        coeffs = f["coeffs"][:]

    # define the properties of the random variables
    parameters = OrderedDict()
    parameters["theta"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0, 180])
    parameters["gradient"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[-20, 20])
    parameters["intensity"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[100, 400])
    parameters["fraction_nmda"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0.25, 0.75])
    parameters["fraction_gaba_a"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0.9, 1.0])
    parameters["fraction_ex"] = pygpc.Beta(pdf_shape=[1, 1], pdf_limits=[0.2, 0.8])

    # get a summary of the sensitivity coefficients
    sobol, gsens = pygpc.get_sens_summary(fn_results, parameters)
    print(sobol)
    print(gsens)

    # plot mean and pdf over time
    ############################################## Fig 13A #################################################################
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

    #plt.savefig(os.path.join(os.path.split(fn_results)[0], "mean_pdf.png"), dpi=600)

    # plot average normalized Sobol indices
    ####################################### Fig 14 #########################################################################
    sobol_vals = np.array(list(sobol.values()))
    sobol_mean = np.mean(sobol_vals[:, t<50], axis=1)
    sobol_sort_idx = np.flip(np.argsort(sobol_mean))
    sobol_names = list(sobol.keys())
    gsens_names = list(gsens.keys())
    sobol_names_sorted = [sobol_names[i] for i in sobol_sort_idx]
    sobol_mean_sorted = sobol_mean[sobol_sort_idx]
    sobol_mean_sorted_condensed = np.hstack((sobol_mean_sorted[:9], 1-np.sum(sobol_mean_sorted[:9])))
    sobol_names_sorted_condensed = [sobol_names_sorted[i] for i in range(9)] + ["remaining"]
    sobol_names_sorted_condensed = [s.replace("[", "") for s in sobol_names_sorted_condensed]
    sobol_names_sorted_condensed = [s.replace("]", "") for s in sobol_names_sorted_condensed]
    sobol_names_sorted_condensed = [s.replace("'", "") for s in sobol_names_sorted_condensed]

    sobol_names_sorted_condensed = [s.replace("fraction_ex", "$f_{ex}$") for s in sobol_names_sorted_condensed]
    sobol_names_sorted_condensed = [s.replace("fraction_nmda", "$f_{NMDA}$") for s in sobol_names_sorted_condensed]
    sobol_names_sorted_condensed = [s.replace("fraction_gaba_a", "$f_{GABAa}$") for s in sobol_names_sorted_condensed]
    sobol_names_sorted_condensed = [s.replace("intensity", "$|\\mathbf{E}|$") for s in sobol_names_sorted_condensed]
    sobol_names_sorted_condensed = [s.replace("theta", "$\\theta$") for s in sobol_names_sorted_condensed]
    sobol_names_sorted_condensed = [s.replace("gradient", "$\\Delta|\\mathbf{E}|$") for s in sobol_names_sorted_condensed]

    gsens_names = [s.replace("fraction_ex", "$f_{ex}$") for s in gsens_names]
    gsens_names = [s.replace("fraction_nmda", "$f_{NMDA}$") for s in gsens_names]
    gsens_names = [s.replace("fraction_gaba_a", "$f_{GABAa}$") for s in gsens_names]
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
    # plt.savefig(os.path.join(os.path.split(fn_results)[0], "sobol_mean.png"), dpi=600)

    # plot normalized Sobol indices over time
    ################################################ Fig 13B ###############################################################
    plt.figure(figsize=[5.6, 5])
    sobol_time = np.array(list(sobol.values()))
    sobol_time_sorted = sobol_time[sobol_sort_idx, :]
    sobol_time_sorted_condensed = np.vstack((sobol_time_sorted[:9, :], 1-np.sum(sobol_time_sorted[:9, :], axis=0)))
    plt.plot(t, sobol_time_sorted_condensed.T)
    plt.xlim([0, 50])
    plt.xlabel("$t$ in ms", fontsize=13)
    plt.ylabel("Normalized Sobol indices", fontsize=13)
    plt.grid()
    plt.legend(sobol_names_sorted_condensed, loc='center left', bbox_to_anchor=(-0.01, 1.15), ncol=4)
    plt.tight_layout()
    #plt.savefig(os.path.join(os.path.split(fn_results)[0], "sobol_time.png"), dpi=600)

    # plot global derivatives over time
    ################################################## Fig 13C #############################################################
    plt.figure(figsize=[5.652, 4.75])
    gsens_time = np.array(list(gsens.values()))
    line = plt.plot(t, gsens_time.T)
    line[0].set_color("C7")
    line[1].set_color("C6")
    line[2].set_color("C2")
    line[3].set_color("C4")
    line[4].set_color("black")
    line[5].set_color("C0")
    plt.xlim([0, 50])
    plt.xlabel("$t$ in ms", fontsize=13)
    plt.ylabel("Global Derivatives", fontsize=13)
    plt.grid()
    plt.legend(gsens_names, loc='center left', bbox_to_anchor=(-0.01, 1.115), ncol=4)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.split(fn_results)[0], "gsens_time.png"), dpi=600)


