
import pathlib
import h5py
import matplotlib
import matplotlib.patches as mpatches
import numpy as np
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from enum import Enum
from helper_functions import __file__
from helper_functions.plotting_lib import round_sig_decimal
MODULE_BASE = pathlib.Path(__file__).parent.parent


if __name__ == '__main__':
    gpc_data_path = MODULE_BASE.joinpath('reference_data/Miller_2025/dendritic_current/dendritic_current_results.hdf5')
    save_folder = pathlib.Path('/path/to/folder')
    # ################################ Load data file ##################################################################
    t_max = 30  # ms
    clockwise = True
    colormap = 'RdYlBu_r'
    SAVE_TITLE = False
    SAVE = False
    NANO_FACTOR = 1e9
    chosen_intensities = [175, 225, 250, 300, 350, 400]
    current_matrices = []
    max_current = 0
    with h5py.File(gpc_data_path, 'r') as f_data:
        theta_mesh = f_data['theta_mesh'][:].astype('float64')  # radians
        r_mesh = f_data['r_mesh'][:].astype('float64')  # ms
        for intensity in chosen_intensities:
            current_matrices.append(f_data[f'i_{intensity}'][:].astype('float64') * NANO_FACTOR)
            if np.max(current_matrices[-1]) > max_current:
                max_current = np.max(current_matrices[-1])
    current_matrices = np.array(current_matrices)
    theta_values_degrees = np.rad2deg(theta_mesh[:, 0])  # degrees
    time_values = r_mesh[0, :]  # ms

    # ######################################## Plot Individual Figure Components #######################################
    plt.ioff()
    cbar_max = round_sig_decimal(max_current)
    plt.rcParams['font.size'] = 16
    # Layer Horizontal
    fig_cbar = plt.figure(figsize=(8.5, 1.5))
    ax_cbar = fig_cbar.add_subplot(111)
    mappable = matplotlib.cm.ScalarMappable(
        norm=matplotlib.colors.Normalize(vmin=0, vmax=cbar_max),
        cmap=colormap
    )
    fig_cbar.colorbar(mappable, cax=ax_cbar, label='Dendritic Current (nA)',
                      orientation='horizontal')
    fig_cbar.tight_layout()
    if SAVE:
        fig_cbar.savefig(save_folder.joinpath('colorbar_current_horizontal.png'), dpi=400, transparent=True)
    plt.close(fig_cbar)
    # ##################################################################################################################
    print('Plotting Theta Dependence')
    for counter, intensity in enumerate(chosen_intensities):
        plt.ioff()
        plt.rcParams['font.size'] = 20
        fig_polar = plt.figure()
        ax = plt.subplot(1, 1, 1, projection='polar')
        if clockwise:
            ax.set_theta_direction(-1)
        else:
            ax.set_theta_direction(1)
        ax.set_theta_offset(np.pi / 2.0)
        levels = np.linspace(0, cbar_max, 200)
        contour = ax.contourf(theta_mesh, r_mesh, current_matrices[counter], levels, cmap=colormap, vmin=0, vmax=cbar_max)
        ax.set_thetamin(0)
        ax.set_thetamax(180)
        ax.set_rmax(t_max)
        ax.set_rticks(np.arange(0, t_max + 10, 10))
        ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180]))
        ax.tick_params(axis='y', pad=5)
        ax.tick_params(axis='x', pad=15)
        ax.grid(axis='y', color='k', alpha=0.7)
        ax.grid(axis='x', color='k', alpha=0.7)
        title = '$|\mathbf{E}|$' + f' = {intensity} V/m'
        if SAVE_TITLE:
            ax.set_title(title, pad=15)
        fig_polar.tight_layout()
        if SAVE:
            fig_polar.savefig(save_folder.joinpath(f'current_polarplot_intensity_{intensity}_theta_dep.png'), dpi=600, transparent=True)
        ax.set_title(title, pad=15)
        ax.set_ylabel('t (ms)')
        fig_polar.tight_layout()
        plt.close(fig_polar)

        plt.ion()
        if intensity in [250, 400]:
            # Plot Crosscut at 60 and 150 degrees
            plt.rcParams['font.size'] = 26
            for crosscut_theta in [150]:
                mask_theta = np.isclose(theta_values_degrees, crosscut_theta)
                theta_index = np.where(mask_theta)[0][0]
                fig_crosscut = plt.figure(figsize=(8, 5))
                ax_cross = fig_crosscut.add_subplot(111)
                ax_cross.plot(time_values, current_matrices[counter, theta_index, :],
                              color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1], linewidth=5)
                ax_cross.set_xlabel('t (ms)', labelpad=15)
                ax_cross.set_ylabel('Dendritic Current (nA)', labelpad=15)
                ax_cross.set_xlim((-1, t_max + 1))
                ax_cross.set_ylim(bottom=-0.01, top=cbar_max)
                ax_cross.set_xticks(np.arange(0, t_max + 10, 10))
                ax_cross.set_yticks(ax_cbar.get_xticks())
                # ax_cross.set_ylim((np.min(kernel_partial_matrix[idx_95, :]), vmax))
                # ax_cross.set_yticks([0, 0.5, 1.0, 1.5, 2.0])
                ax_cross.spines['right'].set_visible(False)
                ax_cross.spines['top'].set_visible(False)
                if SAVE_TITLE:
                    ax_cross.set_title(f'{title} $\\theta$ = {crosscut_theta}$^\\circ$')
                fig_crosscut.tight_layout()
                if SAVE:
                    fig_crosscut.savefig(save_folder.joinpath(f'current_polarplot_crosscut_intensity_{intensity}_theta_{crosscut_theta}.png'), dpi=600, transparent=True)
                ax_cross.set_title(f'{title} $\\theta$ = {crosscut_theta}$^\\circ$')
                fig_crosscut.tight_layout()

    # ################################# Plot Fig 12 ####################################################################
    plt.rcParams['font.size'] = 12
    fig_c = plt.figure(figsize=(7, 7.5))
    widths = [1, 1, 1]
    heights = [1, 0.8, 1, 0.4, 0.15]
    spec = fig_c.add_gridspec(ncols=3, nrows=5, width_ratios=widths, height_ratios=heights)
    ax_175 = fig_c.add_subplot(spec[0, 0], projection='polar')
    ax_225 = fig_c.add_subplot(spec[0, 1], projection='polar')
    ax_250 = fig_c.add_subplot(spec[0, 2], projection='polar')
    ax_300 = fig_c.add_subplot(spec[2, 0], projection='polar')
    ax_350 = fig_c.add_subplot(spec[2, 1], projection='polar')
    ax_400 = fig_c.add_subplot(spec[2, 2], projection='polar')
    ax_cbar_c = fig_c.add_subplot(spec[4, 0:3])
    axes_list = [ax_175, ax_225, ax_250, ax_300, ax_350, ax_400]

    fig_cbar.colorbar(mappable, cax=ax_cbar_c, label='Dendritic Current (nA)',
                      orientation='horizontal')

    for figax, intensity, current_matrix in zip(axes_list, chosen_intensities, current_matrices):
        figax.set_theta_direction(-1)
        figax.set_theta_offset(np.pi / 2.0)
        levels = np.linspace(0, cbar_max, 200)
        contour = figax.contourf(theta_mesh, r_mesh, current_matrix, levels, cmap=colormap, vmin=0, vmax=cbar_max)
        figax.set_thetamin(0)
        figax.set_thetamax(180)
        figax.set_rmax(t_max)
        figax.set_rticks(np.arange(0, t_max + 10, 10))
        figax.set_xticks(np.deg2rad([0, 45, 90, 135, 180]))
        figax.tick_params(axis='y', pad=5)
        figax.tick_params(axis='x', pad=15)
        figax.grid(axis='y', color='k', alpha=0.7)
        figax.grid(axis='x', color='k', alpha=0.7)
        title = '$|\mathbf{E}|$' + f' = {intensity} V/m'
        figax.set_title(title, pad=5)
        figax.set_ylabel('t (ms)')
    fig_c.subplots_adjust(top=0.85)
    fig_c.suptitle('Fig 12. Dendritic Current Directional Sensitivity')

print('fin :)')
