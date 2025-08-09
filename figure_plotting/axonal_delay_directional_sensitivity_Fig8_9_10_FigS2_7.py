import pathlib
import shutil
import matplotlib
import argparse
import matplotlib.patches as mpatches
import numpy as np
import h5py
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from postprocessing_modules.axonal_delay_kernel import get_axonal_delay_kernel
from helper_functions import __file__
MODULE_BASE = pathlib.Path(__file__).parent.parent


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-lr', '--layer', required=False, default=None)
    # args = parser.parse_args()
    # if args.layer is not None:
    #     source_layer = args.layer
    # else:
    #     source_layer = 'L5'
    source_layer = 'L23' # L23, L23_inh, L4, L4_inh, L5, L5_inh

    with h5py.File(MODULE_BASE.joinpath('reference_data/axonal_delay_reduced_biphasic_params.hdf5'), 'r') as f_par:
        params = f_par['params'][:]
    # ######################################## Plot Precomputed Kernels ################################################
    # intensity_vals = np.array(sorted(set(params[:, 2])), dtype=int)
    intensity_vals = np.array([300, 250, 225, 175], dtype=int)  # descending
    # theta_vals_unique = np.array(sorted(set(params[:, 0])), dtype=int)
    theta_vals = np.array([0, 42, 60, 90, 132, 180], dtype=int)
    # theta_vals = np.array([0, 60, 180], dtype=int)
    hist_cell_global_max = 0.0
    hist_layer_global_max = 0.0
    z_step = 100
    t_step = 0.1
    # t_max = kernels[0].t_kernel[-1]  # ms
    global_index = 0
    moon_max = 0
    density = True
    clockwise = True
    PLOT_2DS = True
    PLOT_MOONS = True
    PLOT_2D_COLORBARS = True
    PLOT_2D_AXIS_LABELS = True
    SHOW = True
    SAVE = False

    sweep_save_folder = pathlib.Path(f'/path/to/folder/')
    axonal_kernels_precomputed_folder = MODULE_BASE.joinpath('reference_data/Miller_2025/axon_kernels_precomputed/precomputed_kernels_50_0p1')
    # sweep_save_folder.mkdir(parents=True, exist_ok=True)

    # if source_layer.split('_')[-1] == 'inh':
    #     colormap = 'YlOrRd_r'
    # else:
    #     colormap = 'YlGnBu_r'
    kernels = []
    colormap = 'RdYlBu_r'

    if not SHOW:
        plt.ioff()
    else:
        plt.ion()
    print(f'Plot Sweeps of Theta and multiple Intensities: {source_layer}')
    partial_kernel_matrices = []
    counter = 1
    for idx_intens, intens in enumerate(intensity_vals):
        z_avg_kernel_matrix = []
        for index, thet in enumerate(theta_vals):
            print(f'Loading Kernel {counter}/{len(intensity_vals) * len(theta_vals)}')
            kernel = get_axonal_delay_kernel(source_layer_name=source_layer, delay_z_folder=pathlib.Path(''),
                                             precomputed_kernels_folder=axonal_kernels_precomputed_folder, theta=thet,
                                             gradient=0, intensity=intens, z_step=z_step, t_step=t_step,
                                             density=density, compute_fresh=False)
            kernels.append(kernel)
            z_avg_kernel_matrix.append(np.mean(kernel.layer_kernel, axis=0))

            if intens == 300:
                if np.max(kernel.layer_kernel) > hist_layer_global_max:
                    hist_layer_global_max = np.max(kernel.layer_kernel)
                if np.max(kernel.cell_kernel) > hist_cell_global_max:
                    hist_cell_global_max = np.max(kernel.cell_kernel)
            if intens == 400:
                if np.max(np.mean(kernel.layer_kernel, axis=0)) > moon_max:
                    moon_max = np.max(np.mean(kernel.layer_kernel, axis=0))
            # Compute partial distribution (avg. out z dependence)
            counter += 1
        partial_kernel_matrices.append(np.vstack(z_avg_kernel_matrix))

    t_max = 1.5
    t_min = kernels[0].t_kernel[0]
    radius_array = kernels[0].t_kernel
    theta_array = np.radians(theta_vals)
    r_mesh, theta_mesh = np.meshgrid(radius_array, theta_array)  # cols unique in r, rows unique in phi

    if source_layer == 'L5':
        hist_layer_global_max = 0.3
        hist_cell_global_max = 0.6
        moon_max = 0.12
    if source_layer == 'L23':
        hist_layer_global_max = 0.6
        hist_cell_global_max = 1.4
        moon_max = 0.16
    if source_layer == 'L23_inh':
        hist_layer_global_max = 0.3
        hist_cell_global_max = 0.45
        moon_max = 0.1

    # ############################################# Begin Plotting #####################################################
    print(f'{PLOT_MOONS=} {PLOT_2DS=}')
    # ############################################# Plot Big Colorbars #################################################
    plt.rcParams['font.size'] = 16
    # Layer Vertical
    fig_cbar = plt.figure(figsize=(8.5, 1.5))
    ax_cbar = fig_cbar.add_subplot(111)
    mappable = matplotlib.cm.ScalarMappable(
        norm=matplotlib.colors.Normalize(vmin=0, vmax=hist_layer_global_max),
        cmap=colormap
    )
    fig_cbar.colorbar(mappable, cax=ax_cbar, label='Spike Density (ms$^{-1}$ $\mu$m$^{-1}$)',
                      orientation='horizontal')
    fig_cbar.tight_layout()
    if SAVE:
        fig_cbar.savefig(sweep_save_folder.joinpath('colorbar_layer2D_horizontal.png'), dpi=400, transparent=True)
    # Layer Vertical
    fig_cbar = plt.figure(figsize=(1.65, 8.5))
    ax_cbar = fig_cbar.add_subplot(111)
    mappable = matplotlib.cm.ScalarMappable(
        norm=matplotlib.colors.Normalize(vmin=0, vmax=hist_layer_global_max),
        cmap=colormap
    )
    fig_cbar.colorbar(mappable, cax=ax_cbar, label='Spike Density (ms$^{-1}$ $\mu$m$^{-1}$)',
                      orientation='vertical')
    fig_cbar.tight_layout()
    if SAVE:
        fig_cbar.savefig(sweep_save_folder.joinpath('colorbar_layer2D_vertical.png'), dpi=400, transparent=True)
    # Cell Horizontal
    fig_cbar = plt.figure(figsize=(8.5, 1.5))
    ax_cbar = fig_cbar.add_subplot(111)
    mappable = matplotlib.cm.ScalarMappable(
        norm=matplotlib.colors.Normalize(vmin=0, vmax=hist_cell_global_max),
        cmap=colormap
    )
    fig_cbar.colorbar(mappable, cax=ax_cbar, label='Spike Density (ms$^{-1}$ $\mu$m$^{-1}$)',
                      orientation='horizontal')
    fig_cbar.tight_layout()
    if SAVE:
        fig_cbar.savefig(sweep_save_folder.joinpath('colorbar_cell2D_horizontal.png'), dpi=400, transparent=True)
    # Cell Vertical
    fig_cbar = plt.figure(figsize=(1.65, 8.5))
    ax_cbar = fig_cbar.add_subplot(111)
    mappable = matplotlib.cm.ScalarMappable(
        norm=matplotlib.colors.Normalize(vmin=0, vmax=hist_cell_global_max),
        cmap=colormap
    )
    fig_cbar.colorbar(mappable, cax=ax_cbar, label='Spike Density (ms$^{-1}$ $\mu$m$^{-1}$)',
                      orientation='vertical')
    fig_cbar.tight_layout()
    if SAVE:
        fig_cbar.savefig(sweep_save_folder.joinpath('colorbar_cell2D_vertical.png'), dpi=400, transparent=True)

    # Plot Moons Colorbar Horizontal
    plt.rcParams['font.size'] = 16
    fig_cbar = plt.figure(figsize=(8.5, 1.5))
    ax_cbar = fig_cbar.add_subplot(111)
    mappable = matplotlib.cm.ScalarMappable(
        norm=matplotlib.colors.Normalize(vmin=0, vmax=moon_max),
        cmap=colormap
    )
    fig_cbar.colorbar(mappable, cax=ax_cbar, label='Spike Density ($ms^{-1}$)',
                      orientation='horizontal')
    fig_cbar.tight_layout()
    if SAVE:
        fig_cbar.savefig(sweep_save_folder.joinpath('colorbar_polar_horizontal.png'), dpi=400,
                     transparent=True)
    plt.close(fig_cbar)
    # Moons Colorbar Vertical
    fig_cbar = plt.figure(figsize=(1.8, 8.5))
    ax_cbar = fig_cbar.add_subplot(111)
    mappable = matplotlib.cm.ScalarMappable(
        norm=matplotlib.colors.Normalize(vmin=0, vmax=moon_max),
        cmap=colormap
    )
    fig_cbar.colorbar(mappable, cax=ax_cbar, label='Spike Density ($ms^{-1}$)',
                      orientation='vertical')
    fig_cbar.tight_layout()
    if SAVE:
        fig_cbar.savefig(sweep_save_folder.joinpath('colorbar_polar_vertical.png'), dpi=400, transparent=True)
    ax_cbar.yaxis.set_label_position('left')
    ax_cbar.yaxis.set_ticks_position('left')
    fig_cbar.tight_layout()
    if SAVE:
        fig_cbar.savefig(sweep_save_folder.joinpath('colorbar_polar_vertical_left.png'), dpi=400, transparent=True)
    plt.close(fig_cbar)

    plt.close('all')
    # ####################################### Plot Kernels #############################################################
    for idx_intens, intens in enumerate(intensity_vals):
        print(f'Plotting Intensity {idx_intens + 1}/{len(intensity_vals)} | {intens}')
        subfolder_layer = sweep_save_folder.joinpath(f'lyravg_intensity_{intens}_2D_cbar_{PLOT_2D_COLORBARS}_labels_{PLOT_2D_AXIS_LABELS}')
        subfolder_layer.mkdir(exist_ok=True)
        subfolder_cell = sweep_save_folder.joinpath(f'cellavg_intensity_{intens}_2D_cbar_{PLOT_2D_COLORBARS}_labels_{PLOT_2D_AXIS_LABELS}')
        subfolder_cell.mkdir(exist_ok=True)
        for index, thet in enumerate(theta_vals):
            print(f'{index + 1}/{len(theta_vals)} | {thet}')
            plt.rcParams['font.size'] = 26
            # Plot 2D kernels w.r.t theta
            if PLOT_2DS:
                fig_obj = kernels[global_index].plot_avg_layer_kernel(histogram=False, dynamic=False, crosscuts=False, cmap=colormap, vmax=hist_layer_global_max)
                ax_layer = fig_obj.axes[0]
                ax_layer.set_xticks([0.5, 1, 1.5, 2.0], minor=False)
                ax_layer.set_xticks([0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75], minor=True)
                ax_layer.set_yticks([0, -1000, -2000], minor=False)
                ax_layer.set_yticks([0, -500, -1000, -1500, -2000, -2500], minor=True)
                ax_layer.set_xlim(t_min, t_max)
                ax_layer.set_ylim(-2500, 0)
                ax_layer.grid(axis='x', alpha=1, which='major', linewidth=1.5)
                ax_layer.grid(axis='y', alpha=1, which='major', linewidth=1.5)
                ax_layer.tick_params(axis='both', which='major', pad=10)
                # ax_layer.set_title(f'$|E|$ = {intens}', pad=20)
                fig_width, fig_height = fig_obj.get_figwidth(), fig_obj.get_figheight()
                _, _, ax_width, ax_height = ax_layer.get_position().bounds
                display_ratio = (fig_width * ax_width) / (fig_height * ax_height)  # fig_aspect * axis_aspect
                rect_height = 0.15 * display_ratio
                x_tail, y_tail = 0.825, 0.8
                dx = rect_height / 2 * np.sin(np.radians(thet))
                dy = rect_height / 2 * np.cos(np.radians(thet)) * display_ratio
                x_head, y_head = x_tail + dx, y_tail + dy
                spine = ax_layer.plot(
                    [x_tail, x_tail],
                    [y_tail - rect_height / 2, y_tail + rect_height / 2],
                    'w',
                    alpha=0.5,
                    linewidth=0.75,
                    transform=ax_layer.transAxes
                )
                arrowhead = ax_layer.plot(
                    x_head,
                    y_head,
                    marker=(3, 0, -thet),
                    markersize=7,
                    linestyle='None',
                    color='w',
                    transform=ax_layer.transAxes
                )
                arrow = ax_layer.plot(
                    [x_tail, x_head],
                    [y_tail, y_head],
                    color='w',
                    linewidth=2,
                    transform=ax_layer.transAxes
                )
                arc = mpatches.Arc(
                    (x_tail, y_tail),
                    width=rect_height / 2,
                    height=rect_height * display_ratio / 2,
                    angle=0,
                    theta1=90 - thet,
                    theta2=90,
                    color='w',
                    alpha=0.5,
                    linestyle='-',
                    transform=ax_layer.transAxes
                )
                ax_layer.add_patch(arc)
                ax_layer.plot(x_tail, y_tail, '.w', linewidth=2, transform=ax_layer.transAxes)
                # ax_layer.text(
                #     x_tail - 0.17,
                #     y_tail + rect_height / 2 + 0.05,
                #     f'$\\theta$ = {thet}' + '$^{\circ}$',
                #     verticalalignment='center',
                #     horizontalalignment='left',
                #     fontsize=20,
                #     color='w',
                #     transform=ax.transAxes
                # )
                fig_obj.tight_layout()
                bbox_cbar = fig_obj.axes[1].get_position()
                fig_obj.axes[1].set_position([bbox_cbar.x0 + 0.01, bbox_cbar.y0, bbox_cbar.width, bbox_cbar.height])
                if not PLOT_2D_COLORBARS:
                    fig_obj.axes[1].remove()
                if not PLOT_2D_AXIS_LABELS:
                    if thet != np.max(theta_vals):
                        plt.setp(ax_layer.get_xticklabels(), visible=False)
                        ax_layer.set_xlabel('')
                    if intens != np.min(intensity_vals):
                        plt.setp(ax_layer.get_yticklabels(), visible=False)
                        ax_layer.set_ylabel('')

                if SAVE:
                    fig_obj.savefig(subfolder_layer.joinpath(f'{thet}.png'), dpi=400, transparent=True)

                if not SHOW:
                    plt.close(fig_obj)

                # ################################## Plot Cell Kernel ##################################################
                fig_obj_cell = kernels[global_index].plot_avg_cell_kernel(histogram=False, dynamic=False,
                                                                          crosscuts=False, cmap=colormap,
                                                                          vmax=hist_cell_global_max)
                ax_cell = fig_obj_cell.axes[0]
                ax_cell.set_xticks([0.5, 1, 1.5, 2.0], minor=False)
                ax_cell.set_xticks([0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75], minor=True)
                ax_cell.set_yticks([-2000, -1500, -1000, -500, 0, 500, 1000, 1500, 2000], minor=False)
                ax_cell.set_yticks([-1000, -750, -500, -250, 0, 250, 500, 750, 1000], minor=True)
                ax_cell.set_xlim(t_min, t_max)
                ax_cell.set_ylim(kernels[global_index].z_kernel[0], kernels[global_index].z_kernel[-1])
                ax_cell.grid(axis='x', alpha=1, which='major', linewidth=1.5)
                ax_cell.grid(axis='y', alpha=1, which='major', linewidth=1.5)
                ax_cell.tick_params(axis='both', which='major', pad=10)
                # ax_cell.set_title(f'$|E|$ = {intens}', pad=15)
                fig_width, fig_height = fig_obj_cell.get_figwidth(), fig_obj_cell.get_figheight()
                _, _, ax_width, ax_height = ax_cell.get_position().bounds
                display_ratio = (fig_width * ax_width) / (fig_height * ax_height)  # fig_aspect * axis_aspect
                rect_height = 0.15 * display_ratio
                x_tail, y_tail = 0.835, 0.73
                dx = rect_height/2 * np.sin(np.radians(thet))
                dy = rect_height/2 * np.cos(np.radians(thet)) * display_ratio
                x_head, y_head = x_tail + dx, y_tail + dy
                spine_cell = ax_cell.plot(
                    [x_tail, x_tail],
                    [y_tail - rect_height/2, y_tail + rect_height/2],
                    'w',
                    alpha=0.5,
                    linewidth=0.75,
                    transform=ax_cell.transAxes
                )
                arrowhead_cell = ax_cell.plot(
                    x_head,
                    y_head,
                    marker=(3, 0, -thet),
                    markersize=7,
                    linestyle='None',
                    color='w',
                    transform=ax_cell.transAxes
                )
                arrow_cell = ax_cell.plot(
                    [x_tail, x_head],
                    [y_tail, y_head],
                    color='w',
                    linewidth=2,
                    transform=ax_cell.transAxes
                )
                arc_cell = mpatches.Arc(
                    (x_tail, y_tail),
                    width=rect_height / 2,
                    height=rect_height * display_ratio / 2,
                    angle=0,
                    theta1=90 - thet,
                    theta2=90,
                    color='w',
                    alpha=0.5,
                    linestyle='-',
                    transform=ax_cell.transAxes
                )
                ax_cell.add_patch(arc_cell)
                ax_cell.plot(x_tail, y_tail, '.w', linewidth=2, transform=ax_cell.transAxes)
                # ax_cell.text(
                #     x_tail - 0.17,
                #     y_tail + rect_height / 2 + 0.05,
                #     f'$\\theta$ = {thet}' + '$^{\circ}$',
                #     verticalalignment='center',
                #     horizontalalignment='left',
                #     fontsize=20,
                #     color='w',
                #     transform=ax_cell.transAxes
                # )
                fig_obj_cell.tight_layout()
                bbox_cbar = fig_obj_cell.axes[1].get_position()
                fig_obj_cell.axes[1].set_position([bbox_cbar.x0 + 0.01, bbox_cbar.y0, bbox_cbar.width, bbox_cbar.height])
                if not PLOT_2D_COLORBARS:
                    fig_obj_cell.axes[1].remove()
                if not PLOT_2D_AXIS_LABELS:
                    if thet != np.max(theta_vals):
                        plt.setp(ax_cell.get_xticklabels(), visible=False)
                        ax_cell.set_xlabel('')
                    if intens != np.min(intensity_vals):
                        plt.setp(ax_cell.get_yticklabels(), visible=False)
                        ax_cell.set_ylabel('')
                if SAVE:
                    fig_obj_cell.savefig(subfolder_cell.joinpath(f'{thet}.png'), dpi=400, transparent=True)

                if not SHOW:
                    plt.close(fig_obj_cell)
            global_index += 1

        # Plot 1D kernel avg/half moon plots
        # X, Y = r_mesh * np.cos(phi_mesh), r_mesh * np.sin(phi_mesh)  # transform to cartesian
        if PLOT_MOONS:
            plt.rcParams['font.size'] = 20
            fig_polar = plt.figure()
            ax = plt.subplot(1, 1, 1, projection='polar')
            if clockwise:
                ax.set_theta_direction(-1)
            else:
                ax.set_theta_direction(1)
            ax.set_theta_offset(np.pi / 2.0)
            levels = np.linspace(0, moon_max, 200)
            contour = ax.contourf(theta_mesh, r_mesh, partial_kernel_matrices[idx_intens], levels, cmap=colormap, vmin=0, vmax=moon_max)
            # ax.set_ylabel('t (ms)')
            ax.set_rticks([0.0, 0.5, 1.0, 1.5])
            ax.set_xticks(np.radians([0, 45, 90, 135, 180]))
            ax.set_rmax(t_max)
            ax.set_rmin(t_min)
            ax.set_thetamin(0)
            ax.set_thetamax(180)
            ax.tick_params(axis='y', pad=5)
            ax.tick_params(axis='x', pad=17)
            ax.grid(axis='y', alpha=1, which='both')
            ax.grid(axis='x', alpha=1)
            cbar = fig_polar.colorbar(contour, ax=ax, label='Spike Density ($ms^{-1}$)', location='right',
                                      orientation='vertical', ticks=np.arange(0, moon_max + 0.03, 0.03))
            fig_polar.tight_layout()
            if SAVE:
                fig_polar.savefig(sweep_save_folder.joinpath(f'halfmoon_theta_dep_intensity_{intens}.png'), dpi=400, transparent=True)

            if not SHOW:
                plt.close(fig_polar)

            # ########################################### Plot Crosscuts of Moons ######################################
            for crosscut_theta in [0, 60, 180]:
                index_theta = np.where(crosscut_theta == theta_vals)[0][0]
                fig_crosscut = plt.figure(figsize=(8, 5))
                ax_cross = fig_crosscut.add_subplot(111)
                ax_cross.plot(kernels[0].t_kernel, partial_kernel_matrices[idx_intens][index_theta, :], color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1], linewidth=5)
                ax_cross.set_xlabel('t (ms)', fontsize=26, labelpad=15)
                ax_cross.set_ylabel('Spike Density ($ms^{-1}$)', fontsize=26, labelpad=15)
                ax_cross.set_xlim((0, t_max))
                ax_cross.set_ylim(bottom=-0.005, top=moon_max)
                ax_cross.set_xticks([0.0, 0.5, 1.0, 1.5])
                ax_cross.set_xticklabels([0.0, 0.5, 1.0, 1.5], fontsize=26)
                ax_cross.set_yticks(cbar.get_ticks())
                ax_cross.set_yticklabels(cbar.get_ticks(), fontsize=26)
                # ax_cross.set_ylim((np.min(kernel_partial_matrix[idx_95, :]), vmax))
                # ax_cross.set_yticks([0, 0.5, 1.0, 1.5, 2.0])
                ax_cross.spines['right'].set_visible(False)
                ax_cross.spines['top'].set_visible(False)
                fig_crosscut.tight_layout()
                if SAVE:
                    fig_crosscut.savefig(sweep_save_folder.joinpath(f'halfmoon_crosscut_intensity_{intens}_theta_{crosscut_theta}.png'), dpi=400, transparent=True)
                if not SHOW:
                    plt.close(fig_crosscut)

    plt.ion()

    cellavg_folders = sorted(list(sweep_save_folder.glob('cellavg*')), key=lambda pth: int(pth.stem.split('_')[2]))
    layeravg_folders = sorted(list(sweep_save_folder.glob('lyravg*')), key=lambda pth: int(pth.stem.split('_')[2]))

    for theta in theta_vals:
        denstination_lyr = sweep_save_folder.joinpath(f'lyravg_theta_{theta}_2D_cbar_{PLOT_2D_COLORBARS}_labels_{PLOT_2D_AXIS_LABELS}')
        denstination_cell = sweep_save_folder.joinpath(f'cellavg_theta_{theta}_2D_cbar_{PLOT_2D_COLORBARS}_labels_{PLOT_2D_AXIS_LABELS}')
        denstination_lyr.mkdir(exist_ok=True)
        denstination_cell.mkdir(exist_ok=True)
        for cellfolder, layerfolder in zip(cellavg_folders, layeravg_folders):
            intensity = int(cellfolder.stem.split('_')[2])
            shutil.copy(cellfolder.joinpath(f'{theta}.png'), denstination_cell.joinpath(f'{intensity}.png'))
            shutil.copy(layerfolder.joinpath(f'{theta}.png'), denstination_lyr.joinpath(f'{intensity}.png'))
    print('fin :)')
