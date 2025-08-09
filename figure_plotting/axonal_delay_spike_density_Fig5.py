
import pathlib
import h5py
import matplotlib
import numpy as np
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

from postprocessing_modules.axonal_delay_kernel import get_axonal_delay_kernel
from helper_functions import __file__
from helper_functions.plotting_lib import round_sig_decimal
MODULE_BASE = pathlib.Path(__file__).parent.parent

if __name__ == '__main__':
    # Input Data Files (requires reference_data to be populated from OSF database)
    param_path = MODULE_BASE.joinpath('reference_data/axonal_delay_reduced_biphasic_params.hdf5')
    delays_path = MODULE_BASE.joinpath('reference_data/Miller_2025/delay_z_data/')
    kernels_path = MODULE_BASE.joinpath('reference_data/Miller_2025/axon_kernels_precomputed/precomputed_kernels_50_0p1')
    # Load Parameters
    with h5py.File(param_path, 'r') as f:
        params = f['params'][:]
    # Choose Upstream Population (L23, L23_inh, L4_inh, L5, or L5_inh)
    source_layer = 'L23'
    # Path to save figures
    save_path = pathlib.Path('/path/to/folder/')
    # ####################################### Compute and Plot single param set ########################################
    # Set Parameters for Axonal Delay Kernel computations
    z_bin_width = 100
    t_bin_width = 0.1
    DENSITY = True # If density is false the area under surface may not be conserved by interpolation
    SAVE = False
    # Stepsize of smooth interpolated spike density histogram
    smooth_z_step = 50  # um
    smooth_t_step = 0.1  # ms
    # Electric Field Parameters
    THETA = 30.0 # Polar angle
    GRAD = 0.0 # Electric field gradient
    AMP = 225.0 # Electric field amplitude (intensity)
    e_param_label = '_'.join([f'{int(THETA)}', f'{int(GRAD)}', f'{int(AMP)}'])
    # Compute or recall precomputed AxonalDelayKernel object
    print('Retrieving AxonalDelayKernel')
    kernel_handler = get_axonal_delay_kernel(
        source_layer_name=source_layer,
        delay_z_folder=delays_path,
        precomputed_kernels_folder=kernels_path,
        theta=THETA,
        gradient=GRAD,
        intensity=AMP,
        z_step=z_bin_width,
        t_step=t_bin_width,
        smooth_z_step=smooth_z_step,
        smooth_t_step=smooth_t_step,
        density=True,
        compute_fresh=False
    )
    # OPTIONAL: Re-interpolating spike density to smoother curve for plotting (can take some time)
    print('Re-interpolatings Kernel')
    kernel_handler.smooth_kernel(
        z_step=1,
        t_step=0.005
    )

    volume = np.trapz(
        np.trapz(
            kernel_handler.cell_histogram,
            dx=abs(np.diff(kernel_handler.z_bins)[0]),
            axis=0
        ),
        dx=abs(np.diff(kernel_handler.t_bins)[0])
    )
    volume_layer = np.trapz(
        np.trapz(
            kernel_handler.layer_histogram,
            dx=abs(np.diff(kernel_handler.z_bins_layer)[0]),
            axis=0
        ),
        dx=abs(np.diff(kernel_handler.t_bins)[0])
    )
    sum_cell = np.sum(kernel_handler.cell_histogram)
    sum_layer = np.sum(kernel_handler.layer_histogram)
    if kernel_handler.density:
        print(f"Kernel Volume Under Surface: {volume:.4f}")
        print(f"Layer Kernel Volume Under Surface: {volume_layer:.4f}")
    else:
        print(f"Histogram Sum: {sum_cell:.4f}")
        print(f"Layer Histogram Sum: {sum_layer:.4f}")

    colormap = 'RdYlBu_r'

    print('Plotting Kernels')
    plt.rcParams.update({'font.size': 18})
    # fig_cell = kernel_handler.plot_avg_cell_kernel(crosscuts=False, dynamic=True, cmap=colormap, histogram=True)
    # fig_avg = kernel_handler.plot_avg_layer_kernel(crosscuts=False, dynamic=True, cmap=colormap, histogram=True)
    # #################################### Plot Fig 5 Kernels ##########################################################
    fig_cell_kernel = kernel_handler.plot_avg_cell_kernel(crosscuts=False, dynamic=True, cmap=colormap, histogram=False)
    fig_avg_kernel = kernel_handler.plot_avg_layer_kernel(crosscuts=False, dynamic=True, cmap=colormap, histogram=False)
    fig_avg_kernel.axes[0].set_yticklabels([])
    fig_avg_kernel.axes[0].set_ylabel('')
    fig_avg_kernel.axes[0].grid(axis='y', which='both', alpha=1, linewidth=1.2)

    # fig_cell_kernel.axes[0].set_ylabel('Depth w.r.t. Soma ($\mu$m)')
    # fig_avg_kernel.axes[0].set_yticklabels([])
    # fig_avg_kernel.axes[0].set_ylabel('')
    # fig_avg_kernel.axes[0].grid(axis='y', alpha=0.9)
    # fig_avg_kernel.axes[0].grid(axis='x', alpha=0)
    # fig_avg_kernel.axes[0].set_title('')

    # fig_cell.axes[0].set_title('')

    tlabel = f'{t_bin_width}'.replace('.', 'p')
    # Plot Component Kernels from each Subcell Type
    if source_layer.split('_')[-1] == 'inh':
        plt.rcParams['font.size'] = 16
        num_subtypes = len(kernel_handler.subtype_histograms)
        window = plt.get_current_fig_manager().window
        screen_width = window.width()
        widths = [1.0 for _ in range(num_subtypes)]
        widths.append(0.1)
        fig = plt.figure(constrained_layout=False, figsize=(14, 8))
        widths = [1, 1, 1, 1, 0.1]
        heights = [0.5, 0.5]
        spec_order = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)]
        spec = fig.add_gridspec(ncols=5, nrows=2, width_ratios=widths, height_ratios=heights, hspace=0.4, wspace=0.25)
        kernels = kernel_handler.subtype_histograms
        kernels.append(kernel_handler.cell_histogram)
        kernel_labels = kernel_handler.subtype_labels
        kernel_labels.append('L4 Average Cell')
        hist_max = max([np.max(subhist) for subhist in kernels])
        hist_max = round_sig_decimal(hist_max)
        axes = []
        for ax_index, (subcell_hist, subcell_label, spec_tuple) in enumerate(zip(kernels, kernel_labels, spec_order)):
            axes.append(fig.add_subplot(spec[spec_tuple]))
            subpcolor = axes[-1].pcolor(kernel_handler.t_bin_centers, kernel_handler.z_bin_centers, subcell_hist,
                                        cmap=colormap, vmax=hist_max)
            axes[-1].set_title(f'{subcell_label}')
            axes[-1].grid(axis='both', which='both', alpha=1, linewidth=0.5)
            axes[-1].set_yticks([-1000, -500, 0, 500, 1000])
            axes[-1].set_xticks([0, 0.5, 1, 1.5, 2])
            if ax_index not in [0, 4]:
                plt.setp(axes[-1].get_yticklabels(), visible=False)
            if ax_index > 3:
                axes[-1].set_xlabel('Time (ms)')

        axes[0].set_ylabel('Depth w.r.t Soma ($\mu$m)')
        axes[4].set_ylabel('Depth w.r.t Soma ($\mu$m)')
        cbar_ax = fig.add_subplot(spec[:, 4])
        plt.colorbar(subpcolor, cax=cbar_ax, orientation='vertical', label=kernel_handler.clabel.replace('\n', ' '))
        fig.suptitle(
            f'{source_layer}, $\\theta$ = {THETA}$^\circ$, ' + '$\Delta |\mathbf{E}|$' + f' = {GRAD} %/mm, ' + '$|\mathbf{E}' + f'|$ = {AMP} V/m')
        if SAVE:
            fig.savefig(save_path.joinpath(
                f'{source_layer}_cell_kernel_composition_{e_param_label}_{tlabel}dt_{z_bin_width}dz_{colormap}.png'),
                dpi=600, transparent=True)

    # kernel_handler.plot_kernel_3d()

    # if SAVE:
    #     fig_cell.savefig(save_path.joinpath(f'{source_layer}_avg_cell_hist_{e_param_label}_{tlabel}dt_{z_bin_width}dz_{colormap}.png'), dpi=400, transparent=True)
    #     fig_avg.savefig(save_path.joinpath(f'{source_layer}_avg_layer_hist_{e_param_label}_{tlabel}dt_{z_bin_width}dz_{colormap}.png'), dpi=400, transparent=True)
    #     fig_cell_kernel.savefig(save_path.joinpath(f'{source_layer}_avg_cell_kernel_smooth_{e_param_label}_{tlabel}dt_{z_bin_width}dz_{colormap}.png'), dpi=600, transparent=True)
    #     fig_avg_kernel.savefig(save_path.joinpath(f'{source_layer}_avg_layer_kernel_smooth_{e_param_label}_{tlabel}dt_{z_bin_width}dz_{colormap}.png'), dpi=600, transparent=True)
