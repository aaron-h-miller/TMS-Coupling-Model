
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import pathlib
from helper_functions.load_cell_distributions import compute_distribution
from helper_functions.cell_type_lib import CellType

if __name__ == '__main__':
    matplotlib.rcParams.update({'font.size': 16})
    save_folder = pathlib.Path('/path/to/folder/')

    step = 0.025
    microns = True
    SAVE = True

    # ########################################## Plot Fig 5D Overlays ##################################################
    density_L23, z, _, l23_mean_depth, hist_L23, bins = compute_distribution('L23', step, micrometers=microns)

    fig = plt.figure(figsize=(3.5, 5))
    plt.rcParams['font.size'] = 16
    alph = 0.0
    l23 = plt.stairs(hist_L23, bins, fill=True, alpha=alph, orientation='horizontal')
    plt.plot(density_L23, z, color=l23.get_fc(), linewidth=2, alpha=1)

    if microns:
        plt.ylabel(' Cortical Depth ($\mu m$)')
        plt.ylim(bottom=np.round(np.min(z)), top=0.0)
    else:
        plt.ylabel('Normalized Cortical Depth')
        plt.ylim(bottom=1.0, top=0.0)

    plt.xlabel('Cell Density')
    plt.xticks([0, 2.5e-5])
    plt.gca().set_xticklabels(['0', '2.5e-5'])
    plt.gca().grid(axis='y', which='both', alpha=1, linewidth=1.2)
    plt.tight_layout()
    plt.suptitle('Fig 5D. L23 Cell Distribution', fontsize=16)
    fig.subplots_adjust(top=0.9)


    if SAVE:
        fig.savefig(save_folder.joinpath('L23_cell_density.png'), dpi=400, transparent=True)

    fig_layers = plt.figure(figsize=(3.5, 5))
    plt.rcParams['font.size'] = 16
    ax = fig_layers.add_subplot(111)
    ax.set_ylim(bottom=-CellType.column_depth, top=0)
    ax.set_ylabel(' Cortical Depth ($\mu m$)')
    ax.set_xticks([])
    for layer_name, norm_boundary in CellType.layer_ncd_boundaries.items():
        ax.axhline(norm_boundary[-1] * -CellType.column_depth, color='k', alpha=0.5)
        ax.text(0.5, (norm_boundary[0] + norm_boundary[-1]) / 2 * -CellType.column_depth, layer_name,
                horizontalalignment='center', verticalalignment='center', alpha=0.7)
    fig_layers.tight_layout()
    if SAVE:
        fig_layers.savefig(save_folder.joinpath('layer_boundaries.png'), dpi=400, transparent=True)
    plt.suptitle('Fig 5D. Layer Boundaries', fontsize=16)
    fig_layers.subplots_adjust(top=0.9)
    # ######################################### Plot Fig 7 #############################################################
    density_L5, z_layer_centers, _, l23_mean_depth, hist_L23, bins = compute_distribution('L5', step, micrometers=microns)
    index_upper_bound = np.where(np.invert(np.isclose(density_L5, 0, atol=1e-6)))[0][0]
    index_lower_bound = np.where(np.invert(np.isclose(density_L5, 0, atol=1e-6)))[0][-1]
    upper_bound = z_layer_centers[index_upper_bound]
    lower_bound = z_layer_centers[index_lower_bound]
    idx_step = int((index_lower_bound - index_upper_bound) / 23)
    z_sample_indices = np.arange(index_upper_bound, index_lower_bound + idx_step, idx_step)
    z_sample_depths = np.round(z_layer_centers[z_sample_indices])
    density_sample_values = density_L5[z_sample_indices]
    sample_mean_depth = np.average(z_sample_depths, weights=density_sample_values)

    plt.rcParams['font.size'] = 16
    fig = plt.figure(figsize=(3.5, 5))
    plt.plot(density_L5, z_layer_centers, 'k')
    plt.plot(density_sample_values, z_sample_depths, 'o', color='tab:orange', label='Samples')
    plt.axhline(sample_mean_depth, color='k', linestyle='--', label='Mean Depth')
    # plt.axhline(l5_mean_depth, linestyle='--', label='True mean')
    plt.ylim(-CellType.column_depth, 0)
    plt.ylabel('Cortical Depth ($\\mu m$)')
    plt.xlabel('Cell Density')
    plt.xticks([0, 2.5e-5])
    plt.gca().set_xticklabels(['0', '2.5e-5'])
    plt.legend(fontsize=12)
    plt.tight_layout()
    if SAVE:
        fig.savefig(save_folder.joinpath('sample_soma_depths.png'), dpi=400, transparent=True)
    plt.suptitle('Fig 7. L5 Cell Distribution', fontsize=16)
    fig.subplots_adjust(top=0.9)
