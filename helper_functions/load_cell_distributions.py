import numpy as np
import pandas as pd
import pickle as pkl
import scipy
import pathlib
import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from helper_functions.cell_type_lib import CellType
from helper_functions import __file__
MODULE_BASE = pathlib.Path(__file__).parent.parent


def compute_distribution(cell_class, z_step=0.025, micrometers=True, z_discrete_samples=None):
    """
    Calculates the distribution of cells of the given class/layer in M1 from data provided by Zhang et al 2021
    https://doi.org/10.1038/s41586-021-03705-x. On first run a file is generated for given parameters, and then loaded.
    Distribution is normalized and always sums to 1.
    upon subsequent runs with the same parameters.

    :param cell_class: String classification ('L23', 'L4', 'L5', 'GABA')
    :param z_step: Step size in microns to calculate histogram
    :param micrometers: If true then z is in micrometer units, otherwise is normalized (1, 0)
    :param z_discrete_samples: Depths in -um. If provided, then distribution is calculated exactly at provided samples
    :return: Density in increasing depth, z in increasing depth (normalized if micrometers is False), histogram, histogram bin edges
    """

    csv_file = MODULE_BASE.joinpath("reference_data/Zhang_2021/normalized_depth_MOp_Zhang_2021.csv")
    data = pd.read_csv(csv_file)
    if cell_class == 'L23':
        depth = data["normalized_depth"][(data['subclass'] == "L2/3 IT") & (data['normalized_depth'].notnull())]
        bw = 0.18
    elif cell_class == 'L5':
        depth = data["normalized_depth"][(data['subclass'] == "L5 ET") & (data['normalized_depth'].notnull())]
        bw = 0.18
    elif cell_class == 'L4':
        depth = data["normalized_depth"][(data['subclass'] == "L4/5 IT") & (data['normalized_depth'].notnull())]
        bw = 0.18
    elif cell_class == 'GABA':
        depth = data["normalized_depth"][(data['class_label'] == "GABAergic") & (data['normalized_depth'].notnull())]
        bw = 0.07
    elif cell_class == 'SOM':
        depth = data["normalized_depth"][(data['subclass'] == "Sst") & (data['normalized_depth'].notnull())]
        bw = 0.07
    elif cell_class == 'PV':
        depth = data["normalized_depth"][(data['subclass'] == "Pvalb") & (data['normalized_depth'].notnull())]
        bw = 0.07
    else:
        raise AttributeError("Layer ID is not recognized in (L23, L4, L5, GABA, SOM, PV)")

    zstep_label = f'{z_step}'.replace('.', 'p')
    micro_label = 'microns' if micrometers else 'ncd'
    save_file = MODULE_BASE.joinpath(f'reference_data/cell_distributions/{cell_class}_cell_distribution_zstep{zstep_label}_{micro_label}.pkl')
    # Perform calculation if there is no saved file, or if this is a discrete sample case
    if (not save_file.is_file()) or (z_discrete_samples is not None):
        z_layer = np.arange(0 - z_step / 2, -CellType.column_depth, -z_step)
        z_edges = np.arange(0, -CellType.column_depth - z_step, -z_step)
        z_normalized = np.abs(z_layer / CellType.column_depth)
        z_edges_normalized = np.abs(z_edges / CellType.column_depth)
        bins = np.linspace(z_normalized.min(), z_normalized.max(), 80)
        hist, _ = np.histogram(depth, bins=bins, density=True)
        kde = scipy.stats.gaussian_kde(depth, bw_method=bw)
        if z_discrete_samples is not None:
            z_normalized = np.abs(z_discrete_samples / CellType.column_depth)
            z_layer = z_discrete_samples
        density = kde(z_normalized)
        idx_csf = np.where(np.abs(z_normalized - 0) == np.min(np.abs(z_normalized - 0)))[0][0]
        density[idx_csf] = 0  # set density to zero at top of cortex (z = 0)
        norm_factor = np.sum(density)
        density = density / norm_factor
        hist = hist / norm_factor
        mean_depth = np.average(z_layer, weights=density)
        mean_depth_normalized = abs(mean_depth / CellType.column_depth)

        if micrometers:
            data = {
                'density': density,
                'z_layer': z_layer,
                'z_edges': z_edges,
                'mean_depth': mean_depth,
                'hist': hist,
                'bins': bins * CellType.column_depth
            }
        else:
            data = {
                'density': density,
                'z_layer': z_normalized,
                'z_edges': z_edges_normalized,
                'mean_depth': mean_depth_normalized,
                'hist': hist,
                'bins': bins
            }
        # Only save to file if this is regular grid calculation
        if z_discrete_samples is None:
            with open(save_file, 'wb') as file:
                pkl.dump(data, file)
    else:
        with open(save_file, 'rb') as file:
            data = pkl.load(file)

    return data['density'], data['z_layer'], data['z_edges'], data['mean_depth'], data['hist'], data['bins']


def compute_cell_coordinate_bounds(layer):
    """
    Computes the coordinate limits in x, y, z with respect to soma position for all cells of the given layer
    (L23, L4 or L5). The bounds are returns as [[x_min, y_min, z_min], [x_max, y_max, z_max]]
    :param layer: String Layer name
    :return: Coordinate Limits as [[x_min, y_min, z_min], [x_max, y_max, z_max]]
    """
    if layer == 'L23':
        cell_names = [name for name in CellType.cell_ids.keys() if name.split('_')[0] == 'L23']
    elif layer == 'L5':
        cell_names = [name for name in CellType.cell_ids.keys() if name.split('_')[0] == 'L5']
    elif layer == 'L4':
        cell_names = [name for name in CellType.cell_ids.keys() if name.split('_')[0] == 'L4']
    else:
        raise AttributeError("layer must be ('L23', 'L5' or 'L4')")

    coord_bounds = np.array([[0, 0, 0], [0, 0, 0]])
    for cell_name in cell_names:
        for cell_id in CellType.cell_ids[cell_name]:
            cell = CellType.cell_constructors[cell_name].from_id(cell_id)
            cell.load()
            coords = cell.get_segment_coordinates()
            coords = coords - np.array([cell.soma[0](0.5).x_xtra, cell.soma[0](0.5).y_xtra, cell.soma[0](0.5).z_xtra])
            cell_bounds = np.array([coords.min(axis=0), coords.max(axis=0)])
            for index in range(len(coord_bounds[0])):
                if cell_bounds[0][index] < coord_bounds[0][index]:
                    coord_bounds[0][index] = cell_bounds[0][index]
            for index in range(len(coord_bounds[1])):
                if cell_bounds[1][index] > coord_bounds[1][index]:
                    coord_bounds[1][index] = cell_bounds[1][index]

    return coord_bounds


if __name__ == '__main__':
    matplotlib.rcParams.update({'font.size': 16})
    save_folder = pathlib.Path('/path/to/folder/')

    step = 0.025
    microns = True
    SAVE = True

    density_L23, z, _, l23_mean_depth, hist_L23, bins = compute_distribution('L23', step, micrometers=microns)
    # density_L5, _, _, l5_mean_depth, hist_L5, _ = compute_distribution('L5', step, micrometers=microns)
    # density_L4, _, _, l4_mean_depth, hist_L4, _ = compute_distribution('L4', step, micrometers=microns)
    # density_GABA, _, _, gaba_mean_depth, hist_GABA, _ = compute_distribution('GABA', step, micrometers=microns)
    # density_SOM, _, _, som_mean_depth, hist_SOM, _ = compute_distribution('SOM', step, micrometers=microns)
    # density_PV, _, _, pv_mean_depth, hist_PV, _ = compute_distribution('PV', step, micrometers=microns)

    fig = plt.figure(figsize=(3.5, 5))
    plt.rcParams['font.size'] = 16
    alph = 0.0
    l23 = plt.stairs(hist_L23, bins, fill=True, alpha=alph, orientation='horizontal')
    # l4 = plt.stairs(hist_L4, bins, fill=True, alpha=alph, orientation='horizontal')
    # l5 = plt.stairs(hist_L5, bins, fill=True, alpha=alph, orientation='horizontal')
    # gaba = plt.stairs(hist_GABA, bins, fill=True, alpha=alph, orientation='horizontal')
    # som = plt.stairs(hist_SOM, bins, fill=True, alpha=alph, orientation='horizontal')
    # pv = plt.stairs(hist_PV, bins, fill=True, alpha=alph, orientation='horizontal')
    plt.plot(density_L23, z, color=l23.get_fc(), label='L23 IT', linewidth=2, alpha=1)
    # plt.plot(density_L4, z, color=l4.get_fc(), label='L4 IT', linewidth=2, alpha=1)
    # plt.plot(density_L5, z, color=l5.get_fc(), label='L5 ET', linewidth=2, alpha=1)
    # plt.plot(density_GABA, z, color=gaba.get_fc(), label='GABAergic', linewidth=2, alpha=1)
    # plt.plot(density_SOM, z, color=som.get_fc(), label='SOM', linewidth=2, alpha=1)
    # plt.plot(density_PV, z, color=pv.get_fc(), label='PV', linewidth=2, alpha=1)

    # plt.legend(fontsize=16)
    if microns:
        plt.ylabel(' Cortical Depth ($\mu m$)')
        plt.ylim(bottom=np.round(np.min(z)), top=0.0)
    else:
        plt.ylabel('Normalized Cortical Depth')
        plt.ylim(bottom=1.0, top=0.0)
    plt.xlabel('Cell Density')
    plt.xticks([0, 2.5e-5])
    plt.gca().set_xticklabels(['0', '2.5e-5'])
    # plt.gca().set_xticklabels(['0', '0.5e-5', '1.0e-5'])
    # plt.gca().yaxis.set_ticks_position('right')
    # plt.gca().set_xticks([0, 0.01])
    plt.gca().grid(axis='y', which='both', alpha=1, linewidth=1.2)
    plt.tight_layout()

    # print(f'Average Cortical Depth = {l5_mean_depth:.0f}')

    if not matplotlib.is_interactive():
        plt.show()

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
    # ##################################################################################################################
    # coord_limit_file = MODULE_BASE.joinpath('reference_data/cell_distributions/cell_type_coordinate_limits.pkl')
    #
    # if not coord_limit_file.is_file():
    #     l23_bounds = compute_cell_coordinate_bounds('L23')
    #     l4_bounds = compute_cell_coordinate_bounds('L4')
    #     l5_bounds = compute_cell_coordinate_bounds('L5')
    #     save_data = {
    #         'L23': l23_bounds,
    #         'L4': l4_bounds,
    #         'L5': l5_bounds
    #     }
    #     with open(coord_limit_file, 'wb') as f:
    #         pkl.dump(save_data, f)
    # else:
    #     with open(coord_limit_file, 'rb') as f:
    #         coord_limits = pkl.load(f)
