import matplotlib
import os
# if not matplotlib.is_interactive():
#     matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import pickle as pkl
from neuron_interface.neurosim.cells import L5_TTPC2_cADpyr
from helper_functions.cell_type_lib import CellType
from helper_functions.load_cell_distributions import compute_distribution
from helper_functions import __file__
MODULE_BASE = pathlib.Path(__file__).parent.parent


def compute_dend_length_grid(neuron_cell, z_axis_bins, neuron_depth):
    assert neuron_cell.loaded, 'Neuron Cell is not loaded'
    assert neuron_depth < 0, 'Neuron Cell soma depth must be negative'
    grid_dendL = np.zeros(shape=(len(z_axis_bins) - 1, ))
    soma_z = neuron_cell.soma[0](0.5).z_xtra
    for sec in neuron_cell.all:
        dendL = sec.L / sec.nseg
        for seg in list(sec):
            # Transform neuron coordinate to cortical depth (centered at cell soma depth)
            seg_z = seg.z_xtra - soma_z + neuron_depth
            if (seg_z < z_axis_bins[0]) and (seg_z >= z_axis_bins[-1]):
                bin_idx = np.where(seg_z < z_axis_bins)[0][-1]
                grid_dendL[bin_idx] += dendL
    # Normalize dendritic length grid
    grid_dendL = grid_dendL / np.max(grid_dendL)

    return grid_dendL


def estimate_synaptic_density(neuron_cell, scracm_profile, z_axis_bins, neuron_depth):
    assert neuron_depth < 0, 'Neuron Cell soma depth must be negative'
    grid_dendL = compute_dend_length_grid(neuron_cell=neuron_cell, z_axis_bins=z_axis_bins, neuron_depth=neuron_depth)
    # Calculate synaptic density estimate for this sample cell
    cell_profile = scracm_profile - np.min(scracm_profile)
    cell_profile /= np.max(cell_profile)
    syn_dens_estimate = np.divide(
        cell_profile,
        np.sqrt(grid_dendL),
        out=np.zeros(shape=(len(scracm_profile),)),
        where=np.sqrt(grid_dendL) != 0
    )
    return syn_dens_estimate / np.max(syn_dens_estimate), grid_dendL


def get_l23pt_syn_density(data_path, neuron_cell, normalized_depth):
    assert neuron_cell.loaded, 'Neuron Cell is not loaded'

    fname = 'L23_PT_scracm.pkl'
    with open(data_path.joinpath(fname), 'rb') as f:
        data = pkl.load(f, encoding='latin1')  # (num cells x num z bins)
    sample_soma_depths = np.array([-1100, -1200, -1215, -1230, -1295, -1300, -1318, -1355, -1370, -1383,
                                   -1387, -1402, -1405, -1473, -1478, -1480, -1493, -1508, -1570, -1573,
                                   -1584, -1621, -1858])
    num_z_bins = data.shape[1]
    num_cells = data.shape[0]
    step = 50
    raw_grid_edges = np.arange(0, -step * num_z_bins - step, -step)
    # raw_grid_center = (raw_grid_edges[:-1] + raw_grid_edges[1:]) / 2
    scaled_grid_edges = raw_grid_edges * CellType.column_depth / abs(np.min(raw_grid_edges))
    scaled_grid_center = (scaled_grid_edges[:-1] + scaled_grid_edges[1:]) / 2

    dens_z_step = 0.5
    l5_density, z_density, _, _, _, _ = compute_distribution(
        cell_class='L5',
        z_step=dens_z_step,
        micrometers=True,
    )

    sample_z_idxs = np.array([np.where(np.abs(cell_depth - z_density) == np.min(np.abs(cell_depth - z_density)))[0][0] for cell_depth in sample_soma_depths], dtype=int)
    index_upper_bound = np.where(np.invert(np.isclose(l5_density, 0, atol=1e-5)))[0][0]
    index_lower_bound = np.where(np.invert(np.isclose(l5_density, 0, atol=1e-5)))[0][-1]
    z_sample_bin_ids = np.array((sample_z_idxs[0:-1] + sample_z_idxs[1:]) / 2).astype(int)
    z_sample_bin_ids = np.concatenate((np.array([index_upper_bound]), z_sample_bin_ids, np.array([index_lower_bound])))
    z_sample_bin_edges = z_density[z_sample_bin_ids]
    z_sample_bin_widths = np.abs(np.diff(z_sample_bin_edges))
    z_sample_bin_areas = np.zeros(shape=(len(z_sample_bin_edges) - 1))
    for z_bin in range(len(z_sample_bin_edges) - 1):
        z_sample_bin_areas[z_bin] = np.trapz(l5_density[z_sample_bin_ids[z_bin]:z_sample_bin_ids[z_bin + 1]], dx=dens_z_step)

    # density_weights = l5_density[density_idxs]
    avg_scracm_profile = np.average(data, weights=z_sample_bin_areas, axis=0)
    avg_scracm_profile /= np.max(avg_scracm_profile)

    # Now calculate average DendL profile for all L5 cells which are all at the same depth (the avg depth)

    estimated_syn_dens, dendritic_length_grid = estimate_synaptic_density(
        neuron_cell=neuron_cell,
        scracm_profile=avg_scracm_profile,
        z_axis_bins=scaled_grid_edges,
        neuron_depth=normalized_depth * (-CellType.column_depth)
    )

    return scaled_grid_center, estimated_syn_dens, avg_scracm_profile, z_sample_bin_areas, z_sample_bin_edges, sample_soma_depths, dendritic_length_grid, data


if __name__ == '__main__':
    save_folder = pathlib.Path('/path/to/folder/')
    data_folder = MODULE_BASE.joinpath('/reference_data/Dura_Bernal_2023/conn/')
    layers = {'pia': 0, 'L1-L2 border': 0.1, 'L2/3-L4 border': 0.29, 'L4-L5A border': 0.37, 'L5A-L5B border': 0.47,
              'L5B-L6 border': 0.8, 'L6-WM border': 1.0}

    # depth_range = np.linspace(layers['L4-L5A border'], layers['L5B-L6 border'], 50)
    # Load Morph
    SAVE = True
    cell_ids = L5_TTPC2_cADpyr.get_morphology_ids()
    cell_idx = 15
    cell = L5_TTPC2_cADpyr.from_id(cell_ids[cell_idx])
    cell.load()
    # coords = cell.get_segment_coordinates()
    # coords_layer = coords.copy()
    # coords_layer[:, 2] = coords_layer[:, 2] - soma_z + soma_depth * (-CellType.column_depth)

    mask_start = 0
    mask_stop = 23

    l5_density, z_density, _, _, _ = compute_distribution(
        cell_class='L5',
        z_step=0.025,
        micrometers=True)

    mean_depth = np.average(z_density, weights=l5_density)
    normalized_mean_depth = abs(mean_depth / CellType.column_depth)

    z_grid_center, syn_dens, avg_scracm_1d, bin_area_weights, bin_edges, sample_depths, dendL, raw_scracm = get_l23pt_syn_density(
        data_path=data_folder,
        neuron_cell=cell,
        normalized_depth=normalized_mean_depth,
    )

    fig_s = matplotlib.pyplot.figure(figsize=(7, 7))
    spec = fig_s.add_gridspec(ncols=3, nrows=3, width_ratios=[0.3, 1, 1], height_ratios=[0.07, 1, 0.5], hspace=0.3, wspace=0.15)
    ax_scolor = fig_s.add_subplot(spec[1, 1])
    cplot = ax_scolor.pcolor(np.arange(1, raw_scracm[mask_start:mask_stop, :].shape[0] + 1), z_grid_center, raw_scracm[mask_start:mask_stop, :].T, cmap='RdBu_r')
    ax_scolor.plot(np.arange(1, raw_scracm[mask_start:mask_stop, :].shape[0] + 1), sample_depths[mask_start:mask_stop], 'o', markeredgecolor='w', markerfacecolor="None", markersize=5, markeredgewidth=1.5)
    ax_scolor.set_ylabel('Cortical Depth ($\\mu m$)')
    ax_scolor.set_xticklabels([])
    ax_cbar = fig_s.add_subplot(spec[0, 1])
    plt.colorbar(cplot, cax=ax_cbar, label='SCRACM Intensity', orientation='horizontal')
    ax_cbar.xaxis.set_label_position('top')
    ax_cbar.xaxis.set_ticks_position('top')
    ax_weights = fig_s.add_subplot(spec[2, 1])
    ax_weights.plot(np.arange(1, raw_scracm[mask_start:mask_stop, :].shape[0] + 1), bin_area_weights, 'bo')
    ax_weights.set_xlabel('Sample Cell (a.u.)')
    ax_weights.set_ylabel('L5 Cell Density\nArea under Curve', labelpad=25)
    ax_avg = fig_s.add_subplot(spec[1, 2])
    ax_avg.plot(avg_scracm_1d, z_grid_center)
    ax_avg.set_yticklabels([])
    ax_avg.set_xlabel('Normalized\nAvg. SCRACM')

    fig_comp = plt.figure(figsize=(10, 5))
    ax3 = fig_comp.add_subplot(111)
    ax3.plot(z_grid_center, avg_scracm_1d, '-b', label='Avg. SCRACM', zorder=2)
    ax3.plot(z_grid_center, dendL, '-g', label='Dendritic Length', zorder=3)
    ax3.plot(z_grid_center, syn_dens, '-r', label='Synaptic Density', zorder=4)
    ax3.axvline(mean_depth, linestyle='-', color='tab:gray', zorder=1)
    ax3.text(mean_depth - 25, 0.0, ' - Soma Depth')
    ax3.legend(fontsize=12)
    ax3.set_xlim(0, -CellType.column_depth)
    ax3.set_xlabel('Cortical Depth ($ \\mu m $)')
    ax3.set_ylabel('a.u. (Scaled to Fit)')
    fig_comp.tight_layout()

    # Plot L5 Depth curve with sample positions and average depth
    sample_depth_indices = np.array([np.where(np.abs(sample_depth - z_density) == np.min(np.abs(sample_depth - z_density)))[0][0] for sample_depth in sample_depths])
    dens_weights = l5_density[sample_depth_indices]
    computed_sample_depths = z_density[sample_depth_indices]
    fig_dens = plt.figure(figsize=(3.2*1.5, 5*1.5))
    ax_dens = fig_dens.add_subplot(111)
    ax_dens.plot(l5_density, z_density, 'b-')
    ax_dens.plot(dens_weights[mask_start:mask_stop], sample_depths[mask_start:mask_stop], 'o', markerfacecolor='r', markeredgecolor='k', markersize=4)
    ax_dens.axhline(mean_depth, color='k', linestyle=':')
    ax_dens.set_xlabel('Cell Density')
    ax_dens.set_ylabel('Cortical Depth ($\\mu m)$')
    ax_dens.grid(axis='y', alpha=0.5)
    fig_dens.tight_layout()

    depthlabel = f'{normalized_mean_depth:.3f}'.replace('.', 'p')
    if SAVE:
        fig_s.savefig(save_folder.joinpath(f'L23_L5_avg_scracm_breakdown.png'), dpi=300, transparent=True)
        fig_dens.savefig(save_folder.joinpath(f'L5_cell_density_sample_positions.png'), dpi=300, transparent=True)
        fig_comp.savefig(save_folder.joinpath(f'L23_pt{cell_ids[cell_idx]}_synDens_breakdown_{depthlabel}avgdepth.png'), dpi=300, transparent=True)
