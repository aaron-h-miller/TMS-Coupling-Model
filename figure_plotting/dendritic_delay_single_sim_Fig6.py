
import pathlib
import time
import numpy as np
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib
from matplotlib.collections import LineCollection
import matplotlib.font_manager as fm

from postprocessing_modules.axonal_delay_kernel import get_axonal_delay_kernel
from neuron_interface.neurosim.simulation.dendritic_delay_simulation import DendriticDelaySimulation
from helper_functions.cell_type_lib import CellType
from helper_functions.load_cell_distributions import compute_distribution
from helper_functions.neuron_cell_functions_lib import segment_sublist_mask, collect_lines_section
from helper_functions.plotting_lib import round_sig_decimal
from neuron_interface.neurosim.cells import L5_TTPC2_cADpyr
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from helper_functions import __file__
MODULE_BASE = pathlib.Path(__file__).parent.parent


if __name__ == '__main__':
    # Working directory must be set to module base
    save_folder = pathlib.Path('/path/to/folder/')
    param_path = MODULE_BASE.joinpath('reference_data/axonal_delay_reduced_biphasic_params.hdf5')
    axonal_delay_data_folder = MODULE_BASE.joinpath('reference_data/Miller_2025/delay_z_data/')
    axonal_kernels_precomputed_folder = MODULE_BASE.joinpath('reference_data/Miller_2025/axon_kernels_precomputed/precomputed_kernels_50_0p1/')

    # ############################################## SET PARAMETERS ####################################################
    SIM_DURATION = 50  # ms
    SIM_TIME_STEP = 0.01  # ms
    thet = 30.0
    grad = 10
    intens = 180.0
    FRACTION_NMDA = 0.1
    FRACTION_GABA_A = 0.9
    FRACTION_EXCITATORY = 0.34
    Z_BIN_SIZE = 50
    NA_SUPPRESSED = True
    DENSITY = True
    INCLUDE_AXON = False  # inclusion of axon current in soma current calculation
    na_label = 'noSodium' if NA_SUPPRESSED else 'Sodium'
    SAVE = False
    DYNAMIC = False
    # LOAD L5 MORPHOLOGY
    morph_ids = CellType.cell_ids['L5_TTPC2_cADpyr']
    morph_id = morph_ids[7]
    morph = L5_TTPC2_cADpyr.from_id(morph_id)
    morph.load()
    # LOAD AND PROCESS INPUT FILES
    _, _, _, l5_mean_depth, _, _ = compute_distribution('L5', 0.025, micrometers=True)

    SOMA_DEPTH = l5_mean_depth
    # ###################################### Set Up Simulation #########################################################
    t_start = time.process_time()
    sim = DendriticDelaySimulation(morph, sodium_blocked=NA_SUPPRESSED)
    kernel_l23_ex = get_axonal_delay_kernel(source_layer_name='L23', delay_z_folder=axonal_delay_data_folder,
                                            precomputed_kernels_folder=axonal_kernels_precomputed_folder, theta=thet,
                                            gradient=grad, intensity=intens, z_step=100, t_step=0.1,
                                            smooth_z_step=Z_BIN_SIZE, smooth_t_step=SIM_TIME_STEP, density=DENSITY,
                                            compute_fresh=False)
    kernel_l23_inh = get_axonal_delay_kernel(source_layer_name='L23_inh', delay_z_folder=axonal_delay_data_folder,
                                             precomputed_kernels_folder=axonal_kernels_precomputed_folder, theta=thet,
                                             gradient=grad, intensity=intens, z_step=100, t_step=0.1,
                                             smooth_z_step=Z_BIN_SIZE, smooth_t_step=SIM_TIME_STEP, density=DENSITY,
                                             compute_fresh=False)
    segment_ids_per_bin = DendriticDelaySimulation.get_segment_ids_per_bin(morph, SOMA_DEPTH, kernel_l23_ex.z_layer_edges)
    sim.define_synaptic_inputs_continuous(
        spike_density_ex=kernel_l23_ex.layer_kernel,
        spike_density_in=kernel_l23_inh.layer_kernel,
        z_bin_edges=kernel_l23_ex.z_layer_edges,
        z_bin_centers=kernel_l23_ex.z_layer,
        soma_depth=SOMA_DEPTH,
        fraction_nmda=FRACTION_NMDA,
        fraction_gaba_a=FRACTION_GABA_A,
        fraction_ex=FRACTION_EXCITATORY,
        sim_duration=SIM_DURATION,
        sim_time_step=SIM_TIME_STEP
    )
    _, _, t_axis, voltage, current = sim.simulate(include_axon_current=INCLUDE_AXON)

    t_duration = time.process_time() - t_start
    # Find Axonal Hillock and Neuron Subsection Masks
    nearest_secs = morph.soma[0].children()
    hillock_sec = nearest_secs[np.where([sec in morph.axon for sec in nearest_secs])[0][0]]
    hillock_idx = np.where(segment_sublist_mask(morph, [hillock_sec]))[0][0]

    soma_mask = segment_sublist_mask(morph, morph.soma)
    apic_mask = segment_sublist_mask(morph, morph.apic)
    dend_mask = segment_sublist_mask(morph, morph.dend)
    axon_mask = segment_sublist_mask(morph, morph.axon)
    combined_mask = soma_mask | apic_mask | dend_mask

    hillock_voltage = voltage[hillock_idx, :]
    # Find soma
    soma_idx = np.where(soma_mask)[0][0]
    soma_voltage = voltage[soma_idx, :]
    conductance = sim.conductance_inputs
    mem_current = sim.transmembrane_current_recordings
    for segid in range(len(mem_current)):
        for synid in range(len(mem_current[segid])):
            mem_current[segid][synid] = np.array(mem_current[segid][synid])
        mem_current[segid] = np.array(mem_current[segid])
    mem_current = np.array(mem_current)

    # ################################################ Plotting Prep ###################################################
    # smooth_z_step = 1  # 1 um
    # smooth_t_step = 0.005  # 0.025 ms
    # print('Calculating Smooth Kernel For Plotting')
    # kernel_obj.calculate_kernel(z_step=smooth_z_step, t_step=smooth_t_step, bandwidth=0.15)
    t_bin_centers = kernel_l23_ex.t_kernel
    t_bin_centers = np.concatenate([np.array([0.0]), t_bin_centers, np.array([2.0])])
    z_bin_centers = kernel_l23_ex.z_layer
    spike_rate_ex = kernel_l23_ex.layer_kernel
    spike_rate_ex = np.concatenate([np.zeros(shape=(len(z_bin_centers), 1)), spike_rate_ex, np.zeros(shape=(len(z_bin_centers), 1))], axis=1)
    t_bin_centers_inh = kernel_l23_inh.t_kernel
    t_bin_centers_inh = np.concatenate([np.array([0.0]), t_bin_centers_inh, np.array([2.0])])
    z_bin_centers_inh = kernel_l23_inh.z_layer
    spike_rate_inh = kernel_l23_inh.layer_kernel
    spike_rate_inh = np.concatenate([np.zeros(shape=(len(z_bin_centers), 1)), spike_rate_inh, np.zeros(shape=(len(z_bin_centers), 1))], axis=1)
    t_stop = SIM_DURATION
    # ##### Prepare Coordinate System (w.r.t Soma and w.r.t Layer) #####
    coords = morph.get_segment_coordinates()
    coords_dendrites = coords[combined_mask]
    coords_dendrites_layer = coords_dendrites.copy()
    soma_coord = np.array([morph.soma[0](0.5).x_xtra, morph.soma[0](0.5).y_xtra, morph.soma[0](0.5).z_xtra])
    coords = coords - soma_coord
    coords_dendrites = coords_dendrites - soma_coord
    coords_dendrites_layer = coords_dendrites_layer - soma_coord
    coords_layer = coords.copy()
    coords_layer = coords_layer - soma_coord
    coords_layer[:, 2] = coords_layer[:, 2] + SOMA_DEPTH
    coords_dendrites_layer[:, 2] = coords_dendrites_layer[:, 2] + SOMA_DEPTH
    # cmap = plt.cm.get_cmap("turbo", np.max(calculated_syn_density) + 1)
    cmap = 'turbo'
    # Define Line collection coordinates with respect the layer
    lines_but_axon = collect_lines_section(morph, morph.axon, coords, negate=True)
    lines_axon = collect_lines_section(morph, morph.axon, coords)
    lines_but_axon_layer = []
    lines_axon_layer = []
    for line in lines_but_axon:
        lines_but_axon_layer.append(np.zeros(shape=(2, 2)))
        lines_but_axon_layer[-1][:, 0] = line[:, 0].copy()
        lines_but_axon_layer[-1][:, 1] = SOMA_DEPTH + (line[:, 1] - morph.soma[0](0.5).z_xtra)
    for line in lines_axon:
        lines_axon_layer.append(np.zeros(shape=(2, 2)))
        lines_axon_layer[-1][:, 0] = line[:, 0].copy()
        lines_axon_layer[-1][:, 1] = SOMA_DEPTH + (line[:, 1] - morph.soma[0](0.5).z_xtra)

    kernel_max_e = round_sig_decimal(np.max(spike_rate_ex))
    kernel_max_i = round_sig_decimal(np.max(spike_rate_inh))
    spot_idx = 435
    mask = [spot_idx in sublist for sublist in segment_ids_per_bin]
    bin_idx = np.where(mask)[0][0]
    spot_voltage = voltage[spot_idx, :]
    spot_conductance = conductance[spot_idx, :]
    spot_current = mem_current[spot_idx]
    cmap = 'turbo'
    NEURON_COLOR = 'k'
    lower_bound = coords_dendrites_layer[:, 2].min()
    kernel_stop = 2.0
    plt.rcParams['font.size'] = 14
    black = "#000000"
    lorange = "#E69F00"
    lblue = "#56B4E9"
    green = "#009E73"
    yellow = "#F0E442"
    dblue = "#0072B2"
    dorange = "#D55E00"
    pink = "#CC79A7"
    # ########################################### Plot Fig 6 ###########################################################
    plt.rcParams['font.size'] = 10
    line_col_but_axon_c = LineCollection(
        lines_but_axon_layer,
        linewidths=1,
        linestyles='solid',
        colors=NEURON_COLOR,
        zorder=2
    )
    line_col_axon_c = LineCollection(
        lines_axon_layer,
        linewidths=1,
        linestyles='solid',
        colors=NEURON_COLOR,
        alpha=0.35,
        zorder=1
    )

    # SETUP FIGURE
    fig_c = plt.figure(constrained_layout=False, figsize=(8.5, 6.5))
    widths = [3.5, 2, 3.5, 2, 3.5]
    heights = [0.12, 1, 0.4, 1, 0.4, 1]
    spec = fig_c.add_gridspec(ncols=5, nrows=6, width_ratios=widths, height_ratios=heights)
    ax_colorbar_kernel_c_i = fig_c.add_subplot(spec[0, 0])
    ax_colorbar_kernel_c_e = fig_c.add_subplot(spec[0, 2])
    ax_neuron_c = fig_c.add_subplot(spec[1, 4])
    ax_kernel_c_i = fig_c.add_subplot(spec[1, 0])
    ax_kernel_c_e = fig_c.add_subplot(spec[1, 2])
    ax_avg_c_i = fig_c.add_subplot(spec[3, 0], sharex=ax_kernel_c_i)
    ax_avg_c_e = fig_c.add_subplot(spec[3, 2], sharex=ax_kernel_c_e)
    ax_exp_syn_c = fig_c.add_subplot(spec[3, 4])
    ax_incond_c = fig_c.add_subplot(spec[5, 0])
    ax_incur_c = fig_c.add_subplot(spec[5, 2])
    ax_cur_c = fig_c.add_subplot(spec[5, 4])
    # Plot Neuron Morphology Zoomed In
    line_col_but_axon = LineCollection(
        lines_but_axon_layer,
        linewidths=3,
        linestyles='solid',
        colors='k',
        zorder=2
    )
    edge_coords = np.unique(np.vstack(lines_but_axon_layer), axis=0)
    segment_coords_layer = coords_layer[spot_idx, :]
    z_limits = np.array([segment_coords_layer[2] - 125, segment_coords_layer[2] + 125])
    ax_neuron_c.add_collection(line_col_but_axon)
    ax_neuron_c.plot(edge_coords[:, 0], edge_coords[:, 1], 'or', markersize=2.5, zorder=3)
    raw_xlim = np.array([coords_dendrites_layer[:, 0].min() - 30, coords_dendrites_layer[:, 0].max() + 30])
    raw_ylim = (lower_bound, 0)
    aspect = abs(raw_xlim[1] - raw_xlim[0]) / abs(raw_ylim[1] - raw_ylim[0])  # width / height
    ax_neuron_c.set_xlim(raw_xlim / 5 + segment_coords_layer[0])
    ax_neuron_c.set_ylim(z_limits)
    fontprops = fm.FontProperties(size=16)
    scalebar = AnchoredSizeBar(ax_neuron_c.transData,
                               size=1,
                               size_vertical=25,
                               label='',
                               loc='lower left',
                               pad=1,
                               color='black',
                               frameon=False,
                               fontproperties=fontprops)

    bar = ax_neuron_c.add_artist(scalebar)
    ax_neuron_c.set_xticks([])
    ax_neuron_c.set_yticks([])
    # Plot Synaptic Input Currents
    net_cur_artist = ax_incur_c.plot(t_axis, np.sum(spot_current, axis=0), color='tab:green', label='Net', zorder=2)
    ampa_cur_artist = ax_incur_c.plot(t_axis, spot_current[0, :], color='tab:red', alpha=0.65, label='AMPA', zorder=1)
    nmda_cur_artist = ax_incur_c.plot(t_axis, spot_current[1, :], color='tab:orange', alpha=0.65, label='NMDA', zorder=1)
    gabaa_cur_artist = ax_incur_c.plot(t_axis, spot_current[2, :], color='tab:blue', alpha=0.65, label='GABAa', zorder=1)
    gabab_cur_artist = ax_incur_c.plot(t_axis, spot_current[3, :], color='tab:purple', alpha=0.65, label='GABAb', zorder=1)
    ax_incur_c.set_ylabel('Synaptic Current (nA)', labelpad=10)
    ax_incur_c.set_xlabel('Time (ms)')
    ax_incur_c.set_xlim(0 - SIM_DURATION/10, t_stop + SIM_DURATION/10)
    ax_incur_c.legend(fontsize=6, loc='lower right')
    # Plot Excitatory Kernel Colorplot
    colormap_c_e = ax_kernel_c_e.pcolor(t_bin_centers, z_bin_centers, spike_rate_ex, cmap='YlGnBu_r', vmax=kernel_max_e)
    ax_kernel_c_e.set_xticks([0, 0.5, 1, 1.5, 2])
    ax_kernel_c_e.set_xlim(0, kernel_stop)
    ax_kernel_c_e.yaxis.set_ticks_position('both')
    ax_kernel_c_e.xaxis.set_ticks_position('bottom')
    ax_kernel_c_e.set_xlabel('Time (ms)')
    ax_kernel_c_e.set_ylabel('Cortical Depth ($\mu$m)')
    ax_kernel_c_e.grid(axis='y', which='both', color='tab:gray', alpha=1)
    ax_kernel_c_e.grid(axis='x', which='both', color='tab:gray', alpha=1)
    cbar_label_e = 'Spike Density ($ms^{-1}$ $\mu$$m^{-1}$)' if kernel_l23_ex.density else 'Spike Count (#)'
    plt.colorbar(colormap_c_e, cax=ax_colorbar_kernel_c_e, orientation='horizontal', label=cbar_label_e)
    ax_colorbar_kernel_c_e.xaxis.set_label_position('top')
    ax_colorbar_kernel_c_e.xaxis.set_ticks_position('top')
    ax_colorbar_kernel_c_e.set_xticks([0, kernel_max_e/2, kernel_max_e])
    # Plot Inhibitory Kernel Colorplot
    colormap_c_i = ax_kernel_c_i.pcolor(t_bin_centers_inh, z_bin_centers_inh, spike_rate_inh, cmap='YlOrRd_r', vmax=kernel_max_i)
    ax_kernel_c_i.set_xticks([0, 0.5, 1, 1.5, 2])
    ax_kernel_c_i.set_xlim(0, kernel_stop)
    ax_kernel_c_i.yaxis.set_ticks_position('both')
    ax_kernel_c_i.xaxis.set_ticks_position('bottom')
    ax_kernel_c_i.set_xlabel('Time (ms)')
    ax_kernel_c_i.set_ylabel('Cortical Depth ($\mu$m)')
    plt.setp(ax_kernel_c_i.get_yticklabels(), visible=False)
    ax_kernel_c_i.grid(axis='y', which='both', color='tab:gray', alpha=1)
    ax_kernel_c_i.grid(axis='x', which='both', color='tab:gray', alpha=1)
    cbar_label_c_i = 'Spike Density ($ms^{-1}$ $\mu$$m^{-1}$)' if kernel_l23_inh.density else 'Spike Count (#)'
    plt.colorbar(colormap_c_i, cax=ax_colorbar_kernel_c_i, label=cbar_label_c_i,
                 orientation='horizontal')
    ax_colorbar_kernel_c_i.xaxis.set_label_position('top')
    ax_colorbar_kernel_c_i.xaxis.set_ticks_position('top')
    ax_colorbar_kernel_c_i.set_xticks([0, kernel_max_i/2, kernel_max_i])
    # Plot Dendritic Current
    ax_cur_c.plot(t_axis, current * 1e9, 'k-')
    ax_cur_c.set_xlabel('Time (ms)')
    # ax_cur_c.set_ylim(np.min(current * 1e9) - 0.1 * np.min(current * 1e9),
    #                   np.max(current * 1e9) + 0.1 * np.max(current * 1e9))
    ax_cur_c.set_ylabel('Dendritic Current (nA)')
    ax_cur_c.set_xlim(-2.5, t_stop + 2.5)
    ax_cur_c.yaxis.set_ticks_position('left')
    ax_cur_c.yaxis.set_label_position('left')
    ax_cur_c.set_xlim(0 - SIM_DURATION/10, t_stop + SIM_DURATION/10)
    # Plot Synaptic Conductance Inputs
    ax_incond_c.plot(t_axis, spot_conductance[0], color=black, label='AMPA', zorder=4)
    ax_incond_c.plot(t_axis, spot_conductance[1], color=lorange, label='NMDA', zorder=3)
    ax_incond_c.plot(t_axis, spot_conductance[2], color=lblue, alpha=0.65, label='GABAa', zorder=1)
    ax_incond_c.plot(t_axis, spot_conductance[3], color=pink, alpha=0.65, label='GABAb', zorder=2)
    ax_incond_c.set_xticks(np.arange(0, SIM_DURATION + 10, 10))
    ax_incond_c.set_ylabel('Conductance ($\mu$S)')
    ax_incond_c.set_xlabel('Time(ms)')
    ax_incond_c.legend(loc='upper right', fontsize=6)
    # Plot excitatory bin average
    ax_avg_c_e.plot(t_bin_centers, spike_rate_ex[bin_idx, :] * Z_BIN_SIZE, color='tab:blue', label='ex (L23 PC)')
    ax_avg_c_e.set_xlabel('Time (ms)')
    ax_avg_c_e.set_ylabel('Spike Rate (ms$^{-1}$)')
    # Plot inhibitory bin average
    ax_avg_c_i.plot(t_bin_centers, spike_rate_inh[bin_idx, :] * Z_BIN_SIZE, color='tab:red', label='ex (L23 PC)')
    ax_avg_c_i.set_xlabel('Time (ms)')
    ax_avg_c_i.set_ylabel('Spike Rate (ms$^{-1}$)')
    ax_avg_c_i.set_ylim(ax_avg_c_e.get_ylim())
    # Plot Synaptic Conductance Kernels
    ax_exp_syn_c.plot(t_axis, sim.conductance_AMPA, color=black, alpha=1, label='AMPA', zorder=4)
    ax_exp_syn_c.plot(t_axis, sim.conductance_NMDA, color=lorange, alpha=1, label='NMDA', zorder=3)
    ax_exp_syn_c.plot(t_axis, sim.conductance_GABAa, color=lblue, alpha=1, label='GABAa', zorder=1)
    ax_exp_syn_c.plot(t_axis, sim.conductance_GABAb, color=pink, alpha=1, label='GABAb', zorder=2)
    ax_exp_syn_c.set_xticks(np.arange(0, SIM_DURATION + 10, 10))
    ax_exp_syn_c.set_xlabel('Time (ms)')
    ax_exp_syn_c.set_ylabel('Conductance ($\mu$S)')
    ax_exp_syn_c.legend(loc='center right', fontsize=6)

    fig_c.subplots_adjust(top=0.85)
    fig_c.suptitle('Fig 6. Synaptic inputs and model output of a synapto-dendritic delay simulation.')

    # ################################### Plot Figure Components separately ############################################
    # line_col_but_axon = LineCollection(
    #     lines_but_axon_layer,
    #     linewidths=3,
    #     linestyles='solid',
    #     colors='k',
    #     zorder=2
    # )
    # edge_coords = np.unique(np.vstack(lines_but_axon_layer), axis=0)
    #
    # # segment_id = click_processor_c.seg_idx
    # segment_coords_layer = coords_layer[spot_idx, :]
    # z_limits = np.array([segment_coords_layer[2] - 125, segment_coords_layer[2] + 125])
    # fig_zoom = plt.figure(figsize=(7, 7))
    # ax_zoom = fig_zoom.add_subplot(111)
    # ax_zoom.add_collection(line_col_but_axon)
    # ax_zoom.plot(edge_coords[:, 0], edge_coords[:, 1], 'or', markersize=2.5, zorder=3)
    # raw_xlim = np.array([coords_dendrites_layer[:, 0].min() - 30, coords_dendrites_layer[:, 0].max() + 30])
    # raw_ylim = (lower_bound, 0)
    # aspect = abs(raw_xlim[1] - raw_xlim[0]) / abs(raw_ylim[1] - raw_ylim[0])  # width / height
    # ax_zoom.set_xlim(raw_xlim / 5 + segment_coords_layer[0])
    # ax_zoom.set_ylim(z_limits)
    # fontprops = fm.FontProperties(size=16)
    # scalebar = AnchoredSizeBar(ax_zoom.transData,
    #                            size=1,
    #                            size_vertical=25,
    #                            label='',
    #                            loc='lower left',
    #                            pad=1,
    #                            color='black',
    #                            frameon=False,
    #                            fontproperties=fontprops)
    #
    # bar = ax_zoom.add_artist(scalebar)
    #
    # identical_widths = 3.3
    # identical_heights = 2.6
    # mask = [spot_idx in sublist for sublist in segment_ids_per_bin]
    # bin_idx = np.where(mask)[0][0]
    # zoom_w, zoom_h = get_axis_size(ax_zoom)
    #
    # fig_cut = plt.figure()
    # ax_cut = fig_cut.add_subplot(111)
    # ax_cut.plot(t_bin_centers, spike_rate_ex[bin_idx, :] * Z_BIN_SIZE, color='tab:blue', label='ex (L23 PC)')
    # ax_cut.set_xlabel('Time (ms)')
    # ax_cut.set_ylabel('Spike Rate (ms$^{-1}$)')
    # fig_cut.tight_layout()
    # set_axis_size(identical_widths, identical_heights, ax=ax_cut)
    #
    # fig_cut_in = plt.figure()
    # ax_cut_in = fig_cut_in.add_subplot(111)
    # ax_cut_in.plot(t_bin_centers, spike_rate_inh[bin_idx, :] * Z_BIN_SIZE, color='tab:red', label='inh (L23 BC)')
    # ax_cut_in.set_xlabel('Time (ms)')
    # ax_cut_in.set_ylabel('Spike Rate (ms$^{-1}$)')
    # ax_cut_in.set_ylim(ax_cut.get_ylim())
    # fig_cut_in.tight_layout()
    # set_axis_size(identical_widths, identical_heights, ax=ax_cut_in)
    #
    # fig_synk = plt.figure()
    # ax_synk = fig_synk.add_subplot(111)
    # ax_synk.plot(t_axis, sim.conductance_AMPA, color=black, alpha=1, label='AMPA', zorder=4)
    # ax_synk.plot(t_axis, sim.conductance_NMDA, color=lorange, alpha=1, label='NMDA', zorder=3)
    # ax_synk.plot(t_axis, sim.conductance_GABAa, color=lblue, alpha=1, label='GABAa', zorder=1)
    # ax_synk.plot(t_axis, sim.conductance_GABAb, color=pink, alpha=1, label='GABAb', zorder=2)
    # ax_synk.set_xticks(np.arange(0, SIM_DURATION + 10, 10))
    # ax_synk.set_xlabel('Time (ms)')
    # ax_synk.set_ylabel('Conductance ($\mu$S)')
    # ax_synk.legend(loc='center right')
    # fig_synk.tight_layout()
    # set_axis_size(identical_widths, identical_heights, ax=ax_synk)
    #
    # fig_cond = plt.figure()
    # ax_cond = fig_cond.add_subplot(111)
    # ax_cond.plot(t_axis, spot_conductance[0], color=black, label='AMPA', zorder=4)
    # ax_cond.plot(t_axis, spot_conductance[1], color=lorange, label='NMDA', zorder=3)
    # ax_cond.plot(t_axis, spot_conductance[2], color=lblue, alpha=0.65, label='GABAa', zorder=1)
    # ax_cond.plot(t_axis, spot_conductance[3], color=pink, alpha=0.65, label='GABAb', zorder=2)
    # ax_cond.set_xticks(np.arange(0, SIM_DURATION + 10, 10))
    # ax_cond.set_ylabel('Conductance ($\mu$S)')
    # ax_cond.set_xlabel('Time(ms)')
    # ax_cond.legend(loc='center right')
    # fig_cond.tight_layout()
    # set_axis_size(identical_widths, identical_heights, ax=ax_cond)
    #
    # fig_syncur = plt.figure()
    # ax_syncur = fig_syncur.add_subplot(111)
    # ax_syncur.plot(t_axis, -spot_current[0, :], color=black, alpha=1, label='AMPA', zorder=4)
    # ax_syncur.plot(t_axis, -spot_current[1, :], color=lorange, alpha=1, label='NMDA', zorder=3)
    # ax_syncur.plot(t_axis, -spot_current[2, :], color=lblue, alpha=0.65, label='GABAa', zorder=1)
    # ax_syncur.plot(t_axis, -spot_current[3, :], color=pink, alpha=0.65, label='GABAb', zorder=2)
    # ax_syncur.plot(t_axis, -np.sum(spot_current, axis=0), zorder=5, color='tab:green', linestyle='--', label='Net')
    # # ax_syncur.set_ylim(-0.02, 0.02)
    # ax_syncur.set_xticks(np.arange(0, SIM_DURATION + 10, 10))
    # ax_syncur.axhline(0, color='k', linewidth=0.5, zorder=1)
    # ax_syncur.set_ylabel('Synaptic Current (nA)')
    # ax_syncur.set_xlabel('Time (ms)')
    # ax_syncur.legend(loc='center right' )
    # fig_syncur.tight_layout()
    # set_axis_size(identical_widths, identical_heights, ax=ax_syncur)
    #
    # fig_cmap = plt.figure()
    # ax_cmap = fig_cmap.add_subplot(212)
    # ax_cbar = fig_cmap.add_subplot(211)
    # cmap_e = ax_cmap.pcolor(t_bin_centers, z_bin_centers, spike_rate_ex, cmap='YlGnBu_r', vmax=kernel_max_e)
    # ax_cmap.set_xticks([0, 0.5, 1, 1.5, 2])
    # ax_cmap.set_xlim(0, kernel_stop)
    # ax_cmap.yaxis.set_ticks_position('both')
    # ax_cmap.xaxis.set_ticks_position('bottom')
    # ax_cmap.set_xlabel('Time (ms)')
    # ax_cmap.set_ylabel('Cortical Depth ($\mu$m)')
    # ax_cmap.grid(axis='y', which='both', color='tab:gray', alpha=1)
    # ax_cmap.grid(axis='x', which='both', color='tab:gray', alpha=1)
    # cbar_label_e = 'Spike Density ($ms^{-1}$ $\mu$$m^{-1}$)' if kernel_l23_ex.density else 'Spike Count (#)'
    # # ax_cbar.set_xticks([0, kernel_max_e/2, kernel_max_e])
    # fig_cmap.tight_layout()
    # cbar = plt.colorbar(cmap_e, cax=ax_cbar, orientation='horizontal', label=cbar_label_e)
    # ax_cbar.xaxis.set_ticks_position('top')
    # ax_cbar.xaxis.set_label_position('top')
    # set_axis_size(identical_widths, identical_heights, padding=[1.8, 0.5, 1.2, 0.8], ax=ax_cmap)
    # cmap_pos = ax_cmap.get_position()
    # cleft = cmap_pos.min[0]
    # cbottom = cmap_pos.max[1] + 0.05
    # cwidth = cmap_pos.width
    # cheight = cmap_pos.height
    # bbox = matplotlib.transforms.Bbox.from_bounds(cleft, cbottom, cwidth, cheight / 10)
    # ax_cbar.set_position(bbox)
    #
    # fig_cmap_in = plt.figure()
    # ax_cmap_in = fig_cmap_in.add_subplot(212)
    # ax_cbar_in = fig_cmap_in.add_subplot(211)
    # cmap_in = ax_cmap_in.pcolor(t_bin_centers, z_bin_centers, spike_rate_inh, cmap='YlOrRd_r', vmax=kernel_max_i)
    # ax_cmap_in.set_xticks([0, 0.5, 1, 1.5, 2])
    # ax_cmap_in.set_xlim(0, kernel_stop)
    # ax_cmap_in.yaxis.set_ticks_position('both')
    # ax_cmap_in.xaxis.set_ticks_position('bottom')
    # ax_cmap_in.set_xlabel('Time (ms)')
    # ax_cmap_in.set_ylabel('Cortical Depth ($\mu$m)')
    # ax_cmap_in.grid(axis='y', which='both', color='tab:gray', alpha=1)
    # ax_cmap_in.grid(axis='x', which='both', color='tab:gray', alpha=1)
    # cbar_label_in = 'Spike Density ($ms^{-1}$ $\mu$$m^{-1}$)' if kernel_l23_inh.density else 'Spike Count (#)'
    # # ax_cbar_in.set_xticks([0, kernel_max_i / 2, kernel_max_i])
    # fig_cmap_in.tight_layout()
    # cbar_in = plt.colorbar(cmap_in, cax=ax_cbar_in, orientation='horizontal', label=cbar_label_in)
    # ax_cbar_in.xaxis.set_ticks_position('top')
    # ax_cbar_in.xaxis.set_label_position('top')
    # set_axis_size(identical_widths, identical_heights, padding=[1.8, 0.5, 1.2, 0.8], ax=ax_cmap_in)
    # cmap_pos_in = ax_cmap_in.get_position()
    # cleft_in = cmap_pos_in.min[0]
    # cbottom_in = cmap_pos_in.max[1] + 0.05
    # cwidth_in = cmap_pos_in.width
    # cheight_in = cmap_pos_in.height
    # bbox_in = matplotlib.transforms.Bbox.from_bounds(cleft_in, cbottom_in, cwidth_in, cheight_in / 10)
    # ax_cbar_in.set_position(bbox_in)
    #
    # fig_nrn = plt.figure()
    # line_col_but_axon_nrn = LineCollection(
    #     lines_but_axon_layer,
    #     linewidths=1,
    #     linestyles='solid',
    #     colors='w',
    #     zorder=2
    # )
    # line_col_axon_nrn = LineCollection(
    #     lines_axon_layer,
    #     linewidths=1,
    #     linestyles='solid',
    #     colors='w',
    #     alpha=0.35,
    #     zorder=1
    # )
    # ax_nrn = fig_nrn.add_subplot(111)
    # ax_nrn.add_collection(line_col_axon_nrn)
    # ax_nrn.add_collection(line_col_but_axon_nrn)
    # raw_xlim = np.array([coords_layer[:, 0].min() - 30, coords_layer[:, 0].max() + 30])
    # ax_nrn.set_xlim(raw_xlim[0], raw_xlim[1])
    # # ax_nrn.set_yticks([0, -500, -1000, -1500, -2000, -2500])
    # ax_nrn.set_ylim(ax_cmap.get_ylim())
    # plt.setp(ax_nrn.get_xticklabels(), visible=False)
    # plt.setp(ax_nrn.get_yticklabels(), visible=False)
    # ax_nrn.set_xticks([])
    # ax_nrn.set_yticks([])
    # nrn_width = identical_heights
    # set_axis_size(nrn_width, identical_heights, padding=[0.1, 0.1, 0.1, 0.1], ax=ax_nrn)
    # ax_nrn.spines.top.set_visible(True)
    # ax_nrn.spines.bottom.set_visible(True)
    # ax_nrn.spines.right.set_visible(False)
    # ax_nrn.spines.left.set_visible(False)
    #
    # fig_cur = plt.figure()
    # ax_cur = fig_cur.add_subplot()
    # ax_cur.plot(t_axis, current * 1e9, 'k-')
    # ax_cur.set_xlabel('Time (ms)')
    # ax_cur.set_ylabel('Dendritic Current (nA)')
    # ax_cur.set_xlim(-2.5, t_stop + 2.5)
    # ax_cur.set_xticks(np.arange(0, SIM_DURATION + 10, 10))
    # ax_cur.yaxis.set_ticks_position('left')
    # ax_cur.yaxis.set_label_position('left')
    # ax_cur.set_xlim(0 - SIM_DURATION / 10, t_stop + SIM_DURATION / 10)
    # fig_cur.tight_layout()
    # set_axis_size(identical_widths, identical_heights, ax=ax_cur)
    #
    # if SAVE:
    #     # fig_c.savefig(save_folder.joinpath(f'{morph_id}_{na_label}_dendritic_delay_combined_IO_continuous.png'), dpi=1200, transparent=True)
    #     fig_zoom.savefig(save_folder.joinpath(f'{morph_id}_{na_label}_dendritic_delay_IO_zoomed_cell.png'), dpi=1200, transparent=True)
    #     fig_cut.savefig(save_folder.joinpath(f'{morph_id}_{na_label}_dendritic_delay_compute_inputs_ex_crosscut.png'), dpi=1200, transparent=True)
    #     fig_cut_in.savefig(save_folder.joinpath(f'{morph_id}_{na_label}_dendritic_delay_compute_inputs_inh_crosscut.png'), dpi=1200, transparent=True)
    #     fig_synk.savefig(save_folder.joinpath(f'{morph_id}_{na_label}_dendritic_delay_compute_input_ex_syn_kernels.png'), dpi=1200, transparent=True)
    #     fig_cond.savefig(save_folder.joinpath(f'{morph_id}_{na_label}_dendritic_delay_compute_input_ex_syn_conductance.png'), dpi=1200, transparent=True)
    #     fig_syncur.savefig(save_folder.joinpath(f'{morph_id}_{na_label}_dendritic_delay_compute_input_ex_syn_current.png'), dpi=1200, transparent=True)
    #     fig_cmap.savefig(save_folder.joinpath(f'{morph_id}_{na_label}_dendritic_delay_IO_ex_kernel_only.png'), dpi=1200, transparent=True)
    #     fig_cmap_in.savefig(save_folder.joinpath(f'{morph_id}_{na_label}_dendritic_delay_IO_inh_kernel_only.png'), dpi=1200, transparent=True)
    #     fig_cmap_in.savefig(save_folder.joinpath(f'{morph_id}_{na_label}_dendritic_delay_IO_inh_kernel_only.png'), dpi=1200, transparent=True)
    #     fig_nrn.savefig(save_folder.joinpath(f'{morph_id}_{na_label}_dendritic_delay_IO_inset_neuron_white.png'), dpi=1200, transparent=True)
    #     fig_cur.savefig(save_folder.joinpath(f'{morph_id}_{na_label}_dendritic_delay_IO_dend_cur.png'), dpi=1200, transparent=True)
