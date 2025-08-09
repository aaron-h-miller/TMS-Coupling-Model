import os
import pathlib
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.collections import LineCollection
from numpy.typing import NDArray
from helper_functions.neuron_cell_functions_lib import collect_lines_section, collect_lines
from neuron_interface.neurosim.simulation.voltage_curve_simulation import VoltageCurveSimulation
from neuron_interface.neurosim.cells import L23_PC_cADpyr
from neuron_interface.neurosim.simulation.simulation import WaveformType
from neuron_interface.runner.axonal_delay_runner import AxonalDelayRunner
from postprocessing_modules.axonal_delay_kernel import AxonalDelayKernel
from helper_functions.plotting_lib import hsv_darken_cmap


def extract_delay_from_voltage(neuron_cell, voltage_recordings: NDArray, time_rec: NDArray, stim_delay: float):
    """
    Locates axon terminal segments in all cells, and returns spike arrival times at terminals and at all compartments.
    Parameters
    ----------
    :param neuron_cell:           NeuronCell
    :param voltage_recordings:    Voltage recording at all segments in given cell
    :param time_rec:              Time recording from simulation
    :param stim_delay:            Stimulation Delay
    Returns
    -------
    :return:                      Spike delay (spike time - stimulation delay) for all terminals in this cell and the
                                  z values of the corresponding terminals and the total number of terminals and the terminal indices
                                  and the spike delays for ALL segments in tree order.
    """
    bucket_spike_delays = []

    # determine indexes of axon terminals for this cell
    segment_ids = []
    terminal_inds = []
    terminal_z = []
    soma_z = neuron_cell.soma[0](0.5).z_xtra
    ind = 0
    all_spike_delays = []
    terminal_spike_delays = []

    for section in neuron_cell.all:
        for seg_ind, segment in enumerate(section):
            segment_ids.append(ind)
            voltage = voltage_recordings[ind, :]

            if len(np.where(np.array(voltage) >= 0)[0]) != 0:
                zero_ind = np.where(np.array(voltage) >= 0)[0][0]
                time_zero = time_rec[zero_ind]
                delay = time_zero - stim_delay
            else:
                delay = -1
            all_spike_delays.append(delay)

            if (section in neuron_cell.axon) & (len(section.children()) == 0) & (seg_ind == len(list(section)) - 1):
                terminal_inds.append(ind)
                terminal_z.append(segment.z_xtra - soma_z)
                terminal_spike_delays.append(delay)
            ind += 1

    return terminal_spike_delays, terminal_z, terminal_inds, len(terminal_inds), all_spike_delays


def plot_movie_step(time_ind: int, label_ind, compartment_lines: NDArray, voltage_slice: NDArray,
                    color_min: float, color_max: float, save_path: pathlib.Path, savefigs: bool, show,
                    time_axis: NDArray, timeseries: NDArray, point_coordinates) -> plt.Figure:
    """
    Plots (and saves) a single frame of video of the time evolution of membrane potential across the whole cell.
    A timeseries is plotted beneath the cell corresponding to the compartment at given coordinates. Returns the
    associated figure object.
    Parameters
    ----------
    :param time_ind:            Time index to plot this frame.
    :param label_ind:           Label to name the saved frame file.
    :param compartment_lines:   Array of line coordinates to form cell compartments, Rows = [(x1, y1), (x2, y2)].
    :param timeseries:          Optional timeseries to plot beneath the cell.
    :param voltage_slice:       Voltage values at all compartments at the given time index.
    :param color_min:           Minimum voltage for color scale.
    :param color_max:           Maximum voltage for color scale.
    :param save_path:           Pathlib path to folder to save frame file
    :param savefigs:            Boolean indicating if to save frame file
    :param show:                Plot and show frame
    :param time_axis:           Time axis array of timeseries
    :param timeseries:          Data to plot beneach the cell
    :param point_coordinates:   Coordinates (x, z) in NEURON, of compartment whose data is plotted.

    :return:                    Figure object for the plotted frame
    """
    cmap = hsv_darken_cmap(plt.get_cmap('jet'), 0.9)
    line_col = LineCollection(compartment_lines, array=voltage_slice, cmap=cmap, clim=(color_min, abs(color_max)), linewidths=2)
    if timeseries is not None:
        fig, axs = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={'height_ratios': [5, 1], 'hspace': 0.01}, layout='constrained')
        axs[0].add_collection(line_col)
        axs[0].plot(point_coordinates[0], point_coordinates[1], 'ok')
        axs[0].set_xlim(coords[:, 0].min() - 30, coords[:, 0].max() + 30)
        axs[0].set_ylim(coords[:, 2].min() - 30, coords[:, 2].max() + 30)
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[0].set_aspect('equal')
        axs[0].set_anchor('W')

        plt.colorbar(line_col, ax=axs, orientation='vertical', label='Membrane Potential (mV)', aspect=30, shrink=0.95)

        axs[1].plot(time_axis, timeseries)
        axs[1].plot(time_axis[time_ind], timeseries[time_ind], "o")
        axs[1].set_xlabel('t (ms)')
        # axs[1].set_title(f"t={round(timeseries[time_ind], 3):.3f} ms", fontsize=12)
        axs[1].tick_params(axis='y')
        axs[1].set_yticks([-70, -35, 0, 35, 70])
        axs[1].set_anchor('W')

        if savefigs:
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            plt.savefig(save_path.joinpath(f'{label_ind}.png'), dpi=300)
        if show:
            plt.show(block=True)
        plt.close()

    return fig


if __name__ == '__main__':
    # Global Parameters
    fig_folder = pathlib.Path('/path/to/folder') # Folder to save files
    cell_id = 1 # index of the cell to simulate axonal delay
    cell = L23_PC_cADpyr.from_id(morphology_id=cell_id) # cell object constructed from index
    # Electric field parameters
    theta = 30
    phi = 0
    gradient = 0
    # Histogram bin size for axonal delay kernel
    z_bin_width = 100  # um
    t_bin_width = 0.1  # ms
    # Script Parameters
    SAVE = False
    PLOT_VIDEO = False

    # ################################## Simulate and Plot Fig 4 #######################################################
    intensity = 225
    print(f"Cell: {cell_id}, theta: {theta}, grad: {gradient}, phi: {phi}, int: {intensity}")
    params = np.array([theta, phi, gradient, intensity])
    wavetype = WaveformType.BIPHASIC

    cell.load()
    # VoltageCurveSimulation to simulate and return voltage timeseries at each cell compartment
    simulation = VoltageCurveSimulation(
        neuron_cell=cell,
        waveform_type=wavetype,
        stimulation_delay=0,
        simulation_time_step=0.005,
        simulation_duration=2
    )
    simulation.attach()
    simulation.apply_parametric_e_field(
        e_field_theta=theta,
        e_field_phi=phi,
        relative_mag_change_per_mm=gradient,
        phi_change_per_mm=0,
        theta_change_per_mm=0
    )
    time_recording, voltage_curves = simulation.get_voltage_curves(factor=intensity)
    # AxonalDelayRunner to simulate and return axon terminal delays, terminal z positions, and terminal indices
    delay_runner = AxonalDelayRunner(
        theta=theta,
        relative_mag_change_per_mm=gradient,
        intensity=intensity,
        cell_constructor=L23_PC_cADpyr,
        cell_ids=[cell_id],
        phi_count=1, # default is 60, this example only computes a single simulation for phi=0
        waveform_type=WaveformType.BIPHASIC,
    )
    sim_results, sim_params = delay_runner.run()
    terminal_delays, z_values, terminal_indexes = sim_results[0]
    # sim_results contain one list per cell id and phi value pair (corresponding to sim_params)

    # extract_delay_from_voltage computes action potential arrival delay from voltage for all compartments
    _, _, _, _, all_segment_delays = extract_delay_from_voltage(cell, voltage_curves, time_recording, 0, show=False)

    # AxonalDelayKernel computes the 2D histogram of spike density from the terminal delays
    kernel = AxonalDelayKernel()
    kernel.load_data_direct(
        layer='L23',
        delays=[terminal_delays],
        zs=[z_values],
        cell_type_list=['L23_PC_cADpyr'],
        layer_num_morphs={'L23_PC_cADpyr': 1},
        num_phi_rotations=1,
        theta=theta,
        gradient=gradient,
        intensity=intensity,
        cell_distribution_type='L23'
    )
    kernel.calculate_histogram(z_step=z_bin_width, t_step=t_bin_width, density=False)
    # ########################################## Plot Fig 4 ############################################################
    coords = cell.get_segment_coordinates()

    ind = 0
    # Separating out the axon and non axon delays
    recording_slice_axon = []
    recording_slice_but_axon = []
    for section in cell.all:
        for segment in section:
            if section not in cell.axon:
                recording_slice_but_axon.append(all_segment_delays[ind])
            else:
                recording_slice_axon.append(all_segment_delays[ind])
            ind += 1
    min_v = np.min(all_segment_delays)
    max_v = np.max(all_segment_delays)

    lines = collect_lines(cell, coords)
    lines_axon = collect_lines_section(cell, cell.axon, coords)
    lines_but_axon = collect_lines_section(cell, cell.axon, coords, negate=True)

    cmap = plt.get_cmap('jet')
    matplotlib.rcParams['font.size'] = 16
    fig = plt.figure(figsize=(12, 5))
    line_col_axon = LineCollection(lines_axon, array=recording_slice_axon, cmap=hsv_darken_cmap(cmap, 0.9),
                                   # plot axon segments' voltage
                                   norm=plt.Normalize(vmin=0, vmax=max_v), clim=(0, 0.9), linewidths=4.5)
    # line_col_axon = LineCollection(lines_axon, array=recording_slice_axon, colors='black', linewidths=1.8)
    line_col_but_axon = LineCollection(lines_but_axon, colors='tab:gray',
                                       linewidths=3.5)  # plot all other segments as uniform color
    axs = fig.add_subplot(121)
    ax_cplot = fig.add_subplot(122, sharey=axs)
    # axs.add_collection(line_col)
    axs.add_collection(line_col_but_axon)
    axs.add_collection(line_col_axon)
    axs.set_xlim(coords[:, 0].min() - 30, coords[:, 0].max() + 30)
    axs.set_ylim(coords[:, 2].min() - 30, coords[:, 2].max() + 30)
    axs.set_yticks(np.arange(-1200, 400 + 100, 100), minor=True)
    axs.set_yticks([-1200, -800, -400, 0, 400])
    axs.grid(axis='y', which='major', alpha=0.5)
    axs.grid(axis='y', which='minor', alpha=0.5)
    # axs.set_aspect('equal')
    axs.tick_params(left=False, right=True, bottom=False, labelleft=False, labelright=True, labelbottom=False)
    # fig.colorbar(line_col, ax=axs, orientation='vertical', label='delay (ms)')
    fig.colorbar(line_col_axon, ax=axs, orientation='horizontal', location='top', label='Delay (ms)')

    cplot = ax_cplot.pcolor(kernel.t_bin_centers, kernel.z_bin_centers, kernel.cell_histogram, cmap='ocean_r')
    ax_cplot.set_xlabel('Delay (ms)')
    ax_cplot.set_ylabel('Depth w.r.t Soma ($\mu$m)')
    ax_cplot.set_xlim(0, 0.9)
    ax_cplot.set_xticks([0.05, 0.25, 0.45, 0.65, 0.85])
    # ax_cplot.set_title(f'Sum: {np.sum(hist)},  ({theta}, {gradient}, {intensity})')
    ax_cplot.yaxis.set_label_position('right')
    ax_cplot.yaxis.set_ticks_position('right')
    ax_cplot.grid(axis='y', which='major', alpha=0.5)
    ax_cplot.grid(axis='y', which='minor', alpha=0.5)
    plt.colorbar(cplot, ax=ax_cplot, label='Number of Spikes', location='top', orientation='horizontal')
    # fig.subplots_adjust(bottom=0.18, right=0.85)
    plt.tight_layout()
    if SAVE:
        fig.savefig(fig_folder.joinpath(f'{cell_id}_L23_backpropagation_histogram.png'), dpi=600, transparent=True)
    fig.suptitle('Fig 4: Axonal Delay Histogram Calculation')
    fig.tight_layout()

    # ########################################## Plotting S1 Video and Fig 3B ##########################################
    sample_time = 0.525
    plt.rcParams['font.size'] = 16
    plt_step = int(len(time_recording) / len(time_recording))
    time_plt = time_recording[::plt_step]
    plt_indices = np.arange(0, len(time_recording), plt_step)
    sample_index = np.where(np.isclose(time_plt, sample_time))[0][0]
    print('Plotting Length:', len(time_plt))
    voltage_recordings_plt = voltage_curves[:, ::plt_step]
    stim_index = np.where(coords[:, 0] == np.max(coords[:, 0]))[0][0]
    stim_v = voltage_curves[stim_index, :]
    stim_position = [coords[stim_index, 0], coords[stim_index, 2]]
    min_v, max_v = np.min(voltage_curves), np.max(voltage_curves)
    plt.close('all')
    save_folder = fig_folder.joinpath(f'movie_images/movie_images_{int(theta)}_{int(gradient)}_{int(intensity)}')
    from tqdm import tqdm
    for ind, (plt_ind, recording_slice) in tqdm(enumerate((zip(plt_indices, voltage_recordings_plt.T)))):
        if PLOT_VIDEO:
            frame_obj = plot_movie_step(time_ind=plt_ind, label_ind=ind, compartment_lines=lines, timeseries=time_recording, voltage_slice=recording_slice, color_min=-70, color_max=max_v, stim_voltage=stim_v, timeseries_x_z=stim_position, save_folder=save_folder, savefigs=SAVE, show=False)
        if plt_ind == sample_index:
            frame_obj = plot_movie_step(time_ind=plt_ind, label_ind=ind, compartment_lines=lines, timeseries=time_recording, voltage_slice=recording_slice, color_min=-70, color_max=max_v, stim_voltage=stim_v, timeseries_x_z=stim_position, save_folder=save_folder, savefigs=SAVE, show=False)
            frame_obj.axes[0].set_title(f'Fig 3B: Backpropagation at t = {sample_time} ms')
            frame_obj.show()
    # ################################## Plot Fig 11A ##################################################################
    backprop_folder = pathlib.Path('/path/to/folder/')
    first_terminal_index = 1268
    intensities = [140, 150, 160, 170, 180, 190, 200, 250, 300, 350, 400, 450, 500, 550, 600]
    z_bin_width = 100  # um
    t_bin_width = 0.1  # ms
    max_delay = 0
    first_terminal_voltages = []
    all_delays_per_intensity = []
    first_terminal_activation = []
    for intensity in intensities:
        print(f"Cell: {cell_id}, theta: {theta}, grad: {gradient}, phi: {phi}, int: {intensity}")
        params = np.array([theta, phi, gradient, intensity])
        wavetype = WaveformType.BIPHASIC

        cell.load()
        simulation = VoltageCurveSimulation(neuron_cell=cell, waveform_type=wavetype, stimulation_delay=0,
                                            simulation_time_step=0.005, simulation_duration=2)
        simulation.attach()
        simulation.apply_parametric_e_field(e_field_theta=theta, e_field_phi=phi,
                                            relative_mag_change_per_mm=gradient, phi_change_per_mm=0,
                                            theta_change_per_mm=0)
        time_recording, voltage_curves = simulation.get_voltage_curves(factor=intensity)
        terminal_delays, _, _, _, all_delays = extract_delay_from_voltage(
            cell,
            voltage_curves,
            time_recording,
            0,
            show=False
        )

        local_max = max(terminal_delays)
        if local_max > max_delay:
            max_delay = local_max

        first_terminal_voltages.append(voltage_curves[first_terminal_index, :])
        all_delays_per_intensity.append(all_delays)
        first_terminal_activation.append(all_delays[first_terminal_index])
    first_terminal_voltages = np.vstack(first_terminal_voltages)
    first_terminal_activation = np.array(first_terminal_activation)
    # Plot Fig 11A
    fig_11A = plt.figure(constrained_layout=False, figsize=(8, 5))
    widths = [1, 1, 0.1]
    heights = [0.5, 0.5]
    spec_order = ((0, 0), (0, 1), (1, 0), (1, 1))
    fig11A_intensities = [140, 250, 300, 350]
    intensity_indices = [0, 7, 8, 9]
    spec = fig_11A.add_gridspec(ncols=3, nrows=2, width_ratios=widths, height_ratios=heights, wspace=0.4)
    plt.rcParams['font.size'] = 26
    for counter, (intensity_idx, intensity) in enumerate(zip(intensity_indices, fig11A_intensities)):
        ax_fig_11A = fig_11A.add_subplot(spec[spec_order[counter]])
        ind = 0
        # Separating out the axon and non axon delays
        recording_slice_axon = []
        recording_slice_but_axon = []
        for section in cell.all:
            for segment in section:
                if section not in cell.axon:
                    recording_slice_but_axon.append(all_delays_per_intensity[intensity_idx][ind])
                else:
                    recording_slice_axon.append(all_delays_per_intensity[intensity_idx][ind])
                ind += 1

        lines = collect_lines(cell, coords)
        lines_axon = collect_lines_section(cell, cell.axon, coords)
        lines_but_axon = collect_lines_section(cell, cell.axon, coords, negate=True)
        # Figure Start
        cmap = plt.get_cmap('jet')
        fig = plt.figure(figsize=(8, 5))
        line_col_axon = LineCollection(lines_axon, array=recording_slice_axon, cmap=hsv_darken_cmap(cmap, 0.9),
                                       # plot axon segments' voltage
                                       norm=plt.Normalize(vmin=0, vmax=max_delay), clim=(0, max_delay), linewidths=4.5)
        # line_col_axon = LineCollection(lines_axon, array=recording_slice_axon, colors='black', linewidths=1.8)
        line_col_but_axon = LineCollection(lines_but_axon, colors='tab:gray',
                                           linewidths=3.5)  # plot all other segments as uniform color
        axs = fig.add_subplot(111)
        # axs.add_collection(line_col)
        axs.add_collection(line_col_but_axon)
        axs.add_collection(line_col_axon)
        axs.set_xlim(coords[:, 0].min() - 30, coords[:, 0].max() + 30)
        axs.set_ylim(coords[:, 2].min() - 30, coords[:, 2].max() + 30)
        axs.set_yticks(np.arange(-1200, 400 + 100, 100), minor=True)
        axs.set_yticks([-1200, -800, -400, 0, 400])
        axs.grid(axis='y', which='major', alpha=0.5)
        axs.grid(axis='y', which='minor', alpha=0.5)
        # axs.set_aspect('equal')
        axs.tick_params(left=True, right=False, bottom=False, labelleft=True, labelright=False, labelbottom=False)
        plt.tight_layout()
        if SAVE:
            fig.savefig(backprop_folder.joinpath(f'{cell_id}_backpropagation_cell_{theta}_{gradient}_{intensity}.png'), dpi=600, transparent=True)

        line_col_axon = LineCollection(lines_axon, array=recording_slice_axon, cmap=hsv_darken_cmap(cmap, 0.9),
                                       # plot axon segments' voltage
                                       norm=plt.Normalize(vmin=0, vmax=max_delay), clim=(0, max_delay), linewidths=4.5)
        # line_col_axon = LineCollection(lines_axon, array=recording_slice_axon, colors='black', linewidths=1.8)
        line_col_but_axon = LineCollection(lines_but_axon, colors='tab:gray',
                                           linewidths=3.5)  # plot all other segments as uniform color
        ax_fig_11A.add_collection(line_col_but_axon)
        ax_fig_11A.add_collection(line_col_axon)
        ax_fig_11A.set_xlim(coords[:, 0].min() - 30, coords[:, 0].max() + 30)
        ax_fig_11A.set_ylim(coords[:, 2].min() - 30, coords[:, 2].max() + 30)
        ax_fig_11A.set_yticks(np.arange(-1200, 400 + 100, 100), minor=True)
        ax_fig_11A.set_yticks([-1200, -800, -400, 0, 400])
        ax_fig_11A.grid(axis='y', which='major', alpha=0.5)
        ax_fig_11A.grid(axis='y', which='minor', alpha=0.5)
        ax_fig_11A.set_title(f'{intensity} V/m', fontsize=12)
        # axs.set_aspect('equal')
        if counter % 2 == 0:
            ax_fig_11A.tick_params(left=True, right=False, bottom=False, labelleft=True, labelright=False, labelbottom=False, labelsize=12)
            ax_fig_11A.set_ylabel('Depth w.r.t Soma ($\mu$m)', fontsize=12)
        else:
            ax_fig_11A.tick_params(left=True, right=False, bottom=False, labelleft=False, labelright=False, labelbottom=False, labelsize=12)


        axs.tick_params(left=False, right=False, bottom=False, labelleft=False, labelright=False, labelbottom=False)
        if SAVE:
            fig.savefig(backprop_folder.joinpath(f'{cell_id}_backpropagation_cell_{theta}_{gradient}_{intensity}_noLabel.png'), dpi=600, transparent=True)
        plt.close(fig)
    # Plot Colorbars
    plt.rcParams['font.size'] = 20
    # Vertical
    fig_cbar = plt.figure(figsize=(2.1, 8.5))
    ax_cbar = fig_cbar.add_subplot(111)
    mappable = matplotlib.cm.ScalarMappable(
        norm=matplotlib.colors.Normalize(vmin=0, vmax=1.0),
        cmap=hsv_darken_cmap(cmap, 0.9)
    )
    fig_cbar.colorbar(mappable, cax=ax_cbar, label='Delay (ms)',
                      orientation='vertical')
    fig_cbar.tight_layout()
    if SAVE:
        fig_cbar.savefig(backprop_folder.joinpath('backpropagation_colorbar_vertical.png'), dpi=600, transparent=True)
    plt.close(fig_cbar)
    ax_fig_11A_cbar = fig_11A.add_subplot(spec[0:2, 2])
    plt.colorbar(mappable, cax=ax_fig_11A_cbar, orientation='vertical')
    ax_fig_11A_cbar.set_ylabel('Delay (ms)', fontsize=12)
    ax_fig_11A_cbar.tick_params(left=False, right=True, labelleft=False, labelright=True, labelsize=12)
    fig_11A.suptitle('Fig 11A: Axonal Delay Synchrony', fontsize=12)

    # ####################################### Plot Fig 11B #############################################################
    plt.rcParams['font.size'] = 16
    fig_first = plt.figure(figsize=(10, 3.7))
    ax_f = fig_first.add_subplot(111)
    cmap_f = 'bwr'
    centernorm = mcolors.CenteredNorm(0, clip=True)
    cplot = ax_f.pcolor(time_recording, intensities, first_terminal_voltages, cmap=cmap_f, norm=centernorm)
    ax_f.plot(first_terminal_activation, intensities, 'o:k', markersize=3)
    ax_f.set_xlabel('t (ms)')
    ax_f.set_ylabel('Intensity |$\mathbf{E}$| (V/m)')
    ax_f.set_xlim(0, 0.5)
    plt.colorbar(cplot, ax=ax_f, label='Membrane Potential (mV)')
    fig_first.tight_layout()

    if SAVE:
        fig_first.savefig(backprop_folder.joinpath('backpropagation_first_terminal_activation.png'), dpi=600, transparent=True)
    ax_f.set_title('Fig 11B: Axonal Delay shift at Single Compartment')

    # Plot TMS Waveform
    plt.figure(figsize=(7, 4.5))
    plt.plot(simulation.waveform_time, simulation.waveform)
    plt.xlim(0, 0.5)
    if SAVE:
        plt.savefig(backprop_folder.joinpath('biphasic_E_field.png'), dpi=600, transparent=True)
    plt.title('Biphasic Electric Field Input (normalized)', pad=20, fontsize=20)
    plt.tight_layout()

    # plt.ioff()
    plt.show()