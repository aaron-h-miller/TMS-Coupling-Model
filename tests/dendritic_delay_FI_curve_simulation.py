
import colorsys
import os
import pathlib
import h5py
import numpy as np
import matplotlib
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from neuron import h
from neuron.units import ms, mV
import json
from neuron_interface.neurosim.cells import L5_TTPC2_cADpyr
from helper_functions.neuron_cell_functions_lib import collect_lines, collect_lines_section
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


def hsv_darken_cmap(cmap, value=1.):
    colors = cmap(np.arange(cmap.N))
    hls = np.array([colorsys.rgb_to_hls(*c) for c in colors[:, :3]])
    hls[:, 1] *= value
    rgb = np.clip(np.array([colorsys.hls_to_rgb(*c) for c in hls]), 0, 1)
    return mcolors.LinearSegmentedColormap.from_list("", rgb)


def post_finitialize():
    """ Initialization methode to unsure a steady state before the actual simulation is started.
    """
    temp_dt = h.dt

    h.t = -1e11
    h.dt = 1e9
    # h.t = -1e3
    # h.dt = 0.1
    print('Initialize Start')
    while h.t < -h.dt:  # IF t >= 0 then might trigger events which depend on h.t
        h.fadvance()
    h.dt = temp_dt
    h.t = 0
    h.fcurrent()
    h.frecord_init()
    print('Initialization Complete')


def plot_movie_step(time_ind, label_ind, lines, time_axis, recording_slice, min_v, max_v, save_folder, save, show, trace_data=None):
    plt.rcParams['font.size'] = 14
    cmap = hsv_darken_cmap(plt.get_cmap('jet'), 0.9)
    line_col = LineCollection(lines, array=recording_slice, cmap=cmap, clim=(min_v, abs(max_v)), linewidths=2)
    if trace_data is not None:
        fig, axs = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={'height_ratios': [5, 1], 'hspace': 0.01}, layout='constrained')
        axs[0].add_collection(line_col)
        axs[0].set_xlim(coords[:, 0].min() - 30, coords[:, 0].max() + 30)
        axs[0].set_ylim(coords[:, 2].min() - 30, coords[:, 2].max() + 30)
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[0].set_aspect('equal')
        axs[0].set_anchor('W')

        plt.colorbar(line_col, ax=axs, orientation='vertical', label='membrane potential (mV)', aspect=30, shrink=0.95)

        axs[1].plot(time_axis, trace_data)
        axs[1].plot(time_axis[time_ind], trace_data[time_ind], "o")
        axs[1].set_xlabel('time (ms)')
        # axs[1].set_title(f"t={round(timeseries[time_ind], 3):.3f} ms", fontsize=12)
        axs[1].tick_params(axis='y')
        axs[1].set_ylim(np.min(trace_data), np.max(trace_data) + 0.1 * np.max(trace_data))
        axs[1].set_anchor('W')

        if save:
            movie_folder = save_folder + '/movie_images/'
            if not os.path.isdir(movie_folder):
                os.mkdir(movie_folder)
            plt.savefig(movie_folder + f'{label_ind}.png', dpi=300)
        if show:
            plt.show(block=True)
        plt.close()
    else:
        fig, axs = plt.subplots(2, 1, figsize=(16, 9), gridspec_kw={'height_ratios': [8, 1]})
        axs[0].add_collection(line_col)
        axs[0].set_xlim(coords[:, 0].min() - 30, coords[:, 0].max() + 30)
        axs[0].set_ylim(coords[:, 2].min() - 30, coords[:, 2].max() + 30)
        axs[0].set_aspect('equal')
        fig.colorbar(line_col, ax=axs[0], orientation='vertical', label='membrane potential (mV)')

        axs[1].plot(time_axis, np.ones(len(time_axis)), 'k', linewidth=3)
        axs[1].plot(time_axis[time_ind], 1, "or")
        axs[1].set_xlabel(f'time (ms)')
        axs[1].spines['top'].set_color('none')
        axs[1].spines['left'].set_color('none')
        axs[1].spines['right'].set_color('none')
        axs[1].set_yticks([])

        if save:
            movie_folder = save_folder + '/movie_images/'
            if not os.path.isdir(movie_folder):
                os.mkdir(movie_folder)
            plt.savefig(movie_folder + f'{label_ind}.png', dpi=300, transparent=True)
        if show:
            plt.show(block=True)
        plt.close()


def segment_sublist_mask(neuron_cell, section_sublist, num_segs):
    assert neuron_cell.loaded
    counter = 0
    dendrite_segment_mask = np.zeros(num_segs, dtype=bool)
    for sec in neuron_cell.all:
        for seg in list(sec):
            if sec in section_sublist:
                dendrite_segment_mask[counter] = True
            counter += 1
    return dendrite_segment_mask


def apply_current_extracellular(cell, electrode_position, current_amp):
    assert cell.loaded, 'Cell must be loaded'

    for section in cell.all:
        for segment in section:
            seg_pos = np.array([segment.x_xtra, segment.y_xtra, segment.z_xtra])
            distance = np.sqrt(np.sum((seg_pos - electrode_position)**2))
            # nA / (A / (V um) * um) -> V -> mV (convert nA to A and then V to mV)
            segment.es_xtra = (current_amp / (4 * np.pi * SIGMA * distance)) * 10**3 / 10**9


def simulate_current_injection(cell, current_amplitude, stim_onset=0.5, sim_duration=10.0, sim_time_step=0.025, electrode_pos=None, extracellular=False):
    assert cell.loaded, 'Cell must be loaded'
    assert stim_onset > 0, 'TMS Onset time must be after t=0'
    assert not (extracellular and (electrode_pos is None)), 'Electrode position must be defined for extracellular current'

    time_recording = h.Vector().record(h._ref_t)
    action_potentials = h.Vector()
    hillock_voltage = h.Vector()
    # get axonal hillock
    nearest_secs = cell.soma[0].children()
    hillock_sec = nearest_secs[np.where([sec in cell.axon for sec in nearest_secs])[0][0]]
    hillock_seg = hillock_sec(0)
    # apply spike recorder
    # ap_counter = h.NetCon(hillock_seg._ref_v, None, sec=hillock_sec)
    # ap_counter.threshold = 0 * mV
    # ap_counter.delay = 0
    # ap_counter.record(action_potentials)
    ap_counter = h.APCount(hillock_seg)
    ap_counter.thresh = 0 * mV
    ap_counter.record(action_potentials)
    # apply voltage recorder
    hillock_voltage.record(hillock_seg._ref_v)

    if extracellular:
        apply_current_extracellular(cell=cell, electrode_position=electrode_pos, current_amp=current_amplitude)
    else:
        seg = cell.apic[0](0)
        stimobj = h.IClamp(seg)
        stimobj.delay = stim_onset
        stimobj.dur = sim_duration - stim_onset
        stimobj.amp = current  # nA

    waveform_time = np.arange(0, sim_duration + sim_time_step, sim_time_step)
    ind_onset = np.where(waveform_time >= stim_onset)[0][0]
    step_fn = np.ones(shape=(len(waveform_time, )))
    step_fn[:ind_onset] = 0
    waveform_vector = h.Vector(step_fn)
    waveform_time_vector = h.Vector(waveform_time)
    waveform_vector.play(h._ref_stim_xtra, waveform_time_vector, 1)

    # ################################################ Simulate ########################################################
    # set initial state
    init_handler = h.FInitializeHandler(2, post_finitialize)
    sim_temperature = 34  # degrees C
    initial_voltage = -70.0 * mV  # mV

    h.celsius = sim_temperature
    h.dt = sim_time_step
    # h.tstop = sim_duration
    h.finitialize(initial_voltage)

    # simulate
    print('Starting Simulation')
    h.continuerun(sim_duration)
    print('Simulation Complete')

    time = np.array(list(time_recording))
    hillock_voltage = np.array(list(hillock_voltage))
    action_potentials = np.array(list(action_potentials))

    return time, action_potentials, hillock_voltage


if __name__ == '__main__':
    save_folder = pathlib.Path('/path/to/folder/')

    morph_ids = L5_TTPC2_cADpyr.get_morphology_ids()
    morph_id = morph_ids[15]
    morph = L5_TTPC2_cADpyr.from_id(morph_id)
    morph.load()

    SAVE = False
    ONSET = 1
    DURATION = 600 * ms
    TIME_STEP = 0.025 * ms
    POS_STEP = 100  # um
    SIGMA = 0.276 / 10**6  # S/m -> A/(V*m) -> A/(V * um)

    # Calculate electrode position grid and exclude positions closer than 30um and greater than 1000um
    coords = morph.get_segment_coordinates()
    x_bounds = np.array([POS_STEP * (np.min(coords[:, 0]) // POS_STEP), POS_STEP * (np.max(coords[:, 0]) // POS_STEP)])
    y_bounds = np.array([POS_STEP * (np.min(coords[:, 1]) // POS_STEP), POS_STEP * (np.max(coords[:, 1]) // POS_STEP)])
    z_bounds = np.array([POS_STEP * (np.min(coords[:, 2]) // POS_STEP), POS_STEP * (np.max(coords[:, 2]) // POS_STEP)])

    x_range = np.arange(x_bounds[0], x_bounds[1] + POS_STEP, POS_STEP)
    y_range = np.arange(y_bounds[0], y_bounds[1] + POS_STEP, POS_STEP)
    z_range = np.arange(z_bounds[0], z_bounds[1] + POS_STEP, POS_STEP)

    electrode_coords = []
    for x in x_range:
        for y in y_range:
            for z in z_range:
                diff = coords - np.array([x, y, z]).reshape(1, 3)
                dist = np.sqrt(np.sum(diff**2, axis=1))
                if 30.0 < dist.min() < 1000.0:
                    electrode_coords.append([x, y, z])

    # rand = np.random.randint(0, len(electrode_coords))
    ind = 514
    electrode = np.array(electrode_coords[ind])
    # currents = [0.05 * 10**-3]  # A
    # current = 0.2 * 10**-3
    # currents = np.linspace(0.05 * 10**-3, 0.1 * 10**-3, 5)
    # currents = [1]  # nA
    currents = np.linspace(0.1, 5, 5)
    durations = [600, 600, 600, 600, 600]
    freqs = []
    for current, duration in zip(currents, durations):
        print(f'Current: {current} nA, Duration: {duration} ms')
        t, aps, hillock_v = simulate_current_injection(morph, current, stim_onset=ONSET, sim_duration=duration)
        # Calculate Firing Rate
        T = (DURATION - ONSET) / 10 ** 3  # ms -> s
        freq = len(aps) / T
        string_curr = f'{current:.3f}'
        if len(aps) > 1:
            interspike_interval = np.diff(aps) / 10**3
            frequencies = 1 / interspike_interval
            hist, bins = np.histogram(frequencies, bins=len(frequencies))
            max_idx = np.where(hist == np.max(hist))[0][0]
            max_freq = (bins[max_idx] + bins[max_idx + 1]) / 2

            fig, (ax_freq, ax_time) = plt.subplots(2, 1, figsize=(6, 6), gridspec_kw={'height_ratios': [3, 1]})
            ax_freq.stairs(hist, bins, fill=True, )
            ax_freq.axvline(max_freq, color='r', label=f'{max_freq:.2f} Hz')
            ax_freq.set_xlabel('Spike Rate (Hz)')
            ax_freq.set_title(f'Frequency Spectrum')
            ax_freq.legend()
            ax_time.plot(t, hillock_v)
            ax_time.set_xlabel('t (ms)')
            ax_time.set_ylabel('V (mV)')
            ax_time.set_title(f'Hillock Voltage')
            fig.suptitle(f'I = {current:.3f} nA')
            fig.tight_layout()
            if SAVE:
                fig.savefig(save_folder.joinpath(f'{morph_id}_{string_curr.replace(".", "p")}nA_FI_response.png'), dpi=600, transparent=True)

            freqs.append(max_freq)
        else:
            freqs.append(0)

            fig2 = plt.figure(figsize=(6, 6))
            plt.plot(t, hillock_v)
            plt.xlabel('t (ms)')
            plt.ylabel('V (mV)')
            plt.title(f'Hillock Voltage: I = {current:.3f} nA')
            if SAVE:
                fig2.savefig(save_folder.joinpath(f'{morph_id}_{string_curr.replace(".", "p")}nA_timeseries.png'), dpi=600, transparent=True)

    fig3 = plt.figure()
    plt.plot(currents, freqs, '-')
    plt.plot(currents, freqs, '.r', markersize=10)
    plt.xlabel('I (nA)')
    plt.ylabel('Firing Rate (Hz)')
    plt.title('F-I Curve')
    if SAVE:
        fig3.savefig(save_folder.joinpath(f'{morph_id}_FI_curve.png'), dpi=600, transparent=True)

