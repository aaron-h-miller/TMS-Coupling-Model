""" Test Generating Custom Conductance Inputs
Author: Aaron Miller
Max Planck Institute for Human Cognitive and Brain Sciences, Leipzig Germany
Brain Networks Group

Generates synaptic conductance timeseries for AMPA, NMDA, GABAa, and GABAa synapses based on biexponential functions
convolved with a crosscut of the Axonal Delay Kernel for L23 excitatory cells for an arbitrary cortical depth.

Plotted is the spike rate timeseries used as the input rate, the axonal delay kernal from which this crosscut was taken,
the raw biexponential kernels, the convolved outputs/conductance timeseries, and the normalized comparison between
pre- and post-convolved functions.

"""
import numpy as np
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import pathlib
from postprocessing_modules.axonal_delay_kernel import get_axonal_delay_kernel
from neuron_interface.neurosim.simulation.dendritic_delay_simulation import DendriticDelaySimulation
from helper_functions.cell_type_lib import CellType
from helper_functions.load_cell_distributions import compute_distribution
from neuron_interface.neurosim.cells import L5_TTPC2_cADpyr
from helper_functions import __file__
MODULE_BASE = pathlib.Path(__file__).parent.parent

if __name__ == '__main__':
    precomputed_kernels = MODULE_BASE.joinpath('reference_data/Miller_2025/axon_kernels_precomputed/precomputed_kernels_50_0p1/')
    # Convert spike rate from axonal delay kernel to synaptic/nonspecific current input at each compartment
    # Load Postsynaptic L5 Cell
    morph_ids = CellType.cell_ids['L5_TTPC2_cADpyr']
    morph_id = morph_ids[7]
    morph = L5_TTPC2_cADpyr.from_id(morph_id)
    morph.load()
    _, _, _, l5_mean_depth, _, _ = compute_distribution('L5', 0.025, micrometers=True)

    # Set User Parameters
    SIM_DURATION = 100  # ms
    SIM_TIME_STEP = 0.025  # ms
    thet = 30.0
    grad = 0.0
    intens = 225.0
    FRACTION_NMDA = 0.5
    FRACTION_GABA_A = 0.9
    NA_SUPPRESSED = True
    DENSITY = True
    INCLUDE_AXON = False  # inclusion of axon current in soma current calculation
    na_label = 'noSodium' if NA_SUPPRESSED else 'Sodium'
    SAVE = False
    SOMA_DEPTH = l5_mean_depth + 500
    COLORMAP = 'RdYlBu_r'

    # ###################################### Set Up Simulation #########################################################
    sim = DendriticDelaySimulation(morph, sodium_blocked=NA_SUPPRESSED)

    kernel_l23_ex = get_axonal_delay_kernel(source_layer_name='L23', delay_z_folder=pathlib.Path(''),
                                            precomputed_kernels_folder=precomputed_kernels,
                                            theta=thet, gradient=grad, intensity=intens, z_step=100, t_step=0.1,
                                            smooth_z_step=100, smooth_t_step=SIM_TIME_STEP, density=DENSITY,
                                            compute_fresh=False)

    kernel_data_l23_ex = {
        'density': kernel_l23_ex.density,
        'layer_histogram': kernel_l23_ex.layer_histogram,
        'z_bins_layer': kernel_l23_ex.z_bins_layer,
        'z_bin_layer_centers': kernel_l23_ex.z_bin_layer_centers,
        't_bin_centers': kernel_l23_ex.t_bin_centers
    }

    kernel_l23_ex.plot_avg_layer_kernel(histogram=False, dynamic=True, crosscuts=True, cmap=COLORMAP)
    crosscut_index = 5
    z_crosscut = kernel_l23_ex.z_layer[crosscut_index]
    spike_rate_timeseries = kernel_l23_ex.layer_kernel[crosscut_index, :] * abs(np.diff(kernel_l23_ex.z_layer)[0])

    plt.rcParams['font.size'] = 16
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(kernel_l23_ex.t_kernel, spike_rate_timeseries, '-')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Spike Rate (Hz)')
    ax.set_title('Axonal Delay Kernel Crosscut')
    fig.tight_layout()

    # Compute Synaptic Biexponential Kernels for Membrane Conductivity
    time_conductance = np.arange(0, SIM_DURATION + SIM_TIME_STEP, SIM_TIME_STEP)
    # Values from Rusu et. al. 2014
    # # NMDA
    # NMDA_t_rise = 2  # ms - rise
    # NMDA_t_fall = 26  # ms - fall
    # NMDA_reversal = 0  # mV - reversal
    # NMDA_peak_conductance = 0.03  # mu Ohm^-1
    # # AMPA
    # AMPA_t_rise = 0.2  # ms - rise time
    # AMPA_t_fall = 1.7  # ms - fall time
    # AMPA_reversal = 0  # mV - reversal potential
    # AMPA_peak_conductance = 0.1  # mu Ohm^-1
    # # GABAa
    # GABAa_t_rise = 0.3  # ms - rise time
    # GABAa_t_fall = 2.5  # ms - fall time
    # GABAa_reversal = -70  # mV - reversal potential
    # GABAa_peak_conductance = 0.5  # mu Ohm^-1
    # # GABAb
    # GABAb_t_rise = 45.2  # ms - rise time
    # GABAb_t_fall = 175.16  # ms - fall time
    # GABAb_reversal = -90  # mV - reversal potential
    # GABAb_peak_conductance = 0.5  # mu Ohm^-1

    # Values from Dura-Bernal et al. 2023
    # NMDA
    NMDA_t_rise = 15  # ms - rise
    NMDA_t_fall = 150  # ms - fall
    NMDA_reversal = 0  # mV - reversal
    NMDA_peak_conductance = 0.03  # mu Ohm^-1
    # AMPA
    AMPA_t_rise = 0.05  # ms - rise time
    AMPA_t_fall = 5.3 # ms - fall time
    AMPA_reversal = 0  # mV - reversal potential
    AMPA_peak_conductance = 0.1  # mu Ohm^-1
    # GABAa
    GABAa_t_rise = 0.07  # ms - rise time
    GABAa_t_fall = 18.2  # ms - fall time
    GABAa_reversal = -80  # mV - reversal potential
    GABAa_peak_conductance = 0.5  # mu Ohm^-1
    # GABAb
    GABAb_t_rise = 3.5  # ms - rise time
    GABAb_t_fall = 260.9  # ms - fall time
    GABAb_reversal = -93  # mV - reversal potential
    GABAb_peak_conductance = 0.5  # mu Ohm^-1

    synapse_labels = ['AMPA', 'NMDA', 'GABAa', 'GABAb']
    rise_times = np.array([AMPA_t_rise, NMDA_t_rise, GABAa_t_rise, GABAb_t_rise]).reshape((-1, 1))
    fall_times = np.array([AMPA_t_fall, NMDA_t_fall, GABAa_t_fall, GABAb_t_fall]).reshape((-1, 1))
    peak_conductances = np.array([AMPA_peak_conductance, NMDA_peak_conductance, GABAa_peak_conductance, GABAb_peak_conductance]).reshape((-1, 1))

    # conductance_AMPA = - np.exp(- time_conductance / AMPA_t_rise) + np.exp(- time_conductance / AMPA_t_fall)
    # conductance_AMPA *= AMPA_peak_conductance / np.max(conductance_AMPA)
    # conductance_NMDA = - np.exp(- time_conductance / NMDA_t_rise) + np.exp(- time_conductance / NMDA_t_fall)
    # conductance_NMDA *= NMDA_peak_conductance / np.max(conductance_NMDA)
    # conductance_GABAa = - np.exp(- time_conductance / GABAa_t_rise) + np.exp(- time_conductance / GABAa_t_fall)
    # conductance_GABAa *= GABAa_peak_conductance / np.max(conductance_GABAa)
    # conductance_GABAb = - np.exp(- time_conductance / GABAb_t_rise) + np.exp(- time_conductance / GABAb_t_fall)
    # conductance_GABAb *= GABAb_peak_conductance / np.max(conductance_GABAb)

    conductances = -np.exp(-time_conductance/rise_times) + np.exp(-time_conductance/fall_times)
    peak_prenorm = np.max(conductances, axis=1)
    t_peak = fall_times * rise_times / (fall_times - rise_times) * np.log(fall_times / rise_times)
    calc_peak = np.exp(-t_peak / fall_times) - np.exp(-t_peak / rise_times)
    conductances *= peak_conductances / np.max(conductances, axis=1).reshape((-1, 1))
    fig_g_raw = plt.figure()
    ax_g_raw = fig_g_raw.add_subplot(111)
    # ax_g_raw.plot(time_conductance, conductance_AMPA, label='AMPA')
    # ax_g_raw.plot(time_conductance, conductance_NMDA, label='NMDA')
    # ax_g_raw.plot(time_conductance, conductance_GABAa, label='GABAa')
    # ax_g_raw.plot(time_conductance, conductance_GABAb, label='GABAb')
    ax_g_raw.plot(time_conductance, conductances.T)
    ax_g_raw.set_xlabel('Time (ms)')
    ax_g_raw.set_ylabel('Conductance ($\mu \Omega^-1$)')
    ax_g_raw.legend(synapse_labels)
    ax_g_raw.set_title('Biexpoential Synaptic Kernels')
    fig_g_raw.tight_layout()

    convolved_conductances = np.zeros(shape=(4, len(spike_rate_timeseries) + conductances.shape[1] - 1)) - 1
    for row in range(conductances.shape[0]):
        convolved_conductances[row, :] = np.convolve(conductances[row, :], spike_rate_timeseries, mode='full')
    convolved_conductances = convolved_conductances[:, :len(time_conductance)]

    fig_conv = plt.figure()
    plt.plot(time_conductance, convolved_conductances.T)
    plt.xlabel('Time (ms)')
    plt.ylabel('Conductance ($\mu \Omega^-1$)')
    plt.title('Firing Rate Convolved Conductance')
    plt.legend(synapse_labels)
    plt.tight_layout()

    fig_compare, ax_c = plt.subplots(4, 1, figsize=(7, 8), sharex='all')
    for row, (ax, synapse) in enumerate(zip(ax_c, synapse_labels)):
        ax.plot(time_conductance, conductances[row, :] / np.max(conductances[row, :]), label='Biexponential')
        ax.plot(time_conductance, convolved_conductances[row, :] / np.max(convolved_conductances[row, :]), label='Convolved Conductance')
        ax.legend(fontsize=10)
        ax.set_title(f'{synapse}')
    ax.set_xlim((-0.5, 20))
    fig_compare.tight_layout()

    # FYI, in order for the result to be positioned properly, the window function (second argument) must be symmetric about 0

