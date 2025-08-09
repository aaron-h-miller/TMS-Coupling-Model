""" Custom Synaptic Input Test
Author: Aaron Miller
Max Planck Institute for Human Cognitive and Brain Sciences, Leipzig Germany
Brain Networks Group

Tests the custom synaptic input point processes h.SunCur and h.SynCurNMDA, defined by
neuron_interface/neurosim/mechanisms/synCur.mod
neuron_interface/neurosim/mechanisms/synCur_NMDA.mod

The test cell is single compartment passive only, receiving input conductance timeseries based on biexponential
kernels for AMPA and NMDA synapses, convolved with an arbitrarily chosen crosscut of the L23 excitatory
axonal delay kernel at a respectively arbirary cortical depth.

The plotted result depicts the membrane potential of the compartment cell, the transmembrane current due to each
synaptic input, and the conductance timeseries of each synaptic input.
"""
import pathlib

from neuron import h
import numpy as np

# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from helper_functions import __file__
MODULE_BASE = pathlib.Path(__file__).parent.parent

from postprocessing_modules.axonal_delay_kernel import get_axonal_delay_kernel
from helper_functions.neuron_cell_functions_lib import post_finitialize
from neuron_interface.neurosim.cells import L5_TTPC2_cADpyr
from helper_functions.cell_type_lib import CellType

if __name__ == '__main__':
    axonal_delay_data_folder = MODULE_BASE.joinpath('reference_data/Miller_2025/delay_z_data/')
    axonal_kernels_precomputed_folder = MODULE_BASE.joinpath('reference_data/Miller_2025/axon_kernels_precomputed/precomputed_kernels_50_0p1')
    SIM_DURATION = 100  # ms
    SIM_TIME_STEP = 0.025  # ms
    THETA = 30.0
    GRADIENT = 0.0
    INTENSITY = 225.0
    DENSITY = True
    SCALING_FACTOR = 1
    CONTINUOUS_INPUT = True
    SINGLE_COMPARTMENT = False
    DELTA_SPIKE_RATE = True
    SODIUM_BLOCKED = True
    STIM_OFFSET = 1  # ms
    fraction_AMPA = 0.5

    if SINGLE_COMPARTMENT:
        neuron_cell = h.Section(name='soma')
        # h.psection()  # info about section
        neuron_cell.insert('pas')
        input_segment = neuron_cell(0.5)
        soma_segment = input_segment
        cell_label = 'Single Compartment'
    else:
        morph_ids = CellType.cell_ids['L5_TTPC2_cADpyr']
        morph_id = morph_ids[7]
        neuron_cell = L5_TTPC2_cADpyr.from_id(morphology_id=morph_id)
        neuron_cell.load()
        input_segment = neuron_cell.apic[1](0.5)
        soma_segment = neuron_cell.soma[0](0.5)
        cell_label = 'L5 PC'

        if SODIUM_BLOCKED:
            for sec in neuron_cell.all:
                for seg in list(sec):
                    # Soma Sodium Channel Conductance to Zero
                    seg_dir = dir(seg)
                    for mech_name in seg_dir:
                        if mech_name.find('Na') == 0:
                            mechanism = seg.__getattribute__(mech_name)
                            for parameter in dir(mechanism):
                                if (parameter.find('gNa') != -1) and (parameter.find('bar') != -1):
                                    seg.__setattr__(f'{parameter}_{mech_name}', 0)
            sodium_label = 'Firing Suppressed'
        else:
            sodium_label = 'Firing Allowed'
    # Compute Conductivity for AMPA Synapse
    # Rusu et al 2014 Values
    # AMPA
    AMPA_t_rise = 0.2  # ms - rise time
    AMPA_t_fall = 1.7  # ms - fall time
    AMPA_reversal = 0  # mV - reversal potential
    AMPA_peak_conductance = 0.1  # mu Ohm^-1
    # NMDA
    NMDA_t_rise = 2  # ms - rise
    NMDA_t_fall = 26  # ms - fall
    NMDA_reversal = 0  # mV - reversal
    NMDA_peak_conductance = 0.03  # mu Ohm^-1

    time_conductance = np.arange(0, SIM_DURATION + SIM_TIME_STEP, SIM_TIME_STEP)
    conductance_AMPA = - np.exp(- time_conductance / AMPA_t_rise) + np.exp(- time_conductance / AMPA_t_fall)
    conductance_AMPA *= AMPA_peak_conductance / np.max(conductance_AMPA)
    conductance_NMDA = - np.exp(- time_conductance / NMDA_t_rise) + np.exp(- time_conductance / NMDA_t_fall)
    conductance_NMDA *= NMDA_peak_conductance / np.max(conductance_NMDA)

    # Collect Spike Rate Timeseries
    kernel_l23_ex = get_axonal_delay_kernel(source_layer_name='L23', delay_z_folder=axonal_delay_data_folder,
                                            precomputed_kernels_folder=axonal_kernels_precomputed_folder, theta=THETA,
                                            gradient=GRADIENT, intensity=INTENSITY, z_step=100, t_step=0.1,
                                            smooth_z_step=100, smooth_t_step=SIM_TIME_STEP, density=DENSITY,
                                            compute_fresh=True)

    crosscut_index = 5
    crosscut_z = kernel_l23_ex.z_layer[crosscut_index]
    z_crosscut = kernel_l23_ex.z_layer[crosscut_index]
    kernel_spike_rate = kernel_l23_ex.layer_kernel[crosscut_index, :] * abs(np.diff(kernel_l23_ex.z_layer)[0])
    spike_rate_delta = np.zeros_like(kernel_spike_rate)
    spike_rate_delta[np.where(time_conductance == 1.0)[0][0]] = 1
    area = np.trapz(spike_rate_delta, dx=SIM_TIME_STEP)
    spike_rate_delta = spike_rate_delta / area
    if DELTA_SPIKE_RATE:
        spike_rate_timeseries = spike_rate_delta
        rate_label = 'Delta Function'
    else:
        spike_rate_timeseries = kernel_spike_rate
        rate_label = 'L23 Input Crosscut'
    convolved_conductance_AMPA = np.convolve(conductance_AMPA, spike_rate_timeseries, mode='full') * SIM_TIME_STEP
    convolved_conductance_AMPA = convolved_conductance_AMPA[:len(time_conductance)] * fraction_AMPA
    input_conductance_vector_AMPA = h.Vector(convolved_conductance_AMPA)
    input_conductance_time_vector = h.Vector(time_conductance)
    convolved_conductance_NMDA = np.convolve(conductance_NMDA, spike_rate_timeseries, mode='full') * SIM_TIME_STEP
    convolved_conductance_NMDA = convolved_conductance_NMDA[:len(time_conductance)] * (1 - fraction_AMPA)
    input_conductance_vector_NMDA = h.Vector(convolved_conductance_NMDA)

    # Example Playing into IClamp Point Process
    # clamp = h.IClamp(soma(0.5))
    # clamp.delay = 0
    # clamp.dur = 1e9
    # clamp.amp = 0
    # current_input_vector = h.Vector(conductance_AMPA * 100)
    # current_input_vector.play(clamp._ref_amp, conductance_time_vector, 1)

    # Using Custom SynCur Point Process
    count = 0
    if CONTINUOUS_INPUT:
        syn = h.SynCur(input_segment)
        syn.weight = 1
        syn.e = AMPA_reversal
        input_conductance_vector_AMPA.play(syn._ref_g, input_conductance_time_vector, 1)
        syn_2 = h.SynCur(input_segment)
        syn_2.weight = 1
        syn_2.e = NMDA_reversal
        input_conductance_vector_NMDA.play(syn_2._ref_g, input_conductance_time_vector, 1)
        input_label = 'Continuous Input'
    else:
        syn = h.Exp2Syn(input_segment)
        syn.tau1 = AMPA_t_rise
        syn.tau2 = AMPA_t_fall
        syn.e = AMPA_reversal
        stim = h.NetStim()
        stim.number = 1
        stim.start = STIM_OFFSET
        netcon = h.NetCon(stim, syn)
        netcon.weight[0] = fraction_AMPA * AMPA_peak_conductance
        netcon.delay = 0
        syn_2 = h.Exp2Syn(input_segment)
        syn_2.tau1 = NMDA_t_rise
        syn_2.tau2 = NMDA_t_fall
        syn_2.e = NMDA_reversal
        stim_2 = h.NetStim()
        stim_2.number = 1
        stim_2.start = STIM_OFFSET
        netcon_2 = h.NetCon(stim_2, syn_2)
        netcon_2.weight[0] = (1 - fraction_AMPA) * NMDA_peak_conductance
        netcon_2.delay = 0
        input_label = 'Spike Input'

    # Synatax for Density Mechanism
    # soma(0.5).synCur.weight = 1
    # soma(0.5).synCur.e = AMPA_reversal
    # conductance_vector.play(soma(0.5)._ref_g_synCur, conductance_time_vector, 1)

    # Set up Watch Vector for Time and Membrane Potential
    time_sim = h.Vector().record(h._ref_t)
    measured_potential = h.Vector().record(input_segment._ref_v)
    soma_potential = h.Vector().record(soma_segment._ref_v)
    measured_input_current_AMPA = h.Vector().record(syn._ref_i)
    measured_conductance_AMPA = h.Vector().record(syn._ref_g)
    measured_input_current_NMDA = h.Vector().record(syn_2._ref_i)
    measured_conductance_NMDA = h.Vector().record(syn_2._ref_g)
    # ############ Simulate #################
    # set initial state
    init_handler = h.FInitializeHandler(2, post_finitialize)
    sim_temperature = 37  # degrees C
    initial_voltage = -70.0  # mV

    h.celsius = sim_temperature
    h.dt = SIM_TIME_STEP
    h.tstop = SIM_DURATION
    h.finitialize(initial_voltage)

    # simulate
    print('Starting Simulation')
    h.continuerun(SIM_DURATION)
    print('Simulation Complete')

    # ############# Postprocess and Plot #####
    time_sim = np.array(time_sim)
    potential_plt = np.array(measured_potential)
    soma_potential_plt = np.array(soma_potential)
    input_current_AMPA_plt = np.array(measured_input_current_AMPA)
    conductance_AMPA_plt = np.array(measured_conductance_AMPA)
    input_current_NMDA_plt = np.array(measured_input_current_NMDA)
    conductance_NMDA_plt = np.array(measured_conductance_NMDA)

    fig, ax = plt.subplots(4, 1, sharex='all', figsize=(7, 8))

    if not SINGLE_COMPARTMENT:
        cell_label = cell_label + ' ' + sodium_label
    if CONTINUOUS_INPUT:
        fig.suptitle(f'{cell_label} - {input_label}: {rate_label}')
    else:
        fig.suptitle(f'{cell_label} - {input_label}')

    ax[0].plot(time_sim, potential_plt)
    ax[0].grid(axis='x')
    ax[0].set_ylabel('V (mV)')
    ax[0].set_ylim(np.min(potential_plt) - 5, max(np.max(soma_potential_plt), np.max(potential_plt)) + 5)
    ax[0].set_title('Input Membrane Potential')

    ax[1].plot(time_sim, soma_potential_plt)
    ax[1].grid(axis='x')
    ax[1].set_ylabel('V (mV)')
    ax[1].set_ylim(np.min(soma_potential_plt) - 5, max(np.max(soma_potential_plt), np.max(potential_plt)) + 5)
    ax[1].set_title('Soma Membrane Potential')

    ax[2].plot(time_sim, input_current_AMPA_plt, label='AMPA')
    ax[2].plot(time_sim, input_current_NMDA_plt, label='NMDA')
    ax[2].plot(time_sim, input_current_NMDA_plt + input_current_AMPA_plt, '--', label='Net Current')
    ax[2].grid(axis='x')
    ax[2].legend()
    ax[2].set_ylabel('I (nA)')
    ax[2].set_title('Synaptic Transmembrane Current')

    ax[3].plot(time_sim, conductance_AMPA_plt, label='AMPA')
    ax[3].plot(time_sim, conductance_NMDA_plt, label='NMDA')
    ax[3].grid(axis='x')
    ax[3].legend()
    ax[3].set_xlabel('Time (ms)')
    ax[3].set_ylabel('g ($\mu$S)')
    ax[3].set_title('Synaptic Conductance')
    # ax[3].set_xlim((-1, 20))
    fig.tight_layout()

