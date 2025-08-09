import pathlib
import time

import numpy as np
import h5py

from postprocessing_modules.axonal_delay_kernel import get_axonal_delay_kernel
from neuron_interface.neurosim.simulation.dendritic_delay_simulation import DendriticDelaySimulation
from helper_functions import __file__
from helper_functions.cell_type_lib import CellType
from helper_functions.plotting_lib import block_print, enable_print
from helper_functions.load_cell_distributions import compute_distribution
from neuron_interface.neurosim.cells import L5_TTPC2_cADpyr

MODULE_BASE = pathlib.Path(__file__).parent.parent


def dendritic_delay_simulate(sim_time_step, sim_duration, theta, gradient, intensity, fraction_NMDA, fraction_GABA_a, fraction_excitatory):

    Z_BIN_SIZE = 100
    NA_SUPPRESSED = True
    DENSITY = True
    INCLUDE_AXON = False  # inclusion of axon current in soma current calculation
    na_label = 'noSodium' if NA_SUPPRESSED else 'Sodium'
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
                                            precomputed_kernels_folder=axonal_kernels_precomputed_folder, theta=theta,
                                            gradient=gradient, intensity=intensity, z_step=100, t_step=0.1,
                                            smooth_z_step=Z_BIN_SIZE, smooth_t_step=sim_time_step, density=DENSITY,
                                            compute_fresh=False)
    kernel_l23_inh = get_axonal_delay_kernel(source_layer_name='L23_inh', delay_z_folder=axonal_delay_data_folder,
                                             precomputed_kernels_folder=axonal_kernels_precomputed_folder, theta=theta,
                                             gradient=gradient, intensity=intensity, z_step=100, t_step=0.1,
                                             smooth_z_step=Z_BIN_SIZE, smooth_t_step=sim_time_step, density=DENSITY,
                                             compute_fresh=False)
    sim.define_synaptic_inputs_continuous(
        spike_density_ex=kernel_l23_ex.layer_kernel,
        spike_density_in=kernel_l23_inh.layer_kernel,
        z_bin_edges=kernel_l23_ex.z_layer_edges,
        z_bin_centers=kernel_l23_ex.z_smooth_layer_centers,
        soma_depth=SOMA_DEPTH,
        fraction_nmda=fraction_NMDA,
        fraction_gaba_a=fraction_GABA_a,
        fraction_ex=fraction_excitatory,
        sim_duration=sim_duration,
        sim_time_step=sim_time_step
    )
    seg_syn_array, _, t, voltage, current = sim.simulate(include_axon_current=INCLUDE_AXON)

    return t, current, sim


if __name__ == '__main__':
    param_path = MODULE_BASE.joinpath('reference_data/axonal_delay_reduced_biphasic_params.hdf5')
    axonal_delay_data_folder = MODULE_BASE.joinpath('reference_data/Miller_2025/delay_z_data/')
    axonal_kernels_precomputed_folder = MODULE_BASE.joinpath('reference_data/Miller_2025/axon_kernels_precomputed/precomputed_kernels_50_0p1/')
    save_folder = pathlib.Path('/path/to/folder/')

    # ############################################## SET PARAMETERS ####################################################
    # sim_time_step, sim_duration, theta, gradient, intensity, fraction_NMDA, fraction_GABA_a, fraction_excitatory

    THETA = 30.0
    GRADIENT = 10.0
    INTENSITY = 350
    FRACTION_NMDA = 0.1
    FRACTION_GABAa = 0.9
    FRACTION_EX = 0.34
    nmda_label = f'{FRACTION_NMDA}'.replace('.', 'p')
    gabaa_label = f'{FRACTION_GABAa}'.replace('.', 'p')
    ex_label = f'{FRACTION_EX}'.replace('.', 'p')
    paramlabel = f'{int(THETA)}_{int(GRADIENT)}_{int(INTENSITY)}_{nmda_label}_{gabaa_label}_{ex_label}'

    SAVE = True

    num_samples = 1
    tstep_max = 1
    tstep_min = 0.01
    time_steps = np.linspace(tstep_max, tstep_min, 19)  # ms
    tstep_label = f'{tstep_max}_{tstep_min}'.replace('.','p')

    save_file = save_folder.joinpath(f'{paramlabel}_tstep_opt_{tstep_label}.hdf5')

    time_series = []
    dend_currents = []
    sim_lengths = []
    time_steps_list = []
    if save_file.is_file():
        # Recover Data From File
        with h5py.File(save_file, 'r') as f:
            for i in range(len(time_steps)):
                time_series.append(f[f'time_{i}'][:])
                dend_currents.append(f[f'current_{i}'][:])
                sim_lengths.append(f[f'sim_length_{i}'][()])
                time_steps_list.append(f[f'tstep_{i}'][()])
    else:
        # Compute Data Fresh
        sims = []
        for t_idx, tstep in enumerate(time_steps):
            print(f'{t_idx + 1}/{len(time_steps)}, Time Step: {tstep} ms')
            generator = np.random.default_rng(seed=111)
            block_print()
            tstart = time.process_time()
            time_axis, dend_current, simulation = dendritic_delay_simulate(
                sim_time_step=tstep,
                sim_duration=100,
                theta=THETA,
                gradient=GRADIENT,
                intensity=INTENSITY,
                fraction_NMDA=FRACTION_NMDA,
                fraction_GABA_a=FRACTION_GABAa,
                fraction_excitatory=FRACTION_EX
            )
            tstop = time.process_time()
            enable_print()
            dend_currents.append(dend_current)
            sim_lengths.append(tstop - tstart)
            time_series.append(time_axis)
            sims.append(simulation)
            # dend_currents[-1] = np.vstack(dend_currents[-1])
        # dend_currents = np.vstack(dend_currents)
        if SAVE:
            with h5py.File(save_file, 'a') as f:
                f.attrs.create(name='number', data=len(time_steps), dtype='int32')
                for index, (cur, time_axis, tstep) in enumerate(zip(dend_currents, time_series, time_steps)):
                    print(f'{index}: {tstep} ms')
                    f.create_dataset(name=f'time_{index}', data=time_axis, dtype='float32')
                    f.create_dataset(name=f'current_{index}', data=cur, dtype='float32')
                    f.create_dataset(name=f'tstep_{index}', data=tstep, dtype='float32')
                    f.create_dataset(name=f'sim_length_{index}', data=sim_lengths[index], dtype='float32')

    # matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.size'] = 16

    plt.figure()
    for cur, time_axis, tstep in zip(dend_currents, time_series, time_steps):
        if tstep < 0.5:
            plt.plot(time_axis, cur * 1e9, color='tab:gray', label=f'{tstep} ms')
        else:
            plt.plot(time_axis, cur * 1e9)
    plt.plot(time_axis, )
    plt.xlabel('t (ms)')
    plt.ylabel('I (nA)')
    plt.xlim(0, 40)
    # plt.legend(fontsize=10)
    plt.tight_layout()

    if SAVE:
        plt.savefig(save_folder.joinpath(f'{paramlabel}_tstep_opt_{tstep_label}.png'), dpi=300, transparent=True)


    def calc_mape(actual, predic):
        return np.mean(np.abs((actual - predic) / actual))


    def calc_correlation(actual, predic):
        a_diff = actual - np.mean(actual)
        p_diff = predic - np.mean(predic)
        numerator = np.sum(a_diff * p_diff)
        denominator = np.sqrt(np.sum(a_diff ** 2)) * np.sqrt(np.sum(p_diff ** 2))
        return numerator / denominator

    mape = []
    interp_currents = []
    finest_time = time_series[-1]
    for cur, tm in zip(dend_currents, time_series):
        cur_interp = np.interp(finest_time, tm, cur)
        mape.append(calc_correlation(dend_currents[-1], cur_interp))
        interp_currents.append(cur_interp)

    fig, ax = plt.subplots(2, 1, sharex='all')
    ax[0].plot(time_steps, mape, 'ko-')
    ax[0].set_xlim(time_steps.max(), time_steps.min())
    ax[0].set_ylabel('Correlation')
    ax[0].set_title('Correlation w.r.t 0.01 ms Time Step')
    ax[1].plot(time_steps, np.array(sim_lengths) / 60, 'ro-')
    ax[1].set_xlim(time_steps.max() + 0.01, time_steps.min() - 0.01)
    ax[1].set_xticks(np.arange(1.0, 0.01 - 0.2, -0.2))
    ax[1].set_xlabel('Time Step Size (ms)')
    ax[1].set_ylabel('Simulation Time (min)')
    fig.tight_layout()

    # plt.figure()
    # for sim, time_axis, tstep in zip(sims, time_series, time_steps):
    #     plt.plot(np.array(sim.simulation_time_vector), sim.conductance_inputs[60, 0], label=f'{tstep}')
    # plt.xlabel('t (ms)')
    # plt.ylabel('AMPA G ($\\mu$S)')
    # plt.legend()



