
import argparse
import os
import pathlib
import time
import datetime
import h5py
import numpy as np
from enum import Enum
from neuron_interface.neurosim.simulation.dendritic_delay_simulation import DendriticDelaySimulation
from postprocessing_modules.axonal_delay_kernel import get_axonal_delay_kernel
from helper_functions.cell_type_lib import CellType
from helper_functions.load_cell_distributions import compute_distribution
from neuron_interface.neurosim.cells import L5_TTPC2_cADpyr
from helper_functions.plotting_lib import block_print, enable_print
from helper_functions import __file__
MODULE_BASE = pathlib.Path(__file__).parent.parent


if __name__ == '__main__':
    # ###################################### Retrieve run arguments ####################################################
    parser = argparse.ArgumentParser(description='Runs Dendritic Delay Evaluations')
    parser.add_argument('-sv', '--save_path', help='Path to folder where results hdf5 files are to be saved',
                        required=True)
    parser.add_argument('-pk', '--precom_kernel_path', help='Path to folder where results hdf5 files are to be saved',
                        required=True)
    parser.add_argument('-sd', '--seed', help='Integer seed used to generate parameters', required=True, type=int)
    parser.add_argument('-ns', '--num_samples', help='Integer number of sample parameter sets to simulate', required=True, type=int)
    parser.add_argument('-id', '--proc_id',
                        help='Process/Chunk index, optionally provided to only simulate one process',
                        required=False,
                        type=int,
                        default=None)
    parser.add_argument('-np', '--num_procs',
                        help='Number of processes/chunks, optionally provided to only simulate one process',
                        required=False,
                        type=int,
                        default=None)
    args = parser.parse_args()

    # User provided parameters
    save_path = pathlib.Path(args.save_path)
    precomputed_kernels_path = pathlib.Path(args.precom_kernel_path)
    single_proc_number = args.proc_id
    seed = args.seed
    num_samples = args.num_samples
    total_procs = args.num_procs
    if single_proc_number is None:
        num_procs = int(os.environ['SLURM_NPROCS'])
        process_idx = int(os.environ['SLURM_PROCID'])
    elif (single_proc_number is not None) and (total_procs is not None):
        num_procs = total_procs
        process_idx = single_proc_number
    else:
        raise ValueError('--proc_id and --num_procs must both be provided to simulate a single process, or neither is' +
                         'provided to simulate all processes.')

    # Hard Coded Parameters
    SIM_TIME_STEP = 0.01  # ms
    SIM_DURATION = 100  # ms
    Z_BIN_SIZE = 50

    NA_SUPPRESSED = True
    INCLUDE_AXON = False  # inclusion of axon current in soma current calculation
    # ################################ Process Print Out ###############################################################

    def print_file(message):
        file_path = save_path.joinpath(f'printout_{process_idx}')
        with open(file_path, 'a+') as f_print:
            f_print.write(f"{message}\n")
        print(message)

    # ################################ Generate Parameters #############################################################
    field_param_file = MODULE_BASE.joinpath('reference_data/axonal_delay_reduced_biphasic_params.hdf5')
    with h5py.File(field_param_file, 'r') as f_par:
        field_params = f_par['params'][:].astype(int)
    theta_vals = np.array(sorted(set(field_params[:, 0])))
    gradient_vals = np.array(sorted(set(field_params[:, 1])))
    intensity_vals = np.array(sorted(set(field_params[:, 2])))
    field_params = field_params[field_params[:, 1] == 0, :]
    fraction_nmda_range = [0.25, 0.75]
    fraction_gaba_a_range = [0.9, 1.0]
    fraction_ex_range = [0.2, 0.8]

    cell_ids = CellType.cell_ids['L5_TTPC2_cADpyr']
    SAVE = True
    l5_cell_density, z_layer_centers, _, l5_mean_depth, _, _ = compute_distribution('L5', 0.025, micrometers=True)
    index_upper_bound = np.where(np.invert(np.isclose(l5_cell_density, 0, atol=1e-6)))[0][0]
    index_lower_bound = np.where(np.invert(np.isclose(l5_cell_density, 0, atol=1e-6)))[0][-1]
    upper_bound = z_layer_centers[index_upper_bound]
    lower_bound = z_layer_centers[index_lower_bound]
    idx_step = int((index_lower_bound - index_upper_bound) / 23)
    z_sample_indices = np.arange(index_upper_bound, index_lower_bound + idx_step, idx_step)
    z_sample_depths = np.round(z_layer_centers[z_sample_indices])
    density_sample_values = l5_cell_density[z_sample_indices]

    base_generator = np.random.default_rng(seed=seed)
    param_columns = ['theta', 'gradient', 'intensity', 'fraction_nmda', 'fraction_gaba_a', 'fraction_ex', 'cell_id',
                     'soma_depth']
    total_num_simulations = num_samples * len(cell_ids) * len(z_sample_depths)
    sim_params = np.zeros(shape=(total_num_simulations, len(param_columns)), dtype=float)
    sim_param_free_idx = np.zeros(shape=(total_num_simulations, ), dtype=int)
    free_params = np.zeros(shape=(num_samples, 6), dtype=float)

    row = 0
    for sample_counter in range(num_samples):
        theta = base_generator.choice(theta_vals)
        gradient = base_generator.choice(gradient_vals)
        intensity = base_generator.choice(intensity_vals)
        fraction_nmda = base_generator.uniform(low=fraction_nmda_range[0], high=fraction_nmda_range[1])
        fraction_gaba_a = base_generator.uniform(low=fraction_gaba_a_range[0], high=fraction_gaba_a_range[1])
        fraction_ex = base_generator.uniform(low=fraction_ex_range[0], high=fraction_ex_range[1])
        free_params[sample_counter, :] = np.array(
            [theta, gradient, intensity, fraction_nmda, fraction_gaba_a, fraction_ex]
        )
        for cell_id in cell_ids:
            for cell_depth in z_sample_depths:
                sim_params[row, :] = np.array(
                    [theta, gradient, intensity, fraction_nmda, fraction_gaba_a, fraction_ex, cell_id, cell_depth])
                sim_param_free_idx[row] = sample_counter
                row += 1
    col_enum = Enum('Columns', [col_label for col_label in param_columns], start=0)
    parameter_file = save_path.joinpath(f'dendritic_delay_avg_continuous_gpc_params_seed_{seed}_num_{num_samples}.hdf5')
    if (not parameter_file.is_file()) and (process_idx == 0):
        with h5py.File(parameter_file, 'w') as f_param:
            all_pars = f_param.create_dataset(name='params', data=sim_params, dtype='float32', compression=8, chunks=(1, sim_params.shape[1]))
            all_pars.attrs.create(name='columns', data=param_columns)
            free_pars = f_param.create_dataset(name='free_params', data=free_params, dtype='float32', compression=8, chunks=(1, free_params.shape[1]))
            free_pars.attrs.create(name='columns', data=param_columns[:6])
            f_param.create_dataset(name='free_index_mask', data=sim_param_free_idx, dtype='int32')
    # ################################ Compute Indices for this Chunk/Process ##########################################
    all_chunk_bounds = np.vstack(
        [[chunk_list[0], chunk_list[-1]] for chunk_list in np.array_split(np.arange(sim_params.shape[0]), num_procs)]
    )
    # chunk_lengths = [chunk_bound[-1] - chunk_bound[0] + 1 for chunk_bound in idx_chunk_bounds]
    index_bounds = all_chunk_bounds[process_idx]  # index bounds for this chunk [start, stop] inclusive
    if process_idx == 0:
        print_file(f'##### {num_procs} Processes, {sim_params.shape[0]} Parameter Sets #####')
        for ind, chunk in enumerate(all_chunk_bounds):
            print_file(f'Proccess {ind} | Idx {chunk[0]} - {chunk[1]}')
    chunk_file = save_path.joinpath(
        f"dend_delay_{str(process_idx)}_chunk_{str(index_bounds[0])}_{str(index_bounds[1])}.hdf5"
    )
    # ############################### Determine Starting Index/Continuation ############################################
    if chunk_file.is_file():
        with h5py.File(chunk_file, 'a') as f:
            start_index = int(f.attrs.__getitem__('last_index')) + 1

        if start_index > index_bounds[-1]:
            print_file(f'Process {process_idx} COMPLETE {index_bounds[0]} - {index_bounds[1]}')
        else:
            print_file(f'RESUMING Process {process_idx} from {start_index} | {index_bounds[0]} - {index_bounds[1]}')
    else:
        start_index = index_bounds[0]
        print_file(f'STARTING Process {process_idx} | {index_bounds[0]} - {index_bounds[1]}')

    perf_counter_buffer = []
    for param_index in range(start_index, index_bounds[1] + 1):
        t_start = time.perf_counter()
        # Retrieve parameters
        param_set = sim_params[param_index, :]  # 8 parameters per simulation
        theta = int(param_set[col_enum['theta'].value])
        gradient = int(param_set[col_enum['gradient'].value])
        intensity = int(param_set[col_enum['intensity'].value])
        cell_id = int(param_set[col_enum['cell_id'].value])
        soma_depth = param_set[col_enum['soma_depth'].value]
        fraction_nmda = param_set[col_enum['fraction_nmda'].value]
        fraction_gaba_a = param_set[col_enum['fraction_gaba_a'].value]
        fraction_ex = param_set[col_enum['fraction_ex'].value]
        # load cell
        neuron_cell = L5_TTPC2_cADpyr.from_id(cell_id)
        neuron_cell.load()
        # set up simulation
        block_print()
        kernel_handler_ex = get_axonal_delay_kernel(
            source_layer_name='L23',
            delay_z_folder=pathlib.Path(''),
            precomputed_kernels_folder=precomputed_kernels_path,
            theta=theta, gradient=gradient,
            intensity=intensity,
            smooth_z_step=Z_BIN_SIZE,
            smooth_t_step=SIM_TIME_STEP
        )
        kernel_handler_inh = get_axonal_delay_kernel(
            source_layer_name='L23_inh',
            delay_z_folder=pathlib.Path(''),
            precomputed_kernels_folder=precomputed_kernels_path,
            theta=theta,
            gradient=gradient,
            intensity=intensity,
            smooth_z_step=Z_BIN_SIZE,
            smooth_t_step=SIM_TIME_STEP
        )

        sim = DendriticDelaySimulation(cell=neuron_cell)
        sim.define_synaptic_inputs_continuous(
            spike_density_ex=kernel_handler_ex.layer_kernel,
            spike_density_in=kernel_handler_inh.layer_kernel,
            z_bin_edges=kernel_handler_ex.z_layer_edges,
            z_bin_centers=kernel_handler_ex.z_layer,
            soma_depth=soma_depth,
            fraction_nmda=fraction_nmda,
            fraction_gaba_a=fraction_gaba_a,
            fraction_ex=fraction_ex,
            sim_duration=SIM_DURATION,
            sim_time_step=SIM_TIME_STEP
        )

        _, _, t, _, current = sim.simulate(
            include_axon_current=INCLUDE_AXON
        )
        enable_print()
        neuron_cell.unload()
        t_stop = time.perf_counter()

        with h5py.File(chunk_file, 'a') as f_chunk:
            f_chunk.create_dataset(name=f'current_{param_index}', data=current)
            if param_index == index_bounds[0]:
                f_chunk.create_dataset(name='t', data=t)
                f_chunk.attrs.create(name='simulation_duration', data=SIM_DURATION, dtype='float32')
                f_chunk.attrs.create(name='simulation_time_step', data=SIM_TIME_STEP, dtype='float32')
            f_chunk.attrs.create(name='last_index', data=param_index, dtype='int32')
        run_duration = t_stop - t_start
        perf_counter_buffer.append(run_duration)
        if len(perf_counter_buffer) > 100:
            perf_counter_buffer.pop(0)
        avg_duration = sum(perf_counter_buffer) / len(perf_counter_buffer)
        seconds_remaining = round(avg_duration * (index_bounds[1] - param_index))
        percent_complete = (param_index - index_bounds[0]) / (index_bounds[1] - index_bounds[0])
        print_file(f'Process {process_idx} | Index {param_index}/{index_bounds[1]} Complete in {t_stop - t_start:.1f} s' +
                   f' | {percent_complete:.2%} | Proc Complete in {str(datetime.timedelta(seconds=seconds_remaining))}')

    print_file(f'Process {process_idx} COMPLETE | {index_bounds[0]} - {index_bounds[1]} fin :)')

