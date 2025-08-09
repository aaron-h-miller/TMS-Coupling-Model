import datetime
import os
import time
import h5py
import signal
import argparse
import numpy as np
import psutil
import pathlib
# from memory_profiler import profile
from neuron_interface.neurosim.cells import L23_PC_cADpyr, L4_LBC_dNAC, L4_LBC_cSTUT, L4_LBC_cACint, L4_NBC_dNAC,\
                                                L4_NBC_cACint, L4_SBC_bNAC, L4_SBC_cACint, L5_TTPC2_cADpyr
from neuron_interface.neurosim.simulation.simulation import WaveformType
from neuron_interface.runner.axonal_delay_runner import AxonalDelayRunner
from helper_functions.cell_type_lib import CellType

batch_start = time.perf_counter()

# Get command line arguments
########################################################################################################################
parser = argparse.ArgumentParser(description='Runs Axonal Delay Evaluations')
parser.add_argument('-p', '--parameter_path', help='Path to *.hdf5-file containing the parameters', required=True)
parser.add_argument('-cl', '--cell_type', help='Cell Type (Layer_subtype_electrictype)', required=True)
parser.add_argument('-sv', '--save_path', help='Path to folder where results hdf5 files are to be saved', required=True)
parser.add_argument('-pr', '--process_idx', help='Single process idx to run', required=True)
parser.add_argument('-tot', '--total_procs', help='Total number of processes in batch', required=True)
parser.add_argument('-wv', '--waveform', help='String containing waveform "monophasic" or "biphasic"', required=True)
parser.add_argument('-sk', '--skip', action=argparse.BooleanOptionalAction, help='If provided skips all simulation calculations')
args = parser.parse_args()

process_idx = args.process_idx
num_procs = args.total_procs
param_path = args.parameter_path
save_path = args.save_path
waveform_str = args.waveform
skip = args.skip
cell_name = args.cell_type


def print_file(message):
    file_path = os.path.join(save_path, f'printout_{process_idx}')
    with open(file_path, 'a+') as f_print:
        f_print.write(f"{message}\n")
    print(message)


def calculate_chunk_bounds(chunk_index: int, maximum: int, num: int) -> list:
    """
    Returns the bounds of chunk _chunk_index_ (starting at 0) in an index list from 0 up and including _maximum_,
    split into _num_ roughly even sized chunks. Left over empty chunks are empty.
    :param chunk_index: index of chunk to get
    :param maximum: maximum index
    :param num: number of chunks
    :return: list of (chunk_start, chunk_stop), inclusive. Empty if chunk is empty
    """
    assert maximum >= 0, "maximum must be >= 0"
    assert num > 0, "num must be >= 1"
    assert num < maximum, "maximum must be >= num"
    assert chunk_index < num, "chunk_index must be < num"
    length = maximum + 1
    avg = int(np.round(length / num))
    chunk_ids = np.arange(num)
    last_nonempty_chunk = np.where(chunk_ids * avg <= maximum)[0][-1]
    if chunk_index == last_nonempty_chunk:      # if this index is the final nonempty chunk
        return [(chunk_index * avg), maximum]
    elif chunk_index > last_nonempty_chunk:     # if lower bound is above maximum (past last non-empty chunk)
        return []
    else:                                       # normal chunk with avg number of elements
        return [(chunk_index * avg), (chunk_index * avg) + avg - 1]


def signal_term_handler(sig, frame):
    print_file('####################################')
    print_file('########### SIGTERM ################')


signal.signal(signal.SIGTERM, signal_term_handler)


# @profile
def main():
    print_file(f'Num Procs: {num_procs}')

    # This Process Specific Parameters
    os.makedirs(save_path, exist_ok=True)

    # read length of full dataset
    with h5py.File(param_path, 'r') as f_param:
        full_param_set_length = f_param['params'].shape[0]

    max_digits = len(str(full_param_set_length))
    # Calculate bounds for this process (can return empty or where first and last are the same number)
    chunk_bounds = calculate_chunk_bounds(process_idx, full_param_set_length - 1, num_procs)
    if len(chunk_bounds) != 0:
        delay_z_file = os.path.join(save_path, f"delay_z_{str(process_idx)}_chunk_{str(chunk_bounds[0])}_{str(chunk_bounds[1])}.hdf5")
        raw_results_file = os.path.join(save_path, f"raw_results_{str(process_idx)}_chunk_{str(chunk_bounds[0])}_{str(chunk_bounds[1])}.hdf5")
        tms_wave_types = {
            'monophasic': WaveformType.MONOPHASIC,
            'biphasic': WaveformType.BIPHASIC
        }

        wave_type = tms_wave_types[waveform_str]
        layer_cell_ids = CellType.cell_ids[cell_name]
        layer_cell_constructor = CellType.cell_constructors[cell_name]
        # layer_cell_type_morphology_ratios = CellType.cell_type_morphology_ratios[cell_name]
        SIMULATION_DURATION = 2.0
        SIMULATION_TIME_STEP = 0.005
        STIMULATION_DELAY = 0.0
        PHI_COUNT = 60
        # Checking if simulations are resumed
        ################################################################################################################
        if os.path.exists(delay_z_file):
            with h5py.File(delay_z_file, "a") as f:
                idx_start = int(f.attrs.__getitem__('last_index')) + 1
                f.attrs.create(name='simulation_duration', data=SIMULATION_DURATION)
                f.attrs.create(name='simulation_time_step', data=SIMULATION_TIME_STEP)
                f.attrs.create(name='stimulation_delay', data=STIMULATION_DELAY)
                f.attrs.create(name='phi_count', data=PHI_COUNT)

            if idx_start > chunk_bounds[1]:
                print_file(f"Already Complete Process: {str(process_idx).zfill(len(str(num_procs)))}/{num_procs} indices {str(chunk_bounds[0]).zfill(max_digits)}_{str(chunk_bounds[1]).zfill(max_digits)}")
                continue_run = False
            else:
                print_file(f"Resuming Process: {str(process_idx).zfill(len(str(num_procs)))}/{num_procs} simulating at {idx_start} for parameter indices {str(chunk_bounds[0]).zfill(max_digits)}_{str(chunk_bounds[1]).zfill(max_digits)}")
                print_file(f"Skip = {skip}")
                continue_run = True
        else:
            idx_start = chunk_bounds[0]
            print_file(f"Starting process: {str(process_idx).zfill(len(str(num_procs)))}/{num_procs} simulating parameter indices {str(chunk_bounds[0]).zfill(max_digits)}:{str(chunk_bounds[1]).zfill(max_digits)}")
            print_file(f"Skip = {skip}")
            continue_run = True

            column_labels = 'theta, gradient, intensity, phi, cell_id'
            with h5py.File(raw_results_file, 'a') as f_raw:
                f_raw.attrs.create(name='columns', data=column_labels)

        # Looping over parameter sets if resume/start
        ################################################################################################################
        # Display Chunk Structure
        if process_idx == 0:
            expected_time = 15.0 * 60 * full_param_set_length / num_procs
            print_file(f"Simulating Axonal Delay for: {cell_name} with {waveform_str} tms waveform.")
            print_file(
                f"Expected Total time for {num_procs} procs is conservatively: {datetime.timedelta(seconds=expected_time)}")
            print_file(f"Individual Proc Mode")
            print_file("#################################################################################")
            for i in range(num_procs):
                bounds_i = calculate_chunk_bounds(i, full_param_set_length - 1, num_procs)
                if len(bounds_i) == 0:
                    print_file(f"Process: {i}, NO INDICES")
                else:
                    print_file(f"Process: {i}, indices [{bounds_i[0]}:{bounds_i[1]}]")
            print_file("#################################################################################")

        if continue_run:
            t_delta = 60 * 60  # 1 hour
            for global_idx in range(idx_start, chunk_bounds[1] + 1):
                t_start = time.perf_counter()
                # Assert batch time is < 23:00:00 (or last iteration time)
                if (t_start - batch_start) < (48 * 60 * 60 - t_delta):
                    with h5py.File(param_path, 'r') as f_param:
                        parameter_row = f_param['params'][global_idx]
                    theta = parameter_row[0]
                    gradient = parameter_row[1]
                    intensity = parameter_row[2]

                    time_sim = datetime.datetime.now()
                    time_sim = time_sim.strftime('%d/%m/%Y/ %H:%M:%S')

                    progress_percent = 100 * (global_idx - chunk_bounds[0]) / ((chunk_bounds[1] + 1) - chunk_bounds[0])
                    memory_usage = psutil.virtual_memory()
                    print_file(f"Process {process_idx} Index {global_idx} Memory Usage: {memory_usage.percent}% | {time_sim}")

                    if skip:
                        all_simulation_results = [[[1, 1, 1, 1], [2, 2, 2, 2]], [[3, 3, 3, 3], [4, 4, 4, 4]]]
                        all_simulation_parameters = np.array([[1, 2, 3, 4, 5, 'L23_ex'], [11, 22, 33, 44, 55, 'L23_ex']])
                    else:
                        runner = AxonalDelayRunner(
                            theta=theta,
                            relative_mag_change_per_mm=gradient,
                            intensity=intensity,
                            cell_constructor=layer_cell_constructor,
                            cell_ids=layer_cell_ids,
                            phi_count=PHI_COUNT,
                            waveform_type=wave_type,
                            stimulation_delay=STIMULATION_DELAY,
                            simulation_duration=SIMULATION_DURATION,
                            simulation_time_step=SIMULATION_TIME_STEP
                        )

                        all_simulation_results, all_simulation_parameters = runner.run(processes=1)

                    delay = []
                    z = []
                    for cell_delays, cell_zs, _ in all_simulation_results:
                        delay = delay + cell_delays
                        z = z + cell_zs
                    # Build -1 padded raw results matrices for compartment ids and action potentials
                    raw_result_lengths = [len(raw_results[0]) for raw_results in all_simulation_results]
                    raw_max_length = max(raw_result_lengths)
                    raw_compartment_ids = np.full((len(all_simulation_results), raw_max_length), -1, dtype=float)
                    raw_zs = np.full((len(all_simulation_results), raw_max_length), -1, dtype=float)
                    raw_delays = np.full((len(all_simulation_results), raw_max_length), -1, dtype=float)
                    for row in range(len(all_simulation_results)):
                        raw_compartment_ids[row, :len(all_simulation_results[row][2])] = all_simulation_results[row][2]
                        raw_zs[row, :len(all_simulation_results[row][1])] = all_simulation_results[row][1]
                        raw_delays[row, :len(all_simulation_results[row][0])] = all_simulation_results[row][0]

                    print_file(f'Saving Process {process_idx}, idx {str(global_idx).zfill(max_digits)}/{str(chunk_bounds[1]).zfill(max_digits)} | Memory Usage: {memory_usage.percent}%, ')
                    with h5py.File(delay_z_file, 'a') as f_delay:
                        if len(delay) != 0:
                            f_delay.create_dataset(name=f'delay_{global_idx}', data=delay, dtype='f')
                            f_delay.create_dataset(name=f'z_{global_idx}', data=z, dtype='f')
                        f_delay.attrs.create(name='last_index', data=global_idx, dtype='i')

                    with h5py.File(raw_results_file, 'a') as f_raw:
                        if len(delay) != 0:
                            subgroup = f_raw.create_group(name=f'results_{global_idx}')
                            subgroup.create_dataset(name=f'params', data=all_simulation_parameters, compression=8,
                                                    dtype='f')
                            subgroup.create_dataset(name=f'raw_compartment_ids', data=raw_compartment_ids,
                                                    compression=8, dtype='i')
                            subgroup.create_dataset(name=f'raw_delays', data=raw_delays, compression=8, dtype='f')
                            subgroup.create_dataset(name='raw_zs', data=raw_zs, compression=8, dtype='f')
                        f_raw.attrs.create(name='last_index', data=global_idx, dtype='i')

                    t_end = time.perf_counter()
                    t_delta = t_end - t_start
                    time_remaining = datetime.timedelta(seconds=(t_delta * (chunk_bounds[1] - global_idx)))
                    print_file(f'Saved Process {process_idx}, idx {str(global_idx).zfill(max_digits)}/{str(chunk_bounds[1]).zfill(max_digits)} | {progress_percent:.0f}% Interation Time: {t_delta:.1f}s, Estimated Time Remaining: {time_remaining}')
                else:
                    print_file('########################################################################################################')
                    print_file(f'####### BATCH ELAPSED TIME: {datetime.timedelta(seconds=(t_start - batch_start))} Running Halted #######')
                    break

        print_file(f"Process {process_idx} Complete: fin :)")
    else:
        print_file(f"Process {process_idx} has NO indices, nothing is run: fin :)")


if __name__ == '__main__':
    main()
