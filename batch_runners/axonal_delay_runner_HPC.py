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

batch_start = time.perf_counter()

# Get command line arguments
########################################################################################################################
parser = argparse.ArgumentParser(description='Runs Axonal Delay Evaluations')
parser.add_argument('-p', '--parameter_path', help='Path to *.hdf5-file containing the parameters', required=True)
parser.add_argument('-sv', '--save_path', help='Path to folder where results hdf5 files are to be saved', required=True)
parser.add_argument('-wv', '--waveform', help='String containing waveform "monophasic" or "biphasic"', required=True)
parser.add_argument('-sk', '--skip', action=argparse.BooleanOptionalAction, help='If provided skips all simulation calculations')
args = parser.parse_args()

# Universal/Global Parameters
param_path = pathlib.Path(args.parameter_path)
save_path = pathlib.Path(args.save_path)
morph_file_name = param_path.stem
name_split = morph_file_name.split('_')
layer_name = name_split[0]
subtype_name = name_split[1]
electric_type_name = name_split[2]
cell_name = f"{layer_name}_{subtype_name}_{electric_type_name}"
waveform_str = args.waveform
skip = args.skip
run_single_proc = False
# Process Parameters
num_procs = int(os.environ['SLURM_NPROCS'])
process_idx = int(os.environ['SLURM_PROCID'])


def print_file(message):
    file_path = os.path.join(save_path, f'printout_{process_idx}')
    with open(file_path, 'a+') as f_print:
        f_print.write(f"{message}\n")
    print(message)


def compute_chunk_bounds(num_indices: int, num_chunks: int) -> list:
    """
        Returns the bounds of chunks (starting at 0) in an index list consistent with range(0, num_indices),
        split into num_chunks roughly even sized chunks. Left over empty chunks are empty. Indices are inclusive.
        :param num_indices: number of indices to split into chunks
        :param num_chunks: number of chunks
        :return: list of (chunk_start, chunk_stop) for each chunk, inclusive indices. Empty sublist if chunk is empty
        """
    assert num_chunks <= num_indices, "Number of chunks should equal or exceed number of indices"

    avg = int(np.floor(num_indices / num_chunks))  # avg number of indices per chunk
    chunk_ids = np.arange(num_chunks)
    last_nonempty_chunk = np.where(chunk_ids * avg < num_indices)[0][-1]
    num_occupied_chunks = last_nonempty_chunk + 1

    chunk_lengths = np.full(shape=(num_chunks,), fill_value=avg, dtype=int)
    chunk_lengths[last_nonempty_chunk + 1:] = 0

    # Calculate chunk lengths
    overflow = num_indices - np.sum(chunk_lengths)
    if overflow > 0:  # requires redistribution of excess indices in last chunk
        idx = 0
        while overflow > 0:
            chunk_lengths[idx] += 1
            overflow -= 1
            idx += 1
            if idx > last_nonempty_chunk:
                idx = 0
    else:  # last chunk has a lower # of indices
        chunk_lengths[last_nonempty_chunk] = abs(overflow)
    # Determine bounds from chunk lenths
    chunk_bounds = []
    for chunk in range(num_chunks):
        if chunk <= last_nonempty_chunk:
            lower_bound = np.sum(chunk_lengths[:chunk])
            upper_bound = lower_bound + chunk_lengths[chunk] - 1
            chunk_bounds.append([lower_bound, upper_bound])
        else:
            chunk_bounds.append([])

    return chunk_bounds


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
    chunk_bounds = compute_chunk_bounds(full_param_set_length, num_procs)[process_idx]
    if len(chunk_bounds) != 0:
        delay_z_file = os.path.join(save_path, f"delay_z_{str(process_idx)}_chunk_{str(chunk_bounds[0])}_{str(chunk_bounds[1])}.hdf5")
        raw_results_file = os.path.join(save_path, f"raw_results_{str(process_idx)}_chunk_{str(chunk_bounds[0])}_{str(chunk_bounds[1])}.hdf5")
        tms_wave_types = {
            'monophasic': WaveformType.MONOPHASIC,
            'biphasic': WaveformType.BIPHASIC
        }
        cell_ids = {
            'L23_PC_cADpyr': ["1", "2", "3", "4", "5", "341", "342", "344", "349", "361", "363", "369", "376", "378",
                              "380", "385", "414", "499", "587", "3760", "3761", "3764", "3784", "3811"],
            'L4_LBC_dNAC': ["1", "2", "3", "4", "5", "7899", "7982", "8437", "8567", "8581", "8644", "8702", "8813",
                            "8953", "8971", "9008", "9066", "9114", "9149", "9329", "9341", "9504", "9582", "9806",
                            "9816", "9865", "9947", "9963", "9975", "10423", "10505", "10535", "10796", "10951",
                            "10996"],
            'L4_LBC_cSTUT': ["1", "2", "3", "4", "5", "7977", "7980", "8005", "8339", "8566", "9023", "9391", "9419",
                             "9834", "9900", "10038", "10424", "10590", "10650", "10748", "10793", "10804", "11170",
                             "11328", "11663", "11904", "11915", "11920", "11972", "39288", "39694", "40094", "40096",
                             "40442", "40475"],
            'L4_LBC_cACint': ["1", "2", "3", "4", "5", "7922", "8138", "8792", "8875", "8994", "9000", "9223", "9261",
                              "9477", "10085", "10097", "10208", "11390", "11658", "11773", "11932", "11980", "12299",
                              "12378", "12440", "39523", "40016", "40323", "40452", "40828", "41554", "41710", "41805",
                              "41945", "42081"],
            'L4_NBC_cACint': ["1", "2", "3", "4", "5", "7979", "8169", "8607", "10197", "10679", "11007", "11578",
                              "11845", "11928", "39454", "39487", "41370", "41851", "42164", "42188", "43423", "43840",
                              "70839", "72121", "72510", "74063", "75051", "75055", "102338", "102609", "103307",
                              "103472",
                              "103508", "104946", "105343"],
            'L4_NBC_dNAC': ["1", "2", "3", "4", "5", "7873", "7895", "7949", "8066", "8068", "8465", "8599", "8837",
                            "8989", "9211", "9349", "9398", "9518", "9558", "9875", "9922", "10147", "10313", "10442",
                            "10867", "11111", "11178", "11185", "11368", "11401", "11501", "11533", "12039", "12080",
                            "12137"],
            'L4_SBC_bNAC': ["1", "2", "3", "4", "5", "7990", "8050", "8060", "8241", "8717", "9061", "9231", "9318",
                            "9552", "9845", "10047", "10690", "10824", "10840", "10970", "11047", "11119", "11553",
                            "11669", "11736", "12280", "12324", "39623", "39809", "39819", "39892", "40456", "40662",
                            "40674", "40747"],
            'L4_SBC_cACint': ["1", "2", "3", "4", "5", "8072", "8583", "8751", "8822", "8881", "9091", "9284", "9426",
                              "9594", "9680", "9707", "9744", "10168", "10189", "10502", "10525", "10852", "10950",
                              "11621", "11826", "39265", "39866", "39961", "40008", "40558", "42046", "42385", "42941",
                              "43373", "43391"],
            'L5_TTPC2_cADpyr': ["1", "2", "3", "4", "5", "12521", "12530", "12532", "12535", "12539", "12543", "12545",
                                "12550", "12566", "12570", "12571", "12577", "12578", "12581", "12582", "12583",
                                "12587",
                                "12588", "12593", "12594", "12603", "12604", "12612", "12619", "12764"]
        }

        cell_constructors = {
            'L23_PC_cADpyr': L23_PC_cADpyr,
            'L4_LBC_dNAC': L4_LBC_dNAC,
            'L4_LBC_cSTUT': L4_LBC_cSTUT,
            'L4_LBC_cACint': L4_LBC_cACint,
            'L4_NBC_cACint': L4_NBC_cACint,
            'L4_NBC_dNAC': L4_NBC_dNAC,
            'L4_SBC_bNAC': L4_SBC_bNAC,
            'L4_SBC_cACint': L4_SBC_cACint,
            'L5_TTPC2_cADpyr': L5_TTPC2_cADpyr
        }

        wave_type = tms_wave_types[waveform_str]
        layer_cell_ids = cell_ids[cell_name]
        layer_cell_constructor = cell_constructors[cell_name]
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
                print_file(f"Already Complete Process: {str(process_idx).zfill(len(str(num_procs)))}/{num_procs - 1} indices {str(chunk_bounds[0]).zfill(max_digits)}_{str(chunk_bounds[1]).zfill(max_digits)}")
                continue_run = False
            else:
                print_file(f"Resuming Process: {str(process_idx).zfill(len(str(num_procs)))}/{num_procs - 1} simulating at {idx_start} for parameter indices {str(chunk_bounds[0]).zfill(max_digits)}_{str(chunk_bounds[1]).zfill(max_digits)}")
                print_file(f"Skip = {skip}")
                continue_run = True
        else:
            idx_start = chunk_bounds[0]
            print_file(f"Starting process: {str(process_idx).zfill(len(str(num_procs)))}/{num_procs - 1} simulating parameter indices {str(chunk_bounds[0]).zfill(max_digits)}:{str(chunk_bounds[1]).zfill(max_digits)}")
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
            print_file(f"Simulating Axonal Delay for Layer: {layer_name} with {waveform_str} tms waveform.")
            print_file(
                f"Expected Total time for {num_procs} procs is conservatively: {datetime.timedelta(seconds=expected_time)}")
            print_file(f"Individual Proc Mode: {run_single_proc}")
            print_file("#################################################################################")
            all_bounds = compute_chunk_bounds(full_param_set_length, num_procs)
            for i in range(num_procs):
                if len(all_bounds[i]) == 0:
                    print_file(f"Process: {i}, NO INDICES")
                else:
                    print_file(f"Process: {i}, indices [{all_bounds[i][0]}:{all_bounds[i][1]}]")
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
                        electric_type_results = {subtype_name: {electric_type_name: [[2, 2, 2], [3, 3, 3]]}}
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
                            f_delay.create_dataset(name=f'delay_{global_idx}', data=delay, dtype='float32')
                            f_delay.create_dataset(name=f'z_{global_idx}', data=z, dtype='float32')
                        f_delay.attrs.create(name='last_index', data=global_idx, dtype='int64')

                    with h5py.File(raw_results_file, 'a') as f_raw:
                        if len(delay) != 0:
                            subgroup = f_raw.create_group(name=f'results_{global_idx}')
                            subgroup.create_dataset(name=f'params', data=all_simulation_parameters, compression=8,
                                                    dtype='f')
                            subgroup.create_dataset(name=f'raw_compartment_ids', data=raw_compartment_ids,
                                                    compression=8, dtype='i')
                            subgroup.create_dataset(name=f'raw_delays', data=raw_delays, compression=8, dtype='f')
                            subgroup.create_dataset(name='raw_zs', data=raw_zs, compression=8, dtype='f')
                        f_raw.attrs.create(name='last_index', data=global_idx, dtype='int64')

                    t_end = time.perf_counter()
                    t_delta = t_end - t_start
                    time_remaining = datetime.timedelta(seconds=(t_delta * (chunk_bounds[1] - global_idx)))
                    print_file(f'Saved Process {process_idx}, idx {str(global_idx).zfill(max_digits)}/{str(chunk_bounds[1]).zfill(max_digits)} | {progress_percent:.0f}% Interation Time: {t_delta:.1f}s, Estimated Time Remaining: {time_remaining}')
                else:
                    print_file('########################################################################################################')
                    print_file(f'####### BATCH ELAPSED TIME: {datetime.timedelta(seconds=(t_start - batch_start))} Running Halted #######')
                    break
                if global_idx == chunk_bounds[1]:
                    print_file(f"Process {process_idx} Complete: fin :)")
    else:
        print_file(f"Process {process_idx} has NO indices, nothing is run: fin :)")


if __name__ == '__main__':
    main()
