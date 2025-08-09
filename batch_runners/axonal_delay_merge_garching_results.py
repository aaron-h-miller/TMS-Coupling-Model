# This is the garching script merges raw results and delay/z results from many batches into one file
# padded with -1 for jagged arrays. Run in serial (do not parallelize due to need to open same files simultaneously)
# --
import argparse
import numpy as np
import h5py
import pathlib
from time import perf_counter

# Parse command arguments
parser = argparse.ArgumentParser(description='Runs Axonal Delay Evaluations')
parser.add_argument('-p', '--param_path', help='Path to *.hdf5-file containing the parameters', required=True)
parser.add_argument('-sv', '--save_path', help='Path to folder where two result hdf5 files are to be saved', required=True)
parser.add_argument('-b', '--batch_path', help='Path to *.hdf5-files containing batch results', required=True)
parser.add_argument('-rw', '--merge_raw', action=argparse.BooleanOptionalAction, help='If provided raw results are merged', required=True)
parser.add_argument('-dl', '--merge_delays', action=argparse.BooleanOptionalAction, help='If provided delay/z results are merged', required=True)

args = parser.parse_args()

# Universal/Global Parameters
param_path = pathlib.Path(args.param_path)
cell_type = f"{param_path.stem.split('_')[0]}_{param_path.stem.split('_')[1]}_{param_path.stem.split('_')[2]}"
result_folder = pathlib.Path(args.batch_path)
save_folder = pathlib.Path(args.save_path)
merged_delay_file = save_folder.joinpath(f'{cell_type}_merged_delay_z.hdf5')
merged_raw_file = save_folder.joinpath(f'{cell_type}_merged_raw_results.hdf5')
PROCESS_DELAYS = args.merge_delays
PROCESS_RAW = args.merge_raw
WRITE_THRESHOLD = 50e6  # 15mB


def get_remaining_indices(chunk_bounds, start_idx):
    assert (start_idx >= chunk_bounds[0][0]) and (start_idx <= chunk_bounds[-1][-1]), f"Start index {start_idx} out of bounds [{chunk_bounds[0][0]}, {chunk_bounds[-1][-1]}]"
    # Is the start index within one of the given chunks
    chunk_mask = [(start_idx >= lb) and (start_idx <= up) for lb, up in chunk_bounds]
    if any(chunk_mask):
        # Yes, return the bounds starting from chunk containing start_idx, from start_idx
        start_chunk = np.where(chunk_mask)[0][0]
        bounds_new = chunk_indices[start_chunk:]
        bounds_new[0][0] = start_idx
    else:
        # No, return bounds start from the next chunk with lower bound > start_idx
        chunk_mask = [lb > start_idx for lb, _ in chunk_bounds]
        start_chunk = np.where(chunk_mask)[0][0]
        bounds_new = chunk_indices[start_chunk:]
    return bounds_new, start_chunk


def split_chunks_by_threshold(idx_mask, total_bytes, threshold, height):
    """
    Splits the given index mask into chunks with size no larger than threshold.
    :param idx_mask: NDArray
        Index mask list to split
    :param total_bytes: int
        Total number of bytes in data matrix being processed
    :param threshold: int
        Threshold in bytes that all chunks must be under
    :param height: int
        Height of data matrix being processed

    :returns: tuple(chunked_mask, chunked_raw_bounds)
        chunked_mask: list(NDArray) - List of masks for each chunk
        chunked_raw_bounds: list(list(start, stop)) - List of index bounds [start, stop] starting at 0
    """
    bytes_per_row = int(total_bytes / height)
    rows_per_chunk = int(threshold / bytes_per_row)
    num_complete_chunks = len(idx_mask) // rows_per_chunk
    remainder = len(idx_mask) % rows_per_chunk
    if remainder == len(index_mask):
        # Full dataset in one chunk
        chunked_mask = [index_mask]
        chunked_raw_bounds = [[0, result_height]]
    elif num_complete_chunks == 1:
        # Split chunks evenly if there needs to only be 2 chunks total
        chunked_mask = [index_mask[:int(height / 2)], index_mask[int(height / 2):]]
        chunked_raw_bounds = [[0, int(height / 2)], [height - int(height / 2), height]]
    else:
        # load in chunks of length rows_per_threshold
        chunked_mask = []
        chunked_raw_bounds = []
        for n in range(num_complete_chunks):
            chunk_start = n * rows_per_chunk
            chunk_stop = n * rows_per_chunk + rows_per_chunk
            chunked_mask.append(index_mask[chunk_start:chunk_stop])
            chunked_raw_bounds.append([chunk_start, chunk_stop])
        chunked_mask.append(index_mask[-remainder:])
        chunked_raw_bounds.append([result_height - remainder, result_height])

    return chunked_mask, chunked_raw_bounds


if __name__ == '__main__':
    print(cell_type)
    print(f"Process Delays: {PROCESS_DELAYS}, Process Raw Results: {PROCESS_RAW}")
    print(f"Write Threshold: {WRITE_THRESHOLD:.0E} Bytes")

    with h5py.File(param_path, 'r') as f:
        params = f['params'][:]

    num_params = params.shape[0]
    batch_delay_files = pathlib.Path(result_folder).glob('delay_z_*.hdf5')
    batch_delay_files = sorted(batch_delay_files, key=lambda pth: int(pth.stem.split('_')[2]))
    batch_raw_files = pathlib.Path(result_folder).glob('raw_results_*.hdf5')
    batch_raw_files = sorted(batch_raw_files, key=lambda pth: int(pth.stem.split('_')[2]))
    chunk_indices = [[int(file.stem.split('_')[-2]), int(file.stem.split('_')[-1])] for file in batch_delay_files]
    num_batches = len(batch_delay_files)
    first_batch_num = int(batch_raw_files[0].stem.split('_')[2])
    last_batch_num = int(batch_raw_files[-1].stem.split('_')[2])

    # Collect combined dataset sizes
    max_length_delays = -1
    raw_params_combined_shape = [0, -1]
    raw_aps_combined_shape = [0, -1]
    raw_ids_combined_shape = [0, -1]
    param_id_mask = []
    for file_idx, (raw_file, delay_file) in enumerate(zip(batch_raw_files, batch_delay_files)):
        # Copy all raw result groups from each batch to final files with compression
        if PROCESS_RAW:
            print(f'Processing File {raw_file.name}')
            tstart = perf_counter()
            chunk_num = int(raw_file.stem.split('_')[2])

            with h5py.File(raw_file, 'r') as f_raw:
                for i, group_key in enumerate(f_raw.keys()):
                    key_type = group_key.split('_')[0]
                    # If key is a result which is not already in the output merged raw results file
                    if key_type == 'results':
                        print(f'Raw Param {i}/{len(f_raw.keys()) - 1} | {1 if len(f_raw.keys()) <= 1 else i/(len(f_raw.keys()) - 1):.0%}')
                        param_id = int(group_key.split('_')[1])
                        params_shape = f_raw[group_key]['params'].shape
                        raw_aps_shape = f_raw[group_key][f'raw_aps'].shape
                        raw_ids_shape = f_raw[group_key][f'raw_compartment_ids'].shape

                        raw_params_combined_shape[0] += params_shape[0]
                        raw_params_combined_shape[1] = params_shape[1]
                        [param_id_mask.append(param_id) for _ in range(params_shape[0])]
                        raw_aps_combined_shape[0] += raw_aps_shape[0]
                        raw_ids_combined_shape[0] += raw_ids_shape[0]
                        if raw_aps_shape[1] > raw_aps_combined_shape[1]:
                            raw_aps_combined_shape[1] = raw_aps_shape[1]
                        if raw_ids_shape[1] > raw_ids_combined_shape[1]:
                            raw_ids_combined_shape[1] = raw_ids_shape[1]

            tstop = perf_counter()
            print(f'Complete in {tstop - tstart:.0f}s')

        if PROCESS_DELAYS:
            print(f'Acquiring File {delay_file.name}')
            # Determine maximum length in all files, and thus the final padded rectangular array size
            with h5py.File(delay_file, 'r') as f_delay:
                for i, key in enumerate(f_delay.keys()):
                    key_type, param_id = key.split('_')[0], int(key.split('_')[1])
                    if key_type == 'delay':
                        length = len(f_delay[key])
                        if length > max_length_delays:
                            max_length_delays = length

    param_id_mask = np.array(param_id_mask)

    if PROCESS_RAW:
        print("##### MERGE RAW RESULTS #####")
        # Determine if to continue processing at last_index
        if merged_raw_file.is_file():
            with h5py.File(merged_raw_file, 'r') as f:
                idx_start = int(f.attrs.__getitem__('last_index')) + 1

            if idx_start >= num_params:
                print(f"COMPLETE: Merge operation from already complete for all parameter indices.")
                continue_run = False
            else:
                print(f"CONTINUE: Appending to {merged_raw_file.name} at param index {idx_start}/{num_params}")
                continue_run = True
        else:
            print(f"STARTING: Appending Files to {merged_raw_file.name}")
            continue_run = True
            idx_start = chunk_indices[0][0]
            with h5py.File(merged_raw_file, 'a') as f_raw_merged:
                f_raw_merged.attrs.create(name='last_index', data=idx_start - 1)
                f_raw_merged.attrs.create(name='columns', data="theta, gradient, intensity, phi, cell_id")
                f_raw_merged.create_dataset(
                    name='param_id_mask',
                    data=param_id_mask,
                    dtype='int64',
                    compression=8
                )
                f_raw_merged.create_dataset(
                    name='raw_params',
                    shape=(raw_params_combined_shape[0], raw_params_combined_shape[1]),
                    dtype='float32',
                    fillvalue=-1,
                    compression=8,
                    chunks=(1, raw_params_combined_shape[1])
                )
                f_raw_merged.create_dataset(
                    name='raw_aps',
                    shape=(raw_aps_combined_shape[0], raw_aps_combined_shape[1]),
                    dtype='float32',
                    fillvalue=-1,
                    compression=8,
                    chunks=(1, raw_aps_combined_shape[1])
                )
                f_raw_merged.create_dataset(
                    name='raw_compartment_ids',
                    shape=(raw_ids_combined_shape[0], raw_ids_combined_shape[1]),
                    dtype='float32',
                    fillvalue=-1,
                    compression=8,
                    chunks=(1, raw_ids_combined_shape[1])
                )

        # Continue filling out combined raw files
        if continue_run:
            tstart = perf_counter()

            indices, chunk_num_start = get_remaining_indices(chunk_indices, idx_start)
            file_sublist = batch_raw_files[chunk_num_start:]
            # Loop through remaining chunk files
            for raw_file, (start, stop) in zip(file_sublist, indices):
                print(f'Opening file {raw_file.name}')
                # Loop through remaining indices in this chunk
                for param_id in range(start, stop + 1):
                    print(f'Idx {param_id}/{num_params - 1} | {param_id / (num_params - 1):.0%}')
                    index_mask = np.where(param_id_mask == param_id)[0]
                    with h5py.File(raw_file, 'r') as f_raw, h5py.File(merged_raw_file, 'a') as f_raw_merged:
                        if f'results_{param_id}' in f_raw.keys():
                            result_group = f_raw[f'results_{param_id}']
                            n_bytes = result_group['raw_aps'].nbytes
                            result_height = result_group['raw_aps'].shape[0]
                            mask_chunks, raw_chunk_bounds = split_chunks_by_threshold(index_mask, n_bytes, WRITE_THRESHOLD, result_height)
                            # Write to merged files in given chunks
                            print(f"Chunks: {raw_chunk_bounds}")
                            for mask, raw_bounds in zip(mask_chunks, raw_chunk_bounds):
                                f_raw_merged['raw_params'][mask, :result_group['params'].shape[1]] = result_group['params'][raw_bounds[0]:raw_bounds[1], :]
                                f_raw_merged['raw_aps'][mask, :result_group['raw_aps'].shape[1]] = result_group['raw_aps'][raw_bounds[0]:raw_bounds[1], :]
                                f_raw_merged['raw_compartment_ids'][mask, :result_group['raw_compartment_ids'].shape[1]] = result_group['raw_compartment_ids'][raw_bounds[0]:raw_bounds[1], :]
                            f_raw_merged.attrs.create(name='last_index', data=param_id)
                        else:
                            print(f'No results for index {param_id}')

            tstop = perf_counter()
            print(f"Raw Results Merged in {tstop - tstart:.0f}s")

    if PROCESS_DELAYS:
        print("##### MERGE DELAY RESULTS #####")
        # Determine if to continue processing at last_index
        if merged_delay_file.is_file():
            with h5py.File(merged_delay_file, 'r') as f:
                idx_start = int(f.attrs.__getitem__('last_index')) + 1

            if idx_start >= num_params:
                print(f"COMPLETE: Merge operation from already complete for all parameter indices.")
                continue_run = False
            else:
                print(f"CONTINUE: Appending to {merged_delay_file.name} at param index {idx_start}/{num_params}")
                continue_run = True
        else:
            print(f"STARTING: Appending Files to {merged_delay_file.name}")
            continue_run = True
            idx_start = chunk_indices[0][0]
            with h5py.File(merged_delay_file, 'a') as f_merged:
                f_merged.attrs.create(name='last_index', data=idx_start - 1)
                f_merged.create_dataset(
                    name='delay',
                    shape=(num_params, max_length_delays),
                    dtype='float32',
                    fillvalue=-1,
                    compression=8,
                    chunks=(1, max_length_delays)
                )
                f_merged.create_dataset(
                    name='z',
                    shape=(num_params, max_length_delays),
                    dtype='float32',
                    fillvalue=-1,
                    compression=8,
                    chunks=(1, max_length_delays)
                )

        # Continue filling out combined files
        if continue_run:
            tstart = perf_counter()
            # Determine which files to open based on idx_start
            indices, chunk_num_start = get_remaining_indices(chunk_indices, idx_start)
            file_sublist = batch_delay_files[chunk_num_start:]
            # Loop through remaining chunk files
            for delay_file, (start, stop) in zip(file_sublist, indices):
                print(f'Opening file {delay_file.name}')
                # Loop through remaining indices in this chunk
                for param_id in range(start, stop + 1):
                    print(f'Idx {param_id}/{num_params - 1} | {param_id / (num_params - 1):.0%}')
                    with h5py.File(delay_file, 'r') as f_delay, h5py.File(merged_delay_file, 'a') as f_merged:
                        if f'delay_{param_id}' in f_delay.keys():
                            f_merged['delay'][param_id, :len(f_delay[f'delay_{param_id}'])] = f_delay[f'delay_{param_id}'][:]
                            f_merged['z'][param_id, :len(f_delay[f'z_{param_id}'])] = f_delay[f'z_{param_id}'][:]
                            f_merged.attrs.create(name='last_index', data=param_id)
                        else:
                            print(f'No results for index {param_id}')
            tstop = perf_counter()
            print(f'Append Complete in {tstop - tstart:.1f}s')

    print("fin")
