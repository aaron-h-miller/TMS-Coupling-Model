import pathlib
import h5py
import numpy as np
import argparse
from numpy.typing import NDArray


def all_chunks_complete(chunk_folder: pathlib.Path) -> tuple[bool, NDArray[bool]]:
    """
    Checks if all chunks are complete in the given folder. Chunk files should be hdf5 with the naming scheme:
    '*_<first_index>_<last_index>.hdf5'
    Each file should contain the attribute 'last_index' to indicate the last saved index.
    Returns and True if all chunks are completed, with an additional list of bools indicating each chunk
    :param chunk_folder: path containing chunk files
    :return: bool, list(bool): True if all chunks are completed, a list of bools indicating if each chunk is complete
    """
    glob_list = list(chunk_folder.glob('*chunk_[0-9]*_[0-9]*.hdf5'))
    if len(glob_list) == 0:
        raise AssertionError(
            f"Chunk folder {chunk_folder} does not contain any files of the pattern '*_<int>_<int>.hdf5'"
        )
    else:
        all_chunk_paths = sorted(glob_list, key=lambda pth: int(pth.stem.split('_')[-2]))
        chunk_index_bounds = np.vstack(
            [[int(pth.stem.split('_')[-2]), int(pth.stem.split('_')[-1])] for pth in all_chunk_paths]
        )
        chunks_complete = []
        for chunk_path, chunk_bounds in zip(all_chunk_paths, chunk_index_bounds):
            with h5py.File(chunk_path, 'r') as f_chunk:
                last_index = int(f_chunk.attrs.__getitem__('last_index'))
            chunks_complete.append(last_index >= chunk_bounds[1])

        return all(chunks_complete), np.array(chunks_complete)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-pth', '--path', required=True, help='Path to folder containing chunk files')
    args = parser.parse_args()
    chunk_folder = pathlib.Path(args.path)
    result, chunk_results = all_chunks_complete(chunk_folder)

    if result:
        print(f'All Chunks are Completed {len(chunk_results)} Total Chunks)')
    else:
        unfinished_chunks = np.where(np.invert(chunk_results))[0]
        print(f'NOT Complete. Remaining Chunks are {unfinished_chunks}')
