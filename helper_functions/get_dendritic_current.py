import csv
import pathlib
import matplotlib.pyplot as plt
import numpy as np
NANO_FACTOR = 1e9
from helper_functions import __file__
MODULE_BASE = pathlib.Path(__file__).parent.parent
DEFAULT_DATA_PATH = MODULE_BASE.joinpath('reference_data/Miller_2025/dendritic_current')


def get_dendritic_current(theta: float, gradient: float, intensity: float, data_folder=DEFAULT_DATA_PATH):
    """
    Gets and returns the dendritic current timeseries in nano-Amperes and the time axis in milliseconds for the
    given electric field parameters.

    Parameters
    ----------
    theta: float
        Polar angle in degrees
    gradient: float
        Electric field gradient in % V/m per mm
    intensity: float
        Electric field intensity in V/m
    data_folder: pathlib.Path
        Path to folder containing lookup table parameters, current, and time axis .csv files

    Returns
    -------
    time_axis: NDArray
        Time axis in milliseconds
    dendritic_current: NDArray
        Dendritic current in nanoamperes for the given field parameters

    """
    data_file = data_folder.joinpath('dendritic_delay_current_lookup_table.csv')
    time_axis_file = data_folder.joinpath('dendritic_delay_lookup_table_time_axis.csv')
    parameter_file = data_folder.joinpath('dendritic_delay_lookup_table_params.csv')

    with open(parameter_file, 'r') as f_param:
        reader = csv.reader(f_param, delimiter=',')
        parameter_labels = next(reader)
        nextline = next(reader)
    parameters = np.loadtxt(parameter_file, skiprows=1, delimiter=',', dtype=float)
    field_parameters = parameters[:, :3]
    mask = np.all(np.isclose(field_parameters, np.array([theta, gradient, intensity])), axis=1)

    if not np.any(mask):
        theta_values = sorted(set(parameters[:, 0]))
        gradient_values = sorted(set(parameters[:, 1]))
        intensity_values = sorted(set(parameters[:, 2]))
        raise(AttributeError(f'The field parameter combination [{theta}, {gradient}, {intensity}] can not be found.\n '
                             f'Theta Values: {theta_values}\n '
                             f'Gradient Values: {gradient_values}\n '
                             f'Intensity Values {intensity_values}'))

    parameter_index = np.where(mask)[0][0]

    dendritic_current = np.loadtxt(data_file, skiprows=1, delimiter=',', dtype=float)
    dendritic_current *= NANO_FACTOR

    time_axis = np.loadtxt(time_axis_file, delimiter=',', dtype=float)

    return time_axis, dendritic_current[parameter_index, :]

