
import pathlib
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from helper_functions.get_dendritic_current import get_dendritic_current
from helper_functions import __file__
MODULE_BASE = pathlib.Path(__file__).parent.parent


if __name__ == '__main__':
    data_folder = MODULE_BASE.joinpath('reference_data/Miller_2025/dendritic_current/')
    THETA = 0.0
    GRADIENT = 0.0
    INTENSITY = 300.0
    time_axis, dendritic_current = get_dendritic_current(
        theta=THETA,
        gradient=GRADIENT,
        intensity=INTENSITY,
        data_folder=data_folder
    )

    plt.rcParams['font.size'] = 16
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot()
    ax.plot(time_axis, dendritic_current)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Dendritic Current (nA)')
    ax.set_title(f'$\\theta$ = {THETA}$^\circ$, $\\Delta|E|$ = {GRADIENT} %/mm, $|E|$ = {INTENSITY} V/m', pad=10)
    fig.tight_layout()
