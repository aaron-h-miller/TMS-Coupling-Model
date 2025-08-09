
import numpy as np
import scipy
import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import pathlib
from postprocessing_modules.axonal_delay_kernel import AxonalDelayKernel
from helper_functions.load_cell_distributions import compute_distribution
from helper_functions.cell_type_lib import get_cell_count
from helper_functions import __file__
MODULE_BASE = pathlib.Path(__file__).parent.parent

param_path = MODULE_BASE.joinpath('reference_data/axonal_delay_reduced_biphasic_params.hdf5')
data_file_list = [MODULE_BASE.joinpath('reference_data/Miller_2025/delay_z_data/L23_PC_cADpyr_merged_delay_z.hdf5')]

layers = [[0.1, 0.29], [0.29, 0.37], [0.37, 0.47], [0.47, 0.8], [0.8, 1.0]]
column_depth = 2700  # um https://doi.org/10.1101/2024.03.05.583549 citing https://doi.org/10.1371/journal.pbio.3000678
lyr_idx = 0
z_bin_width = 100
t_bin_width = 0.1
smooth_z_step = 1  # 1 um
smooth_t_step = 0.025  # 0.025 ms
# BANDWIDTH = 0.15
THETA = 30.0
GRAD = 0.0
AMP = 225.0
# smooth_t_step = 0.025
# BANDWIDTH = 0.15
kernel_handler = AxonalDelayKernel()
kernel_handler.load_data(parameter_path=param_path, data_path_list=data_file_list, theta=0.0, gradient=0.0, intensity=200, cell_distribution_type='L23')
kernel_handler.calculate_histogram(z_bin_width, t_bin_width, density=True, scaling_factor=get_cell_count('L23')/get_cell_count('L5'))

cell_hist = kernel_handler.cell_histogram
z_cell = kernel_handler.z_bin_centers
hist_volume = np.trapz(np.trapz(cell_hist, dx=z_bin_width, axis=0), dx=t_bin_width)
crosscut = cell_hist[:, 6].copy()

l23_density, z_layer, _, _, _, _ = compute_distribution(
    cell_class='L23',
    z_step=z_bin_width,
    micrometers=True,
)
# z_layer = np.arange(column_depth, 0 - smooth_z_step, -smooth_z_step)
# mean = column_depth * (layers[lyr_idx][0] + layers[lyr_idx][1]) / 2
# std = column_depth * (layers[lyr_idx][1] - layers[lyr_idx][0]) / 10
# cell_density = scipy.stats.norm.pdf(z_layer, mean, std)
# cell_density /= np.sum(cell_density)
# z_cell = np.arange(-1400, 1400 + smooth_z_step, smooth_z_step)
# win = np.zeros_like(z_cell, dtype=float)
# win[int(len(win)/2) - 70:int(len(win)/2)] = 0.5
# win[int(len(win)/2):int(len(win)/2) + 26] = 2.0
# win_applied = win[::-1]

sig = l23_density
z_sig = z_layer

sig = np.zeros_like(l23_density)
sig[int(len(z_layer)/2) - 5] = 1

win = crosscut
z_win = z_cell
# win[19:21] = 17
# win[20] = 4.7
# win[21] = 4.7
# win = win[15:25]
# z_win = z_win[15:25]

# win = kernel_handler.layer_kernel[:, 10]
# filtered = scipy.signal.convolve(sig, win, mode='same') * smooth_z_step
# filtered = scipy.signal.convolve(sig, win_applied, mode='same')
filtered_correlate = scipy.signal.correlate(sig, win, mode='same')

# filtered *= np.trapz(win, dx=abs(np.diff(z_cell)[0])) / np.trapz(filtered, dx=abs(np.diff(z_layer)[0]))

fig, (ax_orig, ax_win, ax_filt) = plt.subplots(1, 3, figsize=(9, 5))

# sum_conv = np.sum(filtered)
sum_corr = np.sum(filtered_correlate)
sum_win = np.sum(win)
sum_sig = np.sum(sig)

# area_conv = np.trapz(filtered, dx=abs(np.diff(z_sig)[0])) * t_bin_width
area_corr = np.trapz(filtered_correlate, dx=abs(np.diff(z_sig)[0])) * t_bin_width
area_win = np.trapz(win, dx=abs(np.diff(z_win)[0])) * t_bin_width
area_sig = np.trapz(sig, dx=abs(np.diff(z_sig)[0])) * t_bin_width

print(f"Sum Filtered: {sum_corr}")
print(f"Sum Window: {sum_win}")
print(f"Sum Signal: {sum_sig}")
print(f"Sum Window / Sum Filtered: {sum_corr/sum_win}")

print(f"Area Filtered: {area_corr}")
print(f"Area Window: {area_win}")
print(f"Area Signal: {area_sig}")
print(f"Area Window / Sum Filtered: {area_corr/area_win}")

ax_orig.plot(sig, z_layer)
ax_orig.set_ylabel('Cortical Depth (um)')
# ax_orig.set_ylim((z_sig[0], z_sig[-1]))
ax_orig.set_title(f'Signal | Area: {area_sig:.2f}')
ax_orig.margins(0.1, 0)
ax_win.plot(win, z_win)
ax_win.set_title(f'Window | Area: {area_win:.2f}')
ax_win.set_ylabel('z w.r.t Soma (um)')
# ax_win.set_ylim((z_win[0], z_win[-1]))
ax_win.margins(0.1, 0)
ax_filt.plot(filtered_correlate, z_sig, label='Convolution (reversed filter)')
ax_filt.set_title(f'Filtered | Area: {area_corr:.2f}')
ax_filt.margins(0.1, 0)
ax_filt.set_ylabel('Cotical Depth (um)')
# ax_filt.set_ylim((z_sig[0], z_sig[-1]))
fig.suptitle("scipy.convolve(signal, window)")
fig.tight_layout()

# if np.all(np.isclose(filtered - filtered_correlate, 0)):
#     print('Convolve with negated filter and correlate are identical')

if not matplotlib.is_interactive():
    plt.show()
