from typing import Type

import numpy as np
from neuron_interface.neurosim.cells import NeuronCell
from neuron import h


def collect_lines(neuron_cell: Type[NeuronCell], coordinates):
    assert neuron_cell.loaded, "Cell must be loaded"
    lines_collected = []
    idx = 0
    for nrn_section in neuron_cell.all:
        prev = np.array([nrn_section.x3d(0), nrn_section.z3d(0)])
        for segment in list(nrn_section)[:-1]:
            mid = [(coordinates[idx][0] + coordinates[idx + 1][0]) / 2, (coordinates[idx][2] + coordinates[idx + 1][2]) / 2]
            # plt.plot([prev[0], mid[0]], [prev[1], mid[1]], c=cmap(recording_slice[i]))
            lines_collected.append(np.array([(prev[0], prev[1]), (mid[0], mid[1])]))
            prev = mid
            idx += 1
        end = np.array([nrn_section.x3d(nrn_section.n3d() - 1), nrn_section.z3d(nrn_section.n3d() - 1)])
        # plt.plot([prev[0], end[0]], [prev[1], end[1]], c=cmap(recording_slice[i]))
        lines_collected.append(np.array([(prev[0], prev[1]), (end[0], end[1])]))
        idx += 1
    return lines_collected


def collect_lines_3d(neuron_cell: Type[NeuronCell], coordinates):
    assert neuron_cell.loaded, "Cell must be loaded"
    lines_collected = []
    idx = 0
    for nrn_section in neuron_cell.all:
        prev = np.array([nrn_section.x3d(0), nrn_section.y3d(0), nrn_section.z3d(0)])
        for segment in list(nrn_section)[:-1]:
            mid = [(coordinates[idx][0] + coordinates[idx + 1][0]) / 2, (coordinates[idx][1] + coordinates[idx + 1][1]) / 2, (coordinates[idx][2] + coordinates[idx + 1][2]) / 2]
            # plt.plot([prev[0], mid[0]], [prev[1], mid[1]], c=cmap(recording_slice[i]))
            lines_collected.append(np.array([(prev[0], prev[1]), (mid[0], mid[1])]))
            prev = mid
            idx += 1
        end = np.array([nrn_section.x3d(nrn_section.n3d() - 1), nrn_section.y3d(nrn_section.n3d() - 1), nrn_section.z3d(nrn_section.n3d() - 1)])
        # plt.plot([prev[0], end[0]], [prev[1], end[1]], c=cmap(recording_slice[i]))
        lines_collected.append(np.array([(prev[0], prev[1]), (end[0], end[1])]))
        idx += 1
    return lines_collected


def collect_lines_section(neuron_cell: Type[NeuronCell], neuron_cell_section, coordinates, negate=False):
    assert neuron_cell.loaded, "Cell must be loaded"
    lines_collected = []
    idx = 0
    for nrn_section in neuron_cell.all:
        if (nrn_section in neuron_cell_section) & (not negate):
            include_section = True
        elif (nrn_section not in neuron_cell_section) & negate:
            include_section = True
        else:
            include_section = False
        prev = np.array([nrn_section.x3d(0), nrn_section.z3d(0)])
        for segment in list(nrn_section)[:-1]:
            if include_section:
                mid = [(coordinates[idx][0] + coordinates[idx + 1][0]) / 2, (coordinates[idx][2] + coordinates[idx + 1][2]) / 2]
                # plt.plot([prev[0], mid[0]], [prev[1], mid[1]], c=cmap(recording_slice[i]))
                lines_collected.append(np.array([(prev[0], prev[1]), (mid[0], mid[1])]))
                prev = mid
            idx += 1
        if include_section:
            end = np.array([nrn_section.x3d(nrn_section.n3d() - 1), nrn_section.z3d(nrn_section.n3d() - 1)])
            # plt.plot([prev[0], end[0]], [prev[1], end[1]], c=cmap(recording_slice[i]))
            lines_collected.append(np.array([(prev[0], prev[1]), (end[0], end[1])]))
        idx += 1
    return lines_collected


def segment_sublist_mask(neuron_cell, section_sublist):
    assert neuron_cell.loaded
    counter = 0
    dendrite_segment_mask = []
    for sec in neuron_cell.all:
        for seg in list(sec):
            if sec in section_sublist:
                dendrite_segment_mask.append(True)
            else:
                dendrite_segment_mask.append(False)
            counter += 1
    return np.array(dendrite_segment_mask)


def interpolate_segment_sigma_z(neuron_cell, grid_z, grid_sigma) -> list:
    assert neuron_cell.loaded, "Cell is not loaded"

    segNumSyn = []  #
    for sec in neuron_cell.all:
        segNumSyn.append([])
        for seg in list(sec):
            if (sec in neuron_cell.dend) or (sec in neuron_cell.apic):
                z = seg.z_xtra - neuron_cell.soma[0](0.5).z_xtra
                distZ = [abs(gz - z) for gz in grid_z]
                jys = np.array(distZ).argsort()[:2]
                j1, j2 = min(jys), max(jys)
                z1, z2 = grid_z[j1], grid_z[j2]
                sigma_z1 = grid_sigma[j1]
                sigma_z2 = grid_sigma[j2]

                if z1 == z2:
                    # print("ERROR in closest grid points: ", sec, z1, z2)
                    sigma = 0.0
                else:
                    # linear interpolation, see http://en.wikipedia.org/wiki/Linear_interpolation
                    sigma = (sigma_z1 * (z2 - z) + sigma_z2 * (z - z1)) / (z2 - z1)

                numSyn = sigma * sec.L / sec.nseg  # return num syns
                segNumSyn[-1].append(numSyn)
            else:
                segNumSyn[-1].append(0)

    return segNumSyn


def syns_per_segment(neuron_cell, grid_z, syn_dens, total_num_syns):
    assert neuron_cell.loaded, "Cell is not loaded"

    seg_num_syns = interpolate_segment_sigma_z(neuron_cell=neuron_cell, grid_z=grid_z, grid_sigma=syn_dens)
    totSyn = sum([sum(segsyns) for segsyns in seg_num_syns])  # summed density
    scaleNumSyn = float(total_num_syns) / float(totSyn) if totSyn > 0 else 0.0
    diffList = []
    for i, sec in enumerate(seg_num_syns):
        for j, x in enumerate(sec):
            orig = float(x * scaleNumSyn)
            scaled = int(round(x * scaleNumSyn))
            seg_num_syns[i][j] = scaled
            diff = orig - scaled
            if diff > 0:
                diffList.append([diff, i, j])
            else:
                diffList.append([0, i, j])

    totSynRescale = sum([sum(segsyns) for segsyns in seg_num_syns])

    if totSynRescale < total_num_syns:
        extraSyns = total_num_syns - totSynRescale
        diffList = sorted(diffList, key=lambda l: l[0], reverse=True)
        for extra_idx in range(min(extraSyns, len(diffList))):
            sec = diffList[extra_idx][1]
            seg = diffList[extra_idx][2]
            seg_num_syns[sec][seg] += 1

    return seg_num_syns


def post_finitialize():
    """ Initialization methode to unsure a steady state before the actual simulation is started.
    """
    temp_dt = h.dt

    h.t = -1e11
    h.dt = 1e9
    # h.t = -1e3
    # h.dt = 0.1
    print('Starting Steady State Sim')
    while h.t < -h.dt:  # IF t >= 0 then might trigger events which depend on h.t
        h.fadvance()
    h.dt = temp_dt
    h.t = 0
    h.fcurrent()
    h.frecord_init()
    print('Steady State Sim Complete')