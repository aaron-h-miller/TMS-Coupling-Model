from dataclasses import dataclass
from neuron_interface.neurosim.cells import L23_PC_cADpyr, L4_LBC_dNAC, L4_LBC_cSTUT, L4_LBC_cACint, \
                                                      L4_NBC_dNAC, L4_NBC_cACint, L4_SBC_bNAC, L4_SBC_cACint, \
                                                      L5_TTPC2_cADpyr
import pandas as pd
import pathlib
import numpy as np
import pickle as pkl
from helper_functions import __file__
MODULE_BASE = pathlib.Path(__file__).parent.parent


@dataclass
class CellType:
    """A dataclass with cell id lists, constructors, and morphology type ratios"""

    cell_ids = {
        'L23_PC_cADpyr': [1, 2, 3, 4, 5, 341, 342, 344, 349, 361, 363, 369, 376, 378, 380, 385, 414, 499, 587, 3760,
                          3761, 3764, 3784, 3811],
         'L4_LBC_dNAC': [1, 2, 3, 4, 5, 7899, 7982, 8437, 8567, 8581, 8644, 8702, 8813, 8953, 8971, 9008, 9066, 9114,
                         9149, 9329, 9341, 9504, 9582, 9806, 9816, 9865, 9947, 9963, 9975, 10423, 10505, 10535, 10796,
                         10951, 10996],
         'L4_LBC_cSTUT': [1, 2, 3, 4, 5, 7977, 7980, 8005, 8339, 8566, 9023, 9391, 9419, 9834, 9900, 10038, 10424,
                          10590, 10650, 10748, 10793, 10804, 11170, 11328, 11663, 11904, 11915, 11920, 11972, 39288,
                          39694, 40094, 40096, 40442, 40475],
         'L4_LBC_cACint': [1, 2, 3, 4, 5, 7922, 8138, 8792, 8875, 8994, 9000, 9223, 9261, 9477, 10085, 10097, 10208,
                           11390, 11658, 11773, 11932, 11980, 12299, 12378, 12440, 39523, 40016, 40323, 40452, 40828,
                           41554, 41710, 41805, 41945, 42081],
         'L4_NBC_cACint': [1, 2, 3, 4, 5, 7979, 8169, 8607, 10197, 10679, 11007, 11578, 11845, 11928, 39454, 39487,
                           41370, 41851, 42164, 42188, 43423, 43840, 70839, 72121, 72510, 74063, 75051, 75055, 102338,
                           102609, 103307, 103472, 103508, 104946, 105343],
         'L4_NBC_dNAC': [1, 2, 3, 4, 5, 7873, 7895, 7949, 8066, 8068, 8465, 8599, 8837, 8989, 9211, 9349, 9398, 9518,
                         9558, 9875, 9922, 10147, 10313, 10442, 10867, 11111, 11178, 11185, 11368, 11401, 11501, 11533,
                         12039, 12080, 12137],
         'L4_SBC_bNAC': [1, 2, 3, 4, 5, 7990, 8050, 8060, 8241, 8717, 9061, 9231, 9318, 9552, 9845, 10047, 10690, 10824,
                         10840, 10970, 11047, 11119, 11553, 11669, 11736, 12280, 12324, 39623, 39809, 39819, 39892,
                         40456, 40662, 40674, 40747],
         'L4_SBC_cACint': [1, 2, 3, 4, 5, 8072, 8583, 8751, 8822, 8881, 9091, 9284, 9426, 9594, 9680, 9707, 9744, 10168,
                           10189, 10502, 10525, 10852, 10950, 11621, 11826, 39265, 39866, 39961, 40008, 40558, 42046,
                           42385, 42941, 43373, 43391],
         'L5_TTPC2_cADpyr': [1, 2, 3, 4, 5, 12521, 12530, 12532, 12535, 12539, 12543, 12545, 12550, 12566, 12570, 12571,
                             12577, 12578, 12581, 12582, 12583, 12587, 12588, 12593, 12594, 12603, 12604, 12612, 12619,
                             12764]
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

    # Cell frequencies available at https://bbp.epfl.ch/nmc-portal/microcircuit.html
    cell_type_morphology_ratios = {
                'L23': {
                    'PC': {
                        'cADpyr': 1.0
                    }
                },
                'L4': {
                    'LBC': {
                        'dNAC': 46 / (46 + 31 + 22 + 36 + 10 + 21 + 22),
                        'cSTUT': 31 / (46 + 31 + 22 + 36 + 10 + 21 + 22),
                        'cACint': 22 / (46 + 31 + 22 + 36 + 10 + 21 + 22)

                    },
                    'NBC': {
                        'dNAC': 36 / (46 + 31 + 22 + 36 + 10 + 21 + 22),
                        'cACint': 10 / (46 + 31 + 22 + 36 + 10 + 21 + 22)
                    },
                    'SBC': {
                        'bNAC': 21 / (46 + 31 + 22 + 36 + 10 + 21 + 22),
                        'cACint': 22 / (46 + 31 + 22 + 36 + 10 + 21 + 22)
                    }
                },
                'L5': {
                    'TTPC2': {
                        'cADpyr': 1.0
                    }
                }
            }

    column_depth = 2700  # um https://doi.org/10.1101/2024.03.05.583549 citing https://doi.org/10.1371/journal.pbio.3000678

    layer_ncd_boundaries = {
        'L1': [0, 0.1],
        'L23': [0.1, 0.29],
        'L4': [0.29, 0.37],
        'L5A': [0.37, 0.47],
        'L5B': [0.47, 0.8],
        'L6': [0.8, 1.0]
    }


def get_cell_count(cell_class):
    cell_file = MODULE_BASE.joinpath("reference_data/Zhang_2021/normalized_depth_MOp_Zhang_2021.csv")
    data = pd.read_csv(cell_file)
    if cell_class == 'L23':
        return np.sum(data["subclass"] == 'L2/3 IT')
    if cell_class == 'L23_inh':
        gaba_mask = data['class_label'] == "GABAergic"
        l23_upper_bound_mask = data['normalized_depth'] >= CellType.layer_ncd_boundaries['L23'][0]
        l23_lower_bound_mask = data['normalized_depth'] <= CellType.layer_ncd_boundaries['L23'][1]
        return np.sum(gaba_mask & l23_lower_bound_mask & l23_upper_bound_mask)
    elif cell_class == 'L4':
        return np.sum(data['subclass'] == 'L4/5 IT')
    elif cell_class == 'L4_inh':
        gaba_mask = data['class_label'] == "GABAergic"
        l23_upper_bound_mask = data['normalized_depth'] >= CellType.layer_ncd_boundaries['L4'][0]
        l23_lower_bound_mask = data['normalized_depth'] <= CellType.layer_ncd_boundaries['L4'][1]
        return np.sum(gaba_mask & l23_lower_bound_mask & l23_upper_bound_mask)
    elif cell_class == 'L5':
        return np.sum(data['subclass'] == 'L5 ET')
    elif cell_class == 'L5_inh':
        gaba_mask = data['class_label'] == "GABAergic"
        l23_upper_bound_mask = data['normalized_depth'] >= CellType.layer_ncd_boundaries['L5B'][0]
        l23_lower_bound_mask = data['normalized_depth'] <= CellType.layer_ncd_boundaries['L5B'][1]
        return np.sum(gaba_mask & l23_lower_bound_mask & l23_upper_bound_mask)
    elif cell_class == 'L6':
        return np.sum(data['subclass'] == 'L6 IT')
    elif cell_class == 'L6_inh':
        gaba_mask = data['class_label'] == "GABAergic"
        l23_upper_bound_mask = data['normalized_depth'] >= CellType.layer_ncd_boundaries['L6'][0]
        l23_lower_bound_mask = data['normalized_depth'] <= CellType.layer_ncd_boundaries['L6'][1]
        return np.sum(gaba_mask & l23_lower_bound_mask & l23_upper_bound_mask)
    elif cell_class == 'Gluta':
        return np.sum(data['class_label'] == 'Glutamatergic')
    elif cell_class == 'GABA':
        return np.sum(data['class_label'] == "GABAergic")
    elif cell_class == 'SOM':
        return np.sum(data['subclass'] == 'Sst')
    elif cell_class == 'PV':
        return np.sum(data['subclass'] == 'Pvalb')
    else:
        raise AttributeError(
            "Layer ID is not recognized in (L23, L23_inh, L4, L4_inh, L5, L5_inh, L6, L6_inh, Gluta, GABA, SOM, PV)")


def get_cell_coordinate_bounds(layer):
    """
    Computes the coordinate limits in x, y, z with respect to soma position for all cells of the given layer
    (L23, L4 or L5). The bounds are returns as [[x_min, y_min, z_min], [x_max, y_max, z_max]]
    :param layer: String Layer name
    :return: Coordinate Limits as [[x_min, y_min, z_min], [x_max, y_max, z_max]]
    """
    assert layer in ['L23', 'L4', 'L5'], "layer must be ('L23', 'L5' or 'L4')"
    coord_limit_file = MODULE_BASE.joinpath('reference_data/cell_distributions/cell_type_coordinate_limits.pkl')
    with open(coord_limit_file, 'rb') as f:
        coord_bounds = pkl.load(f).get(layer)

    return coord_bounds


def get_connection_probability(pre_cell='L23', post_cell='L5'):
    assert (pre_cell in ['L23', 'L4', 'L5', 'L6']) and (post_cell in ['L23', 'L4', 'L5', 'L6']), \
        "Cell Classes must be 'L23', 'L4', 'L5', or 'L6'."
    label_map = {
        'L23': 0,
        'L4': 0,
        'L5': 2,
        'L6': 3,
    }
    pre_cell = 'L23'
    post_cell = 'L5'
    index_label = label_map[post_cell]
    labels = [('W+AS_norm', 'IT', 'L2/3,4'), ('W+AS_norm', 'IT', 'L5A,5B'), ('W+AS_norm', 'PT', 'L5B'),
              ('W+AS_norm', 'IT', 'L6'), ('W+AS_norm', 'CT', 'L6')]
    labelPostBins = [('W+AS', 'IT', 'L2/3,4'), ('W+AS', 'IT', 'L5A,5B'), ('W+AS', 'PT', 'L5B'),
                     ('W+AS', 'IT', 'L6'), ('W+AS', 'CT', 'L6')]
    labelPreBins = ['W', 'AS', 'AS', 'W', 'W']
    conn_file = MODULE_BASE.joinpath('reference_data/Dura_Bernal_2023/conn/conn.pkl')
    with open(conn_file, 'rb') as f:
        tmp = pkl.load(f)
        # smat = tmp['smat']
        pmat = tmp['pmat']
        # wmat = tmp['wmat']
        bins = tmp['bins']

    layer_bounds = bins['L']  # 6 bounds associated with L23/4, L5ab, L5b, L6
    layer_labels = bins['layerLabels']

    label_pre = labels[index_label]
    bins_pre = bins[labelPreBins[index_label]]  # y axis
    pre_bounds = layer_bounds[label_map[pre_cell]]


    bins_post = bins[labelPostBins[index_label]]  # x axis
    prob_mat = pmat[label_pre]

    maxVal = max([v.max() for k, v in pmat.items() if k in labels])
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(6, 8))
    fig.suptitle('Probability of connection', fontsize=16)
    totsum = 0.0
    for i,(label, preBin, postBin) in enumerate(zip(labels,labelPreBins, labelPostBins)):
        plt.subplot(len(labels),1,i+1)
        xylims = (bins[preBin][0][0], bins[preBin][-1][-1], bins[postBin][0][0], bins[postBin][-1][-1])
        im_pmat=plt.imshow(pmat[label], origin='lower', interpolation='None', aspect='auto', extent=xylims, vmin=0, vmax=maxVal)
        plt.title('E -> '+label[1]+' '+label[2])
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.xlabel('Presynaptic Depth')
        plt.ylabel('Postsynaptic Depth')
        for layer, lbl in zip(bins['layers'][1:-1], bins['layerLabels'][1:-1]):
            plt.axvline(layer, color='r')
            if i == 0:
                plt.text(layer, 0, lbl.split(' ')[0])
        # x-axis (presynaptic)
        # y-axis (postsynaptic)
        subplot_sum = np.sum(pmat[label])
        y_bin_width = np.diff(bins[postBin], axis=1).reshape(-1, 1)
        x_bin_wdith = np.diff(bins[preBin], axis=1).reshape(1, -1)
        bin_area = np.matmul(y_bin_width, x_bin_wdith)
        dens = pmat[label] / bin_area
        dens_sum = np.sum(dens)
        print(f"{label} : {subplot_sum}")
        totsum += dens_sum
    fig.subplots_adjust(right=0.8)
    fig.subplots_adjust(bottom=0.05)
    cbar_ax = fig.add_axes([0.85, 0.25, 0.03, 0.5])
    fig.colorbar(im_pmat, cax=cbar_ax)

    if not mpl.is_interactive():
        plt.show()
