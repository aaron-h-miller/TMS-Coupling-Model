
import pathlib
from typing import Type
import numpy as np
from neuron import h
from neuron_interface.neurosim.cells import NeuronCell
from neuron_interface.neurosim.simulation.simulation import Simulation, WaveformType
from helper_functions.neuron_cell_functions_lib import syns_per_segment
from helper_functions.cell_type_lib import CellType
from helper_functions import __file__

MODULE_BASE = pathlib.Path(__file__).parent.parent


class DendriticDelaySimulation(Simulation):
    def __init__(self, cell: Type[NeuronCell], sodium_blocked=True):
        assert cell.loaded
        self.cell = cell
        self.simulation_duration = 100
        self.simulation_time_step = 0.025
        self.grid_z = None
        self.total_syns = None
        self.syn_dens = None
        self.synapse_netcons = []
        self.synapse_objs = []
        self.hoc_syn_density = None
        self.loaded = False
        self.stim_objs = []
        self.synapse_netcons = []
        self.seg_syn_array = None
        self.segment_ref_array = []
        self.voltage_recordings = []
        self.transmembrane_current_recordings = []
        for sec in self.cell.all:
            for seg in list(sec):
                self.segment_ref_array.append(seg)
                if sodium_blocked:
                    seg_dir = dir(seg)
                    for mech_name in seg_dir:
                        if mech_name.find('Na') == 0:
                            mechanism = seg.__getattribute__(mech_name)
                            for parameter in dir(mechanism):
                                if (parameter.find('gNa') != -1) and (parameter.find('bar') != -1):
                                    seg.__setattr__(f'{parameter}_{mech_name}', 0)
                # Add voltage recording netcon to this segment
                self.voltage_recordings.append(h.Vector().record(seg._ref_v))

        self.segment_ref_array = np.array(self.segment_ref_array)
        self.conductance_inputs = np.empty(shape=(len(self.segment_ref_array), 4), dtype=object) # Columns [AMPA, NMDA, GABAa, GABAb)
        self.conductance_vectors = []
        self.simulation_time_vector = None
        self.conductance_AMPA = None
        self.conductance_NMDA = None
        self.conductance_GABAa = None
        self.conductance_GABAb = None

        super().__init__(
            neuron_cell=self.cell,
            waveform_type=WaveformType.BIPHASIC, # (Not relevant to the dendritic delay simulation),
            stimulation_delay=0.0,
            simulation_time_step=self.simulation_time_step,
            simulation_duration=self.simulation_duration
        )

    def distribute_synapses(self, total_syns, grid_z, syn_dens):
        # Re-initialize cell if synapses already loaded
        if self.loaded:
            self.cell.unload()
            self.cell.load()

        self.total_syns = total_syns
        self.syn_dens = syn_dens

        # Determine number of synapses on each segment
        seg_num_syns = syns_per_segment(self.cell, grid_z, syn_dens, total_syns)
        self.seg_syn_array = np.array([elm for subset in seg_num_syns for elm in subset])

        seg_lengths = []
        self.synapse_objs.clear()
        # segment_currents = []
        for i, sec in enumerate(self.cell.all):
            for j, seg in enumerate(list(sec)):
                seg_lengths.append(sec.L / sec.nseg)
                # Add point process
                num_syns = seg_num_syns[i][j]
                self.synapse_objs.append([])
                for _ in range(num_syns):
                    # AMPA Synapse (from Rusu 2014)
                    self.synapse_objs[-1].append(h.Exp2Syn(seg))
                    # ### AMPA
                    self.synapse_objs[-1][-1].tau1 = 0.2  # ms - rise time
                    self.synapse_objs[-1][-1].tau2 = 1.7  # ms - fall time
                    self.synapse_objs[-1][-1].e = 0  # mV - reversal potential
                    # ### NMDA
                    # synapses[-1].tau1 = 10  # ms - rise time
                    # synapses[-1].tau2 = 26  # ms - fall time
                    # synapses[-1].e = 0  # mV - reversal potential

        seg_lengths = np.array(seg_lengths)
        self.hoc_syn_density = self.seg_syn_array / seg_lengths

        self.loaded = True

    def define_synaptic_inputs_continuous(self, spike_density_ex, spike_density_in, z_bin_edges, z_bin_centers, soma_depth=-0.6*CellType.column_depth, fraction_nmda=0.5, fraction_gaba_a=0.9, fraction_ex=0.5, sim_duration=100, sim_time_step=0.025):
        self.simulation_duration = sim_duration
        self.simulation_time_step = sim_time_step
        # Determine segments within each z_bin
        segment_ids_per_bin = DendriticDelaySimulation.get_segment_ids_per_bin(self.cell, soma_depth, z_bin_edges)
        z_bin_width = abs(np.diff(z_bin_edges)[0])
        # Compute Surface Area of each segment
        segment_surface_area__per_bin = []
        for segment_id_list in segment_ids_per_bin:
            # Within this bin, get surface area of each segment over the total surface area
            bin_total = 0
            bin_area_ratios = np.zeros(shape=(len(segment_id_list), ))
            for counter, segment_id in enumerate(segment_id_list):
                area = np.pi * self.segment_ref_array[segment_id].diam * self.segment_ref_array[segment_id].sec.L / self.segment_ref_array[segment_id].sec.nseg
                bin_area_ratios[counter] = area
                bin_total += area
            bin_area_ratios = bin_area_ratios / bin_total
            segment_surface_area__per_bin.append(list(bin_area_ratios))

        # ############################ Compute AMPA and NMDA Synaptic Kernels ##########################################
        simulation_time = np.arange(0, self.simulation_duration + self.simulation_time_step, self.simulation_time_step)
        self.simulation_time_vector = h.Vector(simulation_time)
        # Values from Dura-Bernal et al. 2023
        # NMDA
        NMDA_t_rise = 15  # ms - rise
        NMDA_t_fall = 150  # ms - fall
        NMDA_reversal = 0  # mV - reversal
        NMDA_peak_conductance = 0.03  # mu Ohm^-1
        conductance_NMDA = -np.exp(-simulation_time / NMDA_t_rise) + np.exp(-simulation_time / NMDA_t_fall)
        conductance_NMDA *= NMDA_peak_conductance / np.max(conductance_NMDA)
        self.conductance_NMDA = conductance_NMDA
        # AMPA
        AMPA_t_rise = 0.05  # ms - rise time
        AMPA_t_fall = 5.3  # ms - fall time
        AMPA_reversal = 0  # mV - reversal potential
        AMPA_peak_conductance = 0.1  # mu Ohm^-1
        conductance_AMPA = -np.exp(-simulation_time / AMPA_t_rise) + np.exp(-simulation_time / AMPA_t_fall)
        conductance_AMPA *= AMPA_peak_conductance / np.max(conductance_AMPA)
        self.conductance_AMPA = conductance_AMPA
        # GABAa
        GABAa_t_rise = 0.07  # ms - rise time
        GABAa_t_fall = 18.2  # ms - fall time
        GABAa_reversal = -80  # mV - reversal potential
        GABAa_peak_conductance = 0.5  # mu Ohm^-1
        conductance_GABAa = -np.exp(-simulation_time / GABAa_t_rise) + np.exp(-simulation_time / GABAa_t_fall)
        conductance_GABAa *= GABAa_peak_conductance / np.max(conductance_GABAa)
        self.conductance_GABAa = conductance_GABAa
        # GABAb
        GABAb_t_rise = 3.5  # ms - rise time
        GABAb_t_fall = 260.9  # ms - fall time
        GABAb_reversal = -93  # mV - reversal potential
        GABAb_peak_conductance = 0.5  # mu Ohm^-1
        conductance_GABAb = -np.exp(-simulation_time / GABAb_t_rise) + np.exp(-simulation_time / GABAb_t_fall)
        conductance_GABAb *= GABAb_peak_conductance / np.max(conductance_GABAb)
        self.conductance_GABAb = conductance_GABAb
        # ############### For each bin, define AMPA and NMDA inputs based on computed conductance ######################
        for bin_row, z_val in enumerate(z_bin_centers):
            # Note: Convolutions are scaled by time step size, such that the convolution conserved the integral under spike density function
            # Convolved AMPA Conductance with Excitatory Input Rate
            convolved_conductance_AMPA = np.convolve(conductance_AMPA, spike_density_ex[bin_row, :] * z_bin_width, mode='full') * self.simulation_time_step
            convolved_conductance_AMPA = convolved_conductance_AMPA[:len(simulation_time)]
            # Convolved NMDA Conductance with Excitatory Input Rate
            convolved_conductance_NMDA = np.convolve(conductance_NMDA, spike_density_ex[bin_row, :] * z_bin_width, mode='full') * self.simulation_time_step
            convolved_conductance_NMDA = convolved_conductance_NMDA[:len(simulation_time)]
            # Convolved GABAa Conductance with Inhibitory Input Rate
            convolved_conductance_GABAa = np.convolve(conductance_GABAa, spike_density_in[bin_row, :] * z_bin_width, mode='full') * self.simulation_time_step
            convolved_conductance_GABAa = convolved_conductance_GABAa[:len(simulation_time)]
            # Convolved GABAb Conductance with Inhibitory Input Rate
            convolved_conductance_GABAb = np.convolve(conductance_GABAb, spike_density_in[bin_row, :] * z_bin_width, mode='full') * self.simulation_time_step
            convolved_conductance_GABAb = convolved_conductance_GABAb[:len(simulation_time)]
            segment_ids = segment_ids_per_bin[bin_row]
            surface_area_ratios = segment_surface_area__per_bin[bin_row]
            for segment_id, surface_area_ratio in zip(segment_ids, surface_area_ratios):
                segment_reference = self.segment_ref_array[segment_id]
                # ############################### SynCur Point Process for AMPA ########################################
                scaling_factor_AMPA = surface_area_ratio * (1 - fraction_nmda) * fraction_ex
                segment_conductance_AMPA = convolved_conductance_AMPA * scaling_factor_AMPA
                conductance_vector_AMPA = h.Vector(segment_conductance_AMPA)
                self.conductance_vectors.append(conductance_vector_AMPA)
                syn_AMPA = h.SynCur(segment_reference)
                syn_AMPA.weight = 1
                syn_AMPA.e = AMPA_reversal
                conductance_vector_AMPA.play(syn_AMPA._ref_g, self.simulation_time_vector, 1)
                self.synapse_objs.append(syn_AMPA)
                self.conductance_inputs[segment_id, 0] = segment_conductance_AMPA
                # ############################### SynCurNMDA Point Process for NMDA ####################################
                scaling_factor_NMDA = surface_area_ratio * fraction_nmda * fraction_ex
                segment_conductance_NMDA = convolved_conductance_NMDA * scaling_factor_NMDA
                conductance_vector_NMDA = h.Vector(segment_conductance_NMDA)
                self.conductance_vectors.append(conductance_vector_NMDA)
                syn_NMDA = h.SynCurNMDA(segment_reference)
                syn_NMDA.weight = 1
                syn_NMDA.e = NMDA_reversal
                conductance_vector_NMDA.play(syn_NMDA._ref_g, self.simulation_time_vector, 1)
                self.synapse_objs.append(syn_NMDA)
                self.conductance_inputs[segment_id, 1] = segment_conductance_NMDA
                # ############################### SynCur Point Process for GABAa #######################################
                scaling_factor_GABAa = surface_area_ratio * fraction_gaba_a * (1 - fraction_ex)
                segment_conductance_GABAa = convolved_conductance_GABAa * scaling_factor_GABAa
                conductance_vector_GABAa = h.Vector(segment_conductance_GABAa)
                self.conductance_vectors.append(conductance_vector_GABAa)
                syn_GABAa = h.SynCur(segment_reference)
                syn_GABAa.weight = 1
                syn_GABAa.e = GABAa_reversal
                conductance_vector_GABAa.play(syn_GABAa._ref_g, self.simulation_time_vector, 1)
                self.synapse_objs.append(syn_GABAa)
                self.conductance_inputs[segment_id, 2] = segment_conductance_GABAa
                # ############################### SynCur Point Process for GABAb #######################################
                scaling_factor_GABAb = surface_area_ratio * (1 - fraction_gaba_a) * (1 - fraction_ex)
                segment_conductance_GABAb = convolved_conductance_GABAb * scaling_factor_GABAb
                conductance_vector_GABAb = h.Vector(segment_conductance_GABAb)
                self.conductance_vectors.append(conductance_vector_GABAb)
                syn_GABAb = h.SynCur(segment_reference)
                syn_GABAb.weight = 1
                syn_GABAb.e = GABAb_reversal
                conductance_vector_GABAb.play(syn_GABAb._ref_g, self.simulation_time_vector, 1)
                self.synapse_objs.append(syn_GABAb)
                self.conductance_inputs[segment_id, 3] = segment_conductance_GABAb

                self.transmembrane_current_recordings.append(
                    [h.Vector().record(syn_AMPA._ref_i),
                     h.Vector().record(syn_NMDA._ref_i),
                     h.Vector().record(syn_GABAa._ref_i),
                     h.Vector().record(syn_GABAb._ref_i)]
                )

    def define_synapses_inputs_kernel_excitatory_discrete(self, kernel_data: dict, soma_depth=-0.6 * CellType.column_depth, spatial_seed=0, type_seed=1, fraction_nmda=0.5, sim_duration=100, sim_time_step=0.025):
        self.simulation_duration = sim_duration
        self.simulation_time_step = sim_time_step
        rng_generator_spatial = np.random.default_rng(seed=spatial_seed)
        rng_generator_syn_type = np.random.default_rng(seed=type_seed)
        # z_layer values are at the center of each bin between the edges (indexed by larger bound)
        try:
            spike_dens = kernel_data['layer_histogram']
            z_bin_edges = kernel_data['z_bins_layer']
            z_layer = kernel_data['z_bin_layer_centers']
            t_bins_center = kernel_data['t_bin_centers']

            if kernel_data['density']:
                bin_scale = abs(np.diff(t_bins_center)[0] * np.diff(z_bin_edges)[0])
            else:
                bin_scale = 1.0

        except KeyError as err:
            raise KeyError(f'kernel_data must contain: "{err.args[0]}"')

        segment_ids_per_bin = DendriticDelaySimulation.get_segment_ids_per_bin(self.cell, soma_depth, z_bin_edges)
        seg_spike_trains = {seg_idx: [] for seg_idx in range(len(self.segment_ref_array))}
        seg_spike_trains_ampa = {seg_idx: [] for seg_idx in range(len(self.segment_ref_array))}
        seg_spike_trains_nmda = {seg_idx: [] for seg_idx in range(len(self.segment_ref_array))}
        spike_count_applied = np.zeros_like(spike_dens, dtype=int)
        for col, t_val in enumerate(t_bins_center):
            for row, z_val in enumerate(z_layer):
                bin_num_spikes = int(np.round(spike_dens[row, col] * bin_scale))
                segment_ids = segment_ids_per_bin[row]
                if (bin_num_spikes > 0) and (len(segment_ids) > 0):
                    spike_count_applied[row, col] = bin_num_spikes
                    # Distribute bin_num_spikes across the bin_num_syns uniformly
                    segment_sub_indices = rng_generator_spatial.integers(low=0, high=len(segment_ids), size=bin_num_spikes)
                    nmda_probabilities = rng_generator_syn_type.random(size=bin_num_spikes)
                    for nmda_probability, segment_sub_index in zip(nmda_probabilities, segment_sub_indices):
                        seg_idx = segment_ids[segment_sub_index]
                        seg = self.segment_ref_array[seg_idx]
                        # Create Synapse Object
                        self.synapse_objs.append(h.Exp2Syn(seg))
                        if nmda_probability <= fraction_nmda:
                            # Biexponential synapse parameters:
                            # Rusu et al. 2014 https://doi.org/10.1016/j.brs.2014.02.009
                            # ### NMDA
                            self.synapse_objs[-1].tau1 = 2  # ms - rise time
                            self.synapse_objs[-1].tau2 = 26  # ms - fall time
                            self.synapse_objs[-1].e = 0  # mV - reversal potential
                            peak_conductance = 0.03  # mu Ohm^-1
                            seg_spike_trains_nmda[seg_idx].append(t_val)
                        else:
                            # ### AMPA
                            self.synapse_objs[-1].tau1 = 0.2  # ms - rise time
                            self.synapse_objs[-1].tau2 = 1.7  # ms - fall time
                            self.synapse_objs[-1].e = 0  # mV - reversal potential
                            peak_conductance = 0.1  # mu Ohm^-1
                            seg_spike_trains_ampa[seg_idx].append(t_val)
                        # Create Stimulation Object
                        stim = h.NetStim()
                        stim.start = t_val  # ms
                        stim.number = 1
                        self.stim_objs.append(stim)
                        # Connect NetCon from Stimulation to Synapse
                        self.synapse_netcons.append(h.NetCon(stim, self.synapse_objs[-1]))
                        self.synapse_netcons[-1].weight[0] = peak_conductance
                        self.synapse_netcons[-1].delay = 0.0
                        self.synapse_netcons[-1].threshold = 0.0

                        seg_spike_trains[seg_idx].append(t_val)

                    # print(f"({row}, {col}) | ({t_val:.3f}, {z_val:.3f}) {bin_num_spikes}")
        seg_spike_trains = [lst for lst in seg_spike_trains.values()]
        ampa_spike_trains = [lst for lst in seg_spike_trains_ampa.values()]
        nmda_spike_trains = [lst for lst in seg_spike_trains_nmda.values()]

        self.hoc_syn_density = np.array([len(lst) / (seg.sec.L / seg.sec.nseg) for lst, seg in zip(seg_spike_trains, self.segment_ref_array)])
        return ampa_spike_trains, nmda_spike_trains, seg_spike_trains, spike_count_applied, self.hoc_syn_density

    def define_synapses_inputs_kernel_inhibitory_discrete(self, kernel_data: dict, soma_depth=-0.6 * CellType.column_depth, spatial_seed=2, type_seed=3, fraction_gaba_a=0.9, sim_duration=100, sim_time_step=0.025):
        self.simulation_duration = sim_duration
        self.simulation_time_step = sim_time_step
        rng_generator_spatial = np.random.default_rng(seed=spatial_seed)
        rng_generator_syn_type = np.random.default_rng(seed=type_seed)
        # z_layer values are at the center of each bin between the edges (indexed by larger bound)
        try:
            spike_dens = kernel_data['layer_histogram']
            z_bin_edges = kernel_data['z_bins_layer']
            z_layer = kernel_data['z_bin_layer_centers']
            t_bins_center = kernel_data['t_bin_centers']

            if kernel_data['density']:
                bin_scale = abs(np.diff(t_bins_center)[0] * np.diff(z_bin_edges)[0])
            else:
                bin_scale = 1.0

        except KeyError as err:
            raise KeyError(f'kernel_data must contain: "{err.args[0]}"')

        segment_ids_per_bin = DendriticDelaySimulation.get_segment_ids_per_bin(self.cell, soma_depth, z_bin_edges)
        seg_spike_trains_a = {seg_idx: [] for seg_idx in range(len(self.segment_ref_array))}
        seg_spike_trains_b = {seg_idx: [] for seg_idx in range(len(self.segment_ref_array))}
        seg_spike_trains = {seg_idx: [] for seg_idx in range(len(self.segment_ref_array))}
        spike_count_applied = np.zeros_like(spike_dens, dtype=int)
        for col, t_val in enumerate(t_bins_center):
            for row, z_val in enumerate(z_layer):
                bin_num_spikes = int(np.round(spike_dens[row, col] * bin_scale))
                segment_ids = segment_ids_per_bin[row]
                if (bin_num_spikes > 0) and (len(segment_ids) > 0):
                    spike_count_applied[row, col] = bin_num_spikes
                    # Distribute bin_num_spikes across the bin_num_syns uniformly
                    segment_sub_indices = rng_generator_spatial.integers(low=0, high=len(segment_ids), size=bin_num_spikes)
                    gaba_a_probabilities = rng_generator_syn_type.random(size=bin_num_spikes)
                    for gaba_a_probability, segment_sub_index in zip(gaba_a_probabilities, segment_sub_indices):
                        seg_idx = segment_ids[segment_sub_index]
                        seg = self.segment_ref_array[seg_idx]
                        # Create Synapse Object
                        self.synapse_objs.append(h.Exp2Syn(seg))
                        if gaba_a_probability <= fraction_gaba_a:
                            # Biexponential synapse parameters:
                            # Rusu et al. 2014 https://doi.org/10.1016/j.brs.2014.02.009
                            # ### GABAa
                            self.synapse_objs[-1].tau1 = 0.3  # ms - rise time
                            self.synapse_objs[-1].tau2 = 2.5  # ms - fall time
                            self.synapse_objs[-1].e = -70  # mV - reversal potential
                            peak_conductance = 0.5  # mu Ohm^-1
                            seg_spike_trains_a[seg_idx].append(t_val)
                        else:
                            # ### GABAb
                            self.synapse_objs[-1].tau1 = 45.2  # ms - rise time
                            self.synapse_objs[-1].tau2 = 175.16  # ms - fall time
                            self.synapse_objs[-1].e = -90  # mV - reversal potential
                            peak_conductance = 0.5  # mu Ohm^-1
                            seg_spike_trains_b[seg_idx].append(t_val)
                        # Create Stimulation Object
                        stim = h.NetStim()
                        stim.start = t_val  # ms
                        stim.number = 1
                        self.stim_objs.append(stim)
                        # Connect NetCon from Stimulation to Synapse
                        self.synapse_netcons.append(h.NetCon(stim, self.synapse_objs[-1]))
                        self.synapse_netcons[-1].weight[0] = peak_conductance
                        self.synapse_netcons[-1].delay = 0.0
                        self.synapse_netcons[-1].threshold = 0.0
                        seg_spike_trains[seg_idx].append(t_val)


                    # print(f"({row}, {col}) | ({t_val:.3f}, {z_val:.3f}) {bin_num_spikes}")
        spike_trains_a = [lst for lst in seg_spike_trains_a.values()]
        spike_trains_b = [lst for lst in seg_spike_trains_b.values()]
        seg_spike_trains = [lst for lst in seg_spike_trains.values()]
        self.hoc_syn_density = np.array([len(lst) / (seg.sec.L / seg.sec.nseg) for lst, seg in zip(seg_spike_trains, self.segment_ref_array)])
        return spike_trains_a, spike_trains_b, seg_spike_trains, spike_count_applied, self.hoc_syn_density

    def clear_synapse_inputs(self):
        self.stim_objs.clear()
        self.synapse_netcons.clear()

    def simulate(self, include_axon_current=False):
        # Collect segments closest to soma for current calculation (voltage recording)
        if include_axon_current:
            nearest_sections = [section for section in self.cell.soma[0].children()]
        else:
            nearest_sections = [section for section in self.cell.soma[0].children() if
                                ((section in self.cell.dend) or (section in self.cell.apic))]
        nearest_segments = [section(0.5) for section in nearest_sections]
        nearest_cross_areas = np.array([(np.pi * segment.diam ** 2 / 4) for segment in nearest_segments])  # um**2
        nearest_resistivities = np.array(
            [(section.Ra * (1e-2 / 1e-6)) for section in nearest_sections])  # ohm um (converted from ohm cm)
        soma_voltage = h.Vector().record(self.cell.soma[0](0.5)._ref_v)  # mV
        nearest_voltages = [h.Vector().record(segment._ref_v) for segment in nearest_segments]
        time_recording = h.Vector().record(h._ref_t)

        # ############ Simulate #################
        # set initial state
        self.attach()

        h.celsius = self.simulation_temperature
        h.dt = self.simulation_time_step
        h.tstop = self.simulation_duration
        h.finitialize(Simulation.INITIAL_VOLTAGE)

        # simulate
        print('Starting Simulation')
        h.continuerun(self.simulation_duration)
        print('Simulation Complete')
        self.detach()

        time = np.array(time_recording)
        soma_voltage = np.array(soma_voltage)
        voltage_recordings = [list(voltage) for voltage in self.voltage_recordings]
        nearest_voltages = [list(voltage) for voltage in nearest_voltages]
        nearest_voltages = np.vstack(nearest_voltages)
        voltage_recordings = np.vstack(voltage_recordings)

        # Calculate Axial Current from voltages, resistivity, and cross-sectional areas (I = -A/r * dV/dx)
        segment_distances = np.array([h.distance(self.cell.soma[0](0.5), nearest_seg) for nearest_seg in nearest_segments])
        gradients = (np.tile(soma_voltage, (nearest_voltages.shape[0], 1)) - nearest_voltages) / segment_distances.reshape((-1, 1))

        curr = - np.transpose(np.tile(nearest_cross_areas, (gradients.shape[1], 1))) * gradients / nearest_resistivities.reshape((-1, 1))
        total_current = np.sum(curr, axis=0) * 1e-3  # A

        return self.seg_syn_array, self.hoc_syn_density, time, voltage_recordings, total_current

    @staticmethod
    def get_segment_ids_per_bin(neuron_cell, soma_depth, z_bin_edges, verbose=False):
        bin_segment_lists = {bin_idx: [] for bin_idx in range(len(z_bin_edges) - 1)}
        seg_index = 0
        ignored_segs = 0
        for sec in neuron_cell.all:
            for seg in list(sec):
                if sec in neuron_cell.dend + neuron_cell.apic:
                    # Calculate segment position in layer space, shifted by postsynaptic soma position
                    z = soma_depth + (seg.z_xtra - neuron_cell.soma[0](0.5).z_xtra)
                    if (z > 0) or (z < -CellType.column_depth):
                        # print(f"ERROR: z out of bounds (0, {CellType.column_depth}) Seg Ignored: {seg}")
                        ignored_segs += 1
                    else:
                        bin_idx = np.where(z_bin_edges >= z)[0][-1]
                        if ((z >= z_bin_edges[bin_idx]) or (z <= z_bin_edges[bin_idx + 1])) and verbose:
                            print(
                                f"##### ERROR {z=} Bin {bin_idx} [{z_bin_edges[bin_idx]}, {z_bin_edges[bin_idx + 1]}] #####")
                        bin_segment_lists[bin_idx].append(seg_index)
                seg_index += 1
        bin_segment_lists = [lst for lst in bin_segment_lists.values()]
        if verbose:
            print(f'{ignored_segs} Ignored Segments out of Bounds')
        return bin_segment_lists



