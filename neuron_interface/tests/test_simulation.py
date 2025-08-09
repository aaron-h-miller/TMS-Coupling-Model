import operator
import numpy as np
from ..neurosim.simulation.threshold_factor_simulation import ThresholdFactorSimulation
from neuron import h

from ..spherical_cartesian_conversion import cartesian_to_spherical


class TestSimulation:
    def test_attach(self, neuron_simulation: ThresholdFactorSimulation):
        assert neuron_simulation.attached
        assert len(neuron_simulation.netcons) == np.sum(
            [sec.nseg for sec in neuron_simulation.neuron_cell.all]
        )
        assert len(neuron_simulation.netcons) == len(h.List("NetCon"))

    def test_detach(self, neuron_simulation: ThresholdFactorSimulation):
        neuron_simulation.detach()
        assert not neuron_simulation.attached
        assert len(neuron_simulation.netcons) == 0
        assert len(h.List("NetCon")) == 0
        neuron_simulation.neuron_cell.unload()
        assert len(list(h.allsec())) == 0

    def test_simple_simulation(self, neuron_simulation: ThresholdFactorSimulation):
        e_field_vecs = np.zeros(
            (np.sum([sec.nseg for sec in neuron_simulation.neuron_cell.all]), 3)
        )
        e_field_vecs[:, 0] = 100
        neuron_simulation.apply_e_field(e_field_vecs)
        np.testing.assert_approx_equal(
            neuron_simulation.find_threshold_factor()[0], 64.013671875, 3
        )
        e_field_vecs[:, 0] = 100 * 2
        neuron_simulation.apply_e_field(e_field_vecs)
        np.testing.assert_approx_equal(
            neuron_simulation.find_threshold_factor()[0], 64.013671875 / 2, 3
        )
        e_field_vecs[:, 0] = 100 * 3
        neuron_simulation.apply_e_field(e_field_vecs)
        np.testing.assert_approx_equal(
            neuron_simulation.find_threshold_factor()[0], 64.013671875 / 3, 3
        )

    def test_apply_parametric_e_field(
        self, neuron_simulation: ThresholdFactorSimulation
    ):
        height = (
            np.ptp(neuron_simulation.neuron_cell.get_segment_coordinates()[:, 2]) / 1000
        )

        test_cases = np.stack(
            np.meshgrid(
                np.linspace(90, 181, 3),
                np.linspace(45, 136, 3),
                np.linspace(10, 101, 5),
                np.linspace(10, 91, 3),
                np.linspace(10, 46, 3),
                indexing="ij",
            ),
            axis=-1,
        ).reshape(-1, 5)

        for (
            phi_soma,
            theta_soma,
            magnitude_change,
            phi_change,
            theta_change,
        ) in test_cases:
            neuron_simulation.apply_parametric_e_field(
                theta_soma,
                phi_soma,
                magnitude_change / height,
                phi_change / height,
                theta_change / height,
            )

            np.testing.assert_allclose(
                cartesian_to_spherical(
                    [
                        [
                            neuron_simulation.neuron_cell.soma[0].Ex_xtra,
                            neuron_simulation.neuron_cell.soma[0].Ey_xtra,
                            neuron_simulation.neuron_cell.soma[0].Ez_xtra,
                        ]
                    ]
                ),
                [phi_soma, theta_soma, 1],
                err_msg=f"E-field at soma of the parametric E-field is wrong ({phi_soma, theta_soma, magnitude_change, phi_change, theta_change})",
            )

            e_field = []
            coords = []
            for section in neuron_simulation.neuron_cell.all:
                for segment in section:
                    e_field.append([segment.Ex_xtra, segment.Ey_xtra, segment.Ez_xtra])
                    coords.append([segment.x_xtra, segment.y_xtra, segment.z_xtra])
            e_field = cartesian_to_spherical(e_field)
            coords = np.array(coords)
            e_field_z_sorted = e_field[np.argsort(coords[:, 2])[::-1]]

            np.testing.assert_almost_equal(
                np.max(e_field[:, 2]) - np.min(e_field[:, 2]),
                magnitude_change / 100,
                err_msg=f"The range of the E-field magnitude is not correct ({phi_soma, theta_soma, magnitude_change, phi_change, theta_change})",
            )
            np.testing.assert_array_less(
                0,
                np.diff(e_field_z_sorted[:, 2]),
                f"The E-field magnitude is not changing monotonic ({phi_soma, theta_soma, magnitude_change, phi_change, theta_change})",
            )

            np.testing.assert_almost_equal(
                np.max(e_field[:, 0]) - np.min(e_field[:, 0]),
                phi_change,
                err_msg=f"The range of the E-field phi is not correct ({phi_soma, theta_soma, magnitude_change, phi_change, theta_change})",
            )
            np.testing.assert_array_less(
                np.diff(e_field_z_sorted[:, 0]),
                0,
                f"E-field phi is not changing monotonic ({phi_soma, theta_soma, magnitude_change, phi_change, theta_change})",
            )

            np.testing.assert_almost_equal(
                np.max(e_field[:, 1]) - np.min(e_field[:, 1]),
                theta_change,
                err_msg=f"The range of the E-field theta is not correct ({phi_soma, theta_soma, magnitude_change, phi_change, theta_change})",
            )
            np.testing.assert_array_less(
                np.diff(e_field_z_sorted[:, 1]),
                0,
                f"E-field theta is not changing monotonic ({phi_soma, theta_soma, magnitude_change, phi_change, theta_change})",
            )

    def test_apply_parametric_e_field_special_cases(
        self, neuron_simulation: ThresholdFactorSimulation
    ):
        height = (
            np.ptp(neuron_simulation.neuron_cell.get_segment_coordinates()[:, 2]) / 1000
        )

        # Test all 0
        phi_soma, theta_soma, magnitude_change, phi_change, theta_change = 0, 0, 0, 0, 0
        neuron_simulation.apply_parametric_e_field(
            theta_soma,
            phi_soma,
            magnitude_change / height,
            phi_change / height,
            theta_change / height,
        )
        np.testing.assert_allclose(
            cartesian_to_spherical(
                [
                    [
                        neuron_simulation.neuron_cell.soma[0].Ex_xtra,
                        neuron_simulation.neuron_cell.soma[0].Ey_xtra,
                        neuron_simulation.neuron_cell.soma[0].Ez_xtra,
                    ]
                ]
            ),
            [phi_soma, theta_soma, 1],
            err_msg=f"E-field at soma of the parametric E-field is wrong ({phi_soma, theta_soma, magnitude_change, phi_change, theta_change})",
        )

        e_field = []
        coords = []
        for section in neuron_simulation.neuron_cell.all:
            for segment in section:
                e_field.append([segment.Ex_xtra, segment.Ey_xtra, segment.Ez_xtra])
                coords.append([segment.x_xtra, segment.y_xtra, segment.z_xtra])
        e_field = cartesian_to_spherical(e_field)
        coords = np.array(coords)
        np.testing.assert_allclose(e_field[:, :2], 0)
        np.testing.assert_allclose(e_field[:, 2], 1)

        # Test high magnitude change
        phi_soma, theta_soma, magnitude_change, phi_change, theta_change = (
            0,
            0,
            1000000,
            0,
            0,
        )
        neuron_simulation.apply_parametric_e_field(
            theta_soma,
            phi_soma,
            magnitude_change / height,
            phi_change / height,
            theta_change / height,
        )
        np.testing.assert_allclose(
            cartesian_to_spherical(
                [
                    [
                        neuron_simulation.neuron_cell.soma[0].Ex_xtra,
                        neuron_simulation.neuron_cell.soma[0].Ey_xtra,
                        neuron_simulation.neuron_cell.soma[0].Ez_xtra,
                    ]
                ]
            ),
            [phi_soma, theta_soma, 1],
            err_msg=f"E-field at soma of the parametric E-field is wrong ({phi_soma, theta_soma, magnitude_change, phi_change, theta_change})",
        )

        e_field = []
        coords = []
        for section in neuron_simulation.neuron_cell.all:
            for segment in section:
                e_field.append([segment.Ex_xtra, segment.Ey_xtra, segment.Ez_xtra])
                coords.append([segment.x_xtra, segment.y_xtra, segment.z_xtra])
        e_field = cartesian_to_spherical(e_field)
        coords = np.array(coords)
        np.testing.assert_array_compare(operator.__le__, 0, e_field[:, 2])

        # Test phi underflow
        phi_soma, theta_soma, magnitude_change, phi_change, theta_change = (
            0,
            90,
            0,
            90,
            0,
        )
        neuron_simulation.apply_parametric_e_field(
            theta_soma,
            phi_soma,
            magnitude_change / height,
            phi_change / height,
            theta_change / height,
        )
        np.testing.assert_allclose(
            cartesian_to_spherical(
                [
                    [
                        neuron_simulation.neuron_cell.soma[0].Ex_xtra,
                        neuron_simulation.neuron_cell.soma[0].Ey_xtra,
                        neuron_simulation.neuron_cell.soma[0].Ez_xtra,
                    ]
                ]
            ),
            [phi_soma, theta_soma, 1],
            err_msg=f"E-field at soma of the parametric E-field is wrong ({phi_soma, theta_soma, magnitude_change, phi_change, theta_change})",
        )

        e_field = []
        coords = []
        for section in neuron_simulation.neuron_cell.all:
            for segment in section:
                e_field.append([segment.Ex_xtra, segment.Ey_xtra, segment.Ez_xtra])
                coords.append([segment.x_xtra, segment.y_xtra, segment.z_xtra])
        e_field = cartesian_to_spherical(e_field)
        coords = np.array(coords)
        assert np.all(np.logical_or(e_field[:, 0] >= 270, e_field[:, 0] <= 90))
        low_range_max = np.max(e_field[e_field[:, 0] <= 90][:, 0])
        high_range_min = np.min(e_field[e_field[:, 0] >= 270][:, 0])
        np.testing.assert_almost_equal(low_range_max + 360 - high_range_min, phi_change)

        # Test theta underflow
        phi_soma, theta_soma, magnitude_change, phi_change, theta_change = (
            90,
            0,
            0,
            0,
            90,
        )
        neuron_simulation.apply_parametric_e_field(
            theta_soma,
            phi_soma,
            magnitude_change / height,
            phi_change / height,
            theta_change / height,
        )
        np.testing.assert_allclose(
            cartesian_to_spherical(
                [
                    [
                        neuron_simulation.neuron_cell.soma[0].Ex_xtra,
                        neuron_simulation.neuron_cell.soma[0].Ey_xtra,
                        neuron_simulation.neuron_cell.soma[0].Ez_xtra,
                    ]
                ]
            )[1:],
            [theta_soma, 1],
            err_msg=f"E-field at soma of the parametric E-field is wrong ({phi_soma, theta_soma, magnitude_change, phi_change, theta_change})",
        )

        e_field = []
        coords = []
        for section in neuron_simulation.neuron_cell.all:
            for segment in section:
                e_field.append([segment.Ex_xtra, segment.Ey_xtra, segment.Ez_xtra])
                coords.append([segment.x_xtra, segment.y_xtra, segment.z_xtra])
        e_field = cartesian_to_spherical(e_field)
        coords = np.array(coords)
        inverted_range_max = np.max(e_field[e_field[:, 0] > 180][:, 1])
        non_inverted_range_max = np.max(e_field[e_field[:, 0] < 180][:, 1])
        np.testing.assert_almost_equal(
            inverted_range_max + non_inverted_range_max, theta_change
        )

        # Test theta overflow
        phi_soma, theta_soma, magnitude_change, phi_change, theta_change = (
            270,
            180,
            0,
            0,
            120,
        )
        neuron_simulation.apply_parametric_e_field(
            theta_soma,
            phi_soma,
            magnitude_change / height,
            phi_change / height,
            theta_change / height,
        )
        np.testing.assert_allclose(
            cartesian_to_spherical(
                [
                    [
                        neuron_simulation.neuron_cell.soma[0].Ex_xtra,
                        neuron_simulation.neuron_cell.soma[0].Ey_xtra,
                        neuron_simulation.neuron_cell.soma[0].Ez_xtra,
                    ]
                ]
            )[1:],
            [theta_soma, 1],
            err_msg=f"E-field at soma of the parametric E-field is wrong ({phi_soma, theta_soma, magnitude_change, phi_change, theta_change})",
        )

        e_field = []
        coords = []
        for section in neuron_simulation.neuron_cell.all:
            for segment in section:
                e_field.append([segment.Ex_xtra, segment.Ey_xtra, segment.Ez_xtra])
                coords.append([segment.x_xtra, segment.y_xtra, segment.z_xtra])
        e_field = cartesian_to_spherical(e_field)
        coords = np.array(coords)
        inverted_range_min = np.min(e_field[e_field[:, 0] < 180][:, 1])
        non_inverted_range_min = np.min(e_field[e_field[:, 0] > 180][:, 1])
        np.testing.assert_almost_equal(
            (180 - inverted_range_min) + (180 - non_inverted_range_min), theta_change
        )

    def test_apply_e_field(self, neuron_simulation: ThresholdFactorSimulation):
        # Test if e-field vectors get applied correctly
        e_field_vecs = [
            [x, x + 1, x + 2]
            for x in range(
                1, np.sum([sec.nseg for sec in neuron_simulation.neuron_cell.all]) + 1
            )
        ]
        neuron_simulation.apply_e_field(e_field_vecs)
        index = 1
        for section in neuron_simulation.neuron_cell.all:
            for segment in section:
                assert segment.Ex_xtra == index
                assert segment.Ey_xtra == index + 1
                assert segment.Ez_xtra == index + 2
                index += 1

    def test_apply_e_field_quasi_potentials(
        self, neuron_simulation: ThresholdFactorSimulation
    ):
        # Test es_xtra changing the right way with simple e-fields
        for e_x, e_y, e_z in [
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, -1],
            [0, 0, 1],
        ]:
            e_field_vecs = np.full(
                (np.sum([sec.nseg for sec in neuron_simulation.neuron_cell.all]), 3),
                [e_x, e_y, e_z],
            )
            neuron_simulation.apply_e_field(e_field_vecs)
            print(e_x, e_y, e_z)

            for section in neuron_simulation.neuron_cell.all:
                segments = [section.parentseg()] if section.parentseg() else []
                segments += list(section)
                for parent_segment, segment in zip(segments, segments[1:]):
                    if e_x != 0:
                        if (segment.x_xtra - parent_segment.x_xtra) * e_x < 0:
                            assert segment.es_xtra > parent_segment.es_xtra
                        else:
                            assert segment.es_xtra <= parent_segment.es_xtra
                    if e_y != 0:
                        if (segment.y_xtra - parent_segment.y_xtra) * e_y < 0:
                            assert segment.es_xtra > parent_segment.es_xtra
                        else:
                            assert segment.es_xtra <= parent_segment.es_xtra
                    if e_z != 0:
                        if (segment.z_xtra - parent_segment.z_xtra) * e_z < 0:
                            assert segment.es_xtra > parent_segment.es_xtra
                        else:
                            assert segment.es_xtra <= parent_segment.es_xtra
