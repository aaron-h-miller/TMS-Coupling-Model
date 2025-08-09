import numpy as np

from scipy.spatial.transform import Rotation as R

from ..neurosim.cells.neuron_cell import NeuronCell
from neuron import h


class TestNeuronCell:
    def test_load(self, neuron_cell: NeuronCell):
        assert len(neuron_cell.all) == 73
        assert len(list(h.allsec())) == len(neuron_cell.all)
        assert neuron_cell.loaded

    def test_unload(self, neuron_cell: NeuronCell):
        neuron_cell.unload()
        assert len(neuron_cell.all) == 0
        assert len(list(h.allsec())) == len(neuron_cell.all)
        assert len(neuron_cell.axon) == 0
        assert len(neuron_cell.myelin) == 0
        assert len(neuron_cell.node) == 0
        assert len(neuron_cell.unmyelin) == 0
        assert len(neuron_cell.dend) == 0
        assert len(neuron_cell.soma) == 0
        assert len(neuron_cell.apic) == 0
        assert not neuron_cell.loaded

    def test_segment_count(self, neuron_cell: NeuronCell):
        NeuronCell._set_segment_count(neuron_cell.all, 34)
        segment_count = []
        section_length = []
        for section in neuron_cell.all:
            segment_count.append(section.nseg)
            section_length.append(section.L)
        segment_count = np.array(segment_count)
        section_length = np.array(section_length)

        np.testing.assert_array_less(section_length / segment_count, 34)

    def test_scale_point_diam(self, neuron_cell: NeuronCell):
        point_diams_before = []
        for section in neuron_cell.all:
            for point_index in range(section.n3d()):
                point_diams_before.append(section.diam3d(point_index))
        NeuronCell._scale_section_point_diameters(2.5, neuron_cell.all)
        point_diams_after = []
        for section in neuron_cell.all:
            for point_index in range(section.n3d()):
                point_diams_after.append(section.diam3d(point_index))
        np.testing.assert_allclose(
            np.array(point_diams_after), np.array(point_diams_before) * 2.5
        )

    def test_scale_section_length(self, neuron_cell: NeuronCell):
        section_length_before = []
        for section in neuron_cell.all:
            section_length_before.append(section.L)
        NeuronCell._scale_section_length(1.5345, neuron_cell.all)
        section_length_after = []
        for section in neuron_cell.all:
            section_length_after.append(section.L)
        np.testing.assert_allclose(
            np.array(section_length_after), np.array(section_length_before) * 1.5345
        )

    def test_scale_soma_area(self, neuron_cell: NeuronCell):
        soma_area_before = neuron_cell.soma[0](0.5).area()
        neuron_cell._scale_soma_area(8.1238)
        soma_area_after = neuron_cell.soma[0](0.5).area()
        np.testing.assert_approx_equal(soma_area_after, soma_area_before * 8.1238)

    def test_rotate_x_90(self, neuron_cell: NeuronCell):
        points = []
        for section in neuron_cell.all:
            for i in range(section.n3d()):
                points.append([section.x3d(i), section.y3d(i), section.z3d(i)])

        neuron_cell._rotate_x_90()

        points_after = []
        for section in neuron_cell.all:
            for i in range(section.n3d()):
                points_after.append([section.x3d(i), section.y3d(i), section.z3d(i)])

        points = np.array(points)
        r = R.from_euler("x", 90, degrees=True)
        points = r.apply(points)

        np.testing.assert_allclose(np.array(points_after), points, atol=1e-9)

    def test_myelinate_axon(self, neuron_cell: NeuronCell):
        if (
            neuron_cell.axon[0].L
            >= NeuronCell.MIN_PRE_MYELIN_AXON_LENGTH
            + neuron_cell.modification_parameters.min_myelin_length
        ):
            assert neuron_cell.axon[0] not in neuron_cell.myelin

        for myelin in neuron_cell.myelin:
            assert myelin.diam > neuron_cell.modification_parameters.min_myelin_diameter
            assert myelin.L > neuron_cell.modification_parameters.min_myelin_length

        for unmyelin in neuron_cell.unmyelin:
            assert (
                unmyelin.diam < neuron_cell.modification_parameters.min_myelin_diameter
                or unmyelin.L
                < (
                    neuron_cell.modification_parameters.min_myelin_length
                    + NeuronCell.NODE_LENGTH
                )
            )

        for node in neuron_cell.node:
            np.testing.assert_almost_equal(node.L, NeuronCell.NODE_LENGTH, 4)

        assert len(list(h.allsec())) == len(neuron_cell.all)
        assert len(neuron_cell.all) == len(neuron_cell.apic) + len(
            neuron_cell.dend
        ) + len(neuron_cell.axon) + len(neuron_cell.soma)

    def test_segment_coordinates(self, neuron_cell: NeuronCell):
        segment_coordinates = neuron_cell.get_segment_coordinates()

        # Test distance between segment positions and section point positions
        index = 0
        for section in neuron_cell.all:
            test_points = np.zeros((section.n3d(), 3))
            for i in range(section.n3d()):
                test_points[i, 0] = section.x3d(i)
                test_points[i, 1] = section.y3d(i)
                test_points[i, 2] = section.z3d(i)
            dists = np.sqrt(((test_points[:, None, :] - test_points) ** 2).sum(-1))
            max_dist = np.max(dists)

            for _ in section:
                distances = np.linalg.norm(
                    test_points - segment_coordinates[index], axis=1
                )
                assert np.min(distances) <= max_dist / 2
                index += 1

        # Test if the segment points are on the same path as the section points
        index = 0
        for section in neuron_cell.all:
            path_points = np.zeros((section.n3d(), 3))
            for i in range(section.n3d()):
                path_points[i, 0] = section.x3d(i)
                path_points[i, 1] = section.y3d(i)
                path_points[i, 2] = section.z3d(i)

            for _ in section:
                segment_coordinate = segment_coordinates[index]
                segment_point_on_line = False
                for current_point, next_point in zip(path_points, path_points[1:]):
                    p1 = next_point - current_point
                    p2 = segment_coordinate - current_point
                    if np.linalg.norm(np.cross(p2, p1)) < 1e-9 and 0 < np.dot(
                        p2, p1
                    ) < np.dot(p1, p1):
                        segment_point_on_line = True
                        break
                np.testing.assert_(
                    segment_point_on_line,
                    f"The segment point {index} {segment_coordinate} is not between any points of the section it is on.",
                )
                index += 1
