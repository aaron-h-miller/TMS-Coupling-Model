
import numpy as np

from ..spherical_cartesian_conversion import cartesian_to_spherical, spherical_to_cartesian


class TestSphericalCartesianConversion:
    def test_cartesian_to_spherical(self):
        np.testing.assert_allclose(cartesian_to_spherical([[0, 0, 1], [0, 0, -1], [0, 0, -20]])[:, 1:], [[0, 1], [180, 1], [180, 20]], atol=1e-9)
        np.testing.assert_allclose(cartesian_to_spherical([[1, 0, 0], [0, 1, 0]]), [[0, 90, 1], [90, 90, 1]], atol=1e-9)
        np.testing.assert_allclose(cartesian_to_spherical([[-1, 0, 0], [0, -1, 0]]), [[180, 90, 1], [270, 90, 1]], atol=1e-9)
        np.testing.assert_allclose(cartesian_to_spherical([[2, 0, 0], [0, 10, 0]]), [[0, 90, 2], [90, 90, 10]], atol=1e-9)

    def test_spherical_to_cartesian(self):
        np.testing.assert_allclose(spherical_to_cartesian([0, 0, 1]), [0, 0, 1])
        np.testing.assert_allclose(spherical_to_cartesian([[0, 0, 1], [0, 90, 1], [0, 180, 1]]), [[0, 0, 1], [1, 0, 0], [0, 0, -1]], atol=1e-9)
        np.testing.assert_allclose(spherical_to_cartesian([[90, 0, 1], [90, 90, 1], [90, 180, 1]]), [[0, 0, 1], [0, 1, 0], [0, 0, -1]], atol=1e-9)
        np.testing.assert_allclose(spherical_to_cartesian([[180, 0, 1], [180, 90, 1], [180, 180, 1]]), [[0, 0, 1], [-1, 0, 0], [0, 0, -1]], atol=1e-9)
        np.testing.assert_allclose(spherical_to_cartesian([[270, 0, 1], [270, 90, 1], [270, 180, 1]]), [[0, 0, 1], [0, -1, 0], [0, 0, -1]], atol=1e-9)
        np.testing.assert_allclose(spherical_to_cartesian([[360, 0, 1], [360, 90, 1], [360, 180, 1]]), [[0, 0, 1], [1, 0, 0], [0, 0, -1]], atol=1e-9)

    def test_reversibility(self):
        np.testing.assert_allclose(cartesian_to_spherical(spherical_to_cartesian([[0, 0, 1], [90, 0, 1]]))[:,1:], [[0, 1], [0, 1]], atol=1e-9)
        np.testing.assert_allclose(spherical_to_cartesian(cartesian_to_spherical([0, 0, 1])), [0, 0, 1], atol=1e-9)

        np.testing.assert_allclose(cartesian_to_spherical(spherical_to_cartesian([[90, 90, 1], [90, 180, 1]])), [[90, 90, 1], [90, 180, 1]], atol=1e-9)
        np.testing.assert_allclose(spherical_to_cartesian(cartesian_to_spherical([[1, 0, 0], [0, 1, 0], [0, 0, 1]])), [[1, 0, 0], [0, 1, 0], [0, 0, 1]], atol=1e-9)

        vecs = np.stack(np.meshgrid(np.linspace(0.01, 10.01, 100), np.linspace(0.01, 10.01, 100), np.linspace(0.01, 10.01, 100), indexing='ij'), axis=-1).reshape(-1, 3)
        np.testing.assert_allclose(spherical_to_cartesian(cartesian_to_spherical(vecs)), vecs, atol=1e-9)

        vecs = np.stack(np.meshgrid(np.linspace(0.01, 360, 360), np.linspace(0.01, 180, 180), np.linspace(1, 11, 100), indexing='ij'), axis=-1).reshape(-1, 3)
        np.testing.assert_allclose(cartesian_to_spherical(spherical_to_cartesian(vecs)), vecs, atol=1e-9)

    def test_special_cases(self):
        np.testing.assert_allclose(cartesian_to_spherical([0,0,0]), [0, 0, 0])
        np.testing.assert_allclose(spherical_to_cartesian([0,0,0]), [0, 0, 0])

