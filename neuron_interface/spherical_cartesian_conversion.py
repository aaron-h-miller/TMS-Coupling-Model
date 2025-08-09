import numpy as np
import numpy.typing as npt


def cartesian_to_spherical(cartesian_vecs: npt.ArrayLike) -> npt.NDArray[np.float_]:
    """Calculates spherical coordinates in degree from cartesian coordinates

    Parameters
    ----------
    vec : npt.ArrayLike (N x 3)
        Cartesian coordinates in the format [x, y, z]

    Returns
    -------
    npt.NDArray[np.int_] (N x 3)
        Spherical coordinates in degree in the format [phi, theta, r]
    """
    cartesian_vecs = np.array(cartesian_vecs).reshape(-1, 3)
    r = np.linalg.norm(cartesian_vecs, axis=1)
    r_non_zero_mask = r != 0
    norm_spherical_vecs = np.zeros((len(cartesian_vecs), 2))
    norm_spherical_vecs[r_non_zero_mask] = norm_cartesian_to_spherical(
        cartesian_vecs[r_non_zero_mask] / r[r_non_zero_mask, None]
    )
    return np.squeeze(np.hstack((norm_spherical_vecs, r[:, None])))


def spherical_to_cartesian(spherical_vecs: npt.ArrayLike) -> npt.NDArray[np.float_]:
    """Calculates cartesian coordinates from spherical coordinates in degree

    Parameters
    ----------
    vec : npt.ArrayLike (N x 3)
        Spherical coordinates in degree in the format [phi, theta, r]

    Returns
    -------
    npt.NDArray[np.int_] (N x 3)
        Cartesian coordinates in the format [x, y, z]
    """
    spherical_vecs = np.array(spherical_vecs).reshape(-1, 3)
    norm_spherical_vecs = norm_spherical_to_cartesian(spherical_vecs[:, :2])
    return np.squeeze(norm_spherical_vecs * spherical_vecs[:, 2][:, None])


def norm_cartesian_to_spherical(
    norm_cartesian_vecs: npt.ArrayLike,
) -> npt.NDArray[np.float_]:
    """Calculates normalized spherical coordinates in degree from normalized cartesian coordinates.

    Parameters
    ----------
    norm_cartesian_vecs : npt.ArrayLike (N x 3)
        Normalized Cartesian coordinates in the format [x, y, z]

    Returns
    -------
    npt.NDArray[np.float_] (N x 2)
        Normalized spherical coordinates in degree in the format [phi, theta]
    """
    x, y, z = np.array(norm_cartesian_vecs).reshape(-1, 3).T
    phi = np.arctan2(y, x) % (2 * np.pi)
    theta = np.arccos(z)

    return np.squeeze(np.column_stack((np.degrees(phi), np.degrees(theta))))


def norm_spherical_to_cartesian(
    norm_spherical_vecs: npt.ArrayLike,
) -> npt.NDArray[np.float_]:
    """Calculates normalized cartesian coordinates from spherical coordinates in degree.

    Parameters
    ----------
    norm_cartesian_vecs : npt.ArrayLike (N x 2)
        Normalized spherical coordinates in degree in the format [phi, theta]

    Returns
    -------
    npt.NDArray[np.float_] (N x 3)
        Normalized Cartesian coordinates in the format [x, y, z]
    """
    norm_spherical_vecs = np.radians(np.array(norm_spherical_vecs).reshape(-1, 2))
    return np.squeeze(
        np.column_stack(
            (
                np.cos(norm_spherical_vecs[:, 0]) * np.sin(norm_spherical_vecs[:, 1]),
                np.sin(norm_spherical_vecs[:, 0]) * np.sin(norm_spherical_vecs[:, 1]),
                np.cos(norm_spherical_vecs[:, 1]),
            )
        )
    )
