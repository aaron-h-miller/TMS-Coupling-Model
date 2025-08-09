import numpy as np
from numpy.typing import NDArray


def correlation_max(a, b, sample_index) -> tuple[int, float, NDArray, NDArray, NDArray]:
    """
    Calculates the correlation (np.correlate(a, b, 'full') between a and b, and returns the index of
    maximum correlation, the maximum correlation, and the correlation curve itself. a and b need not be the same length.
    Also returns a and b after length equalization, padded with zeros on the end of each.
    Parameters
    ----------
    a : NDArray
        Normalized first curve
    b : NDArray
        Normalized second curve
    sample_index : int
        Index to which to sample curved for correlation. (1 = all values, 10 = every ten values i.e 1 order of magnitude)
    Returns
    -------
    ind_max_corr : int
        Index at which correlation between a and b is maximum
    max_corr : float
        Maximum correlation between a and b
    correlation : NDArray
        Correlation curve between a and b for mode "full"
    a_eq : NDArray
        a after length equalization
    b_eq : NDArray
        b after length equalization
    """
    length_diff = len(a) - len(b)
    if length_diff >= 0:
        b = np.concatenate((b, np.zeros(abs(length_diff))))
    else:
        a = np.concatenate((a, np.zeros(abs(length_diff))))
    # correlation and shift value
    correlation = np.correlate(a[::sample_index], b[::sample_index], mode="Full")  # down-sampled
    max_corr = np.max(correlation)
    ind_max_corr = np.where(max_corr == correlation)[0][0]
    return ind_max_corr, max_corr, correlation, a, b


def norm_shift(test_curve, reference_curve):
    """
    Shifts test_curve and reference_curve such that they are aligned by their index of maximum cross correlation.
    Lengths of original arrays are not conserved due to padding.

    Parameters
    ----------
    :param test_curve: NDArray
        First curve to compare.
    :param reference_curve: NDArray
        Reference curve to compare against.

    Returns
    -------
    :returns: tuple(test_curve_shifted, reference_curve_shifted)
        test_curve_shifted: NDArray - Test curve padded and shifted
        reference_curve_shifted: NDArray - Reference curved padded and shifted

    """
    norm_factor_r = np.max(test_curve)
    norm_factor_i_wave = np.max(reference_curve)
    reference_curve = reference_curve / norm_factor_i_wave
    if np.isclose(norm_factor_r, 0):
        test_curve = np.zeros(len(test_curve))
    else:
        test_curve = test_curve / norm_factor_r
    ind_max_corr, max_corr, correlation, test_curve, reference_curve = correlation_max(test_curve, reference_curve, 1)
    correlation_len = len(correlation)
    correlation_center = int(correlation_len / 2)
    # shift curves
    padding_number = int(abs(correlation_center - ind_max_corr))
    if (correlation_center - ind_max_corr) >= 0:  # max is before center, pad left of first array
        test_curve_shifted = np.concatenate((np.zeros(padding_number), test_curve))
        reference_curve_shifted = np.concatenate((reference_curve, np.zeros(padding_number)))
    else:  # max is after center, pad right of first array
        test_curve_shifted = np.concatenate((test_curve, np.zeros(padding_number)))
        reference_curve_shifted = np.concatenate((np.zeros(padding_number), reference_curve))

    return test_curve_shifted, reference_curve_shifted
