r"""Various tools to calculate the entropy either from covariance or samples"""
import numpy as np
import scipy.special
from scipy import spatial
import math


def analytic_entropy_normal(covmat):
    r"""analytically find the entropy of a normal distribution"""
    try:
        dim = float(covmat.shape[0])
        detval = np.linalg.det(covmat)
    # otherwise 1-D, or scalar
    except (IndexError, AttributeError):
        dim = 1.
        detval = covmat

    # factor is 2 pi e
    return 0.5 * np.log( 17.0794684453**dim * detval )


def nearest_1dfast(samples, orig_sort=True, kind="quicksort"):
    r"""Calcluate the 1D nearest neighbors using sort

    orig_sort can be used to ensure the nearest neighbor distances match the
    indices of the inputs. If only the distribution of distances is needed,
    ignore this.
    """
    sort_ind = np.argsort(samples, kind=kind)
    sorted_samp = samples[sort_ind]

    deltas = np.abs(np.roll(sorted_samp, 1) - sorted_samp)
    rdeltas = np.roll(deltas, -1)

    outvec = np.array([min(pair) for pair in zip(deltas, rdeltas)])

    if orig_sort:
        unsort_ind = np.zeros_like(sort_ind)
        unsort_ind[sort_ind] = range(len(sort_ind))
        outvec = outvec[unsort_ind]

    return outvec


def nearest_vector_1d(samples):
    r"""find the 1D separation between nearest neighbors using brute force
    here `samples` is the vector of positions"""
    outvec = np.zeros_like(samples)
    for (sample, idx) in zip(samples, range(len(samples))):
        delta_arr = np.abs(samples - sample)
        outvec[idx] = np.min(delta_arr[delta_arr != 0])

    return outvec


def sample_entropy_nn(samples):
    r"""find the entropy of the sample using nearest neighbors
    Citation: Nearest Neighbor Estimates of Entropy
    Singh et al. Am. J. Mathmetical and Management Science

    Use kdtrees or sort to find the nearest neigh. dist.

    >>> dim = 2
    >>> nsamp = 50000
    >>> mean = np.zeros(dim)

    >>> covmat = np.array([[60., 5.4],[5.4, 10.7]])
    >>> samples = np.random.multivariate_normal(mean, covmat, nsamp)
    >>> result = sample_entropy_nn(samples)/analytic_entropy_normal(covmat)
    >>> np.testing.assert_almost_equal(result, 1., decimal=2)

    >>> variance = 11.
    >>> samples = np.random.normal(0, np.sqrt(variance), nsamp)
    >>> result = sample_entropy_nn(samples)/analytic_entropy_normal(variance)
    >>> np.testing.assert_almost_equal(result, 1., decimal=2)

    """
    nsamp = float(samples.shape[0])
    try:
        ndim = float(samples.shape[1])
    except IndexError:
        ndim = 1.

    if ndim > 1:
        tree = spatial.KDTree(samples)
        # need to find nearest 2 because nearest is point itself
        nn_dist = tree.query(samples, k=2)[0][:, 1]
    else:
        nn_dist = nearest_1dfast(samples)

    ln_nn_dist = np.log(nn_dist)

    factor = math.pi ** (ndim / 2.) / scipy.special.gamma(ndim / 2. + 1)
    result = ndim * np.sum(ln_nn_dist) / nsamp
    result += np.log(factor) + np.log(float(nsamp - 1))
    result += 0.5772156649

    return result


if __name__ == "__main__":
    import doctest

    # run some tests
    OPTIONFLAGS = (doctest.ELLIPSIS |
                   doctest.NORMALIZE_WHITESPACE)
    doctest.testmod(optionflags=OPTIONFLAGS)