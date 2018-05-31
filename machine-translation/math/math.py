import numpy as np


def low_rank_approx(SVD=None, A=None, r=1):
    """
    Computes an r-rank approximation of a matrix
    given the component u, s, and v of it's SVD
    Requires: numpy
    """
    if not SVD:
        SVD = np.linalg.svd(A, full_matrices=False)
    u, s, v = SVD
    Ar = np.zeros((len(u), len(v)))
    for i in xrange(r):
        Ar += s[i] * np.outer(u.T[i], v[i])
    return Ar

if __name__ == "__main__":
    """
    Test: visualize an r-rank approximation of `lena`
    for increasing values of r
    Requires: scipy, matplotlib
    """
    x = np.random.rand(10,10)
    u, s, v = np.linalg.svd(x, full_matrices=False)
    i = 1
    print x[0]
    while i < 10:
        y = low_rank_approx((u, s, v), r=i)
        print y[0]
        i += 1