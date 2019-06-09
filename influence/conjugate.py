from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import math
import numpy as np
from scipy.optimize import fmin_ncg

def conjugate_gradient(Ax_fn, b,
                       debug_callback=None,
                       avextol=None,
                       maxiter=None):
    """
    Computes the solution to Ax - b = 0 by minimizing the conjugate objective
    f(x) = x^T A x / 2 - b^T x. This does not require evaluating the matrix A
    explicitly, only the matrix vector product Ax.

    :param Ax_fn: A function that return Ax given x.
    :param b: The vector b.
    :param debug_callback: An optional debugging function that reports
                           the current optimization function. Takes two parameters:
                           the current solution and a helper function that
                           evaluates the quadratic and linear parts of the conjugate
                           objective separately.
    :return: The conjugate optimization solution.
    """

    cg_callback = None
    if debug_callback:
        cg_callback = lambda x: debug_callback(x, 0.5 * np.dot(x, Ax_fn(x)), -np.dot(b, x))

    result = fmin_ncg(f=lambda x: 0.5 * np.dot(x, Ax_fn(x)) - np.dot(b, x),
                      x0=np.zeros_like(b),
                      fprime=lambda x: Ax_fn(x) - b,
                      fhess_p=lambda x, p: Ax_fn(p),
                      callback=cg_callback,
                      avextol=avextol,
                      maxiter=maxiter)

    return result
