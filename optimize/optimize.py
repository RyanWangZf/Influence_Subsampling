#__docformat__ = "restructuredtext en"
# ******NOTICE***************
# optimize.py module by Travis E. Oliphant
#
# You may copy and use this module as you see fit with no
# guarantee implied provided you keep this notice in all copies.
# *****END NOTICE************

# A collection of optimization algorithms.  Version 0.5
# CHANGES
#  Added fminbound (July 2001)
#  Added brute (Aug. 2002)
#  Finished line search satisfying strong Wolfe conditions (Mar. 2004)
#  Updated strong Wolfe conditions line search to use
#      cubic-interpolation (Mar. 2004)

from __future__ import division, print_function, absolute_import

import warnings
import sys
import numpy

from numpy import (atleast_1d, eye, mgrid, argmin, zeros, shape, squeeze,
                   vectorize, asarray, sqrt, Inf, asfarray, isinf)

from scipy.optimize.linesearch import line_search_wolfe1,line_search_wolfe2,LineSearchWarning
from scipy import sparse
# from scipy.sparse import linalg as sp_linalg

import pdb

_epsilon = sqrt(numpy.finfo(float).eps)

# standard status messages of optimizers
_status_message = {'success': 'Optimization terminated successfully.',
                   'maxfev': 'Maximum number of function evaluations has '
                              'been exceeded.',
                   'maxiter': 'Maximum number of iterations has been '
                              'exceeded.',
                   'pr_loss': 'Desired error not necessarily achieved due '
                              'to precision loss.'}

class OptimizeResult(dict):
    """ Represents the optimization result.

    Attributes
    ----------
    x : ndarray
        The solution of the optimization.
    success : bool
        Whether or not the optimizer exited successfully.
    status : int
        Termination status of the optimizer. Its value depends on the
        underlying solver. Refer to `message` for details.
    message : str
        Description of the cause of the termination.
    fun, jac, hess: ndarray
        Values of objective function, its Jacobian and its Hessian (if
        available). The Hessians may be approximations, see the documentation
        of the function in question.
    hess_inv : object
        Inverse of the objective function's Hessian; may be an approximation.
        Not available for all solvers. The type of this attribute may be
        either np.ndarray or scipy.sparse.linalg.LinearOperator.
    nfev, njev, nhev : int
        Number of evaluations of the objective functions and of its
        Jacobian and Hessian.
    nit : int
        Number of iterations performed by the optimizer.
    maxcv : float
        The maximum constraint violation.

    Notes
    -----
    There may be additional attributes not listed above depending of the
    specific solver. Since this class is essentially a subclass of dict
    with attribute accessors, one can see which attributes are available
    using the `keys()` method.
    """
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in sorted(self.items())])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())

class OptimizeWarning(UserWarning):
    pass
class _LineSearchError(RuntimeError):
    pass

def _check_unknown_options(unknown_options):
    if unknown_options:
        msg = ", ".join(map(str, unknown_options.keys()))
        # Stack level 4: this is called from _minimize_*, which is
        # called from another function in Scipy. Level 4 is the first
        # level in user code.
        warnings.warn("Unknown solver options: %s" % msg, OptimizeWarning, 4)

def wrap_function(function, args):
    ncalls = [0]
    if function is None:
        return ncalls, None

    def function_wrapper(*wrapper_args):
        ncalls[0] += 1
        return function(*(wrapper_args + args))

    return ncalls, function_wrapper

def approx_fhess_p(x0, p, fprime, epsilon, *args):
    f2 = fprime(*((x0 + epsilon*p,) + args))
    f1 = fprime(*((x0,) + args))
    return (f2 - f1) / epsilon

def _line_search_wolfe12(f, fprime, xk, pk, gfk, old_fval, old_old_fval,
                         **kwargs):
    """
    Same as line_search_wolfe1, but fall back to line_search_wolfe2 if
    suitable step length is not found, and raise an exception if a
    suitable step length is not found.

    Raises
    ------
    _LineSearchError
        If no suitable step size is found

    """

    extra_condition = kwargs.pop('extra_condition', None)

    ret = line_search_wolfe1(f, fprime, xk, pk, gfk,
                             old_fval, old_old_fval,
                             **kwargs)

    if ret[0] is not None and extra_condition is not None:
        xp1 = xk + ret[0] * pk
        if not extra_condition(ret[0], xp1, ret[3], ret[5]):
            # Reject step if extra_condition fails
            ret = (None,)

    if ret[0] is None:
        # line search failed: try different one.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', LineSearchWarning)
            kwargs2 = {}
            for key in ('c1', 'c2', 'amax'):
                if key in kwargs:
                    kwargs2[key] = kwargs[key]
            ret = line_search_wolfe2(f, fprime, xk, pk, gfk,
                                     old_fval, old_old_fval,
                                     extra_condition=extra_condition,
                                     **kwargs2)

    if ret[0] is None:
        raise _LineSearchError()

    return ret

"""Precondtioned Truncated newton methods (Newton Conguage Gradient methods)
"""
def fmin_ncg(f, x0, fprime, fhess_p=None, fhess=None, args=(), avextol=1e-5,
             epsilon=_epsilon, maxiter=None, full_output=0, disp=1, retall=0,
             callback=None, preconditioner = None):
    """
    Unconstrained minimization of a function using the Newton-CG method.

    Parameters
    ----------
    f : callable ``f(x, *args)``
        Objective function to be minimized.
    x0 : ndarray
        Initial guess.
    fprime : callable ``f'(x, *args)``
        Gradient of f.
    fhess_p : callable ``fhess_p(x, p, *args)``, optional
        Function which computes the Hessian of f times an
        arbitrary vector, p.
    fhess : callable ``fhess(x, *args)``, optional
        Function to compute the Hessian matrix of f.
    args : tuple, optional
        Extra arguments passed to f, fprime, fhess_p, and fhess
        (the same set of extra arguments is supplied to all of
        these functions).
    epsilon : float or ndarray, optional
        If fhess is approximated, use this value for the step size.
    callback : callable, optional
        An optional user-supplied function which is called after
        each iteration.  Called as callback(xk), where xk is the
        current parameter vector.
    avextol : float, optional
        Convergence is assumed when the average relative error in
        the minimizer falls below this amount.
    maxiter : int, optional
        Maximum number of iterations to perform.
    full_output : bool, optional
        If True, return the optional outputs.
    disp : bool, optional
        If True, print convergence message.
    retall : bool, optional
        If True, return a list of results at each iteration.
    precontioner: numpy.ndarray, used for preconditioning CG (PCG), 
        it is a one dim array on the M's diagnol indices.

    Returns
    -------
    xopt : ndarray
        Parameters which minimize f, i.e. ``f(xopt) == fopt``.
    fopt : float
        Value of the function at xopt, i.e. ``fopt = f(xopt)``.
    fcalls : int
        Number of function calls made.
    gcalls : int
        Number of gradient calls made.
    hcalls : int
        Number of hessian calls made.
    warnflag : int
        Warnings generated by the algorithm.
        1 : Maximum number of iterations exceeded.
    allvecs : list
        The result at each iteration, if retall is True (see below).

    See also
    --------
    minimize: Interface to minimization algorithms for multivariate
        functions. See the 'Newton-CG' `method` in particular.

    Notes
    -----
    Only one of `fhess_p` or `fhess` need to be given.  If `fhess`
    is provided, then `fhess_p` will be ignored.  If neither `fhess`
    nor `fhess_p` is provided, then the hessian product will be
    approximated using finite differences on `fprime`. `fhess_p`
    must compute the hessian times an arbitrary vector. If it is not
    given, finite-differences on `fprime` are used to compute
    it.

    Newton-CG methods are also called truncated Newton methods. This
    function differs from scipy.optimize.fmin_tnc because

    1. scipy.optimize.fmin_ncg is written purely in python using numpy
        and scipy while scipy.optimize.fmin_tnc calls a C function.
    2. scipy.optimize.fmin_ncg is only for unconstrained minimization
        while scipy.optimize.fmin_tnc is for unconstrained minimization
        or box constrained minimization. (Box constraints give
        lower and upper bounds for each variable separately.)

    References
    ----------
    Wright & Nocedal, 'Numerical Optimization', 1999, pg. 140.

    """
    if preconditioner is None:
        preconditioner = numpy.ones(x0.shape[0])

    opts = {'xtol': avextol,
            'eps': epsilon,
            'maxiter': maxiter,
            'disp': disp,
            'return_all': retall,
            'preconditioner': preconditioner,}

    res = _minimize_newton_pcg(f, x0, args, fprime, fhess, fhess_p,callback=callback, **opts)
    # res = _minimize_newtoncg(f, x0, args, fprime, fhess, fhess_p,callback=callback, **opts)

    if full_output:
        retlist = (res['x'], res['fun'], res['nfev'], res['njev'],
                   res['nhev'], res['status'])
        if retall:
            retlist += (res['allvecs'], )
        return retlist
    else:
        if retall:
            return res['x'], res['allvecs']
        else:
            return res['x']


def _minimize_newton_pcg(fun, x0, args=(), jac=None, hess=None, hessp=None,
                       callback=None, xtol=1e-5, eps=_epsilon, maxiter=None,
                       disp=False, return_all=False,
                       **unknown_options):
    """
    Minimization of scalar function of one or more variables using the
    Newton-CG algorithm.

    Note that the `jac` parameter (Jacobian) is required.

    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    xtol : float
        Average relative error in solution `xopt` acceptable for
        convergence.
    maxiter : int
        Maximum number of iterations to perform.
    eps : float or ndarray
        If `jac` is approximated, use this value for the step size.

    """
    _check_unknown_options(unknown_options)
    if jac is None:
        raise ValueError('Jacobian is required for Newton-CG method')
    f = fun
    fprime = jac
    fhess_p = hessp
    fhess = hess
    avextol = xtol
    epsilon = eps
    retall = return_all

    # add a preconditioner
    M = unknown_options['preconditioner']
    # refer to the mixed method proposed by Chih-Yang et al., 2018 ACML
    # construct a mixed preconditioner
    M = 0.1 * M + 0.9 * numpy.ones_like(M)

    M_inv = sparse.diags(1.0/M).tocsr()
    print("Succeed in getting the inverse of preconditioner M.")

    def terminate(warnflag, msg):
        if disp:
            print(msg)
            print("         Current function value: %f" % old_fval)
            print("         Iterations: %d" % k)
            print("         Function evaluations: %d" % fcalls[0])
            print("         Gradient evaluations: %d" % gcalls[0])
            print("         Hessian evaluations: %d" % hcalls)
        fval = old_fval
        result = OptimizeResult(fun=fval, jac=gfk, nfev=fcalls[0],
                                njev=gcalls[0], nhev=hcalls, status=warnflag,
                                success=(warnflag == 0), message=msg, x=xk,
                                nit=k)
        if retall:
            result['allvecs'] = allvecs
        return result

    x0 = asarray(x0).flatten()
    fcalls, f = wrap_function(f, args)
    gcalls, fprime = wrap_function(fprime, args)
    hcalls = 0
    if maxiter is None:
        maxiter = len(x0)*200
    cg_maxiter = 20*len(x0)

    xtol = len(x0) * avextol
    update = [2 * xtol]
    xk = x0
    if retall:
        allvecs = [xk]
    k = 0
    gfk = None
    old_fval = f(x0)
    old_old_fval = None
    float64eps = numpy.finfo(numpy.float64).eps
    while numpy.add.reduce(numpy.abs(update)) > xtol:
        if k >= maxiter:
            msg = "Warning: " + _status_message['maxiter']
            return terminate(1, msg)
        # Compute a search direction pk by applying the CG method to
        #  del2 f(xk) p = - grad f(xk) starting from 0.
        b = -fprime(xk)
        maggrad = numpy.add.reduce(numpy.abs(b))
        eta = numpy.min([0.5, numpy.sqrt(maggrad)])
        termcond = eta * maggrad
        xsupi = zeros(len(x0), dtype=x0.dtype)
        ri = b
        # preconditioning
        zi = M_inv.dot(ri)
        psupi = zi

        i = 0
        # dri0 = numpy.dot(ri, ri)
        dri0 = numpy.dot(ri,zi)

        if fhess is not None:             # you want to compute hessian once.
            A = fhess(*(xk,) + args)
            hcalls = hcalls + 1

        for k2 in range(cg_maxiter):
            iter_diffs = numpy.add.reduce(numpy.abs(ri))

            if k2 % 10 == 0:
                print("iter {} cg iter {} iter_diff {}".format(k,k2,iter_diffs))

            if iter_diffs <= termcond:
                break
            if fhess is None:
                if fhess_p is None:
                    Ap = approx_fhess_p(xk, psupi, fprime, epsilon)
                else:
                    Ap = fhess_p(xk, psupi, *args)
                    hcalls = hcalls + 1
            else:
                Ap = numpy.dot(A, psupi)
            # check curvature
            Ap = asarray(Ap).squeeze()  # get rid of matrices...
            curv = numpy.dot(psupi, Ap)
            if 0 <= curv <= 3 * float64eps:
                break
            elif curv < 0:
                if (i > 0):
                    break
                else:
                    # fall back to steepest descent direction
                    xsupi = dri0 / (-curv) * b
                    break
            alphai = dri0 / curv
            xsupi = xsupi + alphai * psupi
            ri = ri - alphai * Ap

            zi = M_inv.dot(ri)
            dri1 = numpy.dot(ri,zi)
            betai = dri1 / dri0
            psupi = zi + betai * psupi

            i = i + 1
            dri0 = dri1          # update numpy.dot(ri,ri) for next time.
        else:
            # curvature keeps increasing, bail out
            msg = ("Warning: CG iterations didn't converge.  The Hessian is not "
                   "positive definite.")
            return terminate(3, msg)

        pk = xsupi  # search direction is solution to system.
        gfk = -b    # gradient at xk

        try:
            alphak, fc, gc, old_fval, old_old_fval, gfkp1 = \
                     _line_search_wolfe12(f, fprime, xk, pk, gfk,
                                          old_fval, old_old_fval)
        except _LineSearchError:
            # Line search failed to find a better solution.
            msg = "Warning: " + _status_message['pr_loss']
            return terminate(2, msg)

        update = alphak * pk
        xk = xk + update        # upcast if necessary
        if callback is not None:
            callback(xk)
        if retall:
            allvecs.append(xk)
        k += 1
    else:
        msg = _status_message['success']
        return terminate(0, msg)

def _minimize_newtoncg(fun, x0, args=(), jac=None, hess=None, hessp=None,
                       callback=None, xtol=1e-5, eps=_epsilon, maxiter=None,
                       disp=False, return_all=False,
                       **unknown_options):
    """
    Minimization of scalar function of one or more variables using the
    Newton-CG algorithm.

    Note that the `jac` parameter (Jacobian) is required.

    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    xtol : float
        Average relative error in solution `xopt` acceptable for
        convergence.
    maxiter : int
        Maximum number of iterations to perform.
    eps : float or ndarray
        If `jac` is approximated, use this value for the step size.

    """
    _check_unknown_options(unknown_options)
    if jac is None:
        raise ValueError('Jacobian is required for Newton-CG method')
    f = fun
    fprime = jac
    fhess_p = hessp
    fhess = hess
    avextol = xtol
    epsilon = eps
    retall = return_all

    def terminate(warnflag, msg):
        if disp:
            print(msg)
            print("         Current function value: %f" % old_fval)
            print("         Iterations: %d" % k)
            print("         Function evaluations: %d" % fcalls[0])
            print("         Gradient evaluations: %d" % gcalls[0])
            print("         Hessian evaluations: %d" % hcalls)
        fval = old_fval
        result = OptimizeResult(fun=fval, jac=gfk, nfev=fcalls[0],
                                njev=gcalls[0], nhev=hcalls, status=warnflag,
                                success=(warnflag == 0), message=msg, x=xk,
                                nit=k)
        if retall:
            result['allvecs'] = allvecs
        return result

    x0 = asarray(x0).flatten()
    fcalls, f = wrap_function(f, args)
    gcalls, fprime = wrap_function(fprime, args)
    hcalls = 0
    if maxiter is None:
        maxiter = len(x0)*200
    cg_maxiter = 20*len(x0)

    xtol = len(x0) * avextol
    update = [2 * xtol]
    xk = x0
    if retall:
        allvecs = [xk]
    k = 0
    gfk = None
    old_fval = f(x0)
    old_old_fval = None
    float64eps = numpy.finfo(numpy.float64).eps
    while numpy.add.reduce(numpy.abs(update)) > xtol:
        if k >= maxiter:
            msg = "Warning: " + _status_message['maxiter']
            return terminate(1, msg)
        # Compute a search direction pk by applying the CG method to
        #  del2 f(xk) p = - grad f(xk) starting from 0.
        b = -fprime(xk)
        maggrad = numpy.add.reduce(numpy.abs(b))
        eta = numpy.min([0.5, numpy.sqrt(maggrad)])
        termcond = eta * maggrad
        xsupi = zeros(len(x0), dtype=x0.dtype)
        ri = -b
        psupi = -ri
        i = 0
        dri0 = numpy.dot(ri, ri)

        if fhess is not None:             # you want to compute hessian once.
            A = fhess(*(xk,) + args)
            hcalls = hcalls + 1

        for k2 in range(cg_maxiter):

            iter_diffs = numpy.add.reduce(numpy.abs(ri))

            if k2 % 10 == 0:
                print("iter {} cg iter {} iter_diff {}".format(k,k2,iter_diffs))

            if iter_diffs <= termcond:
                break
            if fhess is None:
                if fhess_p is None:
                    Ap = approx_fhess_p(xk, psupi, fprime, epsilon)
                else:
                    Ap = fhess_p(xk, psupi, *args)
                    hcalls = hcalls + 1
            else:
                Ap = numpy.dot(A, psupi)
            # check curvature
            Ap = asarray(Ap).squeeze()  # get rid of matrices...
            curv = numpy.dot(psupi, Ap)
            if 0 <= curv <= 3 * float64eps:
                break
            elif curv < 0:
                if (i > 0):
                    break
                else:
                    # fall back to steepest descent direction
                    xsupi = dri0 / (-curv) * b
                    break
            alphai = dri0 / curv
            xsupi = xsupi + alphai * psupi
            ri = ri + alphai * Ap
            dri1 = numpy.dot(ri, ri)
            betai = dri1 / dri0
            psupi = -ri + betai * psupi
            i = i + 1
            dri0 = dri1          # update numpy.dot(ri,ri) for next time.
        else:
            # curvature keeps increasing, bail out
            msg = ("Warning: CG iterations didn't converge.  The Hessian is not "
                   "positive definite.")
            return terminate(3, msg)

        pk = xsupi  # search direction is solution to system.
        gfk = -b    # gradient at xk

        try:
            alphak, fc, gc, old_fval, old_old_fval, gfkp1 = \
                     _line_search_wolfe12(f, fprime, xk, pk, gfk,
                                          old_fval, old_old_fval)
        except _LineSearchError:
            # Line search failed to find a better solution.
            msg = "Warning: " + _status_message['pr_loss']
            return terminate(2, msg)

        update = alphak * pk
        xk = xk + update        # upcast if necessary
        if callback is not None:
            callback(xk)
        if retall:
            allvecs.append(xk)
        k += 1
    else:
        msg = _status_message['success']
        return terminate(0, msg)