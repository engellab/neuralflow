# -*- coding: utf-8 -*-
"""
Tests PDE_Solve class. Specify test and choose FD or SEM method.
Plots: Error and execution time vs. N and fits curves (log-log plot).
"""

from neuralflow.PDE_Solve import PDESolve
from pkg_resources import working_set
import EV_problems_library as EV_lib
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import time
import numpy as np

GPU_support = any([pkg.key.startswith('cupy') for pkg in working_set])


# Dictionary of availible tests
TestDict = {
    'Sine': EV_lib.SineTest,
    'Sine2': EV_lib.Sine2Test,
    'Cosine': EV_lib.CosineTest,
    'MixedCosineSine': EV_lib.MixedCosineSineTest,
    'SquarePeqMixed': EV_lib.SquarePeqMixedTest,
    'Bessel': EV_lib.BesselTest,
}

# Compare L2 norms of exact and approximate solutions


def L2Error(sol1, sol2):
    return np.linalg.norm(sol1-sol2, axis=0)/np.linalg.norm(sol1, axis=0)

# For curve fitting purpose


def PowerLawFunction(x, b, c):
    return b*x**c


def ErrorAndExecPlot(Nseq, L2Err, ExecTime, fit1, fit2, f2start, f1end):
    """Plot the results
     """
    # Plot error and error fit (if succsessful)
    fig, ax1 = plt.gcf(), plt.gca()
    ax1.loglog(Nseq, L2Err, 'bo', label='L2 Error')
    if fit1 is not None:
        ax1.loglog(
            Nseq[:f1end], PowerLawFunction(Nseq[:f1end], *fit1), '-b',
            label='fit: E=%5.3f*N^%5.3f' % tuple(fit1)
        )

    ax1.set_xlabel('N, number of points')
    ax1.set_ylabel('L2error', color='b')
    ax1.tick_params('y', colors='b')
    ax1.legend(loc='upper center')

    # Plot exec. time and it's fit (if succsessful)
    ax2 = ax1.twinx()
    ax2.loglog(Nseq, ExecTime, 'gx', label='Execution time')
    if fit2 is not None:
        ax2.loglog(
            Nseq[f2start:], PowerLawFunction(Nseq[f2start:], *fit2), '-g',
            label='fit: T=%5.3f*N^%5.3f' % tuple(fit2)
        )

    ax2.set_ylabel('Execution Time', color='r')
    ax2.tick_params('y', colors='r')
    ax2.legend(loc='lower center')
    fig.tight_layout()
    plt.show()


def PerformTest(Ne, Np, Nv, name, with_cuda):
    """ Solve EV problem and compute L2 error with the exact solution
    """

    # obtain interval bounds and BC:
    BC, xbegin, xend = TestDict[name](Nv, None, 'domain_and_bc')

    # Initialize class instance
    solver = PDESolve(
        xbegin, xend, Np, Ne, BoundCond=BC, Nv=Nv, with_cuda=with_cuda
    )

    # Obtain peq, boundary conditions, exact solution
    peq, q, w, EVal_exact, EVec_exact = TestDict[name](
        Nv, solver.grid.x_d, 'exact_solution'
    )

    device = 'GPU' if with_cuda else 'CPU'
    # Solve and scale eigenvectors
    if with_cuda:
        # Need to transfer inputs to GPU
        import cupy as cp
        peq = cp.asarray(peq, dtype='float64')
        if q is not None:
            q = cp.asarray(q, dtype='float64')
        w = cp.asarray(w, dtype='float64')
    start = time.time()
    EVal, EVec = solver.solve_EV(
        peq=peq, q=q, w=w, mode='normal', device=device
    )
    end = time.time()
    if with_cuda:
        # Transfer outputs back to CPU
        EVal = cp.asnumpy(EVal)
        EVec = cp.asnumpy(EVec)
    ExecTime = end-start
    EVec /= np.max(np.abs(EVec), 0)

    # Calculate error based on eigenvectors and error based on eigenvalues:
    L2ErrEVec = np.minimum(L2Error(EVec_exact, EVec),
                           L2Error(EVec_exact, -EVec))
    L2ErrEVal = np.zeros(Nv)
    for j in range(Nv):
        L2ErrEVal[j] = (
            np.abs(EVal_exact[j]-EVal[j])
            if np.abs(EVal_exact[j]) < 10**(-3) else
            np.abs(EVal_exact[j]-EVal[j])/np.abs(EVal_exact[j])
        )

    return ExecTime, L2ErrEVec, L2ErrEVal


if __name__ == '__main__':
    # Test parameters, feel free to change
    # Try other tests (see TestDict in Lines 32-39 for the availible tests)
    TestName = 'Bessel'

    with_cuda = False
    if with_cuda and not GPU_support:
        raise Exception(
            'Cupy package not found. Install cupy package on cuda-enabled '
            'platform and rerun this'
        )

    Nv = 10  # Number of eigenvectors/eigenvalues for error calculation
    Np = 6  # Degree of a single element

    # Specify different Ne (number of elements) for the test
    Ne_seq = np.array(
        [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 30, 40, 50, 75, 100,
         150, 200, 250, 300, 1000],
        dtype='int'
    )

    # Total number of points:
    N_seq = (Np-1) * Ne_seq + 1

    # Allocate Errors based on eigenvectors and eigenvalues, measure execution
    # times
    L2ErrEVec = np.zeros((len(N_seq), Nv))
    L2ErrEVal = np.zeros((len(N_seq), Nv))
    ExecTime = np.zeros(len(N_seq))

    # Now calculaete and compare:
    for i, Ne in enumerate(Ne_seq):
        ExecTime[i], L2ErrEVec[i, :], L2ErrEVal[i, :] = PerformTest(
            Ne, Np, Nv, TestName, with_cuda
        )
        print(f'Ne = {Ne} calculated, execution time = {ExecTime[i]:.5f} s')

    # Curve fit with power function (or linear function in log-log)
    ErrToFit = np.mean(L2ErrEVec[:, 1:], 1)
    endpoint = np.where(ErrToFit > 10**-10)
    if len(endpoint[0]) == 0:
        c1 = None
        endpoint = ([[-1]])
    else:
        c1, pc1 = curve_fit(
            lambda x, a, b: a+b*x,
            np.log(np.array(N_seq)[endpoint]),
            np.log(ErrToFit[endpoint]), bounds=([0, -100], [100, 0])
        )
        # convert coefficient from log to linear for plotting
        c1[0] = np.exp(c1[0])

    startpoint = np.where(ExecTime > 0.05)
    if len(startpoint[0]) == 0:
        c2 = None
        startpoint = ([[-1]])
    else:
        c2, pc2 = curve_fit(
            lambda x, a, b: a+b*x,
            np.log(np.array(N_seq)[startpoint]),  # fit log(exectime))
            np.log(ExecTime[startpoint]),
            bounds=([-100, 0], [0, 10])
        )
        c2[0] = np.exp(c2[0])

    # Plot the results
    plt.clf()
    ErrorAndExecPlot(
        N_seq, ErrToFit, ExecTime, c1, c2, startpoint[0][0], endpoint[0][-1]
    )
