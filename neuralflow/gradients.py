# -*- coding: utf-8 -*-
"""Grads class for gradients and loglikelihood calculation. Include both CPU
   and GPU functions."""

import numpy as np
from scipy import linalg
from neuralflow import c_get_gamma
from neuralflow.PDE_Solve import PDESolve
import math
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)


class Grads:

    """Grad class for loglikelihoods and gradient calculations.


    Parameters
    ----------
    pde_solve_params : dict
        Parameters for solveing FP equation:
            xbegin : float
                The left boundary of the latent state. The default is -1.
            xend : float
                The right boundary of the latent state. The default is 1.
            Np : int
                The degree of Langrange interpolation polynomial, also the
                number of grid points at each element. The default is 8.
            Ne : int
                Number of SEM elements. The default is 64.
            BoundCond : dict
                Boundary conditions. Only include if you want to enforce
                unusual boundary conditions. Otherwise, use boundary_mode
                to specify boundary conditions.
    boundary_mode : ENUM('absorbing', 'reflecting'), optional
        Boundary mode. The default is 'absorbing'.
    grad_list : list, optional
        List of paramters for gradient calculation. Can include 'F', 'F0',
        'D', 'C', 'Fr'. The default is [].
    num_neuron : int, optional
        Number of neurons in the model/data. The default is 1.
    with_trial_end : bool, optional
        Whether to take into account trial end time. The default is True.
    device : ENUM('CPU', 'GPU'), optional
        Device for the computations. The default is 'CPU'. For GPU
        optimization, the platform has to be cuda-enabled, and cupy package
        has to be installed. The default is 'CPU'.
    """

    def __init__(self, pde_solve_params, boundary_mode='absorbing',
                 grad_list=[], num_neuron=1, with_trial_end=True,
                 device='CPU'):
        """

        Public methods
        --------------
        get_grad_data

        """

        self.num_neuron = num_neuron
        self.with_trial_end = with_trial_end

        # Might need to change in place, so deepcopy
        pde_solve_params = deepcopy(pde_solve_params)
        grad_list = deepcopy(grad_list)

        if 'BoundCond' not in pde_solve_params:
            if boundary_mode == 'absorbing':
                pde_solve_params['BoundCond'] = {
                    'leftB': 'Dirichlet', 'rightB': 'Dirichlet'
                }
            elif boundary_mode == 'reflecting':
                pde_solve_params['BoundCond'] = {
                    'leftB': 'Neumann', 'rightB': 'Neumann'
                }

        if boundary_mode == 'absorbing':
            if (
                pde_solve_params['BoundCond']['leftB'] != 'Dirichlet' or
                pde_solve_params['BoundCond']['rightB'] != 'Dirichlet'
            ):
                logger.warning(
                    'Pde solver boundary conditions are inconsistent with '
                    'boundary mode. Absorbing boundary assumes Dirichlet BCs '
                    'on both ends'
                )
        elif boundary_mode == 'reflecting':
            if (
                pde_solve_params['BoundCond']['leftB'] != 'Neumann' or
                pde_solve_params['BoundCond']['rightB'] != 'Neumann'
            ):
                logger.warning(
                    'Pde solver boundary conditions are inconsistent with '
                    'boundary mode. Reflecting boundary assumes Neumann BCs '
                    'on both ends'
                )

        with_cuda = True if device == 'GPU' else False
        self.pde_solver = PDESolve(**pde_solve_params, with_cuda=with_cuda)
        self.boundary_mode = boundary_mode
        self.grad_list = grad_list
        self.device = device

        if device == 'CPU':
            # cython function for elementwise aggregation
            self._G0 = c_get_gamma.G0_d
            self._G1 = c_get_gamma.G1_d
        else:
            # Custom cude kernels for GPU
            import neuralflow.base_cuda as cuda
            self.cuda = cuda
            self.cuda.import_custom_functions()

    # @profile
    def get_grad_data(
            self, data, model, model_num=0, mode='loglik', EV_solution=None
    ):
        """Calculates loglikelihood and/or gradients


        Parameters
        ----------
        data : data : numpy array (num_trials, 2), dtype = np.ndarray.
            Data in ISI format. See spike_data class for details.
        model: model
            An instance of neuralflow.model.
        model_num : int, optional
            Model number to be used. The default is 0.
        mode : ENUM('loglik', 'gradient')
            Whether to only compute loglik, or also a gradient.
        EV_solution : dicionary
            Dictionary with the solution of the eigenvector-eigenvalue (EV)
            problem. The format is {'lQ': lQ, 'QxOrig': QxOrig, 'Qx': Qx,
            'lQd':lQd, 'Qd': Qd}. If not provided, will be calculated.
            The default is None.

        Returns
        -------
        results_all : dictionary
            Dictionary with the results. Possible entries depends on the
            grad_list, and can include 'loglik', 'F', 'F0', 'D', 'Fr', 'C'.
        """

        # number of trials
        num_trials = len(data)

        # Initialize results_all, which will accumulate the gradients and
        # logliklihoods over trials
        results_all = {}
        results_all['loglik'] = 0
        for grad in self.grad_list:
            results_all[grad] = 0

        # Solve EV problaem if the solution is not provided,
        if EV_solution is None:
            EV_solution = self._get_EV_solution(model, model_num)

        if self.device == 'CPU':
            for iSeq in range(num_trials):
                # Get ISI and the corresponding neuron ids
                data_trial = data[iSeq]
                # Compute loglik and gradients
                results = self._get_grad_seq(
                    data_trial, model, model_num, mode, EV_solution
                )
                # Sum the results across trials
                if mode != 'loglik':
                    for grad in self.grad_list:
                        results_all[grad] += results[grad]
                results_all['loglik'] += results['loglik']
        else:
            results_list = []
            # Cuda streams enable parallel computation of gradients on
            # different trials using map-reduce technique.
            if len(self.cuda.streams_) < num_trials:
                self.cuda._update_streams(num_trials)

            for iSeq in range(num_trials):
                stream = self.cuda.streams_[iSeq]
                data_trial = data[iSeq]
                with stream:
                    results = self._get_grad_seq_gpu(
                        data_trial, model, model_num, mode, EV_solution
                    )
                    results_list.append(results)

            self.cuda._free_streams_memory()

            # Sum up contributions from different trials. Do this with the
            # default stream that implicitly syncs the streams
            for i in range(num_trials):
                if mode != 'loglik':
                    for grad in self.grad_list:
                        results_all[grad] += results_list[i][grad]
                results_all['loglik'] += results_list[i]['loglik']

        if mode == 'loglik':
            return results_all['loglik']
        return results_all

    def _get_EV_solution(self, model, model_num):
        """Solve Eigenvector-eigenvalue problem. Needed for likelihood and
        gradients calculation
        """
        peq, _, D, fr = model.get_params(model_num, self.device)
        if self.device == 'CPU':
            fr_cum = np.sum(fr, axis=1)
        else:
            fr_cum = self.cuda.cp.sum(fr, axis=1)
        lQ, QxOrig, Qx, lQd, Qd = self.pde_solver.solve_EV(
            peq, D, None, peq, 'hdark', fr_cum, self.device
        )
        return {'lQ': lQ, 'QxOrig': QxOrig, 'Qx': Qx, 'lQd': lQd, 'Qd': Qd}

    def _get_grad_seq(self, data, model, model_num, mode, EV_solution):
        """This function calculates loglik/gradients for a single trial on CPU
        """

        # Extract ISI and neuron_id for convenience
        seq, nid = data

        # Sequence length
        S_Total = len(seq)

        # Model parameters
        peq, rho0, D, fr = model.get_params_for_grad(model_num, 'CPU')

        # Nv is the dimensionality of H/H0 basis. Usually N - 2
        Nv = self.pde_solver.Nv

        # Number of neurons
        num_neuron = self.num_neuron

        # Whether or not we have absorption in the end of the trial
        if self.boundary_mode == 'absorbing':
            with_absorption = True
        else:
            with_absorption = False

        # take grid from pde_solver
        grid = self.pde_solver.grid

        # Transformation from SEM to H basis
        Qxd = EV_solution["Qx"].dot(EV_solution["Qd"])

        # Transformation from H0 to H basis
        Qd = EV_solution["Qd"]

        # Get the spike operator in the Hdark basis.
        sp_mat = np.zeros((Nv, Nv, num_neuron), dtype=peq.dtype)
        for i in range(num_neuron):
            sp_mat[:, :, i] = (Qxd.T*fr[:, i] * grid.w_d).dot(Qxd)

        # Initialize the atemp and btemp for the forward and backwards passes.
        atemp = Qxd.T.dot(grid.w_d * rho0)
        btemp = Qxd.T.dot(grid.w_d * np.sqrt(peq))

        # Normalization coefficients
        if with_absorption:
            anorm = np.zeros(S_Total + 3, dtype=peq.dtype)
        else:
            anorm = np.zeros(S_Total + 2, dtype=peq.dtype)
        anorm[0] = linalg.norm(atemp)
        atemp /= anorm[0]

        # Store alphas for gradient calculation
        alpha = np.zeros((Nv, anorm.size), dtype=peq.dtype)
        alpha[:, 0] = atemp

        # precalculate exp(-lambda*dt), which is the matrix that propagates
        # probability density with a Fokker-Planck equation for each ISI dt
        prop_mat = np.exp(np.outer(-EV_solution["lQd"], seq))

        # Define the matrix of absorption operator
        if with_absorption:
            abs_mat = Qd.T.dot(np.diag(EV_solution["lQ"])).dot(Qd)

        # Calcuate the alpha vectors (forward pass)
        for i in range(1, S_Total + 1):
            # Propagate in latent space
            atemp *= prop_mat[:, i-1]
            # Spike observation
            if i != S_Total or not self.with_trial_end:
                atemp = atemp.dot(sp_mat[:, :, nid[i-1]])
            # Calculate l2 norm
            # np.linalg.norm(atemp) is faster than np.sqrt(np.sum(atemp**2))
            anorm[i] = np.linalg.norm(atemp)
            # Normalize alpha
            atemp /= anorm[i]
            # save the current alpha vector
            alpha[:, i] = atemp

        # Apply absorption operator
        if with_absorption:
            atemp = atemp.dot(abs_mat)
            anorm[S_Total + 1] = np.sqrt(np.sum(atemp**2))
            atemp /= anorm[S_Total + 1]
            alpha[:, S_Total + 1] = atemp

        # The last anorm coefficient is the product of alpha_N and beta_N
        anorm[-1] = btemp.dot(atemp)
        btemp /= anorm[-1]

        # compute negative log-likelihood
        ll = -np.sum(np.log(anorm))
        if mode == 'likelihood':
            return {'loglik': ll}

        G = np.zeros((Nv, Nv), dtype=peq.dtype)
        if 'Fr' in self.grad_list or 'C' in self.grad_list:
            # Each neuron has it's own G-matrix
            GI1 = np.zeros((Nv, Nv, num_neuron), dtype=peq.dtype)

        # Backward pass also starts with absorption
        if with_absorption:
            btemp /= anorm[S_Total+1]
            Gabs = np.outer(alpha[:, i], btemp)
            btemp = abs_mat.dot(btemp)

        # Backwards pass
        for i in reversed(range(S_Total)):
            dt = seq[i]
            tempExp = prop_mat[:, i]

            # Scaling
            btemp /= anorm[i + 1]

            if i != S_Total - 1 or not self.with_trial_end:
                if 'Fr' in self.grad_list or 'C' in self.grad_list:
                    # Update G-function for Fr
                    self._G1(GI1, Nv, tempExp, alpha, btemp, i, nid[i])
                # Spike emission
                btemp = sp_mat[:, :, nid[i]].dot(btemp)
            # G function
            self._G0(G, Nv, EV_solution["lQd"], tempExp, alpha, btemp, dt, i)
            # Propagation in latent space
            btemp *= tempExp

        if 'Fr' in self.grad_list or 'C' in self.grad_list:
            GI10 = np.zeros_like(GI1)
            for i in range(self.num_neuron):
                GI10[:, :, i] = Qd.dot(GI1[:, :, i]).dot(Qd.T)
            GI20 = Qd.dot(G).dot(Qd.T)

        # add contribution to G from absorption_event
        if with_absorption:
            G -= Gabs

        # Transform G to HO basis
        G0 = Qd.dot(G).dot(Qd.T)
        QxOrig = EV_solution["QxOrig"]
        Qxdx = grid.dmat_d.dot(QxOrig)

        results = {'loglik': ll}

        # Calculate each of the required gradients
        if 'D' in self.grad_list:
            d2Phi = np.sum(Qxdx * Qxdx.dot(G0), 1)
            results['D'] = np.sum(grid.w_d * peq * d2Phi)
        if 'F' in self.grad_list:
            dPhi = (
                np.sum(QxOrig * Qxdx.dot(G0), 1) +
                np.sum(Qxdx * QxOrig.dot(G0), 1)
            )
            # boundary term depends on stationary assumption
            if not model.non_equilibrium:
                boundary_term = grid.Integrate(
                    peq - 0.5 * Qxd.dot(atemp / anorm[-1] + btemp / anorm[0])
                    * np.sqrt(peq)
                )
            else:
                boundary_term = 0.5 * grid.Integrate(
                    Qxd.dot(btemp / anorm[0]) * rho0 -
                    Qxd.dot(atemp / anorm[-1]) * np.sqrt(peq)
                )
            results['F'] = -0.5 * D * peq * dPhi - boundary_term
        if 'F0' in self.grad_list:
            results['F0'] = (
                - grid.Integrate(
                    rho0 * np.sqrt(peq) - rho0 * Qxd.dot(btemp / anorm[0])
                )
            )
        if 'Fr' in self.grad_list or 'C' in self.grad_list:
            Phi = np.sum(QxOrig * QxOrig.dot(GI20), 1)
            PhiY = np.empty((grid.N, num_neuron), dtype=peq.dtype)
            for i in range(num_neuron):
                PhiY[:, i] = (
                    peq * (Phi - np.sum(QxOrig * QxOrig.dot(GI10[:, :, i]), 1))
                )
            gradFr = np.empty((grid.N, num_neuron), dtype=peq.dtype)
            gradC = np.empty(num_neuron, dtype=peq.dtype)
            for i in range(num_neuron):
                C2 = grid.Integrate(fr[:, i]*PhiY[:, i])
                gradFr[:, i] = C2[-1] - C2
                gradC[i] = C2[-1]
        if 'Fr' in self.grad_list:
            results['Fr'] = gradFr
        if 'C' in self.grad_list:
            results['C'] = gradC / fr[0, :]
        return results

#    @profile
    def _get_grad_seq_gpu(self, data, model, model_num, mode, EV_solution):
        """This function calculates loglik/gradients for a single trial on GPU
        """

        # Extract ISI and neuron_id for convenience
        seq, nid = data

        # Sequence length
        S_Total = seq.size

        # Model parameters
        peq, rho0, D, fr = model.get_params_for_grad(model_num, 'GPU')

        # Nv is the dimensionality of H/H0 basis. Usually N - 2
        Nv = self.pde_solver.Nv

        # Number of neurons
        num_neuron = self.num_neuron

        # Number of threads/blocks needed for running custom CUDA kernels
        threads = self.cuda.num_threads_
        blocks = math.ceil(Nv/threads)

        # Whether or not we have absorption in the end of the trial
        if self.boundary_mode == 'absorbing':
            with_absorption = True
        else:
            with_absorption = False

        # take grid from pde_solver
        grid = self.pde_solver.grid

        # Transformation from SEM to H basis
        Qxd = self.cuda.cp.empty((grid.N, Nv), dtype='float64', order='C')
        self.cuda.cp.dot(EV_solution["Qx"], EV_solution["Qd"], out=Qxd)

        # Transformation from H0 to H basis
        Qd = EV_solution["Qd"]

        # Get the spike operator in the Hdark basis.
        sp_mat = self.cuda.cp.empty(
            (num_neuron, Nv, Nv), dtype=peq.dtype, order='C'
        )
        for i in range(num_neuron):
            self.cuda.cp.dot(
                Qxd.T*(fr[:, i] * grid.cuda_var.w_d), Qxd, out=sp_mat[i, :, :]
            )

        # Allocate and initialize forward pass and backward pass. For backward
        # pass we don't need to remember the whole chain
        btemp = self.cuda.cp.empty(Nv, dtype=peq.dtype, order='C')
        self.cuda.cp.dot(
            Qxd.T, grid.cuda_var.w_d * self.cuda.cp.sqrt(peq), out=btemp
        )

        # Normalization coefficients
        if with_absorption:
            anorm = self.cuda.cp.empty(S_Total + 3, dtype=peq.dtype, order='C')
        else:
            anorm = self.cuda.cp.empty(S_Total + 2, dtype=peq.dtype, order='C')

        # Store alphas for gradient calculation
        alpha = self.cuda.cp.empty(
            (anorm.size, Nv), dtype=peq.dtype, order='C')
        self.cuda.cp.dot(Qxd.T, grid.cuda_var.w_d*rho0, out=alpha[0, :])

        # Normalization
        anorm[0] = self.cuda.cp.linalg.norm(alpha[0, :])
        alpha[0, :] /= anorm[0]

        # precalculate exp(-lambda*dt), which is the matrix that propagates
        # probability density with a Fokker-Planck equation for each ISI dt
        prop_mat = self.cuda.cp.empty(
            (S_Total, Nv), dtype=peq.dtype, order='C')
        self.cuda.cp.outer(seq, EV_solution["lQd"], out=prop_mat)
        prop_mat = self.cuda.cp.exp(-prop_mat)

        # Define the matrix of absorption operator
        if with_absorption:
            abs_mat = self.cuda.cp.empty((Nv, Nv), dtype=peq.dtype, order='C')
            self.cuda.cp.dot(
                Qd.T, self.cuda.cp.diag(EV_solution["lQ"]), out=abs_mat
            )
            abs_mat = abs_mat.dot(Qd)

        # Forward pass
        for i in range(1, S_Total + 1):
            # Propagate in latent space
            self.cuda.cp.multiply(
                alpha[i-1, :], prop_mat[i-1, :], out=alpha[i, :]
            )
            # Spike observation
            if i != S_Total or not self.with_trial_end:
                alpha[i, :] = alpha[i, :].dot(sp_mat[nid[i-1], :, :])
            # Normalization
            anorm[i] = self.cuda.cp.linalg.norm(alpha[i, :])
            alpha[i, :] = alpha[i, :] / anorm[i]

        # Apply absorption operator
        if with_absorption:
            self.cuda.cp.dot(
                alpha[S_Total, :], abs_mat, out=alpha[S_Total+1, :]
            )
            anorm[S_Total+1] = self.cuda.cp.linalg.norm(alpha[S_Total+1, :])
            alpha[S_Total+1, :] /= anorm[S_Total+1]

        # The last anorm coefficient is the product of alpha_N and beta_N
        anorm[-1] = btemp.dot(alpha[S_Total+int(with_absorption), :])
        btemp /= anorm[-1]

        # compute negative log-likelihood
        ll = -self.cuda.cp.sum(self.cuda.cp.log(anorm))

        if mode == 'likelihood':
            return {'loglik': ll}

        # For gradient calculation allocate G and GI1
        G = self.cuda.cp.zeros((Nv, Nv), dtype=peq.dtype, order='C')
        if 'Fr' in self.grad_list or 'C' in self.grad_list:
            # Each neuron has it's own matrix
            GI1 = self.cuda.cp.zeros(
                (num_neuron, Nv, Nv), dtype=peq.dtype, order='C'
            )
        # Backward Pass
        # Apply absorption operator
        if with_absorption:
            btemp /= anorm[S_Total+1]
            Gabs = self.cuda.cp.empty((Nv, Nv), dtype=peq.dtype, order='C')
            self.cuda.cp.outer(alpha[S_Total, :], btemp, out=Gabs)
            # self.cuda.cp.dot(absorption_operator,btemp,out=btemp)
            btemp = abs_mat.dot(btemp)

        for i in reversed(range(S_Total)):
            # Scaling
            btemp /= anorm[i+1]

            if i != S_Total-1 or not self.with_trial_end:
                if 'Fr' in self.grad_list or 'C' in self.grad_list:
                    # Update G-function for Fr
                    self.cuda._G1(
                        (blocks, blocks),
                        (threads, threads),
                        (GI1, Nv, alpha, prop_mat, btemp, i, nid)
                    )
                # Spike emission
                btemp = sp_mat[nid[i], :, :].dot(btemp)
            # Update G-function for F
            self.cuda._G0(
                (blocks, blocks),
                (threads, threads),
                (G, Nv, alpha, prop_mat, EV_solution["lQd"], btemp, seq[i], i)
            )
            # Propagate back
            btemp = prop_mat[i, :]*btemp

        # Finish gradients calculation
        if 'Fr' in self.grad_list or 'C' in self.grad_list:
            GI10 = self.cuda.cp.empty(
                (num_neuron, Nv, Nv), dtype=peq.dtype, order='C'
            )
            for i in range(num_neuron):
                self.cuda.cp.dot(
                    EV_solution["Qd"], GI1[i, :, :], out=GI10[i, :, :]
                )
                GI10[i, :, :] = GI10[i, :, :].dot(EV_solution["Qd"].T)
            GI20 = self.cuda.cp.empty((Nv, Nv), dtype=peq.dtype, order='C')
            self.cuda.cp.dot(EV_solution["Qd"], G, out=GI20)
            # self.cuda.cp.dot(GI20,EV_solution["Qd"].T,out=GI20)
            GI20 = GI20.dot(EV_solution["Qd"].T)

        # Gradient from absorption_event
        if with_absorption:
            G -= Gabs
        G0 = self.cuda.cp.empty((Nv, Nv), dtype=peq.dtype, order='C')
        self.cuda.cp.dot(EV_solution["Qd"], G, out=G0)
        # self.cuda.cp.dot(G0,EV_solution["Qd"].T,out=G0)
        G0 = G0.dot(EV_solution["Qd"].T)

        QxOrig = EV_solution["QxOrig"]
        Qxdx = self.cuda.cp.empty((grid.N, Nv), dtype=peq.dtype, order='C')
        self.cuda.cp.dot(grid.cuda_var.dmat_d, QxOrig, out=Qxdx)
        results = {'loglik': ll}
        if 'D' in self.grad_list:
            d2Phi = self.cuda.cp.sum(Qxdx * Qxdx.dot(G0), 1)
            results['D'] = self.cuda.cp.sum(grid.cuda_var.w_d * peq * d2Phi)
        if 'F' in self.grad_list:
            dPhi = self.cuda.cp.sum(QxOrig * Qxdx.dot(G0), 1) + \
                self.cuda.cp.sum(Qxdx * QxOrig.dot(G0), 1)
            # boundary term depends on whether or not we start a trajectory
            # from peq or p0
            if not model.non_equilibrium:
                ABF = grid.Integrate(
                    peq-0.5*Qxd.dot(
                        alpha[S_Total+int(with_absorption), :] /
                        anorm[-1]+btemp/anorm[0]
                    ) * self.cuda.cp.sqrt(peq), device='GPU'
                )
            else:
                ABF = 0.5*grid.Integrate(
                    Qxd.dot(btemp/anorm[0])*rho0-Qxd.dot(
                        alpha[S_Total+int(with_absorption), :] / anorm[-1]
                    ) * self.cuda.cp.sqrt(peq), device='GPU'
                )
            results['F'] = -0.5 * D * peq * dPhi - ABF
        if 'F0' in self.grad_list:
            results['F0'] = -grid.Integrate(
                rho0*self.cuda.cp.sqrt(peq) - rho0*Qxd.dot(btemp/anorm[0]),
                device='GPU'
            )
        if 'Fr' in self.grad_list or 'C' in self.grad_list:
            Phi = self.cuda.cp.sum(QxOrig * QxOrig.dot(GI20), 1)
            PhiY = self.cuda.cp.empty((grid.N, num_neuron), dtype=peq.dtype)
            for i in range(num_neuron):
                PhiY[:, i] = (
                    peq * (Phi - self.cuda.cp.sum(
                        QxOrig * QxOrig.dot(GI10[i, :, :]), 1
                    )
                    )
                )
            gradFr = self.cuda.cp.empty((grid.N, num_neuron), dtype=peq.dtype)
            gradC = self.cuda.cp.empty(num_neuron, dtype=peq.dtype)
            for i in range(num_neuron):
                C2 = grid.Integrate(fr[:, i]*PhiY[:, i], device='GPU')
                gradFr[:, i] = C2[-1] - C2
                gradC[i] = C2[-1]
            if 'Fr' in self.grad_list:
                results['Fr'] = gradFr
            if 'C' in self.grad_list:
                results['C'] = gradC / fr[0, :]
        return results
