
# -*- coding: utf-8 -*-
"""Viterbi algorithm. Only supported on CPU
"""

import numpy as np


class Viterbi:

    """Viterbi algorithm


    Parameters
    ----------
    grad : neuralflow.Grads
        Gradient object is needed for solving FPE.

    Returns
    -------
    None.
    """

    def __init__(self, grad):
        """
        """
        self.grad = grad

    def run_viterbi(self, data, model, model_num=0):
        """Viterbi algorithm
        Finds the latent path X that maximizes joint probability P(X,Y). The
        path X is sampled at trial start/trial end time, and at the time of
        each of spike.


        Parameters
        ----------
        data : neuralflow.spike_data.SpikeData.
            SpikeData object in ISI format.
        model : neuralflow.model.model
            A model object.
        model_num : model number to be used. The default is 0.

        Returns
        -------
        trajectories : numpy array, dtype=np.ndarray
            Latent trajectory sampled at spike times for each trial in the
            data.
        state_inds : indeces of trajectory latent states on the gird.

        """

        # Compute EV solution
        EV_solution = self.grad._get_EV_solution(model, model_num)

        trajectories = np.empty(data.data.shape[0], dtype=np.ndarray)
        state_inds = np.empty(data.data.shape[0], dtype=np.ndarray)

        for trial in range(data.data.shape[0]):
            data_cur = data.data[trial]
            trajectories[trial], state_inds[trial] = self._Viterbi_trial(
                data_cur, model, model_num, EV_solution
            )
        return trajectories, state_inds

    def _Viterbi_trial(self, data, model, model_num, EV_solution):
        """Viterbi Algorithm for a single trial
        """

        # Extract ISI and neuron_id for convenience
        seq, nid = data

        # Sequence length
        S_Total = len(seq)

        grid = self.grad.pde_solver.grid

        # Exclude points on the boundary for absorbing boundary. This is
        # equivalent to setting the probability of latent trajectory to be zero
        # on the boundary before the trial end
        margin = 1

        # For some reason Viterbi does not works well with reflecting mode.
        # For a reflective boundary there is probability accumulation in the
        # domain boundaries, causing the trajectory to always stay exactly at
        # the boundary at all times. I fix it by setting a large margin where
        # the probability is forced to be zero.
        # Ideally, the margin for reflective boundary should be zero
        margin_ref = grid.N//30

        # Qxd is transformation matrix from H-basis to SEM, QxOrig is the same
        # but with scaled EVs
        Qxd = EV_solution["Qx"].dot(EV_solution["Qd"])
        QxdOrig = EV_solution["QxOrig"].dot(EV_solution["Qd"])

        # Model parameters
        peq, p0, D, fr = model.get_params(model_num)

        # Initialize the atemp for forward pass.
        atemp = np.log(p0)

        # Store the latent states for traceback
        states = np.zeros((grid.N, S_Total + 1)).astype(int)

        # Precalculate propagation matrix exp(-lambda_i*dt_j)
        prop_mat = np.exp(np.outer(-EV_solution["lQd"], seq))

        # Forward pass
        for i in range(1, S_Total + 1):
            # Propagate delta-function probability density in latent state
            temp = (
                Qxd.dot(QxdOrig.T * prop_mat[:, i-1][:, np.newaxis]) *
                np.sqrt(peq)[:, np.newaxis]
            )
            # Exclude boundaries
            if self.grad.boundary_mode == 'absorbing' and margin > 0:
                temp[:, 0:margin] = 0
                temp[:, -margin:] = 0
                temp[0:margin, :] = 0
                temp[-margin:, :] = 0
            elif self.grad.boundary_mode == 'reflecting' and margin_ref > 0:
                temp[:, 0:margin_ref] = 0
                temp[:, -margin_ref:] = 0
                temp[0:margin_ref, :] = 0
                temp[-margin_ref:, :] = 0

            # Multiply (sum up logs) with the previous vector of probabilities
            reduced_arg = np.log(np.maximum(temp, 10**-10)) + atemp
            # Max over previous state
            states[:, i] = np.argmax(reduced_arg, axis=1)
            # Subscribe maximum for each state
            atemp = reduced_arg[np.arange(grid.N), states[:, i]]
            # Emit spike
            if i != S_Total or not self.grad.with_trial_end:
                atemp += np.log(fr[:, nid[i-1]])
            # Exclude boundaries for absorbing boundary mode, as we can't hit
            # a boundary before trial end time
            if self.grad.boundary_mode == 'absorbing' and margin > 0:
                atemp[0:margin] = atemp[-margin:] = 0
            elif self.grad.boundary_mode == 'reflecting' and margin_ref > 0:
                atemp[0:margin_ref] = atemp[-margin_ref:] = 0

        # trajectory and state indices
        trajectory = np.zeros(S_Total + 1)
        state_inds = np.zeros(S_Total + 1, dtype=int)

        if self.grad.boundary_mode == 'absorbing':
            # Force to end at the boundary
            idx = margin if atemp[margin] > atemp[-1 -
                                                  margin] else grid.N-1-margin
            trajectory[-1] = grid.x_d[0] if idx == margin else grid.x_d[-1]
            state_inds[-1] = 0 if idx == margin else grid.N-1
        else:
            idx = np.argmax(atemp)
            trajectory[-1] = grid.x_d[idx]
            state_inds[i-1] = idx

        # Traceback
        for i in range(S_Total, 0, -1):
            idx = states[idx, i]
            trajectory[i-1] = grid.x_d[idx]
            state_inds[i-1] = idx

        return trajectory, state_inds.astype(int)
