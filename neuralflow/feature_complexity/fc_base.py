# -*- coding: utf-8 -*-
"""Feature complexity calculation and feature consistency analysis
"""
import numpy as np
import logging
from neuralflow.PDE_Solve import PDESolve

logger = logging.getLogger(__name__)


class FC_tools:

    def __init__(self, non_equilibrium, model, terminal_time=1,
                 number_of_timesteps=10, boundary_mode='absorbing'):
        """


        Parameters
        ----------
        non_equilibrium : bool
            Whether or not the model is non_equilibrium.
        model : neuralflow.model
            Model class variable, which is used to extract those parameters
            that were not optimized. Also used for the integration/differention
            operations on GLL grid.
        terminal_time : float, optional
            Terminal time for non-equilibrium JS divergence calculation in
            seconds. Only used for non-equilibrium analysis. Should set to a
            a typical reaction time for absorbing case. If working with
            reflective mode, set it to trial duration. Different trial
            durations for reflecting are not supported. The default is 1.
        number_of_timesteps : int, optional
            Number of timesteps for non-equilibrium JS divergence calculation.
            Only used for non-equilibrium analysis. It is recommended to set
            this equal to ten times terminal_time. The default is 10.
        boundary_mode : str, optional
            Boundary mode is needed for non-equilibrium analysis, as we need to
            solve FPE to compute feature complexity and JS divergence. The
            default is 'absorbing'.

        """
        self.non_equilibrium = non_equilibrium
        self.model = model
        self.grid = model.grid
        self.terminal_time = terminal_time
        self.number_of_timesteps = number_of_timesteps
        self.boundary_mode = boundary_mode
        if boundary_mode not in ['absorbing', 'reflecting']:
            raise ValueError('Unknown boundary mode')
        if self.non_equilibrium:
            self.pde_solver = PDESolve.init_from_grid(self.grid, boundary_mode)

    def compute_FC(self, data={}, indices=None, invert=False):
        """Compute feature complexities for selected models in the data.


        Parameters
        ----------
        data : dictionary, optional
            Dictionary with the fitting results. The default is {}.
        indices : array, optional
            Indices of models in data dictionary on which FCs will be computed.
            If None, will compute FC on all indices. The default is None.
        invert : bool, optional
            Whether to invert potenial and p0.
            The default is False.

        Returns
        -------
        FCs_array: numpy array
            Evaluated feature complexities.
        lQ_array: numpy array
            Eigenvalues of H0 operator needed for further analysis. Only for
            non-equilibirum mode.
        Qx_array: numpy array
            Eigenvectors of H0 operator needed for further analysis. Only for
            non-equilibirum mode.

        """

        # Either use fitted parameters, or use model parameters if a parameter
        # not in data dict.
        if 'p0' in data:
            with_p0s = True
            num_p0_models = data['p0'][0].shape[0]
        else:
            with_p0s = False
            p0s = self.model.p0
            num_p0_models = p0s.shape[0]

        if 'D' in data:
            with_Ds = True
            num_D_models = data['D'][0].shape[0]
        else:
            with_Ds = False
            Ds = self.model.D
            num_D_models = Ds.shape[0]

        if 'peq' in data:
            with_peqs = True
            num_peq_models = data['peq'][0].shape[0]
        else:
            with_peqs = False
            peqs = self.model.peq
            num_peq_models = peqs.shape[0]
        num_models = max(num_peq_models, num_D_models, num_p0_models)
        if indices is None:
            if with_peqs:
                num_iterations = len(data['peq'])
            elif with_p0s:
                num_iterations = len(data['p0'])
            elif with_Ds:
                num_iterations = len(data['D'])
            else:
                num_iterations = 1
            all_iterations = list(range(num_iterations))
        else:
            num_iterations = len(indices)
            all_iterations = indices
        FCs_array = np.zeros((num_iterations))

        if self.non_equilibrium:
            num_EV = self.grid.N-2
            lQ_array = np.zeros((num_models, num_EV, num_iterations))
            Qx_array = np.zeros(
                (num_models, self.grid.N, num_EV, num_iterations))
            Seq = np.zeros(num_models)

            # Contribution from p0
            if not with_p0s:
                for i in range(num_models):
                    p0 = p0s[min(i, num_p0_models - 1)]
                    Seq[i] = np.sum(p0*np.log(p0) * self.grid.w_d) + np.log(2)
            for j, i in enumerate(all_iterations):
                for k in range(num_models):
                    if with_p0s:
                        # Contribution from p0
                        p0 = data['p0'][i][min(k, num_p0_models - 1)].copy()
                        if invert:
                            p0 = p0[::-1]
                        Seq[k] = (
                            np.sum(p0 * np.log(p0) * self.grid.w_d) + np.log(2)
                        )
                    if with_peqs:
                        peq = data['peq'][i][min(k, num_peq_models - 1)].copy()
                        if invert:
                            peq = peq[::-1]
                    else:
                        peq = peqs[min(k, num_peq_models - 1)]
                    Force = self.grid.dmat_d.dot(np.log(peq))

                    if with_Ds:
                        D = data['D'][i][min(k, num_D_models - 1)]
                    else:
                        D = Ds[min(k, num_D_models - 1)]
                    Fs = D / 4 * Force ** 2
                    # Compute eigenvalues and eigenvectors of H0 operator, i.e.
                    # completely ignoring spikes, as we are only interested in
                    # Langevin dynamics here
                    lQ, _, Qx = self.pde_solver.solve_EV(
                        peq, D, q=None, w=peq, mode='h0', fr=None
                    )
                    # We will need these for JS computation, so save
                    lQ_array[k, ..., j] = lQ.copy()
                    Qx_array[k, ..., j] = Qx.copy()
                    Fd = Qx.T.dot(np.diag(self.grid.w_d)).dot(
                        Fs * np.sqrt(peq)
                    )
                    rho0d = Qx.T.dot(np.diag(self.grid.w_d)).dot(
                        p0 / np.sqrt(peq)
                    )

                    # Contribution from Force
                    if self.boundary_mode == 'absorbing':
                        S2 = np.sum(Fd * rho0d / lQ)
                    elif self.boundary_mode == 'reflecting':
                        # The numerical formula can be generalized by computing
                        # time integral from 0 to terminal time (instead of
                        # infinity). Note that in this case FC depends on the
                        # observation time
                        S2 = (
                            Fd[0] * rho0d[0] * self.terminal_time +
                            np.sum(
                                Fd[1:] * rho0d[1:] * (
                                    1 - np.exp(-lQ[1:] * self.terminal_time)
                                ) / lQ[1:]
                            )
                        )
                    FCs_array[j] += Seq[k] + S2
            return FCs_array, lQ_array, Qx_array
        else:
            for j, i in enumerate(all_iterations):
                for k in range(num_models):
                    if with_peqs:
                        peq = data['peq'][i][min(k, num_peq_models - 1)]
                    else:
                        peq = peqs[min(k, num_peq_models - 1)]
                    rh = np.sqrt(peq)
                    # Use equilibrium formula
                    FCs_array[j] += np.sum(
                        (self.grid.dmat_d.dot(rh)**2) * self.grid.w_d
                    )
            return FCs_array

    def FeatureConsistencyAnalysis(self, data1, data2, JS_thres=0.0015,
                                   FC_stride=5, smoothing_kernel=0,
                                   num_models=10000, epoch_offset=0):
        """Perform feature consistency analysis


        Parameters
        ----------
        data1 : Dictionary
            Dictionary with the fitting results on datasample 1.
        data2 : Dictionary
            Dictionary with the fitting results on datasample 2.
        JS_thres : float, optional
            Threshold JS at which models are considered different. The default
            is 0.0015.
        FC_stride : int, optional
            Slack in the second sequence of models. For each Feature complexity
            calculated from data1, we consider 2*FC_stride + 1 models from
            data2 and choose the one that minimizes JS. The default is 5.
        smoothing_kernel : int, optional
            Smooth JS with moving average before thresholding. The default is
            0.
        num_models : int, optional
            Number of model on which feature complexities will be computed. The
            code will pick the models from data1/data2 on logarithmic scale,
            so that more models on early epochs will be selected. Smaller
            number can accelerate the code, but can be inaccurate. The default
            is 10000.
        epoch_offset : int, optional.
            Ignore threshold breaks for the first epochs_offset epochs.
            The optimization speed is usually faster initially, so JS might
            diverge because of that. The default is 0.

        Returns
        -------
        FCs1 : numpy array
            Feature complexities for data1. Also serves as a "common" feature
            complexity axis.
        min_inds_1 : numpy array
            Indices of models in data1 for each FC1. To get peq of a
            model that corresponds to FC1[i], we take
            data1['peq'][min_inds_1[i]].
        FCs2 : numpy array
            Feature complexities for data2.
        min_inds_2 : numpy array
            Indices of models in data2 for each FC2. To get peq of a
            model that corresponds to FC2[i], we take
            data2['peq'][min_inds_2[i]].
        JS : numpy array
            Jensen-Shannon divergence between models from data1 and data2.
        FC_opt_ind : int
            The index of an optimal model in FC1/JS.

        """

        # Desired iteration numbers on which FC will be calculated
        epochs_desired = np.unique(
            np.concatenate((
                [0],
                np.logspace(0, np.log10(data1['iter_num'][-1]), num_models)
            )).astype(int)
        )

        # Find the closest epoch numbers to the desired ones on which the
        # models are saved for data1 and data 2
        if num_models >= data1['iter_num'].size:
            epochs1 = np.arange(data1['iter_num'].size)
        else:
            epochs1 = np.unique(np.argmin(
                np.abs(np.subtract.outer(data1['iter_num'], epochs_desired)),
                axis=0
            ))
        if num_models >= data2['iter_num'].size:
            epochs2 = np.arange(data2['iter_num'].size)
        else:
            epochs2 = np.unique(np.argmin(
                np.abs(np.subtract.outer(data2['iter_num'], epochs_desired)),
                axis=0
            ))

        # Compute feature complexities at selected epochs
        if self.non_equilibrium:
            # When we are fitting all of the parameters, the models in data1
            # and data2 can be mirror images of each other. In this case, we
            # need to invert models from data2 in order to properly compute JS
            # divergence. In this case inversion is reflection w.r.t. the axis
            # y=0
            if all([el in data2 for el in ['peq', 'p0', 'fr']]):
                invert = self.NeedToReflect(data1, data2)
            else:
                invert = False

            FCs1, lQs1, Qxs1 = self.compute_FC(data1, epochs1)
            FCs2, lQs2, Qxs2 = self.compute_FC(data2, epochs2, invert)
        else:
            FCs1 = self.compute_FC(data1, epochs1)
            FCs2 = self.compute_FC(data2, epochs2)

        # Common feature complexity axis is chosen to be FCs1
        FC = FCs1
        JS = np.zeros(FC.size)
        min_inds_1 = epochs1
        min_inds_2 = np.zeros_like(JS, dtype=int)

        # Iterate through all FCs
        for i in range(FC.size):
            # Identify index of FC in the second array that is closest to the
            # current FC. Consider 2*FC_stride + 1 models around this index
            ind2 = np.argmin(np.abs(FC[i] - FCs2))
            min_ind = max(0, ind2 - FC_stride)
            max_ind = min(FCs2.size, ind2 + 1 + FC_stride)
            FC2_ind = np.arange(min_ind, max_ind)
            if FC2_ind.size > 0:
                JS_cur = np.zeros(FC2_ind.size)

                # Calculate all pairwise JSs and find the minimal one
                for i2, ind2 in enumerate(FC2_ind):
                    if self.non_equilibrium:
                        if 'p0' in data1:
                            p01 = data1['p0'][epochs1[i]].copy()
                            p02 = data2['p0'][epochs2[ind2]].copy()
                            if invert:
                                for samp in range(p02.shape[0]):
                                    p02[samp, :] = p02[samp, :][::-1]
                        else:
                            p01 = self.model.p0
                            p02 = self.model.p0
                        if 'peq' in data1:
                            peq1 = data1['peq'][epochs1[i]].copy()
                            peq2 = data2['peq'][epochs2[ind2]].copy()
                            if invert:
                                for samp in range(peq2.shape[0]):
                                    peq2[samp, :] = peq2[samp, :][::-1]
                        else:
                            peq1 = self.model.peq
                            peq2 = self.model.peq
                        lQ1, lQ2 = lQs1[..., i], lQs2[..., ind2]
                        Qx1, Qx2 = Qxs1[..., i], Qxs2[..., ind2]
                        JS_cur[i2] = self.JS_divergence_timedep(
                            peq1, p01, lQ1, Qx1, peq2, p02, lQ2, Qx2
                        )
                    else:
                        # This option was used in 2020 paper
                        JS_cur[i2] = self.JS_divergence_st(
                            data1['peq'][epochs1[i]],
                            data2['peq'][epochs2[ind2]],
                            True
                        )
                # Record index of the matching model and the corresponding JS
                min_inds_2[i] = epochs2[np.argmin(JS_cur) + min_ind]
                JS[i] = np.min(JS_cur)
            else:
                # Something went wrong...
                JS[i] = -1

        # Smooth JS before thesholding
        if smoothing_kernel > 0:
            JS = self.moving_average(JS, smoothing_kernel)

        # Threshold JS_av
        if epoch_offset >= epochs1[-1]:
            raise ValueError('epochs offset is greater that the maximum epoch')
        elif epoch_offset <= epochs1[0]:
            offset = 0
        else:
            offset = np.where(epochs1 < epoch_offset)[0][-1] + 1

        if len(np.where(JS[offset:] > JS_thres)[0]) > 0:
            FC_opt_ind = np.where(JS[offset:] > JS_thres)[0][0] - 1 + offset
        else:
            # Pick last index if threshold never crossed
            FC_opt_ind = JS.size - 1

        return FCs1, min_inds_1, FCs2, min_inds_2, JS, FC_opt_ind

    def JS_divergence_timedep(self, peq1, p01, lQ1, Qx1, peq2, p02,  lQ2, Qx2):
        """ Compute JS divergence between two non-stationary Langevin models
        """
        time = np.linspace(0, self.terminal_time, self.number_of_timesteps)
        num_peq_models = peq1.shape[0]
        num_p0_models = p01.shape[0]
        num_EVs = lQ1.shape[0]

        num_models = max(num_peq_models, num_p0_models)
        out = 0

        for k in range(num_models):
            p01_c = p01[min(k, num_p0_models - 1)]
            p02_c = p02[min(k, num_p0_models - 1)]
            peq1_c = peq1[min(k, num_peq_models - 1)]
            peq2_c = peq2[min(k, num_peq_models - 1)]
            EV_model = min(k, num_EVs - 1)

            # rho0 = p0/sqrt(peq) in the basis H will be propagated in time
            rho0d1 = Qx1[EV_model].T.dot(np.diag(self.grid.w_d)).dot(
                p01_c / np.sqrt(peq1_c)
            )
            rho0d2 = Qx2[EV_model].T.dot(np.diag(self.grid.w_d)).dot(
                p02_c / np.sqrt(peq2_c)
            )

            for i in range(1, len(time)):
                # propaate rho0 and transform back to SEM basis. scale rho to p
                p1 = (
                    Qx1[EV_model].dot(rho0d1.dot(np.diag(
                        np.exp(-lQ1[EV_model] * (time[i] - time[0]))
                    )))
                ) * np.sqrt(peq1_c)
                p2 = (
                    Qx2[EV_model].dot(rho0d2.dot(np.diag(
                        np.exp(-lQ2[EV_model] * (time[i] - time[0]))
                    )))
                ) * np.sqrt(peq2_c)
                # Accumulate JS
                out += self.JS_divergence_non_st(p1, p2)
        out /= num_models
        return out * (time[1] - time[0])

    def JS_divergence_non_st(self, peq1, peq2):
        """ JS divergence between two distributions peq1 and peq2 that are not
        normalized.
        """
        peq1 = np.maximum(peq1, 10**-10)
        peq2 = np.maximum(peq2, 10**-10)
        peq_av = 0.5*(peq1 + peq2)
        bulk = 0.5 * (
            np.sum(self.grid.w_d * peq1 * np.log(peq1/peq_av)) +
            np.sum(self.grid.w_d * peq2 * np.log(peq2/peq_av))
        )
        if self.boundary_mode == 'reflecting':
            return bulk
        # I1 and I2 takes care of probablity leakage through the left and right
        # boundaries in absorption case. If probability leakage are different,
        # in contributes to JS divergence
        I1 = np.maximum(1 - np.sum(self.grid.w_d * peq1), 10**-10)
        I2 = np.maximum(1 - np.sum(self.grid.w_d * peq2), 10**-10)
        return (
            bulk + 0.5 * (
                I1 * np.log(2 * I1 / (I1 + I2)) +
                I2 * np.log(2 * I2 / (I1 + I2))
            )
        )

    def JS_divergence_st(self, peq1, peq2, normalized=False):
        """ JS divergence between two stationary distributions peq1 and peq2.
        Note that in 2020 paper a slightly different metric was used: we used
        symmetrized KL divergence, which is not the same as JS divergence
        """
        if not normalized:
            peq1 = peq1 / np.sum(peq1*self.grid.w_d)
            peq2 = peq2 / np.sum(peq2*self.grid.w_d)
        peq_av = 0.5*(peq1 + peq2)
        bulk = 0.5 * (
            np.sum(self.grid.w_d * peq1 * np.log(peq1/peq_av)) +
            np.sum(self.grid.w_d * peq2 * np.log(peq2/peq_av))
        )
        return bulk

    def NeedToReflect(self, result1, result2):
        """ Find if model2 has to be reflected around the axis y=0.
        """
        domain_center = int((self.grid.N+1)/2)
        # Iteration on which the p0(x) distribution are compared. Want to take
        # one of the latest iteration
        iteration_compare = min(
            500, result1['iter_num'][-1] - 1, result2['iter_num'][-1] - 1
        )
        score = 0
        # Loop over p0(x) models at specified iterations
        for i in range(result1['p0'][0].shape[0]):
            # Find p0 maxima at both datasamples
            p01max = np.argmax(result1['p0'][iteration_compare][i, :])
            p02max = np.argmax(result2['p0'][iteration_compare][i, :])
            # Require that maxima are not exactly in the domain center
            if (
                    abs(p01max - domain_center) > 5 and
                    abs(p02max - domain_center) > 5
            ):
                # Determine if maxima are at different sides w.r.t. domain
                # center
                if (p01max - domain_center) * (p02max - domain_center) < 0:
                    score += 1
                else:
                    score -= 1
        # If p0 maxima do not provide any information, look at firing rates
        if score == 0:
            for i in range(result1['fr'][0].shape[0]):
                for j in range(result1['fr'][0].shape[2]):
                    fr1 = result1['fr'][iteration_compare][i, :, j]
                    fr2 = result2['fr'][iteration_compare][i, :, j]
                    # Compare relative sum square of the firing rate functions
                    # differences
                    if (
                            np.sum(np.square(fr2-fr1)/np.square(fr1)) >
                            np.sum(np.square(fr2[::-1]-fr1) / np.square(fr1))
                    ):
                        score += 1
                    else:
                        score -= 1
            logger.debug(
                'Maximum p0(x) does not provide reliable result. Using the '
                'difference in firing rate functions to determine if model '
                'reflection is needed'
            )
        else:
            logger.debug(
                'Using the location of p0(x) maxima to determine if model '
                'reflection is needed'
            )
        invert = True if score > 0 else False
        logger.debug(
            'Determined that reflection is '
            f'{"needed" if invert else "not needed"}'
        )
        return invert

    @staticmethod
    def moving_average(x, w):
        out = np.convolve(x, np.ones(w), 'same') / w
        out[-int(w / 2):] = x[-int(w / 2):]
        return out
