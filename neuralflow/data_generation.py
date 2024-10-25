# -*- coding: utf-8 -*-
"""This source file contains SyntheticData class for synthetic data generation.
"""

import logging
import numpy as np
import math
from tqdm import tqdm
from neuralflow.settings import implemented_bms
from scipy.interpolate import interp1d
from neuralflow.spike_data import SpikeData

logger = logging.getLogger(__name__)


class SyntheticData:
    """Spike data generation from a Langevin model


    Parameters
    ----------
    model : model
        An instance of neuralflow.model.
    boundary_mode : ENUM('absorbing', 'reflecting')
        Boundary behavior.
    dt : float, optional
        Time bin size for Langevin ODE integration. The default is 0.0001.
    record_trial_end : bool, optional
        Whether or not to include trial end. Usually True is the best
        choice. The default is True.

    Notes
    ------
    No GPU support. Data generation is usually pretty fast, so CPU is
    sufficient for this purpose.

    """

    def __init__(
            self, model, boundary_mode, dt=0.0001,
            record_trial_end=True
    ):
        self.model = model
        self.grid = model.grid
        if boundary_mode not in implemented_bms:
            raise ValueError(f'Unknown boundary mode {boundary_mode}')
        self.boundary_mode = boundary_mode
        self.num_neuron = model.num_neuron
        self.dt = dt
        self.record_trial_end = record_trial_end

    def generate_data(
            self, trial_start=0, trial_end=1, num_trials=2, model_num=0
    ):
        """Generate spike data and latent trajectories.


        Parameters
        ----------
        trial_start : float or list, optional
            Trial start time. To specify the same trial start time for all
            trials, provide a single float, or provide a list for each trial.
            The default is 0.
        trial_end : float or list, optional
            Trial end time. To specify the same trial end time for all
            trials, provide a single float, or provide a list for each trial.
            Note that for absorbing boundary mode this serves as a timeout and
            the actual trial end time can be smaller if the boundary is reached
            before this time. The default is 1.
        num_trials : int, optional
            Number of trials to be generated. The default is 2.
        model_num : int, optional
            Which model to use for data generation. The default is 0.

        Returns
        -------
        data : numpy array (num_trials, 2), dtype = np.ndarray.
            Data in ISI format. See spike_data class for details.
        time_bins : numpy array (num_trials,), dtype = np.ndarray
            For each trial contains times at which latent trajectory was
            recorded. Each entry is 1D numpy array of floats.
        x : numpy array (num_trials,), dtype = np.ndarray
            Latent trajectories for each trial. Each entry is 1D array of
            floats. Same shape as time_bins
        """

        if not np.isscalar(trial_start) and len(trial_start) != num_trials:
            raise ValueError(
                'trial_start should be float or a list of length num_tials'
            )

        if np.isscalar(trial_start):
            trial_start = [trial_start] * num_trials

        if not np.isscalar(trial_end) and len(trial_end) != num_trials:
            raise ValueError(
                'trial_end should be float or list of length num_tials'
            )
        if np.isscalar(trial_end):
            trial_end = [trial_end] * num_trials

        time_epoch = [(s, e) for s, e in zip(trial_start, trial_end)]

        peq, p0, D, fr = self.model.get_params(model_num)

        # Use firing rate function if availible
        fr_lambda = self.model.get_fr_lambda()

        return self._generate_data(peq, p0, D, fr, fr_lambda, time_epoch)

    def _generate_data(self, peq, p0, D, fr, fr_lambda, time_epoch):
        """Generates synthetic spike data and latent trajectories from a
        given model defined by (peq,p0,D,firing_rate_model).
        """
        num_trial = len(time_epoch)

        # generate diffusion trajectories
        x, time_bins = self._generate_diffusion(peq, p0, D, time_epoch)

        # initialize data arrays
        spikes = np.empty((self.num_neuron, num_trial), dtype=np.ndarray)

        # compute firing rates and spikes
        for iTrial in range(num_trial):
            for iCell in range(self.num_neuron):
                if fr_lambda is not None:
                    # Just evaluate firing rate at each x(t)
                    rt = fr_lambda[iCell](x[iTrial])
                else:
                    # Interpolate firing rate functions between the grid points
                    fr_interp = interp1d(
                        self.grid.x_d, fr[..., iCell], kind='cubic'
                    )
                    rt = fr_interp(x[iTrial])
                # Generate spikes from rate
                spikes[iCell, iTrial] = self._generate_inhom_poisson(
                    time_bins[iTrial][0:rt.shape[0]], rt
                )

        # Calculate the actual time epoch with the actual end of trial times
        time_epoch_actual = [
            (time_epoch[i][0], time_bins[i][-1] + self.dt)
            for i in range(num_trial)
        ]
        # transform spikes to ISIs
        data = SpikeData.transform_spikes_to_isi(
            spikes, time_epoch_actual, self.record_trial_end
        )
        return data, time_bins, x

    def _generate_inhom_poisson(self, time, rate):
        """Generate spike sequence from a given rate of inhomogenious Poisson
        process lambda(t)
        """
        # calculate cumulative rate
        deltaT = time[1:] - time[:-1]
        r = np.cumsum(rate[0:-1] * deltaT)
        r = np.insert(r, 0, 0)
        deltaR = r[1:] - r[:-1]

        # generate 1.5 as many spikes as expected on average for exponential
        # distribution with rate 1
        numX = math.ceil(1.5 * r[-1])

        # generate exponential distributed spikes with the average rate 1
        notEnough = True
        x = np.empty(0)
        xend = 0.0
        while notEnough:
            x = np.append(
                x, xend + np.cumsum(np.random.exponential(1.0, numX)))
            # check that we generated enough spikes
            if (not len(x) == 0):
                xend = x[-1]
            notEnough = xend < r[-1]

        # trim extra spikes
        x = x[x <= r[-1]]

        if len(x) == 0:
            spikes = np.empty(0)
        else:
            # for each x find index of the last rate which is smaller than x
            indJ = [np.where(r <= x[iSpike])[0][-1]
                    for iSpike in range(len(x))]
            # compute rescaled spike times
            spikes = time[indJ] + (x - r[indJ]) * deltaT[indJ] / deltaR[indJ]

        return spikes

    def _generate_diffusion(self, peq, p0, D, time_epoch):
        """Sample latent trajectory by numerical integration of Langevin
        equation
        """

        num_trial = len(time_epoch)

        # pre-allocate output
        x = np.empty(num_trial, dtype=np.ndarray)
        time_bins = np.empty(num_trial, dtype=np.ndarray)

        # sample initial condition from the equilibrium distribution
        x0 = self._sample_from_p(p0, num_trial)

        # compute force profile from the potential
        force = self.model.force_from_peq(peq)

        # fix very high force values on the boundary (so that particle do not
        # travel too far during the time dt)
        ind = (np.abs(force) > 0.05 / (D * self.dt)) & (self.grid.x_d < 0)
        force[ind] = 0.05 / (D * self.dt)
        ind = (np.abs(force) > 0.05 / (D * self.dt)) & (self.grid.x_d > 0)
        force[ind] = -0.05 / (D * self.dt)

        N = len(force)

        for i, iTrial in enumerate(tqdm(range(num_trial))):

            # generate time bins
            time_bins[iTrial] = np.arange(
                time_epoch[iTrial][0], time_epoch[iTrial][1], self.dt
            )
            num_bin = len(time_bins[iTrial]) - 1
            y = np.zeros(num_bin + 1)
            y[0] = x0[iTrial]

            # generate noise
            noise = (np.sqrt(self.dt * 2 * D) * np.random.randn(num_bin))

            # account for absorbing boundary trajectories ending early
            max_ind = num_bin + 1

            # Do Euler integration
            for iBin in range(num_bin):
                # find force at the current position by linear interpolation
                ind = np.argmax(self.grid.x_d - y[iBin] >= 0)
                if ind == 0:
                    f = force[0]
                elif ind == N-1:
                    f = force[-1]
                else:
                    theta = (
                        (y[iBin] - self.grid.x_d[ind - 1]) /
                        (self.grid.x_d[ind] - self.grid.x_d[ind - 1])
                    )
                    f = (1.0 - theta) * force[ind - 1] + theta * force[ind]
                y[iBin + 1] = y[iBin] + D * f * self.dt + noise[iBin]

                # Handle boundaries:
                if self.boundary_mode == "reflecting":
                    # Handle reflection. To verify, check the following:
                    # Provided that abs(y[iBin + 1]) < 3*L:
                    # 1) If y is within x domain, the output is y[iBin + 1]
                    # 2) If y > self.grid.x_d_[-1], the value is
                    #    2 * self.grid.x_d_[-1] - y[iBin + 1]
                    # 3) If y < self.grid.x_d_[0], the value is
                    #    2 * self.grid.x_d_[0] - y[iBin + 1])
                    y[iBin + 1] = min(
                        max(y[iBin + 1], 2 * self.grid.x_d[0] - y[iBin + 1]),
                        2 * self.grid.x_d[-1] - y[iBin + 1]
                    )

                    # Regenerate the values if noise magnitude was very high
                    # This happens when y[iBin + 1] is outside of the domain by
                    # more than domain length L. Should happen very rare
                    # because of clipping of large force values. Repeat until
                    # the noise values make y[iBin + 1] not too large
                    error_state = True
                    while error_state:
                        if (
                            y[iBin + 1] < self.grid.x_d[0] or
                            y[iBin + 1] > self.grid.x_d[-1]
                        ):
                            y[iBin + 1] = (
                                y[iBin] + D * f * self.dt +
                                np.sqrt(self.dt * 2 * D) * np.random.randn()
                            )
                            y[iBin + 1] = min(
                                max(y[iBin + 1],
                                    2 * self.grid.x_d[0] - y[iBin + 1]
                                    ),
                                2 * self.grid.x_d[-1] - y[iBin + 1]
                            )
                        else:
                            error_state = False

                elif self.boundary_mode == "absorbing":
                    # Check termination condition
                    if (
                        y[iBin + 1] < self.grid.x_d[0] or
                        y[iBin + 1] > self.grid.x_d[-1]
                    ):
                        max_ind = iBin
                        break
            x[iTrial] = y[:max_ind]
            time_bins[iTrial] = time_bins[iTrial][:max_ind]

        return x, time_bins

    def _sample_from_p(self, p, num_sample):
        """Generate samples from a given probability distribution. Needed for
        initialization of the latent trajectories. Note: this function does
        not support delta-functions or extremely narrow p-function.
        """
        x = np.zeros(num_sample)
        pcum = np.cumsum(p * self.grid.w_d)
        y = np.random.uniform(0, 1, num_sample)
        for iSample in range(num_sample):
            # find index of the element closest to y[iSample]
            ind = (np.abs(pcum - y[iSample])).argmin()
            x[iSample] = self.grid.x_d[ind]
        return x
