#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Class for spiking data storage and manipulation
"""
import numpy as np
import numbers
from collections.abc import Iterable
import logging

logger = logging.getLogger(__name__)


class SpikeData:
    """A class for storing spike data in an ISI format and also for conversion
    between spiketimes and ISI formats


    Parameters
    ----------
    data : numpy array
        Two formats are supported:
        1) ISIs format: the data is numpy array of the size (N,2) of type
        np.ndarray, where each elements is a 1D array. N is the number of
        trials, and for each trial the first column contains all of the
        inter spike intervals (ISIs) in seconds, and the second column
        contains the corresponding neuronal IDs (trial termination, if
        recorded, is indicated with -1). Neuronal Ids start from zero.
        ISIs are represented as 1D arrays of floats, and neuronal indices
        as 1D array of integers.
            data[i][0] - 1D array, ISIs of type float for the trial i.
                The last entry can optionally be the time interval between
                the last spike and trial termination time.
            data[i][1] - 1D array, neuronal IDs of type int64 for the trial
                i. The last entry is -1 if the trial termination time is
                recorded.
        Example: create a data with 3 neurons (id 0,1,2) and two trials.
            Trial 0 started at time 0 s and ended at 1.55 s, where neuron
            1 spiked at 0.05 s and neuron 2 spikes at 0.55 s. Trial 1 also
            started at 0 s, ended at 10.05 s, where neuron 0 spiked at 0.05
            s, neuron 1 spiked at 3.05 s, and neuron 2 spiked at 1.05 and
            6.05 s.
            ISIs = np.empty((2, 2), dtype=np.ndarray)
            ISIs[0][0] = np.array([0.05,0.5,1])
            ISIs[0][1] = np.array([1,2,-1])
            ISIs[1][0] = np.array([0.05,1,2,3,4])
            ISIs[1][1] = np.array([0,2,1,2,-1])
        2) spiketimes format, where the data is numpy array of size
        (num_neuron, N) of type object, N is the number of trials. Each of
        the entries is 1D array that specify spiketimes of each neuron on
        each trial. In this case time_epoch array specify each trial start
        and end times. For the example above, the spiketimes format would
        be the following:
            spiketimes = np.array(
                [
                    [np.array([], dtype=np.float64), np.array([0.05])],
                    [np.array([0.05]), np.array([3.05])],
                    [np.array([0.55]), np.array([1.05, 6.05])]
                    ],
                dtype=object
                )
            timeepoch = [(0, 1.55), (0, 10.05)]
    dformat : str, optional
        ENUM('spiketimes', 'ISIs'). The default is 'ISIs'.
    time_epoch : list, optional
        For each trial, specify trial start time and trial end time as a
        tuple. Only needed for spiketimes format. The default is None.
    num_neuron : int, optional
        Number of neurons in the data. If not provided, will be inferred.
        The default is None.
    with_trial_end : bool, optional
        Whether trial end time is recorded. The default is True.
    with_cuda : bool, optional
        Whether to include GPU support. For GPU optimization, the platform
        has to be cuda-enabled, and cupy package has to be installed. The
        default is False.
    """

    def __init__(self, data, dformat='ISIs', time_epoch=None,
                 num_neuron=None, with_trial_end=True, with_cuda=False
                 ):
        """
        Public methods
        ------
        to_GPU, trial_average_fr, change_format

        """

        self.data = data

        if dformat not in ['ISIs', 'spiketimes']:
            raise ValueError('Format has to be ISIs or spiketimes')
        self.dformat = dformat
        self.time_epoch = time_epoch
        self.num_neuron = num_neuron
        self.with_trial_end = with_trial_end
        self._check_and_fix_inputs()
        self.with_cuda = with_cuda
        if self.with_cuda:
            import neuralflow.base_cuda as cuda
            self.cuda = cuda
            self.cuda_var = cuda.var()

    def to_GPU(self):
        """ Copy data to GPU memory.
        """
        if not self.with_cuda:
            raise ValueError(
                'Initialize with with_cuda = True for GPU support'
            )

        if self.dformat == 'spiketimes':
            data = SpikeData.transform_spikes_to_isi(
                self.data, self.time_epoch
            )
        else:
            data = self.data

        # In Cuda the data will be stored in a nested list format
        self.cuda_var.data = [[] for _ in range(data.shape[0])]

        for iTrial, seq in enumerate(data):
            self.cuda_var.data[iTrial].append(
                self.cuda.cp.asarray(seq[0], dtype='float64')
            )
            self.cuda_var.data[iTrial].append(
                self.cuda.cp.asarray(seq[1], dtype=int)
            )

    def trial_average_fr(self):
        """For each neuron, compute trial-average firing rate. This is needed
        to scale the initial guess of the firing rate function to match the
        average spike rate in the data.
        """
        if self.dformat != 'ISIs':
            logger.warning('Only ISIs format is supported')
            return None
        tot_spikes = np.zeros(self.num_neuron)
        tot_times = np.empty(len(self.data))
        for trial in range(len(self.data)):
            tot_times[trial] = np.sum(self.data[trial][0])
            sp_count, _ = np.histogram(
                self.data[trial][1],
                bins=np.arange(-0.5, self.num_neuron, 1)
            )
            tot_spikes += sp_count
        tot_time = np.sum(tot_times)
        return tot_spikes/tot_time

    def change_format(self, new_format, record_trial_end=True):
        """Convert the data between spiketimes and ISIs format.


        Parameters
        ----------
        new_format : str
            ENUM('spiketimes', 'ISIs').
        record_trial_end : bool, optional
            Whether to record trial end time in ISIs format.
            The default is True.

        """
        if self.dformat == new_format:
            logger.info(f'The data format is already {new_format}')
        elif new_format == 'ISIs':
            self.data = SpikeData.transform_spikes_to_isi(
                self.data, self.time_epoch, record_trial_end
            )
            self.dformat = new_format
            self.time_epoch = None
            logger.info(f'Data is in {new_format} format')
        elif new_format == 'spiketimes':
            self.data, self.time_epoch = SpikeData.transform_isis_to_spikes(
                self.data
            )
            self.dformat = new_format
            logger.info(f'Data is in {new_format} format')
        else:
            raise ValueError('Unknown format')

    @staticmethod
    def transform_spikes_to_isi(spikes, time_epoch, record_trial_end=True):
        """Convert spike times to ISI format, which is a suitable format for
        optimization.
        """

        num_neuron, num_trial = spikes.shape

        # initialize data array
        data = np.empty((num_trial, 2), dtype=np.ndarray)

        # indices of neurons that spiked
        spike_ind = np.empty(num_neuron, dtype=np.ndarray)

        # transform spikes to interspike intervals format
        for iTrial in range(num_trial):
            for iCell in range(num_neuron):
                spike_ind[iCell] = iCell * \
                    np.ones(len(spikes[iCell, iTrial]), dtype=int)
            all_spikes = np.concatenate(spikes[:, iTrial], axis=0)
            all_spike_ind = np.concatenate(spike_ind[:], axis=0)
            # create data array
            data[iTrial, 0] = np.zeros(len(all_spikes) + record_trial_end)

            if all_spikes.shape[0] == 0:
                data[iTrial, 1] = np.zeros(0)
                # If no spikes emitted, set to trial beginning time
                last_spike_time = time_epoch[iTrial][0]
            else:
                # sort spike times and neuron index arrays
                ind_sort = np.argsort(all_spikes)
                all_spikes = all_spikes[ind_sort]
                all_spike_ind = all_spike_ind[ind_sort]
                data[iTrial, 0][1:len(all_spikes)] = np.diff(all_spikes)
                data[iTrial, 0][0] = all_spikes[0] - time_epoch[iTrial][0]
                last_spike_time = all_spikes[-1]

            if record_trial_end:
                data[iTrial, 0][-1] = time_epoch[iTrial][1] - last_spike_time
                # assign indicies of neurons which fired, trial end is marked
                # with -1
                data[iTrial, 1] = np.concatenate((all_spike_ind, [-1]))
            else:
                data[iTrial, 1] = all_spike_ind
        return data

    @staticmethod
    def transform_isis_to_spikes(data):
        """ Convert ISIs to spikes
        """

        num_neuron = max([seq[1].max() for seq in data]) + 1
        num_trial = data.shape[0]

        spikes = np.empty((num_neuron, num_trial), dtype=np.ndarray)
        time_epoch = []

        for iTrial, seq in enumerate(data):
            all_spikes = np.cumsum(seq[0])
            time_epoch.append((0, all_spikes[-1]))
            for iCell in range(num_neuron):
                spikes[iCell, iTrial] = all_spikes[seq[1] == iCell]

        return spikes, time_epoch

    def _check_and_fix_inputs(self):
        if self.dformat == 'spiketimes':
            if not isinstance(self.data, Iterable):
                raise TypeError(
                    'Spikes should be an ndarray or a list of length '
                    'num_neuron'
                )
            num_neuron = len(self.data)

            # Record number of neurons
            if self.num_neuron is None:
                self.num_neuron = num_neuron
            elif self.num_neuron != num_neuron:
                raise ValueError(
                    f'Provided number of neurons is {num_neuron}, detected '
                    f'number of neurons is {self.num_neuron}'
                )

            # A simple (non-nested) list/array would mean 1 neuron and 1 trial
            if num_neuron == 1 and not isinstance(self.data[0], Iterable):
                self.data = [[self.data]]

            for iseq, seq in enumerate(self.data):
                if not isinstance(seq, Iterable):
                    raise TypeError(
                        'Each entry in spikes should be a list or array of '
                        'length num_trials'
                    )
                if not isinstance(seq[0], Iterable):
                    # This means 1 trial
                    self.data[iseq] = [self.data[iseq]]
            for iseq, seq in enumerate(self.data):
                if iseq == 0:
                    num_trial = len(seq)
                else:
                    if len(seq) != num_trial:
                        raise ValueError(
                            f'Neuron {iseq} has {len(seq)} trials, while '
                            f'neuron 0 has {num_trial} trials'
                        )
                for isubseq, subseq in enumerate(seq):
                    if not isinstance(subseq, np.ndarray):
                        self.data[iseq][isubseq] = np.array(
                            self.data[iseq][isubseq]
                        )
            for iseq, seq in enumerate(self.data):
                for isubseq, subseq in enumerate(seq):
                    if not all(np.diff(subseq) >= 0):
                        raise ValueError(
                            f'Spikes for neuron {iseq} are not sorted on trial'
                            f' {isubseq}'
                        )

            for i, (el1, el2) in enumerate(self.time_epoch):
                if not isinstance(el1, numbers.Number):
                    raise TypeError(
                        'Each of the trial_start should be a number'
                    )
                if not isinstance(el2, numbers.Number):
                    raise TypeError('Each of the trial_end should be a number')
                if el2 < el1:
                    raise ValueError(
                        'trial_end should be greater than trial_start'
                    )
                for neuron in range(num_neuron):
                    if self.data[neuron][i].size == 0:
                        continue
                    if el1 > self.data[neuron][i][0]:
                        raise ValueError(
                            f'On trial {i} one of the neurons spiked before '
                            'trial_start'
                        )
                    if el2 < self.data[neuron][i][-1]:
                        raise ValueError(
                            f'On trial {i} one of the neurons spiked after '
                            'trial_end'
                        )

        elif self.dformat == 'ISIs':
            if isinstance(self.data, list):
                self.data = np.asarray(self.data, dtype='object')
            num_trials = len(self.data)
            if num_trials == 0:
                raise ValueError('No data provided')

            for trial_num, seq in enumerate(self.data):
                if len(seq) != 2:
                    raise ValueError(
                        'For each trial two entries should be provided: ISIs '
                        'and neuron Ids'
                    )
                ISIs = seq[0]
                if isinstance(ISIs, list):
                    self.data[trial_num][0] = np.asarray(
                        self.data[trial_num][0], dtype=np.float64
                    )
                    ISIs = self.data[trial_num][0]
                if not isinstance(ISIs, np.ndarray):
                    raise TypeError('ISIs should be passed as numpy array')
                if len(ISIs.shape) > 1:
                    raise ValueError('ISIs should be 1D array')
                ISIs = ISIs.astype(np.float64)
                if not all(ISIs >= 0):
                    raise ValueError(
                        f'Trial {trial_num} contains negative ISIs'
                    )
                Ids = seq[1]
                if isinstance(Ids, list):
                    self.data[trial_num][1] = np.asarray(Ids, int)
                    Ids = self.data[trial_num][1]
                if not isinstance(Ids, np.ndarray):
                    raise TypeError('Neural ids should be passed as np array')
                if len(Ids.shape) > 1:
                    raise ValueError('Neural ids should be 1D array')
                Ids = Ids.astype('int')
                self.data[trial_num][1] = Ids
                if np.any(Ids[:-1] < 0) or (
                        self.num_neuron is not None and
                        np.any(Ids >= self.num_neuron)
                ):
                    raise ValueError(
                        'Neural ids should be integers that start from 0'
                    )
                if len(Ids) > 0 and (Ids[-1] < -1):
                    raise ValueError(
                        'The last neural id should be -1 or nonnegative int'
                    )
                if ISIs.shape != Ids.shape:
                    raise ValueError(
                        'ISIs and neural ids must have the same size'
                    )
            num_neuron = max([seq[1].max() for seq in self.data]) + 1

            # Record number of neurons
            if self.num_neuron is None:
                self.num_neuron = num_neuron
            elif self.num_neuron != num_neuron:
                raise ValueError(
                    f'Provided number of neurons is {num_neuron}, detected '
                    f'number of neurons is {self.num_neuron}'
                )

            # Ensure that every trial end time is recorded
            if self.with_trial_end:
                if not all([seq[1][-1] == -1 for seq in self.data]):
                    raise ValueError(
                        'On each trial the last entry should be an ISI between'
                        ' the last spike and trial termination time'
                    )
            else:
                # Delete trial end time
                for trial_num, seq in enumerate(self.data):
                    if seq[1][-1] == -1:
                        logger.debug(
                            f'Removing trial end time from trial {trial_num}'
                        )
                        self.data[trial_num][0] = self.data[trial_num][0][:-1]
                        self.data[trial_num][1] = self.data[trial_num][1][:-1]
