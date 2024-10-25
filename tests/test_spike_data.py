#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 20:22:31 2023

@author: mikhailgenkin
"""

import numpy as np
from neuralflow.spike_data import SpikeData
import unittest
import sys
sys.path.insert(0, '/Users/mikhailgenkin/Neuroscience/CSHL/Code/neuralflow')


class TestSpikeData(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Sample spike data in ISI format: 2 trials, 3 neurons (ids = 0, 1, 2)
        cls.ISIs = np.empty((2, 2), dtype=np.ndarray)
        cls.ISIs[0][0] = np.array([0.05, 0.5, 1])
        cls.ISIs[1][0] = np.array([0.05, 1, 2, 3, 4])
        cls.ISIs[1][1] = np.array([0, 2, 1, 2, -1])
        cls.ISIs[0][1] = np.array([1, 2, -1])

        #  Equivalent data in spiketimes format
        cls.spiketimes = np.array(
            [
                [np.array([], dtype=np.float64), np.array([0.05])],
                [np.array([0.05]), np.array([3.05])],
                [np.array([0.55]), np.array([1.05, 6.05])]
            ],
            dtype=object
        )
        cls.timeepoch = [(0, 1.55), (0, 10.05)]

    def testtransformations(self):
        """"A few test for input parameters
        """

        sdata = SpikeData(self.ISIs, dformat='ISIs')

        # Test neuron number autodetection
        self.assertTrue(sdata.num_neuron == 3)

        spikes, timebins = SpikeData.transform_isis_to_spikes(self.ISIs)

        # Test transformation from ISIs to Spikes
        self.assertTrue(
            all(
                [np.allclose(seq1, seq2, atol=10**-8)
                 for seq1, seq2 in zip(spikes.flat, self.spiketimes.flat)]
            )
        )
        self.assertTrue(
            all(
                [np.allclose(np.array(s1), np.array(s2), atol=10**-8)
                 for s1, s2 in zip(timebins, self.timeepoch)]
            )

        )

        # Transform spiketimes back to ISI format
        ISIs = SpikeData.transform_spikes_to_isi(spikes, timebins)

        # Assert equivalence
        self.assertTrue(
            all(
                [np.allclose(seq1, seq2, atol=10**-8)
                 for seq1, seq2 in zip(ISIs.flat, self.ISIs.flat)]
            )
        )


if __name__ == '__main__':
    unittest.main()
