#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Cuda warmup functions and modules. This is a module (not class) since
we need a singleton pattern, and modules implement singleton
"""

import cupy as cp
import numpy as np
import cupyx  # Will be used when imported, do not remove
import os

handle = cp.cuda.device.get_cublas_handle()
streams_ = []
device_ = cp.cuda.Device()
max_threads = cp.cuda.Device().attributes['MaxThreadsPerBlock']

# We will create num_threads_ x num_threads_ 2D grid of threads
num_threads_ = int(np.sqrt(max_threads))

# l2 norm that is faster than linalg.norm. Not used in the current version
l2norm = cp.ReductionKernel(
    'T x', 'T y',  'x * x', 'a + b', 'y = sqrt(a)', '0', 'l2norm'
)


class var:
    pass


def _update_streams(nstreams):
    """If streams is not defined, or there are fewer streams than nstreams -
    destroy the previous streams and create new nstreams streams.
    """
    global streams_
    if len(streams_) < nstreams:
        if len(streams_) > 0:
            memory_pool = cp.cuda.MemoryPool()
            for stream in streams_:
                memory_pool.free_all_blocks(stream=stream)
        streams_ = [cp.cuda.stream.Stream() for _ in range(nstreams)]


def _free_streams_memory():
    device_.synchronize()
    memory_pool = cp.cuda.MemoryPool()
    for stream in streams_:
        memory_pool.free_all_blocks(stream=stream)


def import_custom_functions():
    """Read custom cuda kernels from source files
    """
    global _G0, _G1
    with open(os.path.join(os.path.dirname(__file__), 'cuda_kernels.cu')) as f:
        code = f.read()
        module = cp.RawModule(code=code)
    _G0 = module.get_function('G0_d_gpu')
    _G1 = module.get_function('G1_d_gpu')
