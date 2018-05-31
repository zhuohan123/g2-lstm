#! /usr/bin/python
# -*- encoding: utf-8 -*-

import ctypes

import numpy as np

from utils import Loader


mv_lib = Loader.get_lib()


def init(sync=False):
    """Initialize multiverso.

    This should be called only once before training at the beginning of the
    whole project.
    If sync is True, a sync server will be created. Otherwise an async server
    will be created.
    """

    args = [""]  # the first argument will be ignored. So we put a placeholder here
    if sync:
        args.append("-sync=true")
    n = len(args)
    args_type = ctypes.c_char_p * n
    mv_lib.MV_Init(ctypes.pointer(ctypes.c_int(n)), args_type(*[ctypes.c_char_p(arg) for arg in args]))


def shutdown():
    """Shutdown multiverso.

    This should be called only once after finishing training at the end of the
    whole project.
    """
    mv_lib.MV_ShutDown()


def barrier():
    """Set a barrier for all workers to wait.

    Workers will wait until all workers reach a specific barrier.
    """
    mv_lib.MV_Barrier()


def workers_num():
    """Return the total number of workers."""
    return mv_lib.MV_NumWorkers()


def worker_id():
    """Return the id (zero-based index) for current worker."""
    return mv_lib.MV_WorkerId()


def server_id():
    return mv_lib.MV_ServerId()


def is_master_worker():
    """If the worker is master worker.

    Some things only need one worker process, such as validation, outputting the
    result, initializing the parameters and so on. So we mark the worker 0 as
    the master worker to finish these things.
    """
    return worker_id() == 0


__all__ = [
    'init',
    'shutdown',
    'barrier',
    'workers_num',
    'worker_id',
    'server_id',
    'is_master_worker',
]
