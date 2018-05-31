#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function

import os
import platform
import ctypes
from ctypes.util import find_library

import numpy as np

PACKAGE_PATH = os.path.abspath(os.path.dirname(__file__))


class Loader(object):
    """
    This loader is responsible for loading multiverso dynamic library in both *nix and windows.
    """

    LIB = None

    @classmethod
    def _find_mv_path(cls):
        if platform.system() == 'Windows':
            mv_lib_path = find_library('Multiverso')
            if mv_lib_path is None:
                print('* Fail to load Multiverso.dll from the windows $PATH.'
                      'Because Multiverso.dll can not be found in the $PATH '
                      'directories. Go on loading Multiverso from the package.')
            else:
                return mv_lib_path

            mv_lib_path = os.path.join(PACKAGE_PATH, 'Multiverso.dll')
            if not os.path.exists(mv_lib_path):
                print('* Fail to load Multiverso.dll from the package. '
                      'Because the file {} can not be found.'.format(mv_lib_path))
            else:
                return mv_lib_path
        else:
            mv_lib_path = find_library('Multiverso')
            if mv_lib_path is None:
                print('* Fail to load libmultiverso.so from the system'
                      'libraries. Because libmultiverso.so can\'t be found in'
                      'library paths. Go on loading Multiverso from the package.')
            else:
                return mv_lib_path

            mv_lib_path = os.path.join(PACKAGE_PATH, 'libmultiverso.so')
            if not os.path.exists(mv_lib_path):
                print('* Fail to load libmultiverso.so from the package. '
                      'Because the file {} can not be found.'.format(mv_lib_path))
            else:
                return mv_lib_path

        return None

    @classmethod
    def load_lib(cls):
        mv_lib_path = cls._find_mv_path()
        if mv_lib_path is None:
            print("Fail to load the multiverso library. Please make sure you"
                  "  have installed multiverso successfully")
        else:
            print('Find the multiverso library successfully({})'.format(mv_lib_path))
        return ctypes.cdll.LoadLibrary(mv_lib_path)

    @classmethod
    def get_lib(cls):
        if not cls.LIB:
            cls.LIB = cls.load_lib()
            cls.LIB.MV_NumWorkers.restype = ctypes.c_int
        return cls.LIB


def convert_data(data):
    """Convert the data to float32 ndarray."""
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    return data.astype(np.float32)


__all__ = [
    'Loader',
    'convert_data',
]
