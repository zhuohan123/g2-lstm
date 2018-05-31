#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

from ..constants import fX

__author__ = 'fyabc'


def floatX(value):
    return np.asarray(value, dtype=fX)


__all__ = [
    'floatX',
]
