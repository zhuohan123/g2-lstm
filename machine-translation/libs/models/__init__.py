#! /usr/bin/python
# -*- coding: utf-8 -*-

from .model import *

__author__ = 'fyabc'


def build_and_init_model(model_name, options=None, build=True, model_type='NMTModel'):
    import cPickle as pkl

    from ..config import DefaultOptions
    from ..utility.utils import load_params

    if options is None:
        with open('{}.pkl'.format(model_name), 'rb') as f:
            options = DefaultOptions.copy()
            options.update(pkl.load(f))

    model = eval(model_type)(options)

    # allocate model parameters
    params = model.initializer.init_params()
    # load model parameters and set theano shared variables
    params = load_params(model_name, params)
    model.init_tparams(params)

    if build:
        ret = model.build_model()
        return model, options, ret
    return model, options
