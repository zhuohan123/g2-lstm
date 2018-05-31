#! /usr/bin/python
# -*- coding: utf-8 -*-

from .basic import *
from .gru import *
from .stochastic_lstm import *

__author__ = 'fyabc'

# layers: 'name': ('parameter initializer', 'builder')
layers = {
    'ff': (param_init_feed_forward, feed_forward),
    'gru': (param_init_gru, gru_layer),
    'gru_cond': (param_init_gru_cond, gru_cond_layer),
    'multi_gru': (param_init_gru, gru_layer),
    'multi_gru_cond': (param_init_gru_cond, gru_cond_layer),
    'lstm': (param_init_lstm, lstm_layer),
    'lstm_cond': (param_init_lstm_cond, lstm_cond_layer),
    # todo: implement it
    'multi_lstm': (param_init_lstm, lstm_layer),
    'multi_lstm_cond': (param_init_lstm_cond, lstm_cond_layer),
}


def get_layer(name):
    fns = layers[name]
    return fns[0], fns[1]


def get_init(name):
    return layers[name][0]


def get_build(name):
    return layers[name][1]


__all__ = [
    'layers',
    'get_layer',
    'get_build',
    'get_init',
]
