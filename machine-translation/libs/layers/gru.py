#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import theano
from theano import tensor as T

from ..constants import fX, profile
from .basic import _slice, dropout_layer, _attention
from ..utility.utils import _p, normal_weight, orthogonal_weight

__author__ = 'fyabc'


def param_init_gru(O, params, prefix='gru', nin=None, dim=None, **kwargs):
    if nin is None:
        nin = O['dim_proj']
    if dim is None:
        dim = O['dim_proj']

    layer_id = kwargs.pop('layer_id', 0)
    context_dim = kwargs.pop('context_dim', None)
    multi = 'multi' in O.get('unit', 'gru')
    unit_size = kwargs.pop('unit_size', O.get('unit_size', 2))

    if not multi:
        # embedding to gates transformation weights, biases
        params[_p(prefix, 'W', layer_id)] = np.concatenate([normal_weight(nin, dim), normal_weight(nin, dim)], axis=1)
        params[_p(prefix, 'b', layer_id)] = np.zeros((2 * dim,), dtype=fX)

        # recurrent transformation weights for gates
        params[_p(prefix, 'U', layer_id)] = np.concatenate([orthogonal_weight(dim),
                                                            orthogonal_weight(dim)], axis=1)

        # embedding to hidden state proposal weights, biases
        params[_p(prefix, 'Wx', layer_id)] = normal_weight(nin, dim)
        params[_p(prefix, 'bx', layer_id)] = np.zeros((dim,), dtype=fX)

        # recurrent transformation weights for hidden state proposal
        params[_p(prefix, 'Ux', layer_id)] = orthogonal_weight(dim)

        if context_dim is not None:
            params[_p(prefix, 'Wc', layer_id)] = np.concatenate([normal_weight(context_dim, dim),
                                                                 normal_weight(context_dim, dim)], axis=1)
            params[_p(prefix, 'Wcx', layer_id)] = normal_weight(context_dim, dim)
    else:
        # embedding to gates transformation weights, biases
        params[_p(prefix, 'W', layer_id)] = np.stack([np.concatenate([normal_weight(nin, dim), normal_weight(nin, dim)],
                                                                     axis=1) for _ in xrange(unit_size)], axis=0)
        params[_p(prefix, 'b', layer_id)] = np.zeros((unit_size, 2 * dim), dtype=fX)

        # recurrent transformation weights for gates
        params[_p(prefix, 'U', layer_id)] = np.stack([np.concatenate([orthogonal_weight(dim), orthogonal_weight(dim)],
                                                                     axis=1) for _ in xrange(unit_size)], axis=0)

        # embedding to hidden state proposal weights, biases
        params[_p(prefix, 'Wx', layer_id)] = np.stack([normal_weight(nin, dim) for _ in xrange(unit_size)], axis=0)
        params[_p(prefix, 'bx', layer_id)] = np.zeros((unit_size, dim), dtype=fX)

        # recurrent transformation weights for hidden state proposal
        params[_p(prefix, 'Ux', layer_id)] = np.stack([orthogonal_weight(dim) for _ in xrange(unit_size)], axis=0)

        if context_dim is not None:
            params[_p(prefix, 'Wc', layer_id)] = np.stack([
                np.concatenate([normal_weight(context_dim, dim), normal_weight(context_dim, dim)], axis=1)
                for _ in xrange(unit_size)], axis=0)
            params[_p(prefix, 'Wcx', layer_id)] = np.stack([
                normal_weight(context_dim, dim) for _ in xrange(unit_size)],
                axis=0)

    return params


def _gru_step_slice(
        mask, x_, xx_,
        ht_1,
        U, Ux):
    """GRU step function to be used by scan

    arguments (0) | sequences (3) | outputs-info (1) | non-seqs (2)

    ht_1: ([BS], [H])
    U: ([H], [H] + [H])
    """

    _dim = Ux.shape[1]

    preact = T.dot(ht_1, U) + x_

    # reset and update gates
    r = T.nnet.sigmoid(_slice(preact, 0, _dim))
    u = T.nnet.sigmoid(_slice(preact, 1, _dim))

    # hidden state proposal
    ht_tilde = T.tanh(T.dot(ht_1, Ux) * r + xx_)

    # leaky integrate and obtain next hidden state
    ht = u * ht_1 + (1. - u) * ht_tilde
    ht = mask[:, None] * ht + (1. - mask)[:, None] * ht_1

    return ht


def _gru_step_slice_attention(
        mask, x_, xx_, context,
        ht_1,
        U, Ux, Wc, Wcx):
    """GRU step function with attention.

    context: ([BS], [Hc])
    Wc: ([Hc], [H] + [H])
    Wcx: ([Hc], [H])
    """

    _dim = Ux.shape[1]

    preact = T.nnet.sigmoid(T.dot(ht_1, U) + x_ + T.dot(context, Wc))

    # reset and update gates
    r = _slice(preact, 0, _dim)
    u = _slice(preact, 1, _dim)

    # hidden state proposal
    ht_tilde = T.tanh(T.dot(ht_1, Ux) * r + xx_ + T.dot(context, Wcx))

    # leaky integrate and obtain next hidden state
    ht = u * ht_1 + (1. - u) * ht_tilde
    ht = mask[:, None] * ht + (1. - mask)[:, None] * ht_1

    return ht


def gru_layer(P, state_below, O, prefix='gru', mask=None, **kwargs):
    """GRU layer

    input:
        state_below: ([Ts/t], [BS], x)     # x = [W] for src_embedding
        mask: ([Ts/t], [BS])
        context: ([Tt], [BS], [Hc])
    output: a list
        output[0]: hidden, ([Ts/t], [BS], [H])
    """

    layer_id = kwargs.pop('layer_id', 0)
    dropout_params = kwargs.pop('dropout_params', None)
    context = kwargs.pop('context', None)
    one_step = kwargs.pop('one_step', False)
    init_state = kwargs.pop('init_state', None)
    multi = 'multi' in O.get('unit', 'gru')
    unit_size = kwargs.pop('unit_size', O.get('unit_size', 2))

    kw_ret = {}

    n_steps = state_below.shape[0] if state_below.ndim == 3 else 1
    n_samples = state_below.shape[1] if state_below.ndim == 3 else state_below.shape[0]
    if multi:
        dim = P[_p(prefix, 'Ux', layer_id)][0].shape[1]
    else:
        dim = P[_p(prefix, 'Ux', layer_id)].shape[1]

    mask = T.alloc(1., n_steps, 1) if mask is None else mask

    # state_below is the input word embeddings, to the gates and the hidden state proposal
    if multi:
        state_below_ = T.concatenate([
            T.dot(state_below, P[_p(prefix, 'W', layer_id)][j]) + P[_p(prefix, 'b', layer_id)][j]
            for j in range(unit_size)
        ], axis=-1)
        state_belowx = T.concatenate([
            T.dot(state_below, P[_p(prefix, 'Wx', layer_id)][j]) + P[_p(prefix, 'bx', layer_id)][j]
            for j in range(unit_size)
        ], axis=-1)
    else:
        state_below_ = T.dot(state_below, P[_p(prefix, 'W', layer_id)]) + P[_p(prefix, 'b', layer_id)]
        state_belowx = T.dot(state_below, P[_p(prefix, 'Wx', layer_id)]) + P[_p(prefix, 'bx', layer_id)]

    def _step_slice(mask, x_, xx_, ht_1, U, Ux):
        """
        m_: mask; x_:# W_z*E*y_i, W_r*E*y_i
        xx_: W*E*y_i; h_: s_(i-1)
        """
        h_tmp = ht_1
        for j in range(unit_size):
            x = _slice(x_, j, 2 * dim)
            xx = _slice(xx_, j, dim)
            h = _gru_step_slice(mask, x, xx, h_tmp, U[j], Ux[j])
            h_tmp = h
        return h

    def _step_slice_attention(mask, x_, xx_, context, ht_1, U, Ux, Wc, Wcx):
        h_tmp = ht_1
        for j in range(unit_size):
            x = _slice(x_, j, 2 * dim)
            xx = _slice(xx_, j, dim)
            h = _gru_step_slice_attention(mask, x, xx, context, h_tmp,
                                          U[j], Ux[j], Wc[j], Wcx[j])
            h_tmp = h
        return h

    # prepare scan arguments
    init_states = [T.alloc(0., n_samples, dim) if init_state is None else init_state]

    if context is None:
        seqs = [mask, state_below_, state_belowx]
        shared_vars = [
            P[_p(prefix, 'U', layer_id)],
            P[_p(prefix, 'Ux', layer_id)],
        ]
        if multi:
            _step = _step_slice
        else:
            _step = _gru_step_slice
    else:
        seqs = [mask, state_below_, state_belowx, context]
        shared_vars = [
            P[_p(prefix, 'U', layer_id)],
            P[_p(prefix, 'Ux', layer_id)],
            P[_p(prefix, 'Wc', layer_id)],
            P[_p(prefix, 'Wcx', layer_id)],
        ]
        if multi:
            _step = _step_slice_attention
        else:
            _step = _gru_step_slice_attention

    if one_step:
        outputs = _step(*(seqs + init_states + shared_vars))
    else:
        outputs, _ = theano.scan(
            _step,
            sequences=seqs,
            outputs_info=init_states,
            non_sequences=shared_vars,
            name=_p(prefix, '_layers', layer_id),
            n_steps=n_steps,
            profile=profile,
            strict=True,
        )

    kw_ret['hidden_without_dropout'] = outputs

    if dropout_params:
        outputs = dropout_layer(outputs, *dropout_params)

    return outputs, kw_ret


def param_init_gru_cond(O, params, prefix='gru_cond', nin=None, dim=None, dimctx=None, nin_nonlin=None,
                        dim_nonlin=None, **kwargs):
    if nin is None:
        nin = O['dim']
    if dim is None:
        dim = O['dim']
    if dimctx is None:
        dimctx = O['dim']
    if nin_nonlin is None:
        nin_nonlin = nin
    if dim_nonlin is None:
        dim_nonlin = dim
    layer_id = kwargs.pop('layer_id', 0)
    multi = 'multi' in O.get('unit', 'gru_cond')
    unit_size = kwargs.pop('unit_size', O.get('cond_unit_size', 2))

    if not multi:
        params[_p(prefix, 'W', layer_id)] = np.concatenate([normal_weight(nin, dim),
                                                            normal_weight(nin, dim)], axis=1)
        params[_p(prefix, 'b', layer_id)] = np.zeros((2 * dim,), dtype=fX)
        params[_p(prefix, 'U', layer_id)] = np.concatenate([orthogonal_weight(dim_nonlin),
                                                            orthogonal_weight(dim_nonlin)], axis=1)

        params[_p(prefix, 'Wx', layer_id)] = normal_weight(nin_nonlin, dim_nonlin)
        params[_p(prefix, 'Ux', layer_id)] = orthogonal_weight(dim_nonlin)
        params[_p(prefix, 'bx', layer_id)] = np.zeros((dim_nonlin,), dtype=fX)

        params[_p(prefix, 'U_nl', layer_id)] = np.concatenate([orthogonal_weight(dim_nonlin),
                                                               orthogonal_weight(dim_nonlin)], axis=1)
        params[_p(prefix, 'b_nl', layer_id)] = np.zeros((2 * dim_nonlin,), dtype=fX)

        params[_p(prefix, 'Ux_nl', layer_id)] = orthogonal_weight(dim_nonlin)
        params[_p(prefix, 'bx_nl', layer_id)] = np.zeros((dim_nonlin,), dtype=fX)

        # context to LSTM
        params[_p(prefix, 'Wc', layer_id)] = normal_weight(dimctx, dim * 2)

        params[_p(prefix, 'Wcx', layer_id)] = normal_weight(dimctx, dim)
    else:
        params[_p(prefix, 'W', layer_id)] = np.stack([
            np.concatenate([normal_weight(nin, dim), normal_weight(nin, dim)], axis=1)
            for _ in xrange(unit_size)], axis=0)
        params[_p(prefix, 'b', layer_id)] = np.zeros((unit_size, 2 * dim,), dtype=fX)
        params[_p(prefix, 'U', layer_id)] = np.stack([
            np.concatenate([orthogonal_weight(dim_nonlin), orthogonal_weight(dim_nonlin)], axis=1)
            for _ in xrange(unit_size)], axis=0)

        params[_p(prefix, 'Wx', layer_id)] = np.stack(
            [normal_weight(nin_nonlin, dim_nonlin) for _ in xrange(unit_size)], axis=0)
        params[_p(prefix, 'Ux', layer_id)] = np.stack([
            orthogonal_weight(dim_nonlin) for _ in xrange(unit_size)],
            axis=0)
        params[_p(prefix, 'bx', layer_id)] = np.zeros((unit_size, dim_nonlin,), dtype=fX)

        params[_p(prefix, 'U_nl', layer_id)] = np.stack([
            np.concatenate([orthogonal_weight(dim_nonlin),
                            orthogonal_weight(dim_nonlin)], axis=1)
            for _ in xrange(unit_size)], axis=0)
        params[_p(prefix, 'b_nl', layer_id)] = np.zeros((unit_size, 2 * dim_nonlin,), dtype=fX)

        params[_p(prefix, 'Ux_nl', layer_id)] = np.stack([orthogonal_weight(dim_nonlin) for _ in xrange(unit_size)],
                                                         axis=0)
        params[_p(prefix, 'bx_nl', layer_id)] = np.zeros((unit_size, dim_nonlin,), dtype=fX)

        # context to LSTM
        params[_p(prefix, 'Wc', layer_id)] = np.stack([normal_weight(dimctx, dim * 2) for _ in xrange(unit_size)],
                                                      axis=0)

        params[_p(prefix, 'Wcx', layer_id)] = np.stack([normal_weight(dimctx, dim) for _ in xrange(unit_size)], axis=0)

    # attention: combined -> hidden
    params[_p(prefix, 'W_comb_att', layer_id)] = normal_weight(dim, dimctx)

    # attention: context -> hidden
    params[_p(prefix, 'Wc_att', layer_id)] = normal_weight(dimctx)

    # attention: hidden bias
    params[_p(prefix, 'b_att', layer_id)] = np.zeros((dimctx,), dtype=fX)

    # attention:
    params[_p(prefix, 'U_att', layer_id)] = normal_weight(dimctx, 1)
    params[_p(prefix, 'c_tt', layer_id)] = np.zeros((1,), dtype=fX)

    return params


def gru_cond_layer(P, state_below, O, prefix='gru', mask=None, context=None, one_step=False, init_memory=None,
                   init_state=None, context_mask=None, **kwargs):
    """Conditional GRU layer with Attention

    input:
        state_below: ([Tt], [BS], x)    # x = [W] for tgt_embedding
        mask: ([Tt], [BS])
        init_state: ([BS], [H])
        context: ([Tt], [BS], [Hc])
        context_mask: ([Tt], [BS])

    :return list of 3 outputs
        hidden_decoder: ([Tt], [BS], [H]), hidden states of the decoder gru
        context_decoder: ([Tt], [BS], [Hc]), weighted averages of context, generated by attention module
        alpha_decoder: ([Tt], [Bs], [Tt]), weights (alignment matrix)
    """

    layer_id = kwargs.pop('layer_id', 0)
    multi = 'multi' in O.get('unit', 'gru_cond')
    unit_size = kwargs.pop('unit_size', O.get('cond_unit_size', 2))

    kw_ret = {}

    assert context, 'Context must be provided'
    assert context.ndim == 3, 'Context must be 3-d: #annotation * #sample * dim'
    if one_step:
        assert init_state, 'previous state must be provided'

    # Dimensions
    n_steps = state_below.shape[0] if state_below.ndim == 3 else 1
    n_samples = state_below.shape[1] if state_below.ndim == 3 else state_below.shape[0]
    if multi:
        dim = P[_p(prefix, 'Wcx', layer_id)][0].shape[1]
    else:
        dim = P[_p(prefix, 'Wcx', layer_id)].shape[1]
    dropout_params = kwargs.pop('dropout_params', None)

    # Mask
    if mask is None:
        mask = T.alloc(1., n_steps, 1)

    # Initial/previous state
    if init_state is None:
        init_state = T.alloc(0., n_samples, dim)

    projected_context = T.dot(context, P[_p(prefix, 'Wc_att', layer_id)]) + P[_p(prefix, 'b_att', layer_id)]

    # projected x
    if multi:
        state_belowx = T.concatenate([
            T.dot(state_below, P[_p(prefix, 'Wx', layer_id)][j]) + P[_p(prefix, 'bx', layer_id)][j]
            for j in range(unit_size)
        ], axis=-1)
        state_below_ = T.concatenate([
            T.dot(state_below, P[_p(prefix, 'W', layer_id)][j]) + P[_p(prefix, 'b', layer_id)][j]
            for j in range(unit_size)
        ], axis=-1)
    else:
        state_belowx = T.dot(state_below, P[_p(prefix, 'Wx', layer_id)]) + P[_p(prefix, 'bx', layer_id)]
        state_below_ = T.dot(state_below, P[_p(prefix, 'W', layer_id)]) + P[_p(prefix, 'b', layer_id)]

    def _one_step_att_slice(m_, ctx_, h1, Wc, Wcx, U_nl, Ux_nl, b_nl, bx_nl):
        preact2 = T.nnet.sigmoid(T.dot(h1, U_nl) + b_nl + T.dot(ctx_, Wc))

        r2 = _slice(preact2, 0, dim)
        u2 = _slice(preact2, 1, dim)

        preactx2 = (T.dot(h1, Ux_nl) + bx_nl) * r2 + T.dot(ctx_, Wcx)

        h2 = T.tanh(preactx2)

        h2 = u2 * h1 + (1. - u2) * h2
        h2 = m_[:, None] * h2 + (1. - m_)[:, None] * h1

        return h2

    def _step_slice(m_, x_, xx_,
                    h_, ctx_, alpha_,
                    projected_context_, context_,
                    U, Wc, W_comb_att, U_att, c_tt, Ux, Wcx, U_nl, Ux_nl, b_nl, bx_nl):
        h1 = _gru_step_slice(m_, x_, xx_, h_, U, Ux)

        # attention
        ctx_, alpha = _attention(h1, projected_context_, context_, W_comb_att, U_att, c_tt, context_mask=context_mask)

        # GRU 2 (with attention)
        h2 = _one_step_att_slice(m_, ctx_, h1, Wc, Wcx, U_nl, Ux_nl, b_nl, bx_nl)

        return h2, ctx_, alpha.T

    def _multi_step_slice(m_, x_, xx_,
                          h_, ctx_, alpha_,
                          projected_context_, context_,
                          U, Wc, W_comb_att, U_att, c_tt, Ux, Wcx, U_nl, Ux_nl, b_nl, bx_nl):
        h_tmp = h_
        for j in range(unit_size):
            x = _slice(x_, j, 2 * dim)
            xx = _slice(xx_, j, dim)
            h1 = _gru_step_slice(m_, x, xx, h_tmp, U[j], Ux[j])
            h_tmp = h1

        # attention
        ctx_, alpha = _attention(h1, projected_context_, context_, W_comb_att, U_att, c_tt, context_mask=context_mask)

        # GRU 2 (with attention)
        h_tmp_att = h1
        for j in range(unit_size):
            h2 = _one_step_att_slice(m_, ctx_, h_tmp_att, Wc[j], Wcx[j], U_nl[j], Ux_nl[j], b_nl[j], bx_nl[j])
            h_tmp_att = h2

        return h2, ctx_, alpha.T

    seqs = [mask, state_below_, state_belowx]
    if multi:
        _step = _multi_step_slice
    else:
        _step = _step_slice

    shared_vars = [P[_p(prefix, 'U', layer_id)],
                   P[_p(prefix, 'Wc', layer_id)],
                   P[_p(prefix, 'W_comb_att', layer_id)],
                   P[_p(prefix, 'U_att', layer_id)],
                   P[_p(prefix, 'c_tt', layer_id)],
                   P[_p(prefix, 'Ux', layer_id)],
                   P[_p(prefix, 'Wcx', layer_id)],
                   P[_p(prefix, 'U_nl', layer_id)],
                   P[_p(prefix, 'Ux_nl', layer_id)],
                   P[_p(prefix, 'b_nl', layer_id)],
                   P[_p(prefix, 'bx_nl', layer_id)]]

    if one_step:
        result = _step(*(seqs + [init_state, None, None, projected_context, context] + shared_vars))
    else:
        result, _ = theano.scan(
            _step,
            sequences=seqs,
            outputs_info=[init_state,
                          T.alloc(0., n_samples, context.shape[2]),
                          T.alloc(0., n_samples, context.shape[0])],
            non_sequences=[projected_context, context] + shared_vars,
            name=_p(prefix, '_layers'),
            n_steps=n_steps,
            profile=profile,
            strict=True,
        )

    kw_ret['hidden_without_dropout'] = result[0]

    result = list(result)
    result.append(kw_ret)

    if dropout_params:
        result[0] = dropout_layer(result[0], *dropout_params)

    return result


__all__ = [
    'param_init_gru',
    'gru_layer',
    'param_init_gru_cond',
    'gru_cond_layer',
]