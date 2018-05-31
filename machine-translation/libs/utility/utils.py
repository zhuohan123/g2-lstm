#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

from collections import OrderedDict
import warnings
import os
import cPickle as pkl
from pprint import pprint
import re
import errno
import random
import gzip
import sys
import time

import theano
import theano.tensor as tensor
import numpy as np

from ..constants import *
from .data_iterator import TextIterator

_fp_log = None

def _slice(_x, n, dim):
    """Utility function to slice a tensor."""

    if _x.ndim == 3:
        return _x[:, :, n * dim:(n + 1) * dim]
    return _x[:, n * dim:(n + 1) * dim]

def set_logging_file(logging_filename):
    path, filename = os.path.split(logging_filename)

    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            print('ERROR when creating the logging file: {}'.format(e.message))

    global _fp_log
    _fp_log = open(logging_filename, 'w')


def get_logging_file():
    return _fp_log


def message(*args, **kwargs):
    """Print message both to logging file and stdout."""
    print(*args, **kwargs)

    if _fp_log is not None:
        kwargs['file'] = _fp_log
        print(*args, **kwargs)


def log(*args, **kwargs):
    """Print message to logging file."""
    if _fp_log is not None:
        kwargs['file'] = _fp_log
        print(*args, **kwargs)


def close_logging_file():
    if _fp_log is not None:
        _fp_log.close()


def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)


def zipp(params, tparams):
    """Push parameters to Theano shared variables"""
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)


def unzip(zipped):
    """Pull parameters from Theano shared variables"""
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params


def itemlist(tparams):
    """Get the list of parameters: Note that tparams must be OrderedDict"""
    return [vv for kk, vv in tparams.iteritems()]


def _p(*args, **kwargs):
    """Make prefix-appended name"""

    # FIXME: To be compatible with old model, when the layer id is 0 and open 'layer_id_compatible', omit the layer id.
    layer_id_compatible = kwargs.pop('layer_id_compatible', True)
    if layer_id_compatible and args[-1] == 0:
        args = args[:-1]

    return '_'.join(str(arg) for arg in args)


# These parameters should be duplicated for multiverso.
dup_shared_var_list = ['decoder_c_tt']
dup_size = 100

def is_dup_params(name):
    return name.startswith(tuple(dup_shared_var_list))


def init_tparams(params, given_tparams=None, given_dup_tparams=None, use_mv=False):
    """Initialize Theano shared variables according to the initial parameters"""

    tparams = OrderedDict() if given_tparams is None else given_tparams
    dup_tparams = OrderedDict() if given_dup_tparams is None else given_dup_tparams

    if not use_mv:
        for kk, pp in params.iteritems():
            tparams[kk] = theano.shared(params[kk], name=kk)
    else:
        try:
            from multiverso.theano_ext import sharedvar
        except ImportError:
            from ..multiverso_.theano_ext import sharedvar

        for kk, pp in params.iteritems():
            if any(kk.startswith(var) for var in dup_shared_var_list):
                tparams[kk] = theano.shared(params[kk], name=kk)
                dup_tparams[kk] = sharedvar.mv_shared(value=np.ones(dup_size) * params[kk][0], name=kk, borrow=False)
            else:
                tparams[kk] = sharedvar.mv_shared(value=params[kk], name=kk, borrow=False)
    return tparams, dup_tparams


def sync_tparams(tparams, dup_tparams):
    try:
        from multiverso.theano_ext import sharedvar
    except ImportError:
        from ..multiverso_.theano_ext import sharedvar

    for kk, vv in dup_tparams.iteritems():
        vv.set_value(np.ones(dup_size) * tparams[kk].get_value()[0])
    sharedvar.sync_all_mv_shared_vars()
    for kk, vv in dup_tparams.iteritems():
        tparams[kk].set_value(np.array([vv.get_value()[0]], dtype=fX).reshape((1,)))

def all_reduce_params(sent_shared_params, rec_buffers, average_cnt = 1):
    from mpi4py import MPI
    mpi_communicator = MPI.COMM_WORLD
    commu_time = 0.0
    gpu2cpu_cp_time = 0.0
    for (sent_model, rec_model) in zip(sent_shared_params, rec_buffers):
        cp_start = time.time()
        model_val = sent_model.get_value()
        gpu2cpu_cp_time += time.time() - cp_start

        commu_start = time.time()
        mpi_communicator.Allreduce([model_val, MPI.FLOAT], [rec_model, MPI.FLOAT], op=MPI.SUM)
        commu_time += time.time() - commu_start
        if average_cnt != 1: #try to avoid dividing since it is very cost
            rec_model = rec_model / average_cnt

        cp_start = time.time()
        sent_model.set_value(rec_model)
        gpu2cpu_cp_time += time.time() - cp_start
    return commu_time,  gpu2cpu_cp_time

def all_reduce_params_nccl(nccl_commu, sent_shared_params):
    commu_time = 0.0
    gpu2gpu_cp_time = 0.0

    for sent_model in sent_shared_params:
        cp_start = time.time()
        model_val = sent_model.get_value(borrow = True, return_internal_type=True)
        gpu2gpu_cp_time += time.time() - cp_start

        commu_start = time.time()
        get_value = nccl_commu.all_reduce(model_val, op= "sum")
        commu_time += time.time() - commu_start

        cp_start = time.time()
        sent_model.set_value(get_value)
        gpu2gpu_cp_time += time.time() - cp_start

    return commu_time, gpu2gpu_cp_time

def init_nccl_env(mpi_comm):
    from pygpu import collectives as gpucoll
    from theano import gpuarray as theanoga

    gpu_name = None
    gpu_ctx = theanoga.get_context(gpu_name)
    commid = gpucoll.GpuCommCliqueId(gpu_ctx)
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()

    data = commid.comm_id if mpi_rank == 0 else None
    data = mpi_comm.bcast(data, root = 0)
    commid.comm_id = data

    comm = gpucoll.GpuComm(commid, mpi_size, mpi_rank)
    print('Init pygpu OK, rank %d' % mpi_rank)
    sys.stdout.flush()
    return comm

def load_params(path, params, src_map_file = None, tgt_map_file = None):
    """Load parameters

    :param path: Path of old parameters.
    :param params: New parameters to be updated.
    """

    old_params = np.load(path)
    for key, value in params.iteritems():
        if key not in old_params:
            warnings.warn('{} is not in the archive'.format(key))
            continue
        if params[key].shape == old_params[key].shape:
            params[key] = old_params[key]
            g2 = np.multiply(old_params[key], old_params[key])
            g2 = g2.sum() / (old_params[key].flatten()).shape[0]
            if g2 > 0.1:
                print (g2)
                print (key)


    if src_map_file and tgt_map_file:
        src_map = pkl.load(open(src_map_file, 'rb'))
        tgt_map = pkl.load(open(tgt_map_file, 'rb'))
        params = load_word_params(params, old_params, src_map, tgt_map)

    return params

def low_rank_approx(SVD=None, A=None, r=1):
    """
    Computes an r-rank approximation of a matrix
    given the component u, s, and v of it's SVD
    Requires: numpy
    """
    if not SVD:
        SVD = np.linalg.svd(A, full_matrices=False)
    u, s, v = SVD
    Ar = np.zeros((len(u), len(v)))
    for i in xrange(r):
        Ar += s[i] * np.outer(u.T[i], v[i])
    return Ar

def load_params_v2(path, params, k = -1, type = None):
    """Load parameters

    :param path: Path of old parameters.
    :param params: New parameters to be updated.
    """

    old_params = np.load(path)
    for key, value in params.iteritems():
        if key not in old_params:
            warnings.warn('{} is not in the archive'.format(key))
            continue
        if params[key].shape == old_params[key].shape:
            params[key] = old_params[key]

            if ('U' in key or 'Wc' in key) and params[key].shape[1] != 1:

                dim = old_params[key].shape[1] // 4
                para_i = old_params[key][:, 0 * dim:1 * dim]
                para_f = old_params[key][:, 1 * dim:2 * dim]
                para_o = old_params[key][:, 2 * dim:3 * dim]
                para_c = old_params[key][:, 3 * dim:4 * dim]
                if type == 'precision':
                
                    para_i = np.clip(para_i, -1.0, 1.0)
                    para_f = np.clip(para_f, -1.0, 1.0)
                    
                    para_i = np.round(para_i * k) / k
                    para_f = np.round(para_f * k) / k

                    params[key][:, 0 * dim:1 * dim] = para_i
                    params[key][:, 1 * dim:2 * dim] = para_f
                    params[key][:, 2 * dim:3 * dim] = para_o
                    params[key][:, 3 * dim:4 * dim] = para_c

                if type == 'rank':
                    u, s, v = np.linalg.svd(para_i, full_matrices=False)
                    para_i = low_rank_approx((u, s, v), r=k)
                    u, s, v = np.linalg.svd(para_f, full_matrices=False)
                    para_f = low_rank_approx((u, s, v), r=k)

                    params[key][:, 0 * dim:(1) * dim] = para_i
                    params[key][:, 1 * dim:(2) * dim] = para_f
                    params[key][:, 2 * dim:(3) * dim] = para_o
                    params[key][:, 3 * dim:(4) * dim] = para_c

    return params

def load_embedding(params, embedding_model_file, emb_keys=('Wemb', 'Wemb_dec')):
    embedding_model = np.load(embedding_model_file)

    for key in emb_keys:
        params[key] = embedding_model[key]

    return params

def load_word_params(params, old_params, src_map, tgt_map):
    UNK_ID = 1

    src_vocab_size = params['Wemb'].shape[0]
    Wemb_T = np.tile(old_params['Wemb'][UNK_ID], [src_vocab_size, 1]).T
    old_Wemb_T = old_params['Wemb'].T #transpose to speed up by more efficient cache
    print(Wemb_T.shape, old_Wemb_T.shape)
    for (new_word_id, old_word_id) in src_map.iteritems():
        if new_word_id < src_vocab_size and old_word_id < old_params['Wemb'].shape[0]:
            Wemb_T[:, new_word_id] = old_Wemb_T[:, old_word_id]
        sys.stdout.write('\r%d %d' % (new_word_id, old_word_id))
    params['Wemb'] = Wemb_T.T

    tgt_vocab_size = params['Wemb_dec'].shape[0]
    Wemb_dec_T = np.tile(old_params['Wemb_dec'][UNK_ID], [tgt_vocab_size, 1]).T
    old_Wemb_dec_T = old_params['Wemb_dec'].T
    params['ff_logit_W'] = np.tile(old_params['ff_logit_W'][:,UNK_ID], [tgt_vocab_size,1]).T
    params['ff_logit_b'].fill(old_params['ff_logit_b'][UNK_ID])
    print(params['Wemb_dec'].shape, params['ff_logit_W'].shape, params['ff_logit_b'].shape)
    for (new_word_id, old_word_id) in tgt_map.iteritems():
        if new_word_id < tgt_vocab_size and old_word_id < old_params['Wemb_dec'].shape[0]:
            Wemb_dec_T[:,new_word_id] = old_Wemb_dec_T[:,old_word_id]
            params['ff_logit_W'][:,new_word_id] = old_params['ff_logit_W'][:,old_word_id]
            params['ff_logit_b'][new_word_id] = old_params['ff_logit_b'][old_word_id]
        sys.stdout.write('\r%d %d' % (new_word_id, old_word_id))
    params['Wemb_dec'] = Wemb_dec_T.T

    print('\nLoad previous word related params done')
    return params

# some utilities
def orthogonal_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype('float32')


def normal_weight(nin, nout=None, scale=0.01, orthogonal=True):
    if nout is None:
        nout = nin
    if nout == nin and orthogonal:
        W = orthogonal_weight(nin)
    else:
        W = np.random.randn(nin, nout)
        u, s, v = np.linalg.svd(W)
        if nin > nout:
            W = u[:, :nout]
        else:
            W = v[:nin, :]
    return W.astype('float32')


def uniform_weight(nin, nout=None, scale=0.01):
    if nout is None:
        nout = nin
    return np.random.uniform(-1. * scale, 1. * scale, (nin, nout)).astype('float32')


def concatenate(tensor_list, axis=0):
    """
    Alternative implementation of `theano.tensor.concatenate`.
    This function does exactly the same thing, but contrary to Theano's own
    implementation, the gradient is implemented on the GPU.
    Backpropagating through `theano.tensor.concatenate` yields slowdowns
    because the inverse operation (splitting) needs to be done on the CPU.
    This implementation does not have that problem.
    :usage:
        >>> x, y = theano.tensor.matrices('x', 'y')
        >>> c = concatenate([x, y], axis=1)
    :parameters:
        - tensor_list : list
            list of Theano tensor expressions that should be concatenated.
        - axis : int
            the tensors will be joined along this axis.
    :returns:
        - out : tensor
            the concatenated tensor expression.
    """
    concat_size = sum(tt.shape[axis] for tt in tensor_list)

    output_shape = ()
    for k in range(axis):
        output_shape += (tensor_list[0].shape[k],)
    output_shape += (concat_size,)
    for k in range(axis + 1, tensor_list[0].ndim):
        output_shape += (tensor_list[0].shape[k],)

    out = tensor.zeros(output_shape)
    offset = 0
    for tt in tensor_list:
        indices = ()
        for k in range(axis):
            indices += (slice(None),)
        indices += (slice(offset, offset + tt.shape[axis]),)
        for k in range(axis + 1, tensor_list[0].ndim):
            indices += (slice(None),)

        out = tensor.set_subtensor(out[indices], tt)
        offset += tt.shape[axis]

    return out


def average(l):
    if not l:
        return 0.0
    return sum(l) / len(l)


def apply_gradient_clipping(clip_c, grads, clip_shared=None):
    g2 = 0.
    if clip_c > 0.:
        clip_shared = theano.shared(np.array(clip_c, dtype=fX), name='clip_shared') if clip_shared is None else clip_shared

        for g in grads:
            g2 += (g ** 2).sum()
        new_grads = []
        for g in grads:
            new_grads.append(tensor.switch(g2 > (clip_shared ** 2),
                                           g / tensor.sqrt(g2) * clip_shared,
                                           g))
        grads = new_grads
    return grads, g2

def clip_grad_remove_nan(grads, clip_c_shared, mt_tparams):
    g2 = 0.
    for g in grads:
        g2 += (g*g).sum()
    not_finite = tensor.or_(tensor.isnan(g2), tensor.isinf(g2))
    if clip_c_shared.get_value() > 0.:
        new_grads = []
        for g, p in zip(grads, itemlist(mt_tparams)):
            tmpg = tensor.switch(g2 > (clip_c_shared*clip_c_shared),
                                 g / tensor.sqrt(g2) * clip_c_shared,
                                 g)
            new_grads.append(tensor.switch(not_finite, np.float32(0.1)*p, tmpg))

        return new_grads, tensor.sqrt(g2)
    else:
        return grads, tensor.sqrt(g2)

def make_grads_clip_func(grads_shared, mt_tparams, clip_c_shared):

    new_grads, g2_sqrt = clip_grad_remove_nan(grads_shared, clip_c_shared, mt_tparams)

    zgup = [(zg, g) for zg, g in zip(grads_shared, new_grads)]
    f_grads_clip = theano.function([], g2_sqrt, updates=zgup, on_unused_input='ignore', profile=profile)

    return f_grads_clip


def l2_regularization(cost, tparams, decay_c):
    """Apply L2 regularization on weights."""
    if decay_c > 0.:
        decay_c = theano.shared(np.float32(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in tparams.iteritems():
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay
    return cost


def regularize_alpha_weights(cost, alpha_c, model_options, x_mask, y_mask, opt_ret):
    """Regularize the alpha weights."""
    if alpha_c > 0. and not model_options['decoder'].endswith('simple'):
        alpha_c = theano.shared(np.float32(alpha_c), name='alpha_c')
        alpha_reg = alpha_c * (
            (tensor.cast(y_mask.sum(0) // x_mask.sum(0), fX)[:, None] -
             opt_ret['dec_alphas'].sum(0)) ** 2).sum(1).mean()
        cost += alpha_reg
    return cost


# batch preparation
def prepare_data(seqs_x, seqs_y, maxlen=None):
    # x: a list of sentences
    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [len(s) for s in seqs_y]

    if maxlen is not None:
        new_seqs_x = []
        new_seqs_y = []
        new_lengths_x = []
        new_lengths_y = []
        for l_x, s_x, l_y, s_y in zip(lengths_x, seqs_x, lengths_y, seqs_y):
            if l_x < maxlen and l_y < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
                new_seqs_y.append(s_y)
                new_lengths_y.append(l_y)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
        lengths_y = new_lengths_y
        seqs_y = new_seqs_y

        if len(lengths_x) < 1 or len(lengths_y) < 1:
            return None, None, None, None

    n_samples = len(seqs_x)
    maxlen_x = np.max(lengths_x) + 1
    maxlen_y = np.max(lengths_y) + 1

    x = np.zeros((maxlen_x, n_samples)).astype('int64')
    y = np.zeros((maxlen_y, n_samples)).astype('int64')
    x_mask = np.zeros((maxlen_x, n_samples)).astype('float32')
    y_mask = np.zeros((maxlen_y, n_samples)).astype('float32')
    for idx, [s_x, s_y] in enumerate(zip(seqs_x, seqs_y)):
        x[:lengths_x[idx], idx] = s_x
        x_mask[:lengths_x[idx] + 1, idx] = 1.
        y[:lengths_y[idx], idx] = s_y
        y_mask[:lengths_y[idx] + 1, idx] = 1.

    return x, x_mask, y, y_mask


def prepare_data_x(seqs_x, maxlen=None, pad_eos=True, pad_sos=False, n_word=30000):
    # x: a list of sentences
    lengths_x = [len(s) for s in seqs_x]

    if maxlen is not None:
        new_seqs_x = []
        new_lengths_x = []
        for l_x, s_x in zip(lengths_x, seqs_x):
            if l_x < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)

        lengths_x = new_lengths_x
        seqs_x = new_seqs_x

        if len(lengths_x) < 1:
            return None, None,

    n_samples = len(seqs_x)
    if pad_eos:
        maxlen_x = np.max(lengths_x) + 1
    else:
        maxlen_x = np.max(lengths_x)

    x = np.zeros((maxlen_x, n_samples)).astype('int64')

    x_mask = np.zeros((maxlen_x, n_samples)).astype('float32')

    for idx, s_x in enumerate(seqs_x):
        x[:lengths_x[idx], idx] = s_x
        if pad_eos:
            x_mask[:lengths_x[idx]+1, idx] = 1.
        else:
            x_mask[:lengths_x[idx], idx] = 1.

    if pad_sos:
        x = np.concatenate((
            np.full([1, n_samples], n_word - 1, dtype='int64'), x
        ), axis=0)
        x_mask = np.concatenate((
            np.full([1, n_samples], 1., dtype='float32'), x_mask
        ), axis=0)

    return x, x_mask


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """
    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
        minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if minibatch_start != n:
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return list(enumerate(minibatches))


# Debug utilities
def print_params(params, exit_=False):
    total_parameters = 0

    for k, v in params.iteritems():
        total_parameters += v.size
    print('Total parameters of the network: {}'.format(total_parameters))

    print('Model Parameters:')
    for idx, (k, v) in enumerate(params.iteritems()):
        ratio = v.size * 100.0 / total_parameters
        print('  >', k, v.shape, v.dtype, '%.2f%%' % ratio)
    sys.stdout.flush()

    if exit_:
        exit(0)


def load_options(options, reload_=None, preload=None, maintain_vocab_size = False):
    """Reload options."""

    reload_ = options['reload_'] if reload_ is None else reload_
    preload = options['preload'] if preload is None else preload
    dropout = options['use_dropout']
    valid_datasets = options['valid_datasets']

    src_vocab_size = options['n_words_src']
    tgt_vocab_size = options['n_words']

    if reload_ and os.path.exists(preload):
        print('Reloading model options')
        with open('{}.pkl'.format(preload), 'rb') as f:
            # model_options = pkl.load(f)
            # FIXME: Update the option instead of replace it
            options.update(pkl.load(f))

        # Remain reload_, preload and dropout
        options['reload_'] = reload_
        options['preload'] = preload
        options['use_dropout'] = dropout
        options['valid_datasets'] = valid_datasets
        if maintain_vocab_size:
            options['n_words_src'] = src_vocab_size
            options['n_words'] = tgt_vocab_size

def save_options(options, iteration, saveto=None):
    saveto = options['saveto'] if saveto is None else saveto

    save_filename = '{}.iter{}.npz.pkl'.format(os.path.splitext(saveto)[0], iteration) \
        if iteration != -1 else '{}.pkl'.format(saveto)

    with open(save_filename, 'wb') as f:
        pkl.dump(options, f)


def check_options(options):
    """Check conflict options."""

    assert options['lr_discount_freq'] <= 0 or options['fine_tune_patience'] <= 0, \
        'Cannot enable lr discount and fine-tune at the same time'

    if 'multi' in options['unit']:
        assert options['unit_size'] > 0 and options['cond_unit_size'] > 0, 'Unit size must > 0'

    if options['reload_']:
        assert os.path.exists(options['preload'])


def search_start_uidx(reload_, preload):
    if not reload_:
        return 0

    m = re.search('.+iter(\d+?)\.npz', preload)
    if m:
        return int(m.group(1))
    else:
        return 0


def make_f_train(f_grad_shared, f_update):
    def f_train(x, x_mask, y, y_mask, lr):
        cost = f_grad_shared(x, x_mask, y, y_mask)

        f_update(lr)

        return cost

    return f_train


def get_adadelta_imm_data(optimizer, given_imm, preload):
    if given_imm:
        # [NOTE] preload filename format: filename.iter10000.npz
        _real_filename = os.path.splitext(os.path.splitext(preload)[0])[0]
        given_imm_filename = ImmediateFilename.format(_real_filename)

        # For back compatibility
        given_imm_filename_backup = ImmediateFilenameBackup.format(_real_filename)
        given_imm_filename_backup2 = ImmediateFilenameBackup2.format(_real_filename)
        if os.path.exists(given_imm_filename):
            message('Loading adadelta immediate data')
            with np.load(given_imm_filename) as data:
                data = data['arr_0']
                data_size = len(data)
                if optimizer == 'adadelta':
                    return [
                        [data[i] for i in range(0, data_size // 2)],
                        [data[i] for i in range(data_size // 2, data_size)],
                    ]
                elif optimizer == 'adam':
                    return [
                        data[0],
                        [data[i] for i in range(1, data_size // 2 + 1)],
                        [data[i] for i in range(data_size // 2 + 1, data_size)],
                    ]
                else:
                    pass
        elif os.path.exists(given_imm_filename_backup):
            message('Loading adadelta immediate data')
            return pkl.load(fopen(given_imm_filename_backup, 'rb'))
        elif os.path.exists(given_imm_filename_backup2):
            message('Loading adadelta immediate data')
            return pkl.load(fopen(given_imm_filename_backup2, 'rb'))
        else:
            message('Immediate data not found.')
    return None


def dump_adadelta_imm_data(optimizer, imm_shared, dump_imm, saveto):
    if optimizer == 'sgd':
        return

    if imm_shared is None or dump_imm is None:
        return

    tmp_filename = TempImmediateFilename.format(os.path.splitext(saveto)[0])
    imm_filename = ImmediateFilename.format(os.path.splitext(saveto)[0])

    # Dump to temp file
    message('Dumping adadelta immediate data to temp file...', end='')
    if optimizer == 'adadelta':
        np.savez(tmp_filename,
                 np.array([g.get_value() for g in imm_shared[0]] +
                          [g.get_value() for g in imm_shared[1]], dtype=object))
    elif optimizer == 'adam':
        np.savez(tmp_filename,
                 np.array([imm_shared[0].get_value()] +
                          [g.get_value() for g in imm_shared[1]] +
                          [g.get_value() for g in imm_shared[2]], dtype=object))
    else:
        pass
    message('Done')

    # Move temp file to immediate file
    message('Moving temp file to immediate file...', end='')
    try:
        os.remove(ImmediateFilenameBackup.format(os.path.splitext(saveto)[0]))
    except OSError as e:
        if e.errno == errno.ENOENT:
            pass
        else:
            raise
    try:
        os.remove(ImmediateFilenameBackup2.format(os.path.splitext(saveto)[0]))
    except OSError as e:
        if e.errno == errno.ENOENT:
            pass
        else:
            raise
    try:
        os.remove(imm_filename)
    except OSError as e:
        if e.errno == errno.ENOENT:
            pass
        else:
            raise
    os.rename(tmp_filename, imm_filename)
    message('Done')

def adadelta_set_imm_data(optimizer, given_imm_data, imm_shared):
    if optimizer == 'adadelta':
        running_up2, running_grads2 = imm_shared[0], imm_shared[1]
        for (ru, rg, ru_given, rg_given) in zip(running_up2, running_grads2, given_imm_data[0], given_imm_data[1]):
            ru.set_value(ru_given)
            rg.set_value(rg_given)
    else: #TODO: add adam support
        return

def create_shuffle_data(datasets_orig, dataset_src, dataset_tgt):
    orig_src, orig_tgt = datasets_orig[0], datasets_orig[1]

    with open(orig_src) as orig_f_src:
        l_src = list(orig_f_src)
    with open(orig_tgt) as orig_f_tgt:
        l_tgt = list(orig_f_tgt)

    new_idx = range(len(l_src))
    random.shuffle(new_idx)

    with open(dataset_src, 'w') as new_f_src:
        new_f_src.writelines((l_src[i] for i in new_idx))
    with open(dataset_tgt, 'w') as new_f_tgt:
        new_f_tgt.writelines((l_tgt[i] for i in new_idx))


def load_shuffle_text_iterator(
        epoch, worker_id, text_iterator_list,
        datasets, vocab_filenames, batch_size, maxlen, n_words_src, n_words
):
    e = (epoch + worker_id) % ShuffleCycle

    if text_iterator_list[e] is None:
        # Create new text iterator
        message('Creating text iterator {}...'.format(e), end='')
        dataset_src = '{}_{}'.format(datasets[0], e)
        dataset_tgt = '{}_{}'.format(datasets[1], e)

        if not os.path.exists(dataset_src):
            message('file "{}" and "{}" not exist, creating...'.format(dataset_src, dataset_tgt), end='')
            create_shuffle_data(datasets, dataset_src, dataset_tgt)
            message('Done')

        text_iterator_list[e] = TextIterator(
            dataset_src, dataset_tgt,
            vocab_filenames[0], vocab_filenames[1],
            batch_size, n_words_src, n_words, maxlen
        )
        message('Done')
        return text_iterator_list[e]
    else:
        # Reset current text iterator
        message('Reset text iterator {}'.format(e))
        text_iterator_list[e].reset()
        return text_iterator_list[e]

def get_epoch_batch_cnt(dataset_src, dataset_tgt, vocab_filenames, batch_size, maxlen, n_words_src, n_words):

    text_iterator = TextIterator(
        dataset_src, dataset_tgt,
        vocab_filenames[0], vocab_filenames[1],
        batch_size, n_words_src, n_words,maxlen
    )
    n_batches = 0
    for (x, y) in text_iterator:
        n_batches += 1
    return n_batches

__all__ = [
    'set_logging_file',
    'get_logging_file',
    'message',
    'log',
    'close_logging_file',
    'zipp',
    'unzip',
    'itemlist',
    '_p',
    'init_tparams',
    'sync_tparams',
    'init_nccl_env',
    'all_reduce_params',
    'all_reduce_params_nccl',
    'load_params',
    'load_params_v2',
    'load_embedding',
    'orthogonal_weight',
    'normal_weight',
    'uniform_weight',
    'concatenate',
    'average',
    'apply_gradient_clipping',
    'clip_grad_remove_nan',
    'l2_regularization',
    'regularize_alpha_weights',
    'prepare_data',
    'prepare_data_x',
    'get_minibatches_idx',
    'get_epoch_batch_cnt',
    'print_params',
    'load_options',
    'save_options',
    'check_options',
    'search_start_uidx',
    'make_f_train',
    'get_adadelta_imm_data',
    'dump_adadelta_imm_data',
    'load_shuffle_text_iterator',
    'make_grads_clip_func',
    'adadelta_set_imm_data',
]
