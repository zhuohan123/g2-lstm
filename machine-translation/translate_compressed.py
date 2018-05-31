"""
Build a neural machine translation model with soft attention
"""

import cPickle as pkl
import copy
import os
import sys
import time
import math
from pprint import pprint

import numpy as np
import theano
import theano.tensor as tensor

from libs.config import DefaultOptions
from libs.utility.utils import *

from libs.utility.translate import translate_dev_get_bleu
from libs.models import NMTModel

def test(model_name, beam_size, reload_=True, Hard = False, k = -1, type = None):
    print(1)
    with open('%s.pkl' % model_name, 'rb') as f:
        model_options = DefaultOptions.copy()
        model_options.update(pkl.load(f))
    #model_options['temperature'] = 1.0
    #model_options['scale'] = 1.0
    #model_options['gate_dropout'] = 0.0
    #model_options['fix_dp_bug'] = False
    print(2)
    model = NMTModel(model_options)
    #model.O['small_train_datasets'] = (r'\\GCR\Scratch\RR1\dihe\stochastic_lstm\data\test\test.de-en.bpe.25000.de', r'\\GCR\Scratch\RR1\dihe\stochastic_lstm\data\test\test.de-en.bpe.25000.en',) + (r'\\GCR\Scratch\RR1\dihe\stochastic_lstm\data\test\test.de-en.en',)
    print(3)
    params = model.initializer.init_params()
    print(4)
    params = load_params_v2(model_name, params, k, type)
    print(5)
    model.init_tparams(params)
    print(6)
    print(model_options)
    check_options(model_options)

    trng, use_noise, stochastic_mode, hyper_param,\
        x, x_mask, y, y_mask, \
        opt_ret, \
        cost, test_cost, x_emb, stochastic_updates, _ = model.build_model()

    print 'Building sampler'
    f_init, f_next = model.build_sampler(trng=trng, use_noise=use_noise, batch_mode=True, stochastic_mode=stochastic_mode, hyper_param=hyper_param)

    uidx = search_start_uidx(reload_, model_name)

    if Hard:
        stochastic_mode.set_value(2)
        bleu_hard = translate_dev_get_bleu(model, f_init, f_next, trng, use_noise, beam_size, alpha)
        message('{} {} BLEU = {:.2f} at uidx {} beam_size = {}'.format(type, k, bleu_hard, uidx, beam_size))
        sys.stdout.flush()
        bleu_soft = 0.0
    else:
        stochastic_mode.set_value(0)
        alpha = 1.0
        while True:
          bleu_soft = translate_dev_get_bleu(model, f_init, f_next, trng, use_noise, beam_size, alpha)
          message('{} {} {} BLEU = {:.2f} at uidx {} beam_size = {}'.format(type, k, alpha, bleu_soft, uidx, beam_size))
          sys.stdout.flush()
          break
          
        bleu_hard = 0.0

    return (bleu_soft, bleu_hard)


if __name__ == '__main__':
    _filename = sys.argv[1]
    beam_size = int(sys.argv[2])
    filename = _filename
      
    print filename
    test(filename, beam_size, Hard=False)      
    #test(filename, beam_size, Hard=False, k = 10000.0, type = 'precision')
    #test(filename, beam_size, Hard=False, k = 1000.0, type = 'precision')
    #test(filename, beam_size, Hard=False, k = 100.0, type = 'precision')
    #test(filename, beam_size, beam_size, k=1.0, type='precision')
    #test(filename, beam_size, Hard=False, k = 256, type = 'rank')
    #test(filename, beam_size, Hard=False, k = 128, type = 'rank')
    #test(filename, beam_size, k=32, type='rank')
    #test(filename, beam_size, k=16, type='rank')
      
    #idx += 10000

    '''
    pathDir = os.listdir(sys.argv[1])
    key = sys.argv[1].split('\\')[-1]
    dict = []

    fid = open('res.' + key + '.txt', 'r')
    while True:
        line = fid.readline()
        if line == '':
            break
        line = line.split('\t')
        dict.append(line[0] + line[1])
    fid.close()

    for dir in pathDir:
        child_dir = os.path.join('%s\\%s' % (sys.argv[1], dir))
        _dir = dir.split('\\')[-1]
        pathDir2 = os.listdir(child_dir)
        for file in pathDir2:
            filename = os.path.join('%s\\%s' % (child_dir, file))
            if 'iter' in filename and'npz' in filename and 'pkl' not in filename:
                if dir + file not in dict:
                    print "testing " + dir + " " + file
                    res = test(filename, Hard = True)
                    fid = open('res.' + key + '.txt', 'a')
                    fid.writelines(dir + '\t' + file + '\t' + str(res[0]) + '\t' + str(res[1]) + '\n')
                    fid.close()
    '''