#! /usr/bin/python
# -*- encoding: utf-8 -*-

import argparse
import os
import re
import cPickle as pkl
from pprint import pprint

import numpy as np
import theano

from libs.config import DefaultOptions
from libs.models import build_and_init_model
from libs.utility.translate import load_translate_data, seqs2words, translate, translate_block

__author__ = 'fyabc'


def translate_model_single(input_, model_name, options, k, normalize):
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    trng = RandomStreams(1234)
    use_noise = theano.shared(np.float32(0.))

    model, _ = build_and_init_model(model_name, options=options, build=False)

    # word index
    f_init, f_next = model.build_sampler(trng=trng, use_noise=use_noise)

    return translate(input_, model, f_init, f_next, trng, k, normalize)


def main(model, dictionary, dictionary_target, source_file, saveto, k=5,
         normalize=False, chr_level=False, batch_size=-1, args=None):
    batch_mode = batch_size > 0

    # load model model_options
    option_file = '%s.pkl' % model
    if not os.path.exists(option_file):
        m = re.search("iter(\d+)\.npz", model)
        if m:
            uidx = int(m.group((1)))
            option_file = '%s.iter%d.npz.pkl' % (os.path.splitext(model)[0], uidx)
    assert os.path.exists(option_file)

    with open(option_file, 'rb') as f:
        options = DefaultOptions.copy()
        options.update(pkl.load(f))
        if 'fix_dp_bug' not in options:
            options['fix_dp_bug'] = False
        print 'Options:'
        pprint(options)

    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    trng = RandomStreams(1234)
    use_noise = theano.shared(np.float32(0.))

    model_type = 'NMTModel'
    if args.trg_attention:
        model_type = 'TrgAttnNMTModel'

    model, _ = build_and_init_model(model, options=options, build=False, model_type=model_type)

    f_init, f_next = model.build_sampler(trng=trng, use_noise=use_noise, batch_mode=batch_mode, dropout=options['use_dropout'])

    if not batch_mode:
        word_dict, word_idict, word_idict_trg, input_ = load_translate_data(
            dictionary, dictionary_target, source_file,
            batch_mode=False, chr_level=chr_level, options=options,
        )

        print 'Translating ', source_file, '...'
        trans = seqs2words(
            translate(input_, model, f_init, f_next, trng, k, normalize),
            word_idict_trg,
        )
    else:
        word_dict, word_idict, word_idict_trg, all_src_blocks, m_block = load_translate_data(
            dictionary, dictionary_target, source_file,
            batch_mode=True, chr_level=chr_level, n_words_src=options['n_words_src'],batch_size= batch_size,
        )

        print 'Translating ', source_file, '...'
        all_sample = []
        for bidx, seqs in enumerate(all_src_blocks):
            all_sample.extend(translate_block(seqs, model, f_init, f_next, trng, k))
            print bidx, '/', m_block, 'Done'

        trans = seqs2words(all_sample, word_idict_trg)

    with open(saveto, 'w') as f:
        print >> f, '\n'.join(trans)
    print 'Done'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Translate the source language test file to target language with given model (single thread)')
    parser.add_argument('-k', type=int, default=4,
                        help='Beam size (?), default to 4, can also use 12')
    parser.add_argument('-p', type=int, default=5,
                        help='Number of parallel processes, default to 5')
    parser.add_argument('-n', action="store_true", default=False,
                        help='Use normalize, default to False, set to True')
    parser.add_argument('-c', action="store_true", default=False,
                        help='Char level model, default to False, set to True')
    parser.add_argument('-b', type=int, default=-1,
                        help='Batch size, default to -1, means not to use batch mode')
    parser.add_argument('model', type=str, help='The model path')
    parser.add_argument('dictionary_source', type=str, help='The source dict path')
    parser.add_argument('dictionary_target', type=str, help='The target dict path')
    parser.add_argument('source', type=str, help='The source input path')
    parser.add_argument('saveto', type=str, help='The translated file output path')
    parser.add_argument('--trg_att', action='store_true', dest='trg_attention', default=False,
                        help='Use target attention, default is False, set to True')

    args = parser.parse_args()

    main(args.model, args.dictionary_source, args.dictionary_target, args.source,
         args.saveto, k=args.k, normalize=args.n,
         chr_level=args.c, batch_size=args.b, args=args)
