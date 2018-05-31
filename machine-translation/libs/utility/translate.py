#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function

import os
import re
import cPickle as pkl
import numpy as np
import subprocess

from .utils import prepare_data_x

__author__ = 'fyabc'


def translate(input_, model, f_init, f_next, trng, k, normalize):
    def _trans(seq):
        # sample given an input sequence and obtain scores
        sample, score = model.gen_sample(
            f_init, f_next,
            np.array(seq).reshape([len(seq), 1]),
            trng=trng, k=k, maxlen=200,
            stochastic=False, argmax=False,
        )

        # normalize scores according to sequence lengths
        if normalize:
            lengths = np.array([len(s) for s in sample])
            score = score / lengths
        sidx = np.argmin(score)
        return sample[sidx]

    output = []

    for idx, x in enumerate(input_):
        output.append(_trans(x))

    return output


def translate_block(input_, model, f_init, f_next, trng, k, alpha):
    """Translate for batch sampler.

    :return output: a list of word indices
    """
    x, x_mask = prepare_data_x(input_, maxlen=None, pad_eos=True, pad_sos=False)

    batch_sample, batch_sample_score = model.gen_batch_sample(
        f_init, f_next, x, x_mask, trng,
        k=k, maxlen=200, eos_id=0,
    )
    assert len(batch_sample) == len(batch_sample_score)

    output = []

    for sample, sample_score in zip(batch_sample, batch_sample_score):
        score = sample_score / np.power(np.array([len(s) for s in sample], dtype= np.float32), alpha)
        # sidx = np.argsort(score)
        # output.append([sample[ii] for ii in sidx])
        output.append(sample[np.argmin(score)])

    return output


def load_translate_data(dictionary, dictionary_target, source_file, batch_mode=False, **kwargs):
    chr_level = kwargs.pop('chr_level', False)
    unk_id = kwargs.pop('unk_id', 1)
    n_words_src = kwargs.pop('n_words_src', 30000)
    echo = kwargs.pop('echo', True)
    load_input = kwargs.pop('load_input', True)

    # load source dictionary and invert
    if echo:
        print('Load and invert source dictionary...', end='')
    with open(dictionary, 'rb') as f:
        word_dict = pkl.load(f)
    word_idict = {v: k for k, v in word_dict.iteritems()}
    word_idict[0] = '<eos>'
    word_idict[unk_id] = 'UNK'
    if echo:
        print('Done')

    # load target dictionary and invert
    if echo:
        print('Load and invert target dictionary...', end='')
    with open(dictionary_target, 'rb') as f:
        word_dict_trg = pkl.load(f)
    word_idict_trg = {v: k for k, v in word_dict_trg.iteritems()}
    word_idict_trg[0] = '<eos>'
    word_idict_trg[unk_id] = 'UNK'
    if echo:
        print('Done')

    if not load_input:
        return word_dict, word_idict, word_idict_trg

    if not batch_mode:
        input_ = []

        if echo:
            print('Loading input...', end='')

        with open(source_file, 'r') as f:
            for idx, line in enumerate(f):
                if chr_level:
                    words = list(line.decode('utf-8').strip())
                else:
                    words = line.strip().split()

                x = [word_dict[w] if w in word_dict else unk_id for w in words]
                x = [ii if ii < n_words_src else unk_id for ii in x]
                x.append(0)

                input_.append(x)
        if echo:
            print('Done')

        return word_dict, word_idict, word_idict_trg, input_
    else:
        batch_size = kwargs.pop('batch_size', 128)

        with open(source_file, 'r') as f:
            all_src_sent = [line.strip().split() for line in f]

        all_src_num = []
        for seg in all_src_sent:
            tmp = [word_dict.get(w, unk_id) for w in seg]
            all_src_num.append([w if w < n_words_src else unk_id for w in tmp])

        all_src_blocks = []
        m_block = (len(all_src_num) + batch_size - 1) // batch_size

        for idx in xrange(m_block):
            all_src_blocks.append(all_src_num[batch_size * idx: batch_size * (idx + 1)])

        return word_dict, word_idict, word_idict_trg, all_src_blocks, m_block


def seqs2words(caps, word_idict_trg):
    """Sequences -> Sentences

    :param caps: a list of word indices
    :param word_idict_trg: inverted target word dict
    :return: a list of sentences
    """

    capsw = []
    for cc in caps:
        ww = []
        for w in cc:
            if w == 0:
                break
            ww.append(word_idict_trg[w])
        capsw.append(' '.join(ww))
    return capsw


def _translate_whole(model, f_init, f_next, trng, dictionary, dictionary_target, source_file,
                     k=5, alpha=1.0, normalize=False, chr_level=False, **kwargs):
    n_words_src = kwargs.pop('n_words_src', model.O['n_words_src'])
    batch_mode = kwargs.pop('batch_mode', False)

    # Translate file
    if not batch_mode:
        word_dict, word_idict, word_idict_trg, input_ = load_translate_data(
            dictionary, dictionary_target, source_file,
            batch_mode=batch_mode, chr_level=chr_level, n_words_src=n_words_src,
            echo=False,
        )

        trans = seqs2words(
            translate(input_, model, f_init, f_next, trng, k, normalize),
            word_idict_trg,
        )

        return '\n'.join(trans) + '\n'
    else:
        word_dict, word_idict, word_idict_trg, all_src_blocks, m_block = load_translate_data(
            dictionary, dictionary_target, source_file,
            batch_mode=batch_mode, chr_level=chr_level, n_words_src=n_words_src,
            echo=False, batch_size=128,
        )

        all_sample = []
        for bidx, seqs in enumerate(all_src_blocks):
            all_sample.extend(translate_block(seqs, model, f_init, f_next, trng, k, alpha))
            #print(bidx, '/', m_block, 'Done')

        trans = seqs2words(all_sample, word_idict_trg)

        return '\n'.join(trans) + '\n'


def get_bleu(ref_file, hyp_in=None, type_in='filename'):
    """Get BLEU score, it will call script 'multi-bleu.perl'.

    :param ref_file: standard test filename of target language.
    :param hyp_in: input from _translate_whole script.
    :param type_in: input type, default is 'filename', can be 'filename' or 'string'.
    :return:
    """
    if type_in == 'filename':
        pl_process = subprocess.Popen(
            'perl scripts/moses/multi-bleu.perl {} < {}\n'.format(ref_file, hyp_in), shell=True,
            stdout=subprocess.PIPE, stderr=open(os.devnull, 'w'))
        pl_output = pl_process.stdout.read()
    elif type_in == 'string':
        pl_process = subprocess.Popen(
            'perl scripts/moses/multi-bleu.perl {}\n'.format(ref_file), shell=True, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=open(os.devnull, 'w'))
        pl_output = pl_process.communicate(hyp_in)[0]
    else:
        raise ValueError('Wrong type_in')
    contents = pl_output.split(',')
    if len(contents) == 0:
        return 0.0
    var = contents[0].split(" = ")
    if len(var) <= 1:
        return 0.0
    BLEU = var[1]

    return float(BLEU)


def de_bpe(input_str):
    return re.sub(r'(@@ )|(@@ ?$)', '', input_str)


def translate_dev_get_bleu(model, f_init, f_next, trng, use_noise, beam_size, alpha, **kwargs):
    dataset = kwargs.pop('dataset', model.O['task'])

    # [NOTE]: Filenames here are with path prefix.
    dev1 = kwargs.pop('dev1', model.O['small_train_datasets'][0])
    dev2 = kwargs.pop('dev2', model.O['small_train_datasets'][2])
    dic1 = kwargs.pop('dic1', model.O['vocab_filenames'][0])
    dic2 = kwargs.pop('dic2', model.O['vocab_filenames'][1])
    use_noise.set_value(0.)

    translated_string = _translate_whole(
        model, f_init, f_next, trng,
        dic1, dic2, dev1,
        k=beam_size, alpha=alpha, batch_mode=True,
    )

    use_noise.set_value(1.)

    # first de-truecase, then de-bpe
    if 'tc' in dataset:
        translated_string = subprocess.Popen(
            'perl detruecase.perl',
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(os.devnull, 'w'),
            shell=True,
        ).communicate(translated_string)[0]

    if 'bpe' in dataset:
        translated_string = de_bpe(translated_string)

    return get_bleu(dev2, translated_string, type_in='string')


__all__ = [
    'get_bleu',
    'de_bpe',
    'translate_dev_get_bleu',
]
