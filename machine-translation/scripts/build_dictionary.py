#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function

import numpy

try:
    import cPickle as pkl
except:
    import pickle as pkl

import sys
import os
import fileinput

from collections import OrderedDict
import argparse

__author__ = 'fyabc'


def real_main(args):
    if args.output is None:
        args.output = '{}.pkl'.format(args.input[0])

    tgt_filename = os.path.join('data', 'dic', args.output)

    word_freqs = OrderedDict()
    worddict = OrderedDict()
    worddict['eos'] = 0
    worddict['UNK'] = 1

    for filename in args.input:
        src_filename = os.path.join('data', 'train', filename)

        print('Processing', src_filename)

        with open(src_filename, 'r') as f:
            for line in f:
                words_in = line.strip().split(' ')
                for w in words_in:
                    if w not in word_freqs:
                        word_freqs[w] = 0
                    word_freqs[w] += 1

    words = list(word_freqs.keys())
    freqs = list(word_freqs.values())

    sorted_idx = numpy.argsort(freqs)
    sorted_words = [words[ii] for ii in sorted_idx[::-1]]

    for ii, ww in enumerate(sorted_words):
        worddict[ww] = ii + 2

    with open(tgt_filename, 'wb') as f:
        print('Dump to', tgt_filename)

        pkl.dump(worddict, f)


def main(args=None):
    parser = argparse.ArgumentParser(description='Build dictionary file.')

    parser.add_argument('input', nargs='+',
                        help='input filenames')
    parser.add_argument('-o', '--output', action='store', dest='output', default=None,
                        help='dict output file, default is first input filename + ".pkl"')

    args = parser.parse_args(args)

    real_main(args)


if __name__ == '__main__':
    main()
