#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function

import sys
from collections import OrderedDict
import cPickle as pkl

__author__ = 'fyabc'


def main():
    input_filename = sys.argv[1]

    with open(input_filename, 'r') as f_in:
        d = OrderedDict()

        d['eos'] = 0
        d['UNK'] = 1

        i = 2

        for line in f_in:
            word = line.strip()
            if word:
                d[word] = i
                i += 1

        with open('{}.pkl'.format(input_filename), 'wb') as f_out:
            pkl.dump(d, f_out)

        print('Convert {} -> {}.pkl'.format(input_filename, input_filename))

if __name__ == '__main__':
    main()
