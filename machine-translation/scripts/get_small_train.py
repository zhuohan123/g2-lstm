#! /usr/bin/python
# -*- encoding: utf-8 -*-

from __future__ import print_function

import sys
import os
import random

__author__ = 'fyabc'


def main():
    input_filename1 = sys.argv[1]
    input_filename2 = sys.argv[2]

    if len(sys.argv) >= 4:
        small_size = int(sys.argv[3])
    else:
        small_size = 10000

    with open(input_filename1, 'r') as f_in:
        lines = list(f_in)

        selected_indices = random.sample(range(len(lines)), small_size)

        head, tail = os.path.split(input_filename1)
        output_filename1 = '{}{}small_{}'.format(head, '/' if head else '', tail)
        with open(output_filename1, 'w') as f_out:
            for index in selected_indices:
                print(lines[index], end='', file=f_out)

        print('Extract {} -> {}'.format(input_filename1, output_filename1))

    with open(input_filename2, 'r') as f_in:
        lines = list(f_in)

        head, tail = os.path.split(input_filename2)
        output_filename2 = '{}{}small_{}'.format(head, '/' if head else '', tail)
        with open(output_filename2, 'w') as f_out:
            for index in selected_indices:
                print(lines[index], end='', file=f_out)

        print('Extract {} -> {}'.format(input_filename2, output_filename2))

if __name__ == '__main__':
    main()
