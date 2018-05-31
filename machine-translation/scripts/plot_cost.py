#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import sys
import argparse

import matplotlib.pyplot as plt

__author__ = 'fyabc'


def average(l):
    if not l:
        return 0.0
    return sum(l) / len(l)


def plot(args):
    """Logging format:

    Epoch 0 Update 52 Cost 219.29276 G2 1483.47644 UD 2.78200 Time 127.66500 s
    """

    colors = ['c', 'r', 'y', 'k', 'b', 'g']

    for f_idx, filename in enumerate(args.filenames):
        with open(filename, 'r') as f:
            iterations = []
            costs = []

            valid_iterations = []
            valid_costs = []
            small_train_costs = []

            for line in f:
                if line.startswith('Epoch'):
                    words = line.split()
                    iterations.append(int(words[3]))
                    costs.append(float(words[5]))
                elif line.startswith('Valid'):
                    words = line.split()
                    valid_iterations.append(iterations[-1] if iterations else 0)
                    valid_costs.append(words[2])
                    small_train_costs.append(words[6])

            avg_costs = [average(costs[max(0, i - args.average): i]) for i in xrange(len(costs))]

            # Get intervals
            iterations = [iterations[i] for i in xrange(0, len(iterations), args.interval)]
            avg_costs = [avg_costs[i] for i in xrange(0, len(avg_costs), args.interval)]

            if args.train:
                plt.plot(iterations, avg_costs,
                         '{}-'.format(colors[f_idx]), label='{}_train'.format(filename))
            if args.valid:
                plt.plot(valid_iterations, valid_costs,
                         '{}--'.format(colors[f_idx]), label='{}_valid'.format(filename))
            if args.small_train:
                plt.plot(valid_iterations, small_train_costs,
                         '{}-.'.format(colors[f_idx]), label='{}_small_train'.format(filename))

    plt.xlim(xmin=args.xmin, xmax=args.xmax)
    plt.ylim(ymin=args.ymin, ymax=args.ymax)

    plt.minorticks_on()

    plt.title('Costs')
    plt.legend(loc='upper right')

    plt.grid(which='both')

    plt.show()


def main(args=None):
    parser = argparse.ArgumentParser(description='Plot cost curve.')
    parser.add_argument('filenames', nargs='+', help='The logging filenames')
    parser.add_argument('-a', '--average', action='store', metavar='average', dest='average', type=int, default=40,
                        help='The moving average interval, default is %(default)s')
    parser.add_argument('-i', '--interval', action='store', metavar='interval', dest='interval', type=int, default=100,
                        help='The display interval of train curve, default is %(default)s')
    parser.add_argument('-y', '--ymin', action='store', dest='ymin', type=float, default=None,
                        help='The y min value (default is %(default)s)')
    parser.add_argument('-Y', '--ymax', action='store', dest='ymax', type=float, default=None,
                        help='The y max value (default is %(default)s)')
    parser.add_argument('-x', '--xmin', action='store', dest='xmin', type=int, default=None,
                        help='The x min value (default is %(default)s)')
    parser.add_argument('-X', '--xmax', action='store', dest='xmax', type=int, default=None,
                        help='The x max value (default is %(default)s)')
    parser.add_argument('-T', action='store_false', default=True, dest='train',
                        help='Plot train curve, default is True, set to False')
    parser.add_argument('-V', action='store_false', default=True, dest='valid',
                        help='Plot valid curve, default is True, set to False')
    parser.add_argument('-S', action='store_false', default=True, dest='small_train',
                        help='Plot small train curve, default is True, set to False')

    args = parser.parse_args(args)

    plot(args)


if __name__ == '__main__':
    main()
