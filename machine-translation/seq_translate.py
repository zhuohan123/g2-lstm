# usage: python seq_translate.py model_prefix --start start_iteration --end end_iteration --gap interval --dataset dataset

import argparse
import sys
import os
import subprocess
import operator
import re

from libs.constants import Datasets
from libs.utility.translate import de_bpe


def get_bleu(ref_file, hyp_file):
    pl_output = subprocess.Popen(
        'perl scripts/moses/multi-bleu.perl {} < {}\n'.format(ref_file, hyp_file), shell=True,
        stdout=subprocess.PIPE, stderr=open(os.devnull, 'w')).stdout.read()

    contents = pl_output.split(',')
    if len(contents) == 0:
        return 0.0
    var = contents[0].split(" = ")
    if len(var) <= 1:
        return 0.0
    BLEU = var[1]

    return float(BLEU)


TestDatasets = {'enfr_bpe'}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('model_prefix', nargs='?', default='model/complete/enfr',
                        help='The prefix of nmt model path, default is "%(default)s"')
    parser.add_argument('--start', action="store", metavar="index", dest="start", type=int, default=1,
                        help='The starting index of saved model to test, default is %(default)s')
    parser.add_argument('--end', action="store", metavar="index", dest="end", type=int, default=10,
                        help='The ending index of saved model to test, default is %(default)s')
    parser.add_argument('--gap', action="store", metavar="index", dest="interval", type=int, default=10000,
                        help='The interval between two consecutive tested models\' indexes, default is %(default)s')
    parser.add_argument('--result', action='store', metavar='filename', dest='result_file', type=str,
                        default='trans_result.tsv', help='Target small train file, default is %(default)s')
    parser.add_argument('--beam', action="store", metavar="beam_size", dest="beam_size", type=int, default=4,
                        help='The beam size for translation, default is 4')
    parser.add_argument('--dataset', action='store', dest='dataset', default='en-fr_bpe',
                        help='Dataset, default is "%(default)s"')

    args = parser.parse_args()

    if args.result_file == 'trans_result.tsv':
        model_file_name = os.path.split(args.model_prefix)[-1]
        args.result_file = './translated/complete/{}_bs{}.txt'.format(os.path.splitext(model_file_name)[0],
                                                                      args.beam_size)
    else:
        model_file_name = os.path.split(args.result_file)[-1]

    print args

    bleus = {}
    train1, train2, small1, small2, dev1, dev2, dev3, test1, test2, dic1, dic2 = Datasets[args.dataset]

    for idx in xrange(args.start, args.end + 1):
        trans_model_file = '%s.iter%d.npz' % (os.path.splitext(args.model_prefix)[0], idx * args.interval)
        trans_result_file = '%s.iter%d.txt' % (os.path.splitext(args.result_file)[0], idx * args.interval)

        if not os.path.exists(trans_result_file):
            exec_str = 'python translate_single.py -b 32 -k {} -p 1 -n {} {} {} {} {}\n'.format(
                args.beam_size, trans_model_file, './data/dic/{}'.format(dic1), './data/dic/{}'.format(dic2),
                './data/test/{}'.format(test1), trans_result_file
            )
            print 'Translate model {} '.format(trans_model_file)
            print exec_str
            pl_output = subprocess.Popen(exec_str, shell=True, stdout=subprocess.PIPE).stdout.read()

        if 'tc' in args.dataset:  # first de-truecase, then de-bpe
            exec_str = 'perl scripts/moses/detruecase.perl < {} > {}.detc'.format(trans_result_file, trans_result_file)
            pl_output = subprocess.Popen(exec_str, shell=True, stdout=subprocess.PIPE).stdout.read()
            trans_result_file = '{}.detc'.format(trans_result_file)

        if 'bpe' in args.dataset:
            with open('{}.bpe'.format(trans_result_file), 'w') as fout:
                fout.write(de_bpe(open(trans_result_file, 'r').read()))
            trans_result_file = '{}.bpe'.format(trans_result_file)

        bleus[idx] = get_bleu('./data/test/{}'.format(test2), trans_result_file)

        print 'model %s, bleu %.2f' % (idx * args.interval, bleus[idx])

    args.result_file = './translated/complete/{}_s{}_e{}.txt'.format(os.path.splitext(model_file_name)[0], args.start,
                                                                     args.end)
    bleu_array = sorted(bleus.items(), key=operator.itemgetter(0), reverse=False)
    with open(args.result_file, 'w') as fout:
        fout.write('\n'.join([str(idx) + '\t' + str(score) for (idx, score) in bleu_array]))


if __name__ == '__main__':
    main()
