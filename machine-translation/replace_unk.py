#! /usr/bin/python
# -*- encoding: utf-8 -*-

import argparse
import cPickle as pkl

import numpy as np
import theano

from libs.constants import Datasets
from libs.models import build_and_init_model
from libs.utility.translate import get_bleu, de_bpe
from libs.utility.utils import prepare_data

__author__ = 'fyabc'


def _load_data(args, dic1, dic2, test1):
    with open('{}.pkl'.format(args.model), 'rb') as f:
        options = pkl.load(f)

    # load source dictionary
    with open(dic1, 'rb') as f:
        word_dict_raw = pkl.load(f)
    word_dict = {k: v for k, v in word_dict_raw.iteritems() if v < options['n_words_src']}

    # load target dictionary and invert
    with open(dic2, 'rb') as f:
        word_dict_tgt_raw = pkl.load(f)
    word_dict_tgt = {k: v for k, v in word_dict_tgt_raw.iteritems() if v < options['n_words']}

    if args.nbest == 1:
        with open(args.translated_file, 'r') as f:
            trans_sents_str = [s.strip().split() for s in f]
    else:
        with open(args.translated_file, 'r') as f:
            trans_sents_str = [
                (s.strip().split('|||')[1]).strip().split()
                for idx, s in enumerate(f)
                if idx % args.nbest == 0
            ]

    trans_sents_num = [[word_dict_tgt.get(w, 1) for w in s] for s in trans_sents_str]

    with open(test1, 'r') as f:
        src_sents_str = [s.strip().split() for s in f]
    src_sents_num = [[word_dict.get(w, 1) for w in s] for s in src_sents_str]

    with open(args.table, 'rb') as f:
        src_tgt_table = pkl.load(f)

    return options, src_sents_num, trans_sents_num, src_sents_str, trans_sents_str, src_tgt_table


def replace_unk(args, seq_source, seq_trans, src_sents, trans_sents, src_tgt_table):
    print 'Load and build models...',
    model, _, (trng, use_noise, x, x_mask, y, y_mask, opt_ret, cost, context_mean) = \
        build_and_init_model(args.model, build=True)
    f_get_attention = theano.function([x, x_mask, y, y_mask], opt_ret['dec_alphas'])
    print 'Done'

    print 'Start to calculate the scores...'

    current_id = 0
    batch_size = 80

    while True:
        block_x = seq_source[current_id * batch_size: (current_id + 1) * batch_size]
        block_y = seq_trans[current_id * batch_size: (current_id + 1) * batch_size]
        block_x_str = src_sents[current_id * batch_size: (current_id + 1) * batch_size]
        block_y_str = trans_sents[current_id * batch_size: (current_id + 1) * batch_size]

        if len(block_x) == 0:
            break

        x, x_mask, y, y_mask = prepare_data(block_x, block_y)
        attn_score_ = f_get_attention(x, x_mask, y, y_mask)
        srcWordsByAttn = attn_score_.argmax(axis=2)
        for idx, (sentx, senty, strx, stry) in enumerate(zip(block_x, block_y, block_x_str, block_y_str)):
            attn_mapping = srcWordsByAttn[:, idx]
            unk_pos = np.where(np.array(senty, dtype='int64') == 1)
            badder = 0
            end_pos = -1
            for ii in unk_pos[0].tolist():
                srcidx = attn_mapping[ii]
                if srcidx < len(sentx):
                    trans_sents[idx + current_id * batch_size][ii] = src_tgt_table.get(strx[srcidx], strx[srcidx])
                else:
                    badder += 1
                    if badder == 1:
                        end_pos = ii
                    if badder > 1:
                        trans_sents[idx + current_id * batch_size] = \
                            trans_sents[idx + current_id * batch_size][:end_pos]
                        break

        print 'Minibatch', current_id, ' Done'
        current_id += 1


def main():
    parser = argparse.ArgumentParser(description='Replace UNK in the translated file, and get BLEU.')
    parser.add_argument('model', help='The model path')
    parser.add_argument('translated_file', help='The translated file with UNK')
    parser.add_argument('table', nargs='?', default='./data/dic/fastAlign_en2fr.pkl',
                        help='Source-Target table path, default is %(default)s')
    parser.add_argument('--dataset', action='store', dest='dataset', default='en-fr_bpe',
                        help='Dataset, default is "%(default)s"')
    parser.add_argument('--nbest', action="store", metavar="N", dest="nbest", type=int, default=1,
                        help='number of best, default is %(default)s')
    parser.add_argument('-B', action='store_false', default=True, dest='bleu',
                        help='Get BLEU, default is True, set to False')
    parser.add_argument('-d', '--dump', action='store_true', default=False, dest='dump',
                        help='Dump translated file without UNK, default is False, set to True')

    args = parser.parse_args()

    print 'model: {}, translated file: {}'.format(args.model, args.translated_file)

    train1, train2, small1, small2, valid1, valid2, test1, test2, dic1, dic2 = Datasets[args.dataset]

    options, src_sents_num, trans_sents_num, src_sents_str, trans_sents_str, src_tgt_table = _load_data(
        args, './data/dic/{}'.format(dic1), './data/dic/{}'.format(dic2), './data/test/{}'.format(test1),
    )

    replace_unk(args, src_sents_num, trans_sents_num, src_sents_str, trans_sents_str, src_tgt_table)

    translated_string = '\n'.join(' '.join(w for w in s) for s in trans_sents_str) + '\n'

    postfix = '.nounk'

    if 'bpe' in args.dataset:
        translated_string = de_bpe(translated_string)
        postfix = '.bpe' + postfix

    if args.dump:
        with open('{}{}'.format(args.translated_file, postfix), 'w') as f:
            print >>f, translated_string,

    if args.bleu:
        bleu = get_bleu(
            './data/test/{}'.format(test2),
            translated_string,
            type_in='string',
        )

        print 'BLEU: {:.2f}'.format(bleu)


if __name__ == '__main__':
    main()
