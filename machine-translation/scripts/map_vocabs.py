#! /usr/bin/python
# -*- coding: utf-8 -*-
#used to map new dataset vocab id to old dataset vocab id

import sys
import cPickle as pkl

def main():

    new_src_dic_file = sys.argv[1]
    new_tgt_dic_file = sys.argv[2]
    old_src_dic_file = sys.argv[3]
    old_tgt_dic_file = sys.argv[4]

    new_to_old_src_map = {}
    new_to_old_tgt_map = {}

    o_src_dic = pkl.load(open(old_src_dic_file, 'rb'))
    o_tgt_dic = pkl.load(open(old_tgt_dic_file, 'rb'))

    new_src_dic = pkl.load(open(new_src_dic_file, 'rb'))
    new_tgt_dic = pkl.load(open(new_tgt_dic_file, 'rb'))

    for (word, id) in new_src_dic.iteritems():
        if word in o_src_dic:
            new_to_old_src_map[id] = o_src_dic[word]

    print 'Find %d vocabs in total %d src vocabs' % (len(new_to_old_src_map), len(new_src_dic))

    for (word, id) in new_tgt_dic.iteritems():
        if word in o_tgt_dic:
            new_to_old_tgt_map[id] = o_tgt_dic[word]

    print 'Find %d vocabs in total %d target vocabs' % (len(new_to_old_tgt_map), len(new_tgt_dic))

    pkl.dump(new_to_old_src_map, open('../resources/enfr_large2small_src_vocab_map.pkl', 'wb'))
    pkl.dump(new_to_old_tgt_map, open('../resources/enfr_large2small_tgt_vocab_map.pkl', 'wb'))


if __name__ == '__main__':
    main()