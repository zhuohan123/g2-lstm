#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

__author__ = 'fyabc'

profile = False
fX = 'float32'

ImmediateFilenameBackup = '{}_imm.pkl'
ImmediateFilenameBackup2 = '{}_imm.pkl.gz'
ImmediateFilename = '{}_imm.npz'
TempImmediateFilename = '{}_imm_tmp.npz'

# Cycle of shuffle data.
ShuffleCycle = 7

# Set datasets
# train1, train2, small1, small2, valid1, valid2(postprocessed, e.g., bpe and truecase), valid3(original), test1, test2, dic1, dic2
Datasets = {
    'en-fr': [
        'filtered_en-fr.en', 'filtered_en-fr.fr',
        'small_en-fr.en', 'small_en-fr.fr',
        'dev_en.tok', 'dev_fr.tok', 'dev_fr.tok',
        'test_en-fr.en.tok','test_en-fr.fr.tok',
        'filtered_dic_en-fr.en.pkl', 'filtered_dic_en-fr.fr.pkl',
    ],
    'en-fr_tc': [
        'tc_filtered_en-fr.en', 'tc_filtered_en-fr.fr',
        'small_tc_en-fr.en', 'small_tc_en-fr.fr',
        'tc_dev_en.tok', 'tc_dev_fr.tok', 'dev_fr.tok',
        '', '',
        'tc_filtered_en-fr.en.pkl', 'tc_filtered_en-fr.fr.pkl',
    ],

    'en-fr_bpe': [
        'en-fr_en.tok.bpe.32000', 'en-fr_fr.tok.bpe.32000',
        'small_en-fr_en.tok.bpe.32000', 'small_en-fr_fr.tok.bpe.32000',
        'dev_en-fr_en.tok.bpe.32000', 'dev_en-fr_fr.tok.bpe.32000', 'dev_fr.tok',
        'test_en-fr.en.tok.bpe.32000', 'test_en-fr.fr.tok',
        'en-fr_vocab.bpe.32000.pkl', 'en-fr_vocab.bpe.32000.pkl',
    ],
    'en-fr_bpe_tc': [
        'tc_en-fr_en.tok.bpe.32000', 'tc_en-fr_fr.tok.bpe.32000',
        'tc_test_en-fr.en.tok.bpe.32000', 'tc_test_en-fr.fr.tok.bpe.32000',
        'tc_dev_en-fr_en.tok.bpe.32000', 'tc_dev_en-fr_fr.tok.bpe.32000','dev_fr.tok',
        'tc_test_en-fr.en.tok.bpe.32000', 'test_en-fr.fr.tok',
        'tc_en-fr_vocab.bpe.32000.pkl', 'tc_en-fr_vocab.bpe.32000.pkl',
    ],

    'large_en-fr_bpe_tc': [
        'tc_train_enfr_large_bpe.en', 'tc_train_enfr_large_bpe.fr',
        'tc_small_train_enfr_large_bpe.en', 'tc_small_train_enfr_large_bpe.fr',
        'tc_valid_enfr_bpe_by_large.en', 'tc_valid_enfr_bpe_by_large.fr', 'dev_fr.tok',
        'tc_test_enfr_bpe_by_large.en', 'test_en-fr.fr.tok',
        'tc_enfr_large_bpe.vocab.pkl', 'tc_enfr_large_bpe.vocab.pkl',
    ],

    'large_fr-en_bpe_tc': [
        'tc_train_enfr_large_bpe.fr', 'tc_train_enfr_large_bpe.en',
        'tc_small_train_enfr_large_bpe.fr', 'tc_small_train_enfr_large_bpe.en',
        'tc_valid_enfr_bpe_by_large.fr', 'tc_valid_enfr_bpe_by_large.en', 'dev_en.tok',
        'tc_test_enfr_bpe_by_large.fr', 'test_en-fr.en.tok',
        'tc_enfr_large_bpe.vocab.pkl', 'tc_enfr_large_bpe.vocab.pkl',
    ],

    'fr-en_bpe_tc': [
        'tc_en-fr_fr.tok.bpe.32000','tc_en-fr_en.tok.bpe.32000',
        'tc_small_en-fr_fr.tok.bpe.32000', 'tc_small_en-fr_en.tok.bpe.32000',
        'tc_dev_en-fr_fr.tok.bpe.32000', 'tc_dev_en-fr_en.tok.bpe.32000', 'dev_en.tok',
        'tc_test_en-fr.fr.tok.bpe.32000', 'test_en-fr.en.tok',
        'tc_en-fr_vocab.bpe.32000.pkl', 'tc_en-fr_vocab.bpe.32000.pkl',
    ],


    'en-de': [
        'en-de.en_0', 'en-de.de_0',
        'small_en-de.en_0', 'small_en-de.de_0',
        'dev_en.tok', 'dev_de.tok', '',
        '','',
        'en-de.en.pkl', 'en-de.de.pkl',
    ],
    'en-de_tc': [
        '', '',
        '', '',
        '', '', '',
        '', '',
        '', '',
    ],

    'en-de_bpe': [
        'train.tok.clean.bpe.32000.en', 'train.tok.clean.bpe.32000.de',
        'newstest2014.tok.bpe.32000.en', 'newstest2014.tok.bpe.32000.de',
        'newstest2013.tok.bpe.32000.en', 'newstest2013.tok.bpe.32000.de', '',
        'newstest2014.tok.bpe.32000.en', 'test_en-de.de.tok',
        'vocab.bpe.32000.pkl', 'vocab.bpe.32000.pkl',
    ],

    'en-de-s2s_bpe_tc': [
        'tc_train.tok.clean.bpe.32000.en', 'tc_train.tok.clean.bpe.32000.de',
        'tc_small_train.tok.clean.bpe.32000.en', 'tc_small_train.tok.clean.bpe.32000.de',
        'tc_newstest2013.tok.bpe.32000.en', 'tc_newstest2013.tok.bpe.32000.de', '',
        'tc_newstest2014.tok.bpe.32000.en', 'test_en-de.de.tok',
        'ende_s2s_vocab.bpe.32000.pkl', 'ende_s2s_vocab.bpe.32000.pkl',
    ],

    'de-en':[
       'train.de-en.de', 'train.de-en.en',
        'small_train.de-en.de','small_train.de-en.en',
        'dev.de-en.de', 'dev.de-en.en','dev.de-en.en',
        'test.de-en.de','test.de-en.en',
        'de-en_vocab.de.pkl','de-en_vocab.en.pkl',
    ],

    'en-de_small':[
       'train.de-en.en', 'train.de-en.de',
       'small_train.de-en.en', 'small_train.de-en.de',
       'dev.de-en.en', 'dev.de-en.de', 'dev.de-en.de',
       'test.de-en.en', 'test.de-en.de',
       'de-en_vocab.en.pkl', 'de-en_vocab.de.pkl',
    ],

    'de-en_bpe':[
       'train.de-en.bpe.25000.de', 'train.de-en.bpe.25000.en',
        'test.de-en.bpe.25000.de', 'test.de-en.bpe.25000.en',
        'dev.de-en.bpe.25000.de', 'dev.de-en.bpe.25000.en','dev.de-en.en',
        'test.de-en.bpe.25000.de', 'test.de-en.en',
        'de-en_vocab.bpe.25000.pkl', 'de-en_vocab.bpe.25000.pkl',
    ],

    'en-de_small_bpe' :[
       'train.de-en.bpe.25000.en', 'train.de-en.bpe.25000.de',
       '', '',
       'dev.de-en.bpe.25000.en', 'dev.de-en.bpe.25000.de', 'dev.de-en.de',
       'test.de-en.bpe.25000.en', 'test.de-en.de',
       'de-en_vocab.bpe.25000.pkl', 'de-en_vocab.bpe.25000.pkl',
    ],

    'zh-en': [
        'zh-en.1.25M.zh', 'zh-en.1.25M.en',
        'small_zh-en.1.25M.zh', 'small_zh-en.1.25M.en',
        'Nist2003.chs.word.max50.snt', 'Nist2003.enu.word.max50.snt', '',
        '','',
        'zh-en.1.25M.zh.pkl', 'zh-en.1.25M.en.pkl',
    ],
    'zh-en_tc': [
        'tc_zh-en.1.25M.zh', 'tc_zh-en.1.25M.en',
        'small_tc_zh-en.1.25M.zh', 'small_tc_zh-en.1.25M.en',
        'tc_Nist2003.chs.word.max50.snt', 'tc_Nist2003.enu.word.max50.snt', '',
        '', '',
        'tc_zh-en.1.25M.zh.pkl', 'tc_zh-en.1.25M.en.pkl',
    ],
}
