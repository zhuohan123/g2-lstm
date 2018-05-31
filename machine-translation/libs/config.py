#! /usr/bin/python
# -*- coding: utf-8 -*-

__author__ = 'fyabc'

# Dict of default options (Copied from nmt.py)
DefaultOptions = dict(
    dim_word=100,  # word vector dimensionality
    dim=1000,  # the number of LSTM units

    # These 2 options are deprecated
    encoder='gru',
    decoder='gru_cond',

    n_words_src=30000,
    n_words=30000,
    patience=10,  # early stopping patience
    max_epochs=5000,
    finish_after=10000000,  # finish after this many updates
    dispFreq=100,
    decay_c=0.,  # L2 regularization penalty
    alpha_c=0.,  # alignment regularization
    clip_c=-1.,  # gradient clipping threshold
    lrate=1.,  # learning rate
    maxlen=100,  # maximum length of the description
    optimizer='rmsprop',
    batch_size=16,
    valid_batch_size=80,
    saveto='model.npz',
    saveFreq=1000,  # save the parameters after every saveFreq updates
    validFreq=10000,
    datasets=('/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.en.tok',
              '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.fr.tok'),
    valid_datasets=('./data/dev/dev_en.tok',
                    './data/dev/dev_fr.tok'),
    small_train_datasets=('./data/train/small_en-fr.en',
                          './data/train/small_en-fr.fr'),
    picked_train_idxes_file=r'',

    # The dropout rate
    # If False, do not use dropout.
    # If float value, this is the dropout rate.
    use_dropout=False,
    reload_=False,
    overwrite=False,
    preload='',
    sort_by_len=False,

    # Options below are from v-yanfa
    convert_embedding=True,
    dump_before_train=False,
    plot_graph=None,
    vocab_filenames=('./data/dic/filtered_dic_en-fr.en.pkl',
                     './data/dic/filtered_dic_en-fr.fr.pkl'),
    map_filename='./data/dic/mapFullVocab2Top1MVocab.pkl',
    lr_discount_freq=80000,

    # Options of deeper encoder and decoder
    n_encoder_layers=1,
    n_decoder_layers=1,

    # The connection type:
    #     1. encoder_many_bidirectional = True (default):
    #         forward1 -> forward2 -> forward3 -> ...
    #         backward1 -> backward2 -> backward3 -> ...
    #     2. encoder_many_bidirectional = False:
    #         forward1 + backward1 -> forward2 -> forward3 -> ...
    encoder_many_bidirectional=True,

    # Attention at which decoder layer (default is 0)
    attention_layer_id=0,

    # Unit type, LSTM or GRU (Attention unit type = unit type + '_cond')
    # Add new unit types: multi_gru, multi_lstm
    unit='gru',

    # Residual connection type
    # Candidates:
    #   None:           not any residual connection
    #   "layer_wise":   connect to next layer
    #   "last":         all connect to the last layer (average)
    residual_enc=None,
    residual_dec=None,

    use_zigzag=False,

    # todo: implement it
    # Initializer type
    # Candidates:
    #   "orthogonal":   Current type
    #   "baidu":        Baidu initializer
    initializer='orthogonal',

    # Given embedding file
    given_embedding=None,

    # Options for sync distribution
    #Set it to none to run single GPU version. Other options include 'mv' and 'mpi_reduce'
    dist_type=None,
    #The sync frequency. Will be automatically fixed to be 1 when syncing gradients
    sync_batch=1,
    #From start to dist_recover_lr iteration, linearly increase lr to normal lr, s.t. nan is avoided
    dist_recover_lr = 10000,
    #Whether to sync models (i.e., model average) or gradients (per batch)
    sync_models = False,

    # Options for multi-gru/lstm
    # Used only when unit is "multi_gru" or "multi_lstm"

    # Depth of common unit
    unit_size=2,

    # Depth of cond unit
    cond_unit_size=2,

    # Given immediate file of adadelta? Only effective in adadelta
    given_imm=False,
    # Dump immediate file of adadelta? Only effective in adadelta
    # (Dump when save model)
    dump_imm=False,

    # Shuffle data per epoch?
    shuffle_data=False,

    # Attention at each layer of decoder?
    decoder_all_attention=False,

    # Average context vector? default is False (use last context vector)
    # Used only when decoder_all_attention is True.
    average_context=False,

    #The file storing physical_gpu_id -> theano_id information. Per gpu infor, per line
    gpu_map_file = None,
    task='en-fr',

    # MPI options
    dist_recover_lr_iter=False,

    # Fine-tune options
    fine_tune_patience=8,

    # Target attention options
    # Target attention layer id, default is None, means not use target attention.
    trg_attention_layer_id=None,
)


# Dict of dual learning default options
DualLearningDefaultOptions = {

}
