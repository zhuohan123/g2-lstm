import argparse
import sys
import os

from libs.constants import Datasets
from libs.gpu_manager import get_gpu_usage


def main():
    parser = argparse.ArgumentParser(
        description='Train the deep NMT model.',
        fromfile_prefix_chars='@',
    )

    parser.add_argument('-R', action="store_false", default=True, dest='reload',
                        help='Reload old model, default to True, set to False')
    parser.add_argument('-d', action='store_true', default=False, dest='dump_before_train',
                        help='Dump before train default to False, set to True')
    parser.add_argument('--lr', action="store", metavar="learning_rate", dest="learning_rate", type=float, default=1.0,
                        help='Start learning rate, default is %(default)s')
    parser.add_argument('--optimizer', action='store', default='adadelta')
    parser.add_argument('--plot', action='store', default=None,
                        help='Plot filename, default is None (not plot) (deprecated).')
    parser.add_argument('--save_freq', action='store', default=10000, type=int, dest='save_freq',
                        help='Model save frequency, default is %(default)s')
    parser.add_argument('--dev_bleu_freq', action='store', default=20000, type=int, dest='dev_bleu_freq',
                        help='Get dev set BLEU frequency, default is %(default)s')
    parser.add_argument('--dim', action='store', default=512, type=int, dest='dim',
                        help='Dim of hidden units, default is %(default)s')
    parser.add_argument('--bs', action='store', default=128, type=int, dest='batch_size',
                        help='Train batch size, default is %(default)s')
    parser.add_argument('--valid_bs', action='store', default=128, type=int, dest='valid_batch_size',
                        help='Valid batch size, default is %(default)s')
    parser.add_argument('--dim_word', action='store', default=512, type=int, dest='dim_word',
                        help='Dim of word embedding, default is %(default)s')
    parser.add_argument('--maxlen', action='store', default=80, type=int, dest='maxlen',
                        help='Max sentence length, default is %(default)s')
    parser.add_argument('-S', action='store_false', default=True, dest='shuffle',
                        help='Shuffle data per epoch, default is True, set to False')
    parser.add_argument('--train1', action='store', metavar='filename', dest='train1', type=str,
                        default='filtered_en-fr.en',
                        help='Source train file, default is %(default)s')
    parser.add_argument('--train2', action='store', metavar='filename', dest='train2', type=str,
                        default='filtered_en-fr.fr',
                        help='Target train file, default is %(default)s')
    parser.add_argument('--small1', action='store', metavar='filename', dest='small1', type=str,
                        default='small_en-fr.en',
                        help='Source small train file, default is %(default)s')
    parser.add_argument('--small2', action='store', metavar='filename', dest='small2', type=str,
                        default='small_en-fr.fr',
                        help='Target small train file, default is %(default)s')
    parser.add_argument('--valid1', action='store', metavar='filename', dest='valid1', type=str,
                        default='dev_en.tok',
                        help='Source valid file, default is %(default)s')
    parser.add_argument('--valid2', action='store', metavar='filename', dest='valid2', type=str,
                        default='dev_fr.tok',
                        help='Target valid file, default is %(default)s')
    parser.add_argument('--dic1', action='store', metavar='filename', dest='dic1', type=str,
                        default='filtered_dic_en-fr.en.pkl',
                        help='Source dict file, default is %(default)s')
    parser.add_argument('--dic2', action='store', metavar='filename', dest='dic2', type=str,
                        default='filtered_dic_en-fr.fr.pkl',
                        help='Target dict file, default is %(default)s')
    parser.add_argument('--n_words_src', action='store', default=30000, type=int, dest='n_words_src',
                        help='Vocabularies in source side, default is %(default)s')
    parser.add_argument('--n_words_tgt', action='store', default=30000, type=int, dest='n_words_tgt',
                        help='Vocabularies in target side, default is %(default)s')

    parser.add_argument('model_file', nargs='?', default='model/baseline/baseline.npz',
                        help='Generated model file, default is "%(default)s"')
    parser.add_argument('pre_load_file', nargs='?', default='model/en2fr.iter160000.npz',
                        help='Pre-load model file, default is "%(default)s"')
    parser.add_argument('--src_vocab_map', action='store', metavar='filename', dest='src_vocab_map_file', type=str,
                        default=None, help='The file containing source vocab mapping information' 
                                           'used to initialize a model on large dataset from small one')
    parser.add_argument('--tgt_vocab_map', action='store', metavar='filename', dest='tgt_vocab_map_file', type=str,
                        default=None, help='The file containing target vocab mapping information'
                                           'used to initialize a model on large dataset from small one')

    parser.add_argument('--enc', action='store', default=1, type=int, dest='n_encoder_layers',
                        help='Number of encoder layers, default is 1')
    parser.add_argument('--dec', action='store', default=1, type=int, dest='n_decoder_layers',
                        help='Number of decoder layers, default is 1')
    parser.add_argument('--conn', action='store', default=2, type=int, dest='connection_type',
                        help='Connection type, '
                             'default is 2 (bidirectional only in first layer, other layers are forward);'
                             '1 is divided bidirectional GRU')
    parser.add_argument('--max_epochs', action='store', default=100, type=int, dest='max_epochs',
                        help='Maximum epoches, default is 100')
    parser.add_argument('--unit', action='store', metavar='unit', dest='unit', type=str, default='lstm',
                        help='The unit type, default is "lstm", can be set to "gru".')
    parser.add_argument('--attention', action='store', metavar='index', dest='attention_layer_id', type=int, default=0,
                        help='Attention layer index, default is 0')
    parser.add_argument('--residual_enc', action='store', metavar='type', dest='residual_enc', type=str, default=None,
                        help='Residual connection of encoder, default is None, candidates are "layer_wise", "last"')
    parser.add_argument('--residual_dec', action='store', metavar='type', dest='residual_dec', type=str,
                        default='layer_wise',
                        help='Residual connection of decoder, default is "layer_wise", candidates are None, "last"')
    parser.add_argument('-z', '--zigzag', action='store_false', default=True, dest='use_zigzag',
                        help='Use zigzag in encoder, default is True, set to False')
    parser.add_argument('--dropout', action="store", metavar="dropout", dest="dropout", type=float, default=False,
                        help='Dropout rate, default is False (not use dropout)')
    parser.add_argument('--unit_size', action='store', default=2, type=int, dest='unit_size',
                        help='Number of unit size, default is %(default)s')
    # TODO: rename this option to decoder_unit_size in future
    parser.add_argument('--cond_unit_size', action='store', default=2, type=int, dest='cond_unit_size',
                        help='Number of decoder unit size (will rename in future), default is %(default)s')
    parser.add_argument('--clip', action='store', metavar='clip', dest='clip', type=float, default=1.0,
                        help='Gradient clip rate, default is 1.0.')
    parser.add_argument('--manual', action='store_false', dest='auto', default=True,
                        help='Set dropout rate and grad clip rate manually.')
    parser.add_argument('--emb', action='store', metavar='filename', dest='given_embedding', type=str, default=None,
                        help='Given embedding model file, default is None')
    parser.add_argument('--lr_discount', action='store', metavar='freq', dest='lr_discount_freq', type=int,
                        default=-1, help='The learning rate discount frequency, default is -1')

    parser.add_argument('--distribute', action = 'store', metavar ='type', dest = 'dist_type', type = str, default= None,
                        help = 'The distribution version, default is None (singe GPU mode), candiates are "mv", "mpi_reduce"')
    parser.add_argument('--nccl', action="store_true", default=False, dest='nccl',
                        help='Use NCCL in distributed mode, default to False, set to True')
    parser.add_argument('--clip_grads_local', action="store_true", default=False, dest='clip_grads_local',
                        help='Whether to clip grads in distributed mode, default to False, set to True')
    parser.add_argument('--recover_lr_iter', action='store', dest='dist_recover_lr', type = int, default=10000,
                        help='The mini-batch index to recover lrate in distributed mode, default is 10000.')

    parser.add_argument('--all_att', action='store_true', dest='all_att', default=False,
                        help='Generate attention from all decoder layers, default is False, set to True')
    parser.add_argument('--avg_ctx', action='store_true', dest='avg_ctx', default=False,
                        help='Average all context vectors to get softmax, default is False, set to True')
    parser.add_argument('--dataset', action='store', dest='dataset', default='en-fr',
                        help='Dataset, default is "%(default)s"')
    parser.add_argument('--gpu_map_file', action='store', metavar='filename', dest='gpu_map_file', type=str,
                        default=None, help='The file containing gpu id mapping information, '
                                           'each line is in the form physical_gpu_id\\theano_id')
    parser.add_argument('--ft_patience', action='store', metavar='N', dest='fine_tune_patience', type=int, default=-1,
                        help='Fine tune patience, default is %(default)s, set 8 to enable it')
    parser.add_argument('--valid_freq', action='store', metavar='N', dest='valid_freq', type=int, default=5000,
                        help='Validation frequency, default is 5000')
    parser.add_argument('--trg_att', action='store', metavar='N', dest='trg_attention_layer_id', type=int, default=None,
                        help='Target attention layer id, default is None (not use target attention)')
    parser.add_argument('--fix_dp_bug', action="store_true", default=False, dest='fix_dp_bug',
                        help='Fix previous dropout bug, default to False, set to True')
    parser.add_argument('--abandon_imm', action="store_true", default=False, dest='abandon_imm',
                        help='Whether to load previous immediate params, default to True, set to False')
    parser.add_argument('--tp', action="store", metavar="temperature", dest="temperature", type=float, default=1.0,
                        help='temperature, default is %(default)s')
    parser.add_argument('--scale', action="store", metavar="scale", dest="scale", type=float, default=1.0,
                        help='scale, default is %(default)s')
    parser.add_argument('--gate_dp', action="store", metavar="gate_dropout", dest="gate_dropout", type=float, default=1.0,
                        help='gate_dropout, default is %(default)s')

    args = parser.parse_args()
    print args

    if args.residual_enc == 'None':
        args.residual_enc = None
    if args.residual_dec == 'None':
        args.residual_dec = None
    if args.dist_type != 'mv' and args.dist_type != 'mpi_reduce':
        args.dist_type = None

    # FIXME: Auto mode
    if args.auto:
        if args.n_encoder_layers <= 2:
            args.dropout = False
            args.clip = 1.0
        else:
            args.dropout = 0.1
            args.clip = 5.0

        if args.n_encoder_layers <= 1:
            args.residual_enc = None
        if args.n_decoder_layers <= 1:
            args.residual_dec = None
            args.attention_layer_id = 0

        args.cond_unit_size = args.unit_size

    # If dataset is not 'en-fr', old value of dataset options like 'args.train1' will be omitted
    if args.dataset != 'en-fr':
        args.train1, args.train2, args.small1, args.small2, args.valid1, args.valid2, args.valid3, args.test1, args.test2, args.dic1, args.dic2 = \
            Datasets[args.dataset]

    print 'Command line arguments:'
    print args
    sys.stdout.flush()

    # Init multiverso or mpi and set theano flags.
    if args.dist_type == 'mv':
        try:
            import multiverso as mv
        except ImportError:
            import libs.multiverso_ as mv

        # FIXME: This must before the import of theano!
        mv.init(sync=True)
        worker_id = mv.worker_id()
        workers_cnt = mv.workers_num()
    elif args.dist_type == 'mpi_reduce':
        from mpi4py import MPI

        communicator = MPI.COMM_WORLD
        worker_id = communicator.Get_rank()
        workers_cnt = communicator.Get_size()

    if args.dist_type:
        available_gpus = get_gpu_usage(workers_cnt)
        gpu_maps_info = {idx: idx for idx in available_gpus}
        if args.gpu_map_file:
            for line in open(os.path.join('resources', args.gpu_map_file), 'r'):
                phy_id, theano_id = line.split()
                gpu_maps_info[int(phy_id)] = int(theano_id)
        theano_id = gpu_maps_info[available_gpus[worker_id]]
        print 'worker id:%d, using theano id:%d, physical id %d' % (worker_id, theano_id, available_gpus[worker_id])
        os.environ['THEANO_FLAGS'] = 'device=cuda{},floatX=float32'.format(theano_id)
        sys.stdout.flush()

    from libs.nmt import train

    train(
        max_epochs= args.max_epochs,
        saveto=args.model_file,
        preload=args.pre_load_file,
        reload_=args.reload,
        dim_word=args.dim_word,
        dim=args.dim,
        decay_c=0.,
        clip_c=args.clip,
        lrate=args.learning_rate,
        optimizer=args.optimizer,
        maxlen=args.maxlen,
        batch_size=args.batch_size,
        valid_batch_size=args.valid_batch_size,
        dispFreq=1,
        saveFreq=args.save_freq,
        validFreq=args.valid_freq,
        datasets=(r'data/train/{}'.format(args.train1),
                  r'data/train/{}'.format(args.train2)),
        valid_datasets=(r'data/dev/{}'.format(args.valid1),
                        r'data/dev/{}'.format(args.valid2)),
        small_train_datasets=(r'data/test/{}'.format(args.small1),r'data/test/{}'.format(args.small2),
                              r'data/test/{}'.format(args.test2)),
        vocab_filenames=(r'data/dic/{}'.format(args.dic1),
                         r'data/dic/{}'.format(args.dic2)),
        task=args.dataset,
        use_dropout=args.dropout,
        overwrite=False,
        n_words=args.n_words_tgt,
        n_words_src=args.n_words_src,

        # Options from v-yanfa
        dump_before_train=args.dump_before_train,
        plot_graph=args.plot,
        lr_discount_freq=args.lr_discount_freq,

        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        encoder_many_bidirectional=args.connection_type == 1,

        attention_layer_id=args.attention_layer_id,
        unit=args.unit,
        residual_enc=args.residual_enc,
        residual_dec=args.residual_dec,
        use_zigzag=args.use_zigzag,
        given_embedding=args.given_embedding,

        unit_size=args.unit_size,
        cond_unit_size=args.cond_unit_size,

        given_imm = not args.abandon_imm,
        dump_imm=True,
        shuffle_data=args.shuffle,

        decoder_all_attention=args.all_att,
        average_context=args.avg_ctx,

        dist_type=args.dist_type,
        dist_recover_lr_iter = args.dist_recover_lr,

        fine_tune_patience=args.fine_tune_patience,
        nccl= args.nccl,
        src_vocab_map_file= args.src_vocab_map_file,
        tgt_vocab_map_file= args.tgt_vocab_map_file,

        trg_attention_layer_id=args.trg_attention_layer_id,
        dev_bleu_freq = args.dev_bleu_freq,
        fix_dp_bug= args.fix_dp_bug,
        temperature=args.temperature,
        scale=args.scale,
        gate_dropout=args.gate_dropout,
    )


if __name__ == '__main__':
    main()
