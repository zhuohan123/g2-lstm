import argparse
import time
import math
import numpy as np
np.random.seed(331)
import torch
import torch.nn as nn
from torch.autograd import Variable

import data
import model

from utils import batchify, get_batch, repackage_hidden, message, set_log_file

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--load', type=str,  default='',
                    help='path to load the fine-tune model')
parser.add_argument('--log-file', type=str,  default='',
                    help='path to save the log')
parser.add_argument('--compression', type=str,  default='',
                    help='compress the model, (svd, precision)')
parser.add_argument('--compression-k', type=float,  default=None,
                    help='compression parameter')
args = parser.parse_args()

set_log_file(args.log_file)

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        message("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################
args.bptt = 70
corpus = data.Corpus(args.data)
train_batch_size = 20
eval_batch_size = 10
test_batch_size = 1
train_data = batchify(corpus.train, train_batch_size, args)
val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)
criterion = nn.CrossEntropyLoss()
def evaluate(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    # if args.model == 'QRNN': model.reset()
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        output, hidden = model(data, hidden)
        output_flat = output.view(-1, ntokens)
        total_loss += len(data) * criterion(output_flat, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss[0] / len(data_source)

# Load the best saved model.
with open(args.load, 'rb') as f:
    model = torch.load(f)

def compression(x):
    if args.compression == 'precision':
        x.mul_(1.0 / args.compression_k).round_().mul_(args.compression_k)
    if args.compression == 'svd':
        u, s, v = torch.svd(x)
        r = int(args.compression_k)
        x_low_rank = torch.mm((u * s).narrow(1, 0, r),
                              v.narrow(1, 0, r).transpose_(0, 1))
        x.copy_(x_low_rank)

for x in model.rnns:
    weight_hh = getattr(x, 'cell_0').weight_hh.data
    hidden_size = weight_hh.size()[1] // 4
    print('hidden_size =', hidden_size)
    compression(weight_hh.narrow(1, 0, hidden_size))
    compression(weight_hh.narrow(1, hidden_size, hidden_size))
    print('weight_hh', weight_hh)
    weight_ih = getattr(x, 'cell_0').weight_ih.data
    hidden_size = weight_ih.size()[1] // 4
    compression(weight_ih.narrow(1, 0, hidden_size))
    compression(weight_ih.narrow(1, hidden_size, hidden_size))
    print('weight_ih', weight_ih)

# Run on test data.
test_loss = evaluate(test_data, test_batch_size)
message('=' * 89)
message('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
message('=' * 89)
