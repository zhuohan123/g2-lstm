"""Implementation of batch-normalized LSTM."""
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional, init
import math
import numpy as np

class GumbelNoise(nn.Module):

    def __init__(self, p=0.0, t=1.0, eps=1e-3, noise_type='new_U_B'):
        super(GumbelNoise, self).__init__()
        self.p = p
        self.t = t
        self.eps = eps
        self.noise_type = noise_type
        self.U = None
        self.B = None
        self.noise = None

    def update_noise(self, input_):
        if not self.training or self.p == 0.0:
            return
        if self.noise_type == 'new_U_B':
            pass
        elif self.noise_type == 'new_U':
            self.B = input_.data.new(input_.size()).bernoulli_(self.p)
        elif self.noise_type == 'no_new':
            self.U = input_.data.new(input_.size()).uniform_()
            self.U = torch.log(self.U + self.eps) - torch.log(1 + self.eps - self.U)
            self.B = input_.data.new(input_.size()).bernoulli_(self.p)
            self.noise = self.U * self.B
        else:
            raise ValueError('Unknown noise_type', self.noise_type)

    def forward(self, input_):
        if not self.training or self.p == 0.0:
            return input_
        if self.noise_type == 'new_U_B':
            self.U = input_.data.new(input_.size()).uniform_()
            self.U = torch.log(self.U + self.eps) - torch.log(1 + self.eps - self.U)
            self.B = input_.data.new(input_.size()).bernoulli_(self.p)
            self.noise = self.U * self.B
        elif self.noise_type == 'new_U':
            self.U = input_.data.new(input_.size()).uniform_()
            self.U = torch.log(self.U + self.eps) - torch.log(1 + self.eps - self.U)
            self.noise = self.U * self.B
        elif self.noise_type == 'no_new':
            pass
        else:
            raise ValueError('Unknown noise_type', self.noise_type)
        return (input_ + Variable(self.noise, requires_grad=False)) * (1/self.t)


class LSTMCell(nn.Module):

    """A basic LSTM cell."""

    def __init__(self, input_size, hidden_size, use_bias=True,
                 gumbel_noise_p=0.0, gumbel_noise_t=1.0, gumbel_noise_type='new_U_B',
                 divide_temp=None):
        """
        Most parts are copied from torch.nn.LSTMCell.
        """

        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.weight_ih = nn.Parameter(
            torch.FloatTensor(input_size, 4 * hidden_size))
        self.weight_hh = nn.Parameter(
            torch.FloatTensor(hidden_size, 4 * hidden_size))
        self.weight_hh_wdrop = None
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(4 * hidden_size))
        else:
            self.register_parameter('bias', None)
        self.noisef = GumbelNoise(p=gumbel_noise_p, t=gumbel_noise_t,
                                  noise_type=gumbel_noise_type)
        self.noisei = GumbelNoise(p=gumbel_noise_p, t=gumbel_noise_t,
                                  noise_type=gumbel_noise_type)
        self.divide_temp = divide_temp
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters following the way proposed in the paper.
        """
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

        # init.orthogonal(self.weight_ih.data)
        # weight_hh_data = torch.eye(self.hidden_size)
        # weight_hh_data = weight_hh_data.repeat(1, 4)
        # self.weight_hh.data.set_(weight_hh_data)
        # # The bias is just set to zero vectors.
        # if self.use_bias:
        #     init.constant(self.bias.data, val=0)

    def forward(self, input_, hx, update_noise=True):
        """
        Args:
            input_: A (batch, input_size) tensor containing input
                features.
            hx: A tuple (h_0, c_0), which contains the initial hidden
                and cell state, where the size of both states is
                (batch, hidden_size).
        Returns:
            h_1, c_1: Tensors containing the next hidden and cell state.
        """

        h_0, c_0 = hx
        batch_size = h_0.size(0)
        bias_batch = (self.bias.unsqueeze(0)
                      .expand(batch_size, *self.bias.size()))
        new_w_hh = self.weight_hh_wdrop \
            if self.weight_hh_wdrop is not None else self.weight_hh
        # act = torch.addmm(bias_batch,
        #                   torch.cat((h_0, input_), dim=1),
        #                   torch.cat((new_w_hh, self.weight_ih), dim=0))
        wh_b = torch.addmm(bias_batch, h_0, new_w_hh)
        wi = torch.mm(input_, self.weight_ih)
        f, i, o, g = torch.split(wh_b + wi,
                                 split_size=self.hidden_size, dim=1)

        if hasattr(self, 'noisef') and hasattr(self, 'noisei'):
            if update_noise:
                self.noisef.update_noise(f)
                self.noisei.update_noise(i)
            f = self.noisef(f)
            i = self.noisei(i)

        if getattr(self, 'divide_temp', None) is not None:
            f = f * (1 / self.divide_temp)
            i = i * (1 / self.divide_temp)
        sigm_i = torch.sigmoid(i)
        sigm_f = torch.sigmoid(f)
        c_1 = sigm_f*c_0 + sigm_i*torch.tanh(g)
        h_1 = torch.sigmoid(o) * torch.tanh(c_1)
        return h_1, c_1

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

class LSTM(nn.Module):

    """A module that runs multiple steps of LSTM."""

    def __init__(self, cell_class, input_size, hidden_size, num_layers=1,
                 use_bias=True, batch_first=False, dropout=0, wdrop=None, **kwargs):
        super(LSTM, self).__init__()
        self.cell_class = cell_class
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_bias = use_bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.wdrop = wdrop

        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            cell = cell_class(input_size=layer_input_size,
                              hidden_size=hidden_size,
                              **kwargs)
            setattr(self, 'cell_{}'.format(layer), cell)
        self.dropout_layer = nn.Dropout(dropout)
        self.reset_parameters()

    def get_cell(self, layer):
        return getattr(self, 'cell_{}'.format(layer))

    def reset_parameters(self):
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            cell.reset_parameters()

    def reset_gumbel_noise(self, gumbel_noise_p=0.0, gumbel_noise_t=1.0,
                           gumbel_noise_type='new_U_B'):
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)

            if not hasattr(cell, 'noisef'):
                cell.noisef = GumbelNoise(p=gumbel_noise_p, t=gumbel_noise_t,
                                          noise_type=gumbel_noise_type)
            else:
                cell.noisef.p = gumbel_noise_p
                cell.noisef.t = gumbel_noise_t
                cell.noisef.noise_type = gumbel_noise_type

            if not hasattr(cell, 'noisei'):
                cell.noisei = GumbelNoise(p=gumbel_noise_p, t=gumbel_noise_t,
                                          noise_type=gumbel_noise_type)
            else:
                cell.noisei.p = gumbel_noise_p
                cell.noisei.t = gumbel_noise_t
                cell.noisei.noise_type = gumbel_noise_type

    @staticmethod
    def _forward_rnn(cell, input_, hx):
        max_time = input_.size(0)
        output = []
        for time in range(max_time):
            h_next, c_next = cell(input_=input_[time], hx=hx,
                                  update_noise=(time == 0))
            hx_next = (h_next, c_next)
            output.append(h_next)
            hx = hx_next
        output = torch.stack(output, 0)
        return output, hx

    def forward(self, input_, hx=None):
        if self.batch_first:
            input_ = input_.transpose(0, 1)
        max_time, batch_size, _ = input_.size()
        print("max_time:", max_time)
        print("batch_size:", batch_size)
        if hx is None:
            hx = Variable(input_.data.new(batch_size, self.hidden_size).zero_())
            hx = [(hx, hx) for _ in range(self.num_layers)]
        layer_output = None
        new_hx = []
        for layer in range(self.num_layers):
            global global_layer
            global_layer = layer
            print("layer:", layer)
            cell = self.get_cell(layer)
            if self.wdrop is not None:
                cell.weight_hh_wdrop = torch.nn.functional.dropout(
                    cell.weight_hh, self.wdrop, training=self.training)
            layer_output, (layer_h_n, layer_c_n) = LSTM._forward_rnn(
                cell=cell, input_=input_, hx=hx[layer])
            input_ = self.dropout_layer(layer_output)
            new_hx.append((layer_h_n, layer_c_n))
        output = layer_output
        return output, new_hx
