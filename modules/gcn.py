import torch
import torch.nn as nn
from utils import constant
from torch.autograd import Variable
import torch.nn.functional as F

class GCN(nn.Module):
    """
    A GCN/Contextualized GCN module operated on dependency graphs.
    """
    def __init__(self, opt, embeddings, mem_dim, num_layers):
        """

        :param opt: hyperparameter
        :param embeddings: tuple, including word embedding, pos embedding, ner embedding
        :param mem_dim: GCN state dimension
        :param num_layers: the number of GCN layer
        """
        super(GCN, self).__init__()
        self.opt = opt
        self.layers = num_layers # the number of GCN layer
        self.use_cuda = opt['cuda']
        self.mem_dim = mem_dim # GCN state dimension
        # input vector dimension, contencationg of word embedding, pos embedding, ner embedding
        self.in_dim = opt['emb_dim'] + opt['pos_dim'] + opt['ner_dim']

        self.emb, self.pos_emb, self.ner_emb = embeddings

        # rnn layer
        if self.opt.get('rnn', False):
            input_size = self.in_dim
            self.rnn = nn.LSTM(input_size, opt['rnn_hidden'], opt['rnn_layers'],
                               batch_first=True, dropout=opt['rnn_dropout'], bidirectional=True)
            self.in_dim = opt['rnn_hidden'] * 2 # output of Bi-LSTM the input of GCN
            self.rnn_drop = nn.Dropout(opt['rnn_dropout']) # Apply dropout on the last output

        self.in_drop = nn.Dropout(opt['input_dropout']) # dropout on input
        self.gcn_drop = nn.Dropout(opt['gcn_dropout']) # dropout on every gcn layer except the last layer

        # gcn layer
        self.W = nn.ModuleList()

        for layer in range(self.layers):
            input_dim = self.in_dim if layer == 0 else self.mem_dim
            self.W.append(nn.Linear(input_dim, self.mem_dim))


    def conv_l2(self):
        """
        l2 regulation on every gcn layer
        :return:
        """
        conv_weights = []
        for w in self.W:
            conv_weights += [w.weight, w.bias]
        return sum([x.pow(2).sum() for x in conv_weights])


    def encode_with_rnn(self, rnn_inputs, masks, batch_size):
        """
        Encoding sentence with Bi-LSTM
        :param rnn_inputs: torch tensor, of shape (batch_size, max_length, input_dim)
        :param masks: torch tensor, of shape (batch_size, max_length), mask tokens
        :param batch_size: int
        :return: rnn_outputs
        """
        seq_lens = list(masks.data.eq(constant.PAD_ID).long().sum(1).squeeze())
        h0, c0 = rnn_zero_state(batch_size, self.opt['rnn_hidden'], self.opt['rnn_layers'])
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True) # this API can not be unstandood
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs

    def forward(self, adj, inputs):
        words, masks, pos, ner, deprel, head, subj_pos, obj_pos, subj_type, obj_type = inputs
        word_embs = self.emb(words)
        embs = [word_embs]
        if self.opt['pos_dim'] > 0:
            embs += [self.pos_emb(pos)]
        if self.opt['ner_dim'] > 0:
            embs += [self.ner_emb(ner)]
        embs = torch.cat(embs, dim=2)
        embs = self.in_drop(embs)

        # rnn layer
        if self.opt.get('rnn', False):
            gcn_inputs = self.rnn_drop(self.encode_with_rnn(embs, masks, words.size()[0]))
        else:
            gcn_inputs = embs

        # gcn layer
        denom = adj.sum(2).unsqueeze(2) + 1
        # because adj shape is (max_len, max_len), when we do pooling operation
        # we should judge which tokens are in dependency path and use this tokens feature to do pooling
        # and other tokens we don't care
        mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)

        # zeros out adj for ablation
        if self.opt.get('no_adj', False):
            adj = torch.zeros_like(adj)

        for l in range(self.layers):
            Ax = adj.bmm(gcn_inputs)
            AxW = self.W[l](Ax)
            AxW = AxW + self.W[l](gcn_inputs)  # self loop
            AxW = AxW / denom

            gAxW = F.relu(AxW)
            gcn_inputs = self.gcn_drop(gAxW) if l < self.layers - 1 else gAxW

        return gcn_inputs, mask


def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True, use_cuda=True):
    """
    Initiliaze rnn h0, c0 with zero
    :param batch_size: int
    :param hidden_dim: the dimension of rnn hidden state
    :param num_layers: the number of rnn layers
    :param bidirectional:
    :param use_cuda:
    :return: h0, c0 : torch tensor, of shape(total_layer, batch_size, hidden_dim)
    """
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    return h0, c0




