import torch
import torch.nn as nn
from utils import constant
from utils import torch_utils
from .gcn import GCN
import numpy as np
from .tree_utils import head_to_tree, tree_to_adj
from torch.autograd import Variable

class GCNRelationModel(nn.Module):
    def __init__(self, opt, emb_matrix=None):
        """

        :param opt: hyperparameter
        :param emb_matrix: pretraining embedding, we use glove embedding here
        """
        super(GCNRelationModel, self).__init__()
        self.opt = opt
        self.emb_matrix = emb_matrix

        # create embedding layers
        self.emb = nn.Embedding(opt['vocab_size'], opt['emb_dim'], padding_idx=constant.PAD_ID)
        self.pos_emb = nn.Embedding(len(constant.POS_TO_ID), opt['pos_dim']) if opt['pos_dim'] > 0 else None
        self.ner_emb = nn.Embedding(len(constant.NER_TO_ID), opt['ner_dim']) if opt['ner_dim'] > 0 else None
        embeddings = (self.emb, self.pos_emb, self.ner_emb)
        self.init_embeddings()

        # gcn layer
        self.gcn = GCN(self.opt, embeddings, opt['hidden_dim'], opt['num_layers'])

        # gcn output go through feed-forward neural network
        in_dim = opt['hidden_dim'] * 3
        layers = [nn.Linear(in_dim, opt['hidden_dim']), nn.ReLU()]
        for _ in range(self.opt['mlp_layers'] - 1):
            layers += [nn.Linear(opt['hidden_dim'], opt['hidden_dim']), nn.ReLU()]
        self.out_mlp = nn.Sequential(*layers)

    def init_embeddings(self):
        """
        Initiliaze word embedding
        :return:
        """
        if self.emb_matrix is None:
            self.emb.weight.data[1:, :].uniform_(-1.0, 1.0)
        else:
            self.emb.weight.data.copy_(self.emb_matrix)

        # decide finetuning
        if self.opt['topn'] <= 0:
            print('Do not finetune word embedding layer.')
            self.emb.weight.requires_grad = False
        elif self.opt['topn'] < self.opt['vocab_size']:
            print('Finetune top {} word embeddings.'.format(self.opt['topn']))
            self.emb.weight.register_hook(lambda grad : torch_utils.keep_partial_grad(grad, self.opt['topn']))
        else:
            print('Finetune all embeddings.')

    def forward(self, inputs):
        words, masks, pos, ner, deprel, head, subj_pos, obj_pos, subj_type, obj_type = inputs
        l = (masks.data.cpu().numpy() == 0).astype(np.int64).sum(1) # the token length of sentences(except PAD)
        maxlen = max(l) # max length among all sentences

        def inputs_to_tree_reps(head, words, l, prune, subj_pos, obj_pos):
            """
            Obtain the adjacency matrix of the dependency path
            :param head: torch tensor, of shape (batch_size, maxlen), every row is head of one sentence
            :param words: torch tensor, of shape (batch_size, maxlen), every row is token_id of one sentence
            :param l:the token length of sentences(except PAD)
            :param prune: int
            :param subj_pos: torch tensor, of shape (batch_size, maxlen), every row is subject position of one sentence
            :param obj_pos:torch tensor, of shape (batch_size, maxlen), every row is object position of one sentence
            :return: adj : torch tensor, of shape (maxlen, maxlen), dependency path of argument
            """
            trees = [head_to_tree(head[i], words[i], l[i], prune, subj_pos[i], obj_pos[i]) for i in range(len(l))]
            adj = [tree_to_adj(maxlen, tree, directed=False, self_loop=False).reshape(1, maxlen, maxlen) for tree in trees]
            adj = np.concatenate(adj, axis=0) # so the the first dimension means batch_size
            adj = torch.from_numpy(adj)
            return Variable(adj)

        adj = inputs_to_tree_reps(head.data, words.data, l, self.opt['prune_k'], subj_pos.data, obj_pos.data)
        h, pool_mask = self.gcn(adj, inputs)

        # pooling
        # print('subj_pos:', subj_pos.size(), 'obj_pos:', obj_pos.size())
        subj_mask, obj_mask = subj_pos.eq(0).eq(0).unsqueeze(2), obj_pos.eq(0).eq(0).unsqueeze(2)
        # print('subj_mask:', subj_mask.size(), 'obj_mask:', obj_mask.size())
        pool_type = self.opt['pooling']
        h_out = pool(h, pool_mask, type=pool_type)
        subj_out = pool(h, subj_mask, type=pool_type)
        obj_out = pool(h, obj_mask, type=pool_type)
        outputs = torch.cat([subj_out, h_out, obj_out], dim=1)
        outputs = self.out_mlp(outputs)
        return outputs, h_out


def pool(h, mask, type='max'):
    """
    pool operation
    :param h: torch tensor, gcn output, of shape (batch_size, max_len, mem_dim)
    :param mask: torch tensor, judge mask some tokens, of shape(batch, max_len, 1)
    :param type: pool type
    :return: h: torch tensor, of shape(batch_size, mem_dim)
    """
    if type == 'max':
        h = h.masked_fill(mask, -constant.INFINITY_NUMBER)
        return torch.max(h, 1)[0]
    elif type == 'avg':
        h = h.masked_fill(mask, 0)
        return h.sum(1) / (mask.size(1) - mask.float().sum(1))
    else:
        h = h.masked_fill(mask, 0)
        return h.sum(1)

