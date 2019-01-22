"""
GCN model for relation extraction.
"""

import torch
import torch.nn as nn
import torch.functional as F
from torch.autograd import Variable
import numpy as np
from modules.GCNRelationModel import GCNRelationModel
from modules.gcn import GCN

class GCNClassifier(nn.Module):
    """
    A wrapper classifier for GCNRelationModel.
    """
    def __init__(self, opt, emb_matrix=None):
        """

        :param opt: hyperparameter
        :param emb_matrix: pretraining embedding, we use glove embedding here
        """
        super(GCNClassifier, self).__init__()
        self.gcn_model = GCNRelationModel(opt, emb_matrix)
        in_dim = opt['hidden_dim']
        self.classifier = nn.Linear(in_dim, opt['num_class'])
        self.opt = opt

    def conv_l2(self):
        return self.gcn_model.gcn.conv_l2()

    def forward(self, input):
        outputs, pooling_output = self.gcn_model(input)
        logits = self.classifier(outputs)
        return logits, pooling_output

