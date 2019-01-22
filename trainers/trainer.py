"""
A trainer class.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from utils import torch_utils
from models.GCNClassifier import GCNClassifier

class Trainer(object):
    def __init__(self, opt, emb_matrix=None):
        raise NotImplementedError

    def update(self, batch):
        """
        Update the parameter with one batch size data
        """
        raise NotImplementedError

    def predict(self, batch):
        """
        Predict one batch
        """
        raise NotImplementedError

    def update_lr(self, new_lr):
        """
        Change learning rate with new_lr
        :param new_lr:
        :return:
        """
        torch_utils.change_lr(self.optimizer, new_lr)

    def load(self, filename):
        """
        load the checkpoint
        :param filename : checkpoint file path
        """
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print('Cannot load model from {}'.format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']

    def save(self, filename, epoch):
        """
        save the model from epoch
        :param filename: distination file path
        :param epoch: int, epoch num
        """
        # save model parameter and hyperparameter
        params = {
            'model': self.model.state_dict(),
            'config': self.opt
        }
        try:
            torch.save(params, filename)
            print('model saved to {}'.format(filename))
        except BaseException:
            print('[Warning: Saving failed...continueing anyway.')


def unpack_batch(batch):
    """
    unpack batch data
    :param batch: list type
    :return:
    """
    inputs = [Variable(b) for b in batch[:10]]
    labels = Variable(batch[10])
    tokens = batch[0]
    head = batch[5]
    subj_pos = batch[6]
    obj_pos = batch[7]
    lens = batch[1].eq(0).long().sum(1).squeeze() # sentences length
    return inputs, labels, tokens, head, subj_pos, obj_pos, lens


class GCNTrainer(Trainer):
    def __init__(self, opt, emb_matrix=None):
        """
        GCN Trainer
        :param opt:
        :param emb_matrix: word embedding matrix, torch tensor
        """
        self.opt = opt
        self.emb_matrix = emb_matrix
        self.model = GCNClassifier(opt, emb_matrix=emb_matrix)
        self.criterion = nn.CrossEntropyLoss()
        self.parameters = [p for p in self.model.parameters() if p.requires_grad] # only update some parameter, because we may not update some parameter
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['lr'])

    def update(self, batch):
        """
        Update the parameter with one batch size data
        :param batch:
        :return: loss_val: real value
        """
        inputs, labels, tokens, head, subj_pos, obj_pos, lens = unpack_batch(batch)

        # step forward
        self.model.train() # set the train mode
        # before updating the parameter, we should clear the exsiting grad
        self.optimizer.zero_grad()
        logits, pooling_output = self.model(inputs)
        loss = self.criterion(logits, labels)
        # l2 decay on all conv layers
        if self.opt.get('conv_l2', 0) > 0:
            loss += self.model.conv_l2() * self.opt['conv_l2']
        # l2 penalty on output representations
        if self.opt.get('pooling_l2', 0) > 0:
            loss += self.opt['pooling_l2'] * (pooling_output ** 2).sum(1).mean()
        loss_val = loss.item()
        # backward
        loss.backward()
        # clip the gradient
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt['max_grad_norm'])
        self.optimizer.step()
        return loss_val

    def predict(self, batch):
        """
        Predict one batch
        :param batch: batch size data
        :return: prediction, probs, loss: list, list, real value
        """
        inputs, labels, tokens, head, subj_pos, obj_pos, lens = unpack_batch(batch)
        orig_idx = batch[11]
        # forward
        self.model.eval()
        logits, _ = self.model(inputs)
        loss = self.criterion(logits, labels)
        probs = F.softmax(logits, dim=1).data.cpu().numpy().tolist()
        predictions = np.argmax(logits.data.cpu().numpy(), axis=1).tolist()
        _, predictions, probs = [list(t) for t in zip(*sorted(zip(orig_idx, predictions, probs)))]
        return predictions, probs, loss.item()
