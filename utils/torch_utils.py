import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer


def keep_partial_grad(grad, topk):
    """
    Keep only the topk rows of grads, others' grads should not be computed, we set it to zero,
    so these weight will not be updated
    :param grad: grad tensor
    :param topk: int
    :return: grad: processed grad
    """
    assert topk < grad.size(0)
    grad.data[topk:].zero_()
    return grad

def change_lr(optimizer, new_lr):
    """
    Change optimizer learning rate with new_lr
    :param optimizer:
    :param new_lr:
    :return:
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


### torch specific functions
def get_optimizer(name, parameters, lr, l2=0):
    if name == 'sgd':
        return torch.optim.SGD(parameters, lr=lr, weight_decay=l2)
    elif name == 'adam':
        return torch.optim.Adam(parameters, weight_decay=l2) # use default lr
    elif name == 'adamax':
        return torch.optim.Adamax(parameters, weight_decay=l2) # use default lr
    elif name == 'adadelta':
        return torch.optim.Adadelta(parameters, lr=lr, weight_decay=l2)
    else:
        raise Exception("Unsupported optimizer: {}".format(name))

def load_config(filename):
    try:
        dump = torch.load(filename)
    except BaseException:
        print("[ Fail: model loading failed. ]")
    return dump['config']

