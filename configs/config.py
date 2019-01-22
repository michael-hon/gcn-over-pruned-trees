"""
Define arguments
"""
import argparse
import torch

def parses_args():
    """
    Define arguments
    :return: args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/tacred')
    parser.add_argument('--vocab_dir', type=str, default='data/vocab')
    parser.add_argument('--emb_dir', type=str, default='data/glove')
    parser.add_argument('--emb_dim', type=int, default=300, help='Word embedding dimension.')
    parser.add_argument('--ner_dim', type=int, default=30, help='NER embedding dimension.')
    parser.add_argument('--pos_dim', type=int, default=30, help='POS embedding dimension.')
    parser.add_argument('--hidden_dim', type=int, default=200, help='RNN hidden state size.')
    parser.add_argument('--num_layers', type=int, default=2, help='Num of RNN layers.')
    parser.add_argument('--input_dropout', type=float, default=0.5, help='Input dropout rate.')
    parser.add_argument('--gcn_dropout', type=float, default=0.5, help='GCN layer dropout rate.')
    parser.add_argument('--word_dropout', type=float, default=0.04,
                        help='The rate at which randomly set a word to UNK.')
    parser.add_argument('--topn', type=int, default=1e10, help='Only finetune top N word embeddings.')
    parser.add_argument('--lower', dest='lower', action='store_true', help='Lowercase all words.')
    parser.add_argument('--no-lower', dest='lower', action='store_false')
    parser.set_defaults(lower=False)

    parser.add_argument('--prune_k', default=-1, type=int,
                        help='Prune the dependency tree to <= K distance off the dependency path; set to -1 for no pruning.')
    parser.add_argument('--conv_l2', type=float, default=0, help='L2-weight decay on conv layers only.')
    parser.add_argument('--pooling', choices=['max', 'avg', 'sum'], default='max',
                        help='Pooling function type. Default max.')
    parser.add_argument('--pooling_l2', type=float, default=0, help='L2-penalty for all pooling output.')
    parser.add_argument('--mlp_layers', type=int, default=2, help='Number of output mlp layers.')
    parser.add_argument('--no_adj', dest='no_adj', action='store_true', help="Zero out adjacency matrix for ablation.")

    parser.add_argument('--no-rnn', dest='rnn', action='store_false', help='Do not use RNN layer.')
    parser.add_argument('--rnn_hidden', type=int, default=200, help='RNN hidden state size.')
    parser.add_argument('--rnn_layers', type=int, default=1, help='Number of RNN layers.')
    parser.add_argument('--rnn_dropout', type=float, default=0.5, help='RNN dropout rate.')

    parser.add_argument('--lr', type=float, default=1.0, help='Applies to sgd and adagrad.')
    parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate decay rate.')
    parser.add_argument('--decay_epoch', type=int, default=5, help='Decay learning rate after this epoch.')
    parser.add_argument('--optim', choices=['sgd', 'adagrad', 'adam', 'adamax'], default='sgd',
                        help='Optimizer: sgd, adagrad, adam or adamax.')
    parser.add_argument('--num_epoch', type=int, default=100, help='Number of total training epochs.')
    parser.add_argument('--batch_size', type=int, default=50, help='Training batch size.')
    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
    parser.add_argument('--log_step', type=int, default=20, help='Print log every k steps.')
    parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
    parser.add_argument('--save_epoch', type=int, default=100, help='Save model checkpoints every k epochs.')
    parser.add_argument('--save_dir', type=str, default='./saved_models', help='Root dir for saving models.')
    parser.add_argument('--id', type=str, default='00', help='Model ID under which to save models.')
    parser.add_argument('--info', type=str, default='', help='Optional info for the experiment.')

    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
    parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')

    parser.add_argument('--load', dest='load', action='store_true', help='Load pretrained model.')
    parser.add_argument('--model_file', type=str, help='Filename of the pretrained model.')
    # parser.add_argument('gold_file', help='The gold relation file; one relation per line')
    # parser.add_argument('pred_file', help='A prediction file; one relation per line, in the same order as the gold file.')
    args = parser.parse_args()
    return args


