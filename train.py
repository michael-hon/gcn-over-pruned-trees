from configs import config
import torch
import numpy as np
import random
from utils import constant
from utils import util
import os
from utils.vocab import Vocab
from data_loader.TacredDataset import TacredDataset
from trainers.trainer import GCNTrainer
from utils import helper
from utils import torch_utils
import time
from datetime import datetime
from utils import metrics

args = config.parses_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(1234)

# make opt
opt = vars(args)
label2id = constant.LABEL_TO_ID
opt['num_class'] = len(label2id)

# load vocab
print('='*10 + 'Build Vocab' + '=' * 10)
if not os.path.exists(opt['vocab_dir']):
    os.makedirs(opt['vocab_dir'])

vocab_file = opt['vocab_dir'] + '/vocab.txt'
if not os.path.isfile(vocab_file):
    util.build_vocab([opt['data_dir'] + '/train.json', opt['data_dir'] + '/dev.json',
                      opt['data_dir'] + '/test.json'], vocab_file, -1)

vocab = Vocab(vocab_file, special_words=constant.VOCAB_PREFIX)
opt['vocab_size'] = vocab.size

print('Build Vocab Done !')
# load embedding
print('='*10 + 'Loading Embedding' + '='*10)
emb_file = opt['data_dir'] + '/tacred.pth'
if os.path.isfile(emb_file):
    emb = torch.load(emb_file)
else:
    glove_emb_file = opt['emb_dir'] + '/glove.840B.300d.txt'
    glove_vocab, glove_vector = util.load_glove_vector(glove_emb_file)
    emb = util.get_embedding(vocab, glove_vector, glove_vocab)
    assert glove_vector.size(1) == opt['emb_dim']
    torch.save(emb, emb_file)

# load data and define data loader
print('=' * 10 + 'Loading Data' + '=' * 10)
print('batch_size:', opt['batch_size'])
train_batch = TacredDataset(opt['data_dir'] + '/train.json', opt['batch_size'], opt, vocab,
                            evaluation=False)
dev_batch = TacredDataset(opt['data_dir'] + '/dev.json', opt['batch_size'], opt, vocab,
                          evaluation=True)


model_id = opt['id'] if len(opt['id']) > 1 else '0' + opt['id']
model_save_dir = opt['save_dir'] + '/' + model_id
opt['model_save_dir'] = model_save_dir
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

helper.save_config(opt, model_save_dir + '/config.json', verbose=True)

# Define the model
if not opt['load']:
    trainer = GCNTrainer(opt, emb_matrix=emb)
else:
    # load pretrained model
    model_file = opt['model_file']
    print('Loading model from {}'.format(model_file))
    model_opt = torch_utils.load_config(model_file)
    model_opt['optim'] = opt['optim']
    trainer = GCNTrainer(model_opt)
    trainer.load(model_file)

id2label = dict([(v,k) for k,v in label2id.items()])
dev_score_history = []
current_lr = opt['lr']

global_step = 0
global_start_time = time.time()
format_str = '{}: step {}/{} (epoch {}/{}), loss = {:.6f} ({:.3f} sec/batch), lr: {:.6f}'
max_steps = len(train_batch) * opt['num_epoch']

# start training
for epoch in range(1, opt['num_epoch'] + 1):
    train_loss = 0
    for i, batch in enumerate(train_batch):
        start_time = time.time()
        global_step += 1
        loss = trainer.update(batch)
        train_loss += loss
        if global_step % opt['log_step'] == 0:
            duration = time.time() - start_time
            print(format_str.format(datetime.now(), global_step, max_steps, epoch,\
                    opt['num_epoch'], loss, duration, current_lr))
        # duration = time.time() - start_time
        # print(format_str.format(datetime.now(), global_step, max_steps, epoch,\
        #             opt['num_epoch'], loss, duration, current_lr))

    # eval on dev
    print('Evaluating on dev set...')
    predictions = []
    dev_loss = 0
    for i, batch in enumerate(dev_batch):
        preds, _, loss = trainer.predict(batch)
        predictions += preds
        dev_loss += loss
    predictions = [id2label[p] for p in predictions]
    train_loss = train_loss / train_batch.num_example * opt['batch_size'] # avg loss per batch
    dev_loss = dev_loss / dev_batch.num_example * opt['batch_size']

    dev_p, dev_r, dev_f1 = metrics.score(dev_batch.gold(), predictions)
    print("epoch {}: train_loss = {:.6f}, dev_loss = {:.6f}, dev_f1 = {:.4f}".format(epoch,\
        train_loss, dev_loss, dev_f1))
    dev_score = dev_f1

    # save
    model_file = model_save_dir + '/checkpoint_epoch_{}.pt'.format(epoch)
    trainer.save(model_file, epoch)
    if epoch % opt['save_epoch'] != 0:
        os.remove(model_file)

    # lr schedule
    if len(dev_score_history) > opt['decay_epoch'] and dev_score <= dev_score_history[-1] and \
            opt['optim'] in ['sgd', 'adagrad', 'adadelta']:
        current_lr *= opt['lr_decay']
        trainer.update_lr(current_lr)

    dev_score_history += [dev_score]
    print("")

print("Training ended with {} epochs.".format(epoch))