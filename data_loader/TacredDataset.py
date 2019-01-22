import torch
import torch.utils.data as data
from utils import constant
import json
import random
import numpy as np

class TacredDataset(data.Dataset):
    """
    Tacred dataset
    """
    def __init__(self, filename, batch_size, opt, vocab, evaluation=False):
        """
        Load data from json files, preprocess and prepare batches
        :param filename: file path
        :param batch_size: int
        :param opt:
        :param vocab: Vocab object
        :param evaluation: boolean value, train or test
        """
        self.batch_size = batch_size
        self.opt = opt
        self.vocab = vocab
        self.eval = evaluation
        self.label2id = constant.LABEL_TO_ID

        with open(filename) as infile:
            data = json.load(infile)
        self.raw_data = data
        data = self.preprocess(data, vocab, opt)

        # shuffle for training
        if not evaluation:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
        self.id2label = dict([(v, k) for k, v in self.label2id.items()])
        self.labels = [self.id2label[d[-1]] for d in data]
        self.num_example = len(data)

        # chunk into batch
        data = [data[i: i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data
        print('{} batches created for {}'.format(len(data), filename))

    def gold(self):
        """
        Return gold labels as a list
        :return:
        """
        return self.labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        """
        Get a batch with index
        :param item:
        :return:
        """
        if not isinstance(item, int):
            raise TypeError
        if item < 0 or item > len(self.data):
            raise IndexError
        batch = self.data[item]
        batch_size = len(batch)
        batch = list(zip(*batch))
        assert len(batch) == 10

        # sort all fields by tokens lens for easy RNN operations
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = sort_all(batch, lens)

        # word dropout
        if not self.eval:
            words = [word_dropout(sent, self.opt['word_dropout']) for sent in batch[0]]
        else:
            words = batch[0]

        # convert to tensors
        words = get_long_tensor(words, batch_size)
        # mask all the tokens except PAD,it means PAD_ID = 1,others is equal to 0
        masks = torch.eq(words, 0)
        pos = get_long_tensor(batch[1], batch_size)
        ner = get_long_tensor(batch[2], batch_size)
        deprel = get_long_tensor(batch[3], batch_size)
        head = get_long_tensor(batch[4], batch_size)
        subj_position = get_long_tensor(batch[5], batch_size)
        obj_position = get_long_tensor(batch[6], batch_size)
        subj_type = get_long_tensor(batch[7], batch_size)
        obj_type = get_long_tensor(batch[8], batch_size)

        relations = torch.LongTensor(batch[9])

        return (words, masks, pos, ner, deprel, head, subj_position, obj_position, subj_type, obj_type, relations, orig_idx)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def preprocess(self, data, vocab, opt):
        """
        Preprocess the data and convert into idx
        :param: data: list
        :param: vocab: vocabulary
        :param: opt
        :return: processed, list type, every element is list, containing token_id, dependency hea,and so on
        """
        processed = []
        for d in data:
            tokens = d['token']
            if opt['lower']:
                tokens = [t.lower() for t in tokens]
            # mask
            ss, se = d['subj_start'], d['subj_end']
            os, oe = d['obj_start'], d['obj_end']
            tokens[ss: se + 1] = ['SUBJ-' + d['subj_type']] * (se - ss + 1)
            tokens[os: oe + 1] = ['OBJ-' + d['obj_type']] * (oe - os + 1)
            tokens = map_to_ids(tokens, vocab.WordsToIdx)
            pos = map_to_ids(d['stanford_pos'], constant.POS_TO_ID)
            ner = map_to_ids(d['stanford_ner'], constant.NER_TO_ID)
            deprel = map_to_ids(d['stanford_deprel'], constant.DEPREL_TO_ID)
            head = list(map(int, d['stanford_head']))
            l = len(tokens)
            subj_positions = get_position(d['subj_start'], d['subj_end'], l)
            obj_positions = get_position(d['obj_start'], d['obj_end'], l)
            subj_type = [constant.SUBJ_NER_TO_ID[d['subj_type']]]
            obj_type = [constant.OBJ_NER_TO_ID[d['obj_type']]]
            relation = self.label2id[d['relation']]
            processed += [(tokens, pos, ner, deprel, head, subj_positions, obj_positions, subj_type, obj_type, relation)]
        return processed


def map_to_ids(tokens, vocab):
    """
    map tokens into idx, if token is not in vocab, then replace it with UNK_ID
    :param tokens: list
    :param vocab: dict, key is token while value is idx
    :return: indices: list
    """
    ids = [vocab[t] if t in vocab else constant.UNK_ID for t in tokens]
    return ids

def get_position(start, end, length):
    """
    Obtain entity mention position in token, entity mention pos is 0 while ohters
    is not 0
    :param start: entity mention start position
    :param end: entity mention end position
    :param length: tokens length
    :return: pos: list, entity mention pos is 0 while other position is not 0
    """
    return list(range(-start, 0)) + [0] * (end - start + 1) + list(range(1, length - end))


def word_dropout(tokens, dropout):
    """
    Randomly dropout tokens (IDs) and replace them with <UNK> tokens
    :param tokens: list, every element is ID
    :param dropout: int, probability
    :return: list type: every element is ID
    """
    return [constant.UNK_ID if x != constant.UNK_ID and np.random.random() < dropout else x for x in tokens]


def sort_all(batch, lens):
    """
    Sort all fields by descending order of lens, and return the original indices.
    :param batch: will sort with respect to lens
    :param lens: list
    :return: sorted batch, original indices
    """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]


def get_long_tensor(token_list, batch_size):
    """
    Convert list of list of tokens to a padded LongTensor.
    Not only token need pad, but also POS, dependency, subj_position, obj_position and so on does .
    :param token_list: list, every element is list
    :param batch_size: int
    :return: tokens: torch LongTensor, of shape (batch_size, max length of token_list)
    """
    token_len = max(len(x) for x in token_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(constant.PAD_ID)
    # add PAD at the end of tokens to equal to token_len
    for i, s in enumerate(token_list):
        tokens[i, :len(s)] = torch.LongTensor(s)
    return tokens
