import json
from collections import Counter
import torch
import os
from .vocab import Vocab
from . import constant

def build_vocab(filenames, vocabFile, min_freq):
    """
    Build the vocabulary from filenames(including train, dev, test) , and save in vocabFile
    :param filenames: list, including train file path, dev file path, test file path
    :param vocabFile: filepath--- vocab file path
    :param min_freq: int, if word freq is less than min_freq, then remove it
    :return: None
    """
    tokens = []
    for filename in filenames:
        tokens += load_tokens(filename)
    counter = Counter(token for token in tokens)
    vocab = sorted([t for t in counter if counter.get(t) >= min_freq], key=counter.get, reverse=True)
    # we add entity mask and special words into vocab in the begining of vocab while abadoning entity tokens
    vocab = constant.VOCAB_PREFIX + entity_mask() + vocab
    with open(vocabFile, 'w') as f:
        for v in vocab:
            f.write(v + '\n')


def entity_mask():
    """
    Get all entity mask tokens as a list
    :return: masks: list type
    """
    # we mask all entity types
    masks = []
    sub_entities = list(constant.SUBJ_NER_TO_ID.keys())[2:]
    obj_entities = list(constant.OBJ_NER_TO_ID.keys())[2:]
    sub_masks = ['SUBJ-' + subj for subj in sub_entities]
    obj_masks = ['OBJ-' + obj for obj in obj_entities]
    masks += sub_masks
    masks += obj_masks
    return masks

def load_tokens(filename):
    """
    load tokens from filename
    :param filename: file path
    :return: tokens: list, every element is list that its elements are words
    """
    with open(filename, 'r', encoding='utf8', errors='ignore') as infile:
        data = json.load(infile)
        tokens = []
        for d in data:
            ts = d['token']
            ss, se, os, oe = d['subj_start'], d['subj_end'], d['obj_start'], d['obj_end']
            # remove entity tokens, i.e. it can not be appeared in vocabulary
            ts[ss:se+1] = ['<PAD>'] * (se-ss+1)
            ts[os:oe+1] = ['<PAD>'] * (oe-os+1)
            tokens += list(filter(lambda x : x != '<PAD>', ts))
    print('{} tokens from {} examples loaded from {}.'.format(len(tokens), len(data), filename))
    return tokens


def load_glove_vector(path):
    """
    loading word vector(this project employs GLOVE word vector), save GLOVE word, vector as file
    respectively
    :param path: GLOVE word vector path
    :return: glove vocab: vocab object, glove_vector(torch tensor, of shape(words_num, word_dim))
    """
    base = os.path.splitext(os.path.basename(path))[0]
    glove_word_path = os.path.join('./data/glove/', base + '.vocab')
    glove_vector_path = os.path.join('./data/glove/', base + '.pth')
    if os.path.isfile(glove_word_path) and os.path.isfile(glove_vector_path):
        print('======> File found, loading memory !')
        glove_vocab = Vocab(glove_word_path)
        glove_vector = torch.load(glove_vector_path)
        return glove_vocab, glove_vector

    print('======>Loading glove word vector<======')
    with open(path, 'r', encoding='utf8', errors='ignore') as f:
        content = f.readline().rstrip('\n').split(' ')
        word_dim = len(content[1:])
        word_count = 1
        for _ in f:
            word_count += 1
    glove_tokens = [None] * word_count
    glove_vector = torch.zeros(word_count, word_dim, dtype=torch.float)
    with open(path, 'r', encoding='utf8', errors='ignore') as f:
        count = 0
        for content in f:
            content = content.rstrip('\n').split(' ')
            glove_tokens[count] = content[0]
            vectors = list(map(float, content[1:]))
            glove_vector[count] = torch.tensor(vectors, dtype=torch.float)
            count += 1

    with open(glove_word_path, 'w') as f:
        for token in glove_tokens:
            f.write(token + '\n')

    torch.save(glove_vector, glove_vector_path)
    glove_vocab =Vocab(glove_word_path)
    return glove_vocab, glove_vector


def get_embedding(vocab, pre_embedding, pre_vocab):
    """
    Obtain the word embedding. If words are in vocab and glove at the same time, then using
    glove_embedding.Otherwise we can use random vector.Noting that <pad> should be all 0
    :param vocab: Vocab object
    :param pre_embedding: in this project, we use glove embedding, tensor
    :param pre_vocab: vocab of pre_embedding, Vocab object
    :return: embedding: torch tensor, of shape (vocab size, word dim)
    """
    embedding = torch.zeros(len(vocab.WordsToIdx), pre_embedding.size(1), dtype=torch.float)
    # embedding.normal_(0, 0.05)
    embedding.uniform_(-1, 1)
    # <pad> should be all 0
    embedding[constant.PAD_ID].zero_()
    for word in vocab.WordsToIdx:
        if pre_vocab.get_index(word) is not None:
            embedding[vocab.get_index(word)] = pre_embedding[pre_vocab.get_index(word)]

    return embedding



