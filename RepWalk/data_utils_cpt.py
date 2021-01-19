import os
import json
import pickle
import numpy as np
from torch.utils.data import Dataset
import sys


class Vocab(object):
    ''' vocabulary for tokens '''

    def __init__(self, vocab_list, add_pad=True, add_unk=True):
        self._vocab_dict = dict()
        self._reverse_vocab_dict = dict()
        self._length = 0
        if add_pad:  # pad_id should be zero (for masks)
            self.pad_word = '<pad>'
            self.pad_id = self._length
            self._length += 1
            self._vocab_dict[self.pad_word] = self.pad_id
        if add_unk:
            self.unk_word = '<unk>'
            self.unk_id = self._length
            self._length += 1
            self._vocab_dict[self.unk_word] = self.unk_id
        for w in vocab_list:
            self._vocab_dict[w] = self._length
            self._length += 1
        for w, i in self._vocab_dict.items():
            self._reverse_vocab_dict[i] = w

    def word_to_id(self, word):  # transform tokens to indices
        if hasattr(self, 'unk_id'):
            return self._vocab_dict.get(word, self.unk_id)
        return self._vocab_dict[word]

    def id_to_word(self, idx):  # transform indices to tokens
        if hasattr(self, 'unk_word'):
            return self._reverse_vocab_dict.get(idx, self.unk_word)
        return self._reverse_vocab_dict[idx]

    def has_word(self, word):
        return word in self._vocab_dict

    def __len__(self):
        return self._length


class Tokenizer(object):
    ''' transform tokens to indices and pad sequences '''

    def __init__(self, word_vocab, pos_vocab, deprel_vocab, max_slen, max_plen, lower, max_tlen):
        self.vocab = {
            'word': word_vocab,  # vocabulary for words
            'pos': pos_vocab,  # vocabulary for part-of-speech tags
            'deprel': deprel_vocab  # vocabulary for dependency relations
        }
        self.max_slen = max_slen  # maximum length of sentences
        self.max_plen = max_plen  # maximum length of dependency paths
        self.lower = lower  # whether lowering the words in sentences
        self.max_tlen = max_tlen

    @classmethod
    def from_files(cls, fnames, lower=True):
        all_tokens, all_pos, all_deprel = set(), set(), set()  # parpare for vocabularies
        max_slen, max_plen = 0, 0
        max_tlen = 0
        for fname in fnames:
            fdata = json.load(open(fname, 'r', encoding='utf-8'))
            for data in fdata:
                all_tokens.update(map(lambda x: x.lower() if lower else x, data['token']))
                all_pos.update(data['pos'])
                all_deprel.update(data['deprel'])
                max_slen = max(max_slen, len(data['token']) + 2)  # sentence length should add 2 for position indicators
                max_plen = max(max_plen, max([len(p) + 1 for aspect in data['aspects'] for p in
                                              aspect['path']]))  # path length should add 1 for extra relation
                max_tlen = max(max_tlen, max([len(p) for aspect in data['aspects'] for p in
                                              aspect['term']]))

        all_tokens.update(['<p>', '</p>'])  # add position indicators
        all_pos.update(['<p>', '</p>'])  # add position indicators
        all_deprel.update([f"rev#{rel}" for rel in all_deprel])
        word_vocab, pos_vocab, deprel_vocab = Vocab(all_tokens), Vocab(all_pos), Vocab(all_deprel)
        return cls(word_vocab, pos_vocab, deprel_vocab, max_slen, max_plen, lower, max_tlen)

    @staticmethod
    def _pad_sequence(sequence, pad_id, maxlen, dtype, padding, truncating):
        x = (np.zeros(maxlen) + pad_id).astype(dtype)
        if truncating == 'pre':
            trunc = sequence[-maxlen:]
        else:
            trunc = sequence[:maxlen]
        trunc = np.asarray(trunc, dtype=dtype)
        if padding == 'post':
            x[:len(trunc)] = trunc
        else:
            x[-len(trunc):] = trunc
        return x

    def to_sequence(self, tokens, vocab_name, maxlen=None, reverse=False, dtype='int64', padding='post',
                    truncating='post'):
        if vocab_name == 'word' and self.lower:
            tokens = map(lambda x: x.lower(), tokens)
        sequence = list(map(self.vocab[vocab_name].word_to_id, tokens))
        pad_id = self.vocab[vocab_name].pad_id
        if maxlen is None:
            maxlen = self.max_slen
        elif maxlen == 'aspects':
            maxlen = self.max_tlen
        if reverse:
            sequence.reverse()
        return Tokenizer._pad_sequence(sequence, pad_id, maxlen, dtype, padding, truncating)

    def pad_sequence(self, sequence, pad_id, maxlen=None, reverse=False, dtype='int64', padding='post',
                     truncating='post'):
        if dtype == 'int64':
            sequence = list(map(int, sequence))
        elif dtype == 'float32':
            sequence = list(map(float, sequence))
        if maxlen is None:
            maxlen = self.max_slen
        if reverse:
            sequence.reverse()
        return Tokenizer._pad_sequence(sequence, pad_id, maxlen, dtype, padding, truncating)


class MyDataset(Dataset):
    ''' PyTorch standard dataset class '''

    def __init__(self, fname, tokenizer):
        data_file = os.path.join('dats', os.path.split(fname)[-1].replace('.json', '.dat'))  # dataset cache
        if os.path.exists(data_file):
            print(f"loading dataset: {data_file}")
            dataset = pickle.load(open(data_file, 'rb'))
        else:
            print('building dataset...')
            dataset = self._read_data(fname, tokenizer)
            pickle.dump(dataset, open(data_file, 'wb'))
        self._dataset = dataset

    @staticmethod
    def _read_data(fname, tokenizer):
        dataset = list()
        polarity_dict = {'positive': 0, 'negative': 1, 'neutral': 2}
        fdata = json.load(open(fname, 'r', encoding='utf-8'))
        for data in fdata:
            for aspect in data['aspects']:
                start, end = int(aspect['from']), int(aspect['to'])
                ''' position indicator '''
                text_left = data['token'][0:start]
                text_aspect = data['token'][start:end]
                text_right = data['token'][end:]
                pos_left = data['pos'][0:start]
                pos_aspect = data['pos'][start:end]
                pos_right = data['pos'][end:]
                text_pi = text_left + ['<p>'] + text_aspect + ['</p>'] + text_right
                pos_pi = pos_left + ['<p>'] + pos_aspect + ['</p>'] + pos_right
                ''' padding sequence '''
                text = tokenizer.to_sequence(text_pi, 'word')
                aspect_ids = tokenizer.to_sequence(text_aspect, 'word', maxlen='aspects')

                max_t = 40.0
                distance = []
                for pos in range(len(text_pi)):
                    if text_pi[pos] == '<p>' or text_pi[pos] == '</p>':
                        distance.append(-1)
                    elif pos < start:
                        distance.append(end - 1 - pos)
                    else:
                        distance.append(pos - start)

                pad = tokenizer.max_slen - len(text_pi)
                distance = distance + pad * [-1]
                position_weight = []
                for d in distance:
                    if d == -1 or d > max_t:
                        position_weight.append(0.0)
                    else:
                        position_weight.append(1 - float(d) / max_t)
                position_weight = np.asarray(position_weight, dtype=np.float32)

                pos = tokenizer.to_sequence(pos_pi, 'pos')
                deprel = tokenizer.to_sequence(aspect['deprel'], 'deprel')
                gather_idx = [i + 1 for i in range(len(text_left))] + [i + 2 + start for i in
                                                                       range(len(text_aspect))] + [i + 3 + end for i in
                                                                                                   range(
                                                                                                       len(text_right))]
                gather_idx = tokenizer.pad_sequence(gather_idx, 0)
                aspect_mask = [1 if start <= i < end else 0 for i in range(len(data['token']))]
                aspect_head = tokenizer.pad_sequence(aspect['head'], 0)
                aspect_mask = tokenizer.pad_sequence(aspect_mask, 0)
                ''' compute dependency path '''
                path = list()
                for i in range(len(data['token'])):
                    aspect['path'][i].append(i + 1 + tokenizer.max_slen)  # add a stop node with extra relation
                    path.append(tokenizer.pad_sequence(aspect['path'][i], 0, maxlen=tokenizer.max_plen))
                for i in range(tokenizer.max_slen - len(data['token'])):  # pad the paths
                    path.append(tokenizer.pad_sequence([], 0, maxlen=tokenizer.max_plen))
                path = np.asarray(path)
                polarity = polarity_dict[aspect['polarity']]
                dataset.append(((text, pos, deprel, aspect_head, aspect_mask, gather_idx, path,
                                 aspect_ids, position_weight), polarity))  # samples in (x, y) tuples
        return dataset

    def __getitem__(self, index):
        return self._dataset[index]

    def __len__(self):
        return len(self._dataset)


def build_tokenizer(fnames, dataset):
    if not os.path.exists('dats'):
        os.mkdir('dats')
    data_file = os.path.join('dats', f"{dataset}_tokenizer.dat")  # tokenizer cache
    if os.path.exists(data_file):
        print(f"loading tokenizer: {data_file}")
        tokenizer = pickle.load(open(data_file, 'rb'))
    else:
        print('building tokenizer...')
        tokenizer = Tokenizer.from_files(fnames)
        pickle.dump(tokenizer, open(data_file, 'wb'))
    return tokenizer


def _load_wordvec(data_path, word_dim, vocab=None):
    with open(data_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        word_vec = dict()
        word_vec['<pad>'] = np.zeros(word_dim).astype('float32')  # embedding vector for <pad> is always zero
        for line in f:
            tokens = line.rstrip().split()
            if (len(tokens) - 1) != word_dim:
                continue
            if tokens[0] == '<pad>' or tokens[0] == '<unk>':  # avoid them
                continue
            if vocab is None or vocab.has_word(tokens[0]):
                word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
        return word_vec


def build_embedding_matrix(vocab, dataset, word_dim=300, dir_path='..ss'):
    data_file = os.path.join('dats', f"{dataset}_embedding_matrix.dat")  # embedding matrix cache
    glove_file = os.path.join(dir_path, 'glove.840B.300d.txt')  # pre-trained glove embedding file
    if os.path.exists(data_file):
        print(f"loading embedding matrix: {data_file}")
        embedding_matrix = pickle.load(open(data_file, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.random.uniform(-0.25, 0.25, (len(vocab), word_dim)).astype(
            'float32')  # sample from U(-0.25,0.25)
        word_vec = _load_wordvec(glove_file, word_dim, vocab)
        for i in range(len(vocab)):
            vec = word_vec.get(vocab.id_to_word(i))
            if vec is not None:
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(data_file, 'wb'))
    return embedding_matrix
