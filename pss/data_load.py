import numpy as np
import pickle
import os

from util import get_embedding


def read(path):
    data = []
    with open(path, encoding='utf-8') as f:
        # rid = 0
        for line in f:
            record = {}
            tokens = line.strip().split()
            words, target_words = [], []
            d = []
            flag = False
            for t in tokens:
                if '/p' in t or '/n' in t or '/0' in t:
                    end = 'xx'
                    y = 0
                    if '/p' in t:
                        end = '/p'
                        y = 1
                    elif '/n' in t:
                        end = '/n'
                        y = 0
                    elif '/0' in t:
                        end = '/0'
                        y = 2

                    words.append(t.strip(end))
                    target_words.append(t.strip(end))

                    if not flag:
                        flag = True
                        record['y'] = y
                        start_t = end_t = tokens.index(t)
                    else:
                        end_t += 1
                else:
                    words.append(t)

            for pos in range(len(tokens)):
                if pos < start_t:
                    d.append(end_t - pos)
                else:
                    d.append(pos - start_t)

            # record['sent'] = line.strip()
            record['words'] = words
            record['targets'] = target_words
            record['word_count'] = len(words)
            record['target_word_count'] = len(target_words)

            record['distance'] = d
            # record['sid'] = rid
            # record['beg'] = start_t
            # record['end'] = end_t + 1
            # rid += 1
            data.append(record)
    return data


def add_pos_weight(data, max_len):
    max_t = 40
    for i in range(len(data)):
        data[i]['position_weight'] = []
        pad = max_len - len(data[i]['distance'])
        data[i]['distance'] = data[i]['distance'] + pad * [-1]
        for d in data[i]['distance']:
            if d == -1 or d > max_t:
                data[i]['position_weight'].append(0.0)
            else:
                data[i]['position_weight'].append(1 - float(d) / max_t)

    return data


def get_vocab(data):
    word_to_id = {}
    word_set = set()

    for d in data:
        for word in d['words']:
            word_set.add(word)
        for word in d['targets']:
            word_set.add(word)

    word_list = ['<pad>'] + list(word_set)
    for i, word in enumerate(word_list):
        word_to_id[word] = i

    return word_to_id, word_list


def data_word_to_id(data, word_to_id, max_len, max_len_t):
    for i in range(len(data)):
        word_list = []
        for word in data[i]['words']:
            word_list.append(word_to_id[word])
        pad = max_len - len(word_list)
        word_list += pad * [0]
        data[i]['word_ids'] = word_list

        word_list = []
        for word in data[i]['targets']:
            word_list.append(word_to_id[word])
        pad = max_len_t - len(word_list)
        target_mask = [1] * len(word_list) + pad * [0]
        word_list += pad * [0]
        data[i]['target_ids'] = word_list
        data[i]['target_mask'] = target_mask

    return data


def get_attention_mask_init(dataset, alphas_list):
    max_entropy = 3.0

    for i in range(len(dataset)):
        masks = []
        for w in dataset[i]['distance']:
            if w == -1:  # -1 is the padding part
                masks.append(0.0)
            else:
                masks.append(1.0)

        for alphas in alphas_list:
            if alphas is not None:
                alpha = alphas[i][alphas[i] != 0]
                entropy = - np.sum(np.log2(abs(alpha)) * abs(alpha))
                if entropy < max_entropy:
                    index = abs(alphas[i]).argmax()
                    masks[index] = 0.0
                    dataset[i]['word_ids'][index] = 0

        dataset[i]['mask'] = masks

    return dataset


def get_attention_mask_final(dataset, alphas_list):
    max_entropy = 3.0

    for i in range(len(dataset)):
        masks = []
        amasks = []
        avalues = []
        for d in dataset[i]['distance']:
            if d == -1:
                masks.append(0.0)
            else:
                masks.append(1.0)
            amasks.append(0.0)
            avalues.append(0.0)

        for alphas in alphas_list:
            if alphas is not None:
                alpha = alphas[i][alphas[i] != 0]
                entropy = - np.sum(np.log2(abs(alpha)) * abs(alpha))
                if entropy < max_entropy:
                    index = abs(alphas[i]).argmax()
                    amasks[index] = 1.0
                    if alphas[i][index] > 0:
                        avalues[index] = 1.0
        dataset[i]['mask'] = masks
        dataset[i]['amask'] = amasks
        dataset[i]['avalue'] = avalues

    return dataset


def get_attention_mask_test(dataset):
    for i in range(len(dataset)):
        masks = []
        for d in dataset[i]['distance']:
            if d == -1:
                masks.append(0.0)
            else:
                masks.append(1.0)
        dataset[i]['mask'] = masks
    return dataset


def load_data(dataset_name, alphas_list, erase=True):
    embeddings, test_data, train_data, word_list, word_to_id = get_dataset(dataset_name)

    # alphas_list = []
    # for i in range(len(alpha_files)):
    #     if alpha_files[i] is not None:
    #         alphas_list.append(np.loadtxt(alpha_files[i]))
    #     else:
    #         alphas_list.append(None)

    if erase:
        train_data = get_attention_mask_init(dataset=train_data, alphas_list=alphas_list)
    else:
        train_data = get_attention_mask_final(dataset=train_data, alphas_list=alphas_list)
    test_data = get_attention_mask_test(dataset=test_data)

    return {'train': train_data, 'test': test_data}, word_to_id, word_list, embeddings


def get_dataset(dataset_name):

    pickle_file = './embeddings/%s_dump.pkl' % dataset_name
    if os.path.exists(pickle_file):
        return pickle.load(open(pickle_file, 'rb'))

    train_file = './dataset/' + dataset_name + '/train.txt'
    test_file = './dataset/' + dataset_name + '/test.txt'

    train_data = read(train_file)
    test_data = read(test_file)

    train_wc = [t['word_count'] for t in train_data]
    test_wc = [t['word_count'] for t in test_data]
    max_len = max(train_wc + test_wc)

    train_wc_t = [t['target_word_count'] for t in train_data]
    test_wc_t = [t['target_word_count'] for t in test_data]
    max_len_t = max(train_wc_t + test_wc_t)

    train_data = add_pos_weight(train_data, max_len)
    test_data = add_pos_weight(test_data, max_len)

    word_to_id, word_list = get_vocab(train_data + test_data)
    train_data = data_word_to_id(train_data, word_to_id, max_len, max_len_t)
    test_data = data_word_to_id(test_data, word_to_id, max_len, max_len_t)
    embeddings = get_embedding(word_to_id)

    data = embeddings, test_data, train_data, word_list, word_to_id

    pickle.dump(data, open(pickle_file, 'wb'))

    return embeddings, test_data, train_data, word_list, word_to_id

