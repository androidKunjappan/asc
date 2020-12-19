import argparse
import time
from cmath import inf

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from data_load import load_data
from model import Model


criterion = nn.CrossEntropyLoss()
mse_loss = nn.MSELoss()


def pass_data_iteratively(model, x, initial, batch_size=128):
    outputs = []
    alphas = []
    targets = []
    total_size = len(x)
    idx = np.arange(total_size)
    for i in range(0, total_size, batch_size):
        idx_tmp = idx[i: i+batch_size]

        with torch.no_grad():
            output, alpha, target, a_mask, a_value = model([x[i] for i in idx_tmp], initial)
        outputs.append(output)
        alphas.append(alpha)
        targets.append(target)

    return torch.cat(outputs), torch.cat(alphas), torch.cat(targets)


def main_init(args, alphas_list, k, device):
    max_test_acc = 0
    num_clasees = 3
    dataset, word_to_id, word_list, word_embeddings = load_data(args.dataset_name, alphas_list, True)
    args.embed_dim = len(word_embeddings[1])
    args.sent_len = len(dataset['train'][0]['word_ids'])
    args.target_len = len(dataset['train'][0]['target_ids'])

    train_data = dataset['train']
    test_data = dataset['test']
    train_size = len(train_data)

    # alpha_files[k] = 'alpha-' + str(k)
    model = Model(args, num_clasees, word_embeddings, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    alphas_to_save = None
    batch_size = args.batch_size
    for epoch in range(1, args.epochs + 1):
        train_init(epoch, model, optimizer, train_data, train_size, batch_size)

        alphas_to_save, max_test_acc = test_init(epoch, model, scheduler, train_data, test_data, alphas_to_save,
                                                 max_test_acc, k)
        print('')
    print('max test accuracy : ', max_test_acc)
    print('=' * 100)

    alphas_list.append(alphas_to_save)
    return alphas_list, max_test_acc


def test_init(epoch, model, scheduler, train_data, test_data, alphas_to_save, max_test_acc, k):
    model.eval()
    outputs_train, alphas_train, targets_train = pass_data_iteratively(model, train_data, batch_size=128,
                                                                       initial=True)
    pred_train = outputs_train.max(1, keepdim=True)[1]
    correct = pred_train.eq(targets_train.view_as(pred_train)).sum().cpu().item()
    train_acc = correct / float(len(targets_train))
    outputs_test, alpha_test, targets_test = pass_data_iteratively(model, test_data, batch_size=128, initial=True)
    pred_test = outputs_test.max(1, keepdim=True)[1]
    correct = pred_test.eq(targets_test.view_as(pred_test)).sum().cpu().item()
    test_acc = correct / float(len(targets_test))
    if test_acc > max_test_acc:
        max_test_acc = test_acc
        pred_train = pred_train.view_as(targets_train)
        alphas_to_save = alphas_train.detach().cpu().numpy()
        for ind in range(len(pred_train)):
            if pred_train[ind] != targets_train[ind]:
                alphas_to_save[ind] = -alphas_to_save[ind]
    else:
        scheduler.step()
    print('initial', k, 'epoch :', epoch, 'accuracy train :', train_acc, 'test :', test_acc, flush=True)
    return alphas_to_save, max_test_acc


def train_init(epoch, model, optimizer, train_data, train_size, batch_size):
    model.train()
    train_idx = np.random.permutation(train_size)
    loss_accum = 0
    for i in range(0, train_size, batch_size):
        idx = train_idx[i: i + batch_size]
        output, alpha, target, a_mask, a_value = model([train_data[i] for i in idx], initial=True)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()

        loss = loss.detach().cpu().numpy()
        loss_accum += loss
    print('epoch :', epoch, 'loss :', loss_accum, flush=True)


def main_final(args, alphas_list, k, device):
    max_test_acc = 0
    num_clasees = 3
    dataset, word_to_id, word_list, word_embeddings = load_data(args.dataset_name, alphas_list, False)
    # args.embed_dim = len(word_embeddings[1])
    # args.sent_len = len(dataset['train'][0]['word_ids'])
    # args.target_len = len(dataset['train'][0]['target_ids'])

    train_data = dataset['train']
    test_data = dataset['test']
    train_size = len(train_data)

    model = Model(args, num_clasees, word_embeddings, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    batch_size = args.batch_size
    for epoch in range(1, args.epochs + 1):
        train_final(epoch, args, model, optimizer, train_data, train_size, batch_size)

        max_test_acc = test_final(epoch, model, scheduler, train_data, test_data, max_test_acc, k)

        print('')

    print('max test accuracy : ', max_test_acc)
    print('=' * 100)

    return alphas_list, max_test_acc


def test_final(epoch, model, scheduler, train_data, test_data, max_test_acc, k):
    model.eval()
    outputs_train, alphas_train, targets_train = pass_data_iteratively(model, train_data, batch_size=128,
                                                                       initial=True)
    pred_train = outputs_train.max(1, keepdim=True)[1]
    correct = pred_train.eq(targets_train.view_as(pred_train)).sum().cpu().item()
    train_acc = correct / float(len(targets_train))
    outputs_test, alpha_test, targets_test = pass_data_iteratively(model, test_data, batch_size=128, initial=True)
    pred_test = outputs_test.max(1, keepdim=True)[1]
    correct = pred_test.eq(targets_test.view_as(pred_test)).sum().cpu().item()
    test_acc = correct / float(len(targets_test))
    if test_acc > max_test_acc:
        max_test_acc = test_acc
    else:
        scheduler.step()
    print('final', k, 'epoch :', epoch, 'accuracy train :', train_acc, 'test :', test_acc, flush=True)
    return max_test_acc


def train_final(epoch, args, model, optimizer, train_data, train_size, batch_size):
    model.train()
    train_idx = np.random.permutation(train_size)
    loss_accum = 0
    for i in range(0, train_size, batch_size):
        idx = train_idx[i: i + batch_size]
        output, alpha, target, a_mask, a_value = model([train_data[i] for i in idx], initial=False)
        loss1 = criterion(output, target)
        loss2 = mse_loss(alpha * a_mask, a_value)
        loss = loss1 + args.beta * loss2

        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()

        loss = loss.detach().cpu().numpy()
        loss_accum += loss
    print('epoch :', epoch, 'loss :', loss_accum, flush=True)


def main():
    parser = argparse.ArgumentParser(description='Pytorch for RTER')
    parser.add_argument("--dataset_name", type=str, default="14semeval_rest", help="dataset name")
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--hidden_dim', type=int, default=50, help='hidden dimension')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-4)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout (default: 0.3)')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay (default: 0.3)')
    parser.add_argument('--early_stop', type=int, default=5, help='early stop')
    parser.add_argument('--beta', type=float, default=.1, help='beta')

    args = parser.parse_args()

    # if args.dataset_name == '14semeval_rest' or args.ds_name == '14semeval_rest_val':
    #     args.beta = 0.5

    print(args, flush=True)

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(0)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    print('device : ', device, flush=True)

    max_k = 5
    alphas_list = []
    init_accuracies = []
    for k in range(max_k):
        alphas_list, max_test_acc = main_init(args, alphas_list, k, device)
        init_accuracies.append(max_test_acc)

    alphas_list_new = []
    final_accuracies = []
    for k in range(max_k):
        alphas_list_new.append(alphas_list[k])
        alphas_list_new, max_test_acc = main_final(args, alphas_list_new, k, device)
        final_accuracies.append(max_test_acc)

    print('=' * 150)
    print(args)
    for i in range(max_k):
        print('k :', i, '\taccuracy int :', init_accuracies[i], '\taccuracy final :', final_accuracies[i])

    print('=' * 150)


if __name__ == '__main__':
    main()
