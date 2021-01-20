import os
import sys
import math
import torch
import random
import argparse
import numpy as np
from sklearn import metrics
from torch.utils.data import DataLoader
from model import RepWalk
from loss_func import CrossEntropy
from data_utils import build_tokenizer, build_embedding_matrix, MyDataset
import torch.nn.functional as F


def retrieve_args():
    dataset_files = {
        'restaurant': {
            'train': os.path.join('../RepWalk/datasets', 'Restaurants_Train.json'),
            'test': os.path.join('../RepWalk/datasets', 'Restaurants_Test.json')
        },
        'laptop': {
            'train': os.path.join('../RepWalk/datasets', 'Laptops_Train.json'),
            'test': os.path.join('../RepWalk/datasets', 'Laptops_Test.json')
        },
        'twitter': {
            'train': os.path.join('../RepWalk/datasets', 'Tweets_Train.json'),
            'test': os.path.join('../RepWalk/datasets', 'Tweets_Test.json')
        },
        'restaurant16': {
            'train': os.path.join('../RepWalk/datasets', 'Restaurants16_Train.json'),
            'test': os.path.join('../RepWalk/datasets', 'Restaurants16_Test.json')
        }
    }
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='restaurant', type=str, help=', '.join(dataset_files.keys()))
    parser.add_argument('--word_dim', default=300, type=int)
    parser.add_argument('--pos_dim', default=30, type=int)
    parser.add_argument('--dep_dim', default=50, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_epoch', default=100, type=int)
    parser.add_argument('--iterations', default=3, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--wt_decay', default=1e-5, type=float)
    parser.add_argument('--embed_dropout', default=0.5, type=float)
    parser.add_argument('--bilinear_dropout', default=0, type=float)
    parser.add_argument('--fc_dropout', default=0, type=float)
    parser.add_argument('--beta', default=0.01, type=float)
    parser.add_argument('--phi', default=100, type=float)
    parser.add_argument('--eps', default=0.01, type=float)
    parser.add_argument('--entropy', default=3, type=float)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--device', default=None, type=str, help='cpu, cuda')
    parser.add_argument('--compress', default='F', type=str, help='T, F')

    parser.add_argument('--cpt', action="store_true", help='run with cpt')
    parser.add_argument("--rnn_type", type=str, default="LSTM", help="lstm or gru")
    parser.add_argument('--early_stop', type=int, default=5, help='early stop')
    args = parser.parse_args()
    args.dataset_file = dataset_files[args.dataset]
    args.device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # ~ torch.backends.cudnn.deterministic = True #TODO:
    # ~ torch.backends.cudnn.benchmark = False    #TODO:
    print(args, flush=True)
    return args


def weight_init(model):
    for name, param in model.named_parameters():
            if param.requires_grad:
                if 'embedding' in name:
                    weight = torch.nn.init.xavier_uniform_(torch.zeros_like(param))
                    weight[0] = torch.tensor(0, dtype=param.dtype, device=param.device)
                    setattr(param, 'data', weight)
                else:
                    if len(param.shape) > 1:
                        torch.nn.init.xavier_uniform_(param)
                    else:
                        stdv = 1. / math.sqrt(param.size(0))
                        torch.nn.init.uniform_(param, a=-stdv, b=stdv)


def run_train(dataloader, model, criterion, optimizer, device, compress, att_neg_mask):
    train_loss, n_correct, n_train = 0, 0, 0
    model.train()
    att_weights = []
    correctness = []
    i = 0
    for sample_batched in dataloader:
        batch_size = len(sample_batched[0][0])
        attention_neg_mask = att_neg_mask[i:i+batch_size]
        i += batch_size
        inputs = list(map(lambda x: x.to(device), sample_batched[0]))
        labels = sample_batched[1].to(device)
        outputs = model(inputs, compress, attention_neg_mask)
        att_weights.append(outputs[1])
        #print(outputs[1].shape)
        #print(torch.sum(outputs[1]!=0, dim=1))

        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * len(labels)
        n_correct += (torch.argmax(outputs[0], -1) == labels).sum().item()
        n_train += len(labels)
        #print(torch.argmax(outputs[0], -1) == labels)
        correctness.append(torch.argmax(outputs[0], -1) == labels)
    return train_loss / n_train, n_correct / n_train, torch.cat(att_weights), torch.cat(correctness)


def run_train_final(dataloader, model, criterion, optimizer, device, compress, att_neg_mask, att_full_mask, att_target, phi):
    train_loss, n_correct, n_train = 0, 0, 0
    model.train()
    att_weights = []
    correctness = []
    i = 0
    for sample_batched in dataloader:
        batch_size = len(sample_batched[0][0])
        attention_neg_mask = att_neg_mask[i:i+batch_size]
        attention_full_mask = att_full_mask[i:i+batch_size]
        attention_target = att_target[i:i+batch_size]
        i += batch_size
        inputs = list(map(lambda x: x.to(device), sample_batched[0]))
        labels = sample_batched[1].to(device)
        outputs = model(inputs, compress, attention_neg_mask)
        att_weights.append(outputs[1])
        #print(outputs[1].shape)
        #print(torch.sum(outputs[1]!=0, dim=1))

        optimizer.zero_grad()
        loss1 = criterion(outputs, labels)
        loss2 = phi * F.mse_loss(outputs[1] * attention_full_mask, attention_target)
        #print(loss1.item(), loss2.item())
        loss = loss1 + loss2

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * len(labels)
        n_correct += (torch.argmax(outputs[0], -1) == labels).sum().item()
        n_train += len(labels)
        #print(torch.argmax(outputs[0], -1) == labels)
        correctness.append(torch.argmax(outputs[0], -1) == labels)
    return train_loss / n_train, n_correct / n_train, torch.cat(att_weights), torch.cat(correctness)


def run_test(dataloader, model, criterion, device, compress):
    test_loss, n_correct, n_test = 0, 0, 0
    labels_all, predicts_all = None, None
    model.eval()
    with torch.no_grad():
        for sample_batched in dataloader:
            inputs = list(map(lambda x: x.to(device), sample_batched[0]))
            labels = sample_batched[1].to(device)
            outputs = model(inputs, compress)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * len(labels)
            n_correct += (torch.argmax(outputs[0], -1) == labels).sum().item()
            n_test += len(labels)
            labels_all = torch.cat((labels_all, labels), dim=0) if labels_all is not None else labels
            predicts_all = torch.cat((predicts_all, outputs[0]), dim=0) if predicts_all is not None else outputs[0]
    f1 = metrics.f1_score(labels_all.cpu(), torch.argmax(predicts_all, -1).cpu(), labels=[0, 1, 2], average='macro') # compute f1 score
    return test_loss / n_test, n_correct / n_test, f1


def calc_entropy_index(weights):
    index = 0
    entropy = 0
    for i in range(weights.shape[0]):
        if weights[i] > 0:
            entropy -= torch.log(weights[i])*weights[i]
        if weights[i] > weights[index]:
            index = i
    return entropy, index


def get_mask_target(new_weights, curr_neg_mask, curr_full_mask, curr_target, correctness, max_entropy):
    n = new_weights.shape[0]
    new_neg_mask = curr_neg_mask.clone().detach()
    new_full_mask = curr_full_mask.clone().detach()
    new_target = curr_target.clone().detach()
    count_entropy = 0
    for i in range(n):
        weights = new_weights[i]
        entropy, index = calc_entropy_index(weights)
        if entropy < max_entropy:
            count_entropy += 1
            new_full_mask[i][index] = 1
            if correctness[i] == False:
                new_neg_mask[i][index] = 1
            else:
                new_target[i][index] = 1
    print("Skipped", n-count_entropy, "out of", n, "due to entropy above threshold")
    return new_neg_mask, new_full_mask, new_target


def main():
    args = retrieve_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.tokenizer = build_tokenizer(fnames=args.dataset_file.values(), dataset=args.dataset) # transfrom tokens to indices
    embedding_matrix = build_embedding_matrix(vocab=args.tokenizer.vocab['word'], dataset=args.dataset, dir_path='../../') # pre-trained glove embeddings
    trainset = MyDataset(fname=args.dataset_file['train'], tokenizer=args.tokenizer) #TODO
    testset = MyDataset(fname=args.dataset_file['test'], tokenizer=args.tokenizer) #TODO
    model = RepWalk(embedding_matrix, args).to(args.device)

    n_train = len(trainset._dataset)
    f_train = len(trainset._dataset[0][0][0])

    #print(len(trainset._dataset))
    #print(trainset._dataset[0][0][0])
    #print(trainset._dataset[0].shape, "shapeeeeeeeeeeeeeeeeeE")
    _params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(_params, lr=args.lr, weight_decay=args.wt_decay)
    criterion = CrossEntropy(beta=args.beta, eps=args.eps)

    train_dataloader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(dataset=testset, batch_size=args.batch_size, shuffle=False)

    if args.compress == 'T':
        compress = True
        print("Starting to train compressed model")
    else:
        compress = False
        print("Starting to train extended model")
    att_weights_list = []
    att_neg_mask_list = []  #for masking attention weights leading to negative prediction
    att_full_mask_list = [] #mask of positive and negative attention weights. used for loss calculation
    att_target_list = []

    #att_weights_list.append((torch.LongTensor(n_train, f_train)*0).to(args.device))
    #att_neg_mask_list.append((torch.LongTensor(n_train, f_train)*0).to(args.device))
    #att_full_mask_list.append((torch.LongTensor(n_train, f_train)*0).to(args.device))
    #att_target_list.append((torch.FloatTensor(n_train, f_train)*0).to(args.device))

    att_weights_list.append(torch.zeros((n_train, f_train), dtype = torch.float32).to(args.device))
    att_neg_mask_list.append(torch.zeros((n_train, f_train), dtype = torch.int32).to(args.device))
    att_full_mask_list.append(torch.zeros((n_train, f_train), dtype = torch.int32).to(args.device))
    att_target_list.append(torch.zeros((n_train, f_train), dtype = torch.float32).to(args.device))

    init_accs = []
    final_accs = []
    for outer_iter_init in range(args.iterations):
        print('\n')
        print('*' * 20)
        print("\nStarting initial iteration :", outer_iter_init + 1)
        weight_init(model)
        best_test_acc, best_test_f1 = 0, 0
        best_att_weights = None
        best_correctness = None

        for epoch in range(args.num_epoch):
            train_loss, train_acc, att_weights, correctness = run_train(train_dataloader, model, criterion, optimizer, args.device, compress, att_neg_mask_list[-1])
            test_loss, test_acc, test_f1 = run_test(test_dataloader, model, criterion, args.device, compress)
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_test_f1 = test_f1
                best_att_weights = att_weights
                best_correctness = correctness
                print("Best accuracy in this iteration..")
            print(f"{100*(epoch+1)/args.num_epoch:6.2f}% > loss: {train_loss:.4f}, acc: {train_acc:.4f}, test acc: {test_acc:.4f}, test f1: {test_f1:.4f}", flush=True)

        print(f"best test acc: {best_test_acc:.4f}, best test f1: {best_test_f1:.4f}")
        init_accs.append(best_test_acc)
        print("\nRetrieving attention weight masks for this iteration...")
        best_att_neg_mask, best_att_full_mask, best_att_target = get_mask_target(best_att_weights, att_neg_mask_list[-1], att_full_mask_list[-1], att_target_list[-1], best_correctness, args.entropy)
        att_weights_list.append(best_att_weights)
        att_neg_mask_list.append(best_att_neg_mask)
        att_full_mask_list.append(best_att_full_mask)
        att_target_list.append(best_att_target)

    print('#' * 50)
    print(f"best test acc: {best_test_acc:.4f}, best test f1: {best_test_f1:.4f}")

    #print(att_weights_list[0])
    #print(att_weights_list[1])
    #print(torch.sum(att_neg_mask_list[1]))
    #print(torch.sum(att_full_mask_list[1]))
    #print(torch.sum(att_target_list[1]))

    for outer_iter_final in range(args.iterations+1):
        print('*' * 20)
        print("\n\nStarting final iteration :", outer_iter_final)
        weight_init(model)
        best_test_acc, best_test_f1 = 0, 0
        best_att_weights = None

        for epoch in range(args.num_epoch):
            train_loss, train_acc, _, _ =  run_train_final(train_dataloader, model, criterion, optimizer, args.device, compress, att_neg_mask_list[outer_iter_final], \
                                                           att_full_mask_list[outer_iter_final], att_target_list[outer_iter_final], args.phi)
            test_loss, test_acc, test_f1 = run_test(test_dataloader, model, criterion, args.device, compress)
            _, train_acc, _ = run_test(train_dataloader, model, criterion, args.device, compress)
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_test_f1 = test_f1
                print("Best accuracy in this iteration..")
            print(f"{100*(epoch+1)/args.num_epoch:6.2f}% > loss: {train_loss:.4f}, acc: {train_acc:.4f}, test acc: {test_acc:.4f}, test f1: {test_f1:.4f}", flush=True)
            print('', flush=True)
        print(f"best test acc: {best_test_acc:.4f}, best test f1: {best_test_f1:.4f}")
        final_accs.append(best_test_acc)

    print("\n\nFinal Accuracies per iteration")
    print(init_accs)
    print(final_accs)
    print("dataset:", args.dataset)
    print("Arguments used")
    print("batch_size:", args.batch_size, "lr:", args.lr, "phi:", args.phi, "entropy:", args.entropy, "eps:", args.eps, "beta:", args.beta, flush=True)


if __name__ == '__main__':
    main()
