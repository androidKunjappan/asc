import os
import math
import torch
import random
import argparse
import numpy as np
from sklearn import metrics
from torch.utils.data import DataLoader
from model_cpt import RepWalk
from loss_func import CrossEntropy
from data_utils_cpt import build_tokenizer, build_embedding_matrix, MyDataset


def retrieve_args():
    dataset_files = {
        'restaurant': {
            'train': os.path.join('datasets', 'Restaurants_Train.json'),
            'test': os.path.join('datasets', 'Restaurants_Test.json')
        },
        'laptop': {
            'train': os.path.join('datasets', 'Laptops_Train.json'),
            'test': os.path.join('datasets', 'Laptops_Test.json')
        },
        'twitter': {
            'train': os.path.join('datasets', 'Tweets_Train.json'),
            'test': os.path.join('datasets', 'Tweets_Test.json')
        },
        'restaurant16': {
            'train': os.path.join('datasets', 'Restaurants16_Train.json'),
            'test': os.path.join('datasets', 'Restaurants16_Test.json')
        }
    }
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='twitter', type=str, help=', '.join(dataset_files.keys()))
    parser.add_argument('--word_dim', default=300, type=int)
    parser.add_argument('--pos_dim', default=30, type=int)
    parser.add_argument('--dep_dim', default=50, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_epoch', default=100, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--wt_decay', default=1e-5, type=float)
    parser.add_argument('--embed_dropout', default=0.5, type=float)
    parser.add_argument('--bilinear_dropout', default=0, type=float)
    parser.add_argument('--fc_dropout', default=0, type=float)
    parser.add_argument('--beta', default=0.01, type=float)
    parser.add_argument('--eps', default=0.01, type=float)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--device', default=None, type=str, help='cpu, cuda')
    parser.add_argument('--compress', default='F', type=str, help='T, F')

    parser.add_argument("--rnn_type", type=str, default="LSTM", help="lstm or gru")
    parser.add_argument('--early_stop', type=int, default=5, help='early stop')

    args = parser.parse_args()
    args.dataset_file = dataset_files[args.dataset]
    args.seed = args.seed if args.seed else random.randint(0, 4294967295)
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


def run_train(dataloader, model, criterion, optimizer, device, compress):
    train_loss, n_correct, n_train = 0, 0, 0
    model.train()
    for sample_batched in dataloader:
        inputs = list(map(lambda x: x.to(device), sample_batched[0]))
        labels = sample_batched[1].to(device)
        outputs = model(inputs, compress)

        optimizer.zero_grad()
        loss = criterion(outputs[0], labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * len(labels)
        n_correct += (torch.argmax(outputs[0], -1) == labels).sum().item()
        n_train += len(labels)
    return train_loss / n_train, n_correct / n_train

    
def run_test(dataloader, model, criterion, device, compress):
    test_loss, n_correct, n_test = 0, 0, 0
    labels_all, predicts_all = None, None
    model.eval()
    with torch.no_grad():
        for sample_batched in dataloader:
            inputs = list(map(lambda x: x.to(device), sample_batched[0]))
            labels = sample_batched[1].to(device)
            outputs = model(inputs, compress)
            loss = criterion(outputs[0], labels)
            
            test_loss += loss.item() * len(labels)
            n_correct += (torch.argmax(outputs[0], -1) == labels).sum().item()
            n_test += len(labels)
            labels_all = torch.cat((labels_all, labels), dim=0) if labels_all is not None else labels
            predicts_all = torch.cat((predicts_all, outputs[0]), dim=0) if predicts_all is not None else outputs[0]
    f1 = metrics.f1_score(labels_all.cpu(), torch.argmax(predicts_all, -1).cpu(), labels=[0, 1, 2], average='macro') # compute f1 score
    return test_loss / n_test, n_correct / n_test, f1


def main():
    args = retrieve_args()
    args.tokenizer = build_tokenizer(fnames=args.dataset_file.values(), dataset=args.dataset) # transfrom tokens to indices
    print(args.tokenizer.max_slen)
    print(args.tokenizer.max_tlen)
    print(args.tokenizer.max_plen)

    embedding_matrix = build_embedding_matrix(vocab=args.tokenizer.vocab['word'], dataset=args.dataset, dir_path = '../..') # pre-trained glove embeddings
    trainset = MyDataset(fname=args.dataset_file['train'], tokenizer=args.tokenizer) #TODO
    testset = MyDataset(fname=args.dataset_file['test'], tokenizer=args.tokenizer) #TODO
    model = RepWalk(embedding_matrix, args).to(args.device)

    _params = filter(lambda p: p.requires_grad, model.parameters())
    print(_params)
    optimizer = torch.optim.Adam(_params, lr=args.lr, weight_decay=args.wt_decay)
    # criterion = CrossEntropy(beta=args.beta, eps=args.eps)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    print(model)
    print(optimizer)
    
    train_dataloader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=testset, batch_size=args.batch_size, shuffle=False)
    print('train', len(trainset))
    print('test', len(testset))
    # weight_init(model)
    best_test_acc, best_test_f1 = 0, 0
    best_epoch = 0
    if args.compress == 'T':
        compress = True
        print("Starting to train compressed model")
    else:
        compress = False
        print("Starting to train extended model")
    for epoch in range(args.num_epoch):
        train_loss, train_acc = run_train(train_dataloader, model, criterion, optimizer, args.device, compress)
        test_loss, test_acc, test_f1 = run_test(test_dataloader, model, criterion, args.device, compress)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_test_f1 = test_f1
            best_epoch = epoch
        # else:
        #     scheduler.step()
        print(f"{100*(epoch+1)/args.num_epoch:6.2f}% > loss: {train_loss:.4f}, acc: {train_acc:.4f}, test acc: {test_acc:.4f}, test f1: {test_f1:.4f}")
        print('', flush=True)
        if epoch > best_epoch + 10:
            break

    print('#' * 50)
    print(f"best test acc: {best_test_acc:.4f}, best test f1: {best_test_f1:.4f}")


if __name__ == '__main__':
    main()
