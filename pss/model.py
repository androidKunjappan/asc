import torch
import torch.nn as nn
import torch.nn.functional as F

from rnn import DynamicLSTM


def get_features(inputs, device, initial):
    feature_ids, aspect_ids, feature_lens, aspect_lens, position_weight, masks, target = [], [], [], [], [], [], []
    a_mask, a_value, target_masks = [], [], []
    for d in inputs:
        feature_ids.append(d['word_ids'])
        aspect_ids.append(d['target_ids'])
        feature_lens.append(d['word_count'])
        aspect_lens.append(d['target_word_count'])
        position_weight.append(d['position_weight'])
        masks.append(d['mask'])
        target.append(d['y'])
        target_masks.append(d['target_mask'])
        if not initial:
            a_mask.append(d['amask'])
            a_value.append(d['avalue'])

    feature_ids = torch.tensor(feature_ids).long().to(device)
    aspect_ids = torch.tensor(aspect_ids).long().to(device)
    feature_lens = torch.tensor(feature_lens).long().to(device)
    aspect_lens = torch.tensor(aspect_lens).long().to(device)
    position_weight = torch.tensor(position_weight).float().to(device)
    masks = torch.tensor(masks).float().to(device)
    # masks = masks.eq(0)
    target = torch.tensor(target).long().to(device)

    target_masks = torch.tensor(target_masks).float().to(device)
    # target_masks = target_masks.eq(0)
    if not initial:
        a_mask = torch.tensor(a_mask).float().to(device)
        a_value = torch.tensor(a_value).float().to(device)
    else:
        a_mask = None
        a_value = None

    return feature_ids, aspect_ids, feature_lens, aspect_lens, position_weight, masks, target, a_mask, a_value, target_masks


class Model(nn.Module):
    def __init__(self, args, num_clasees, word_embeddings, device):
        super(Model, self).__init__()
        self.device = device

        self.word_embeddings = nn.Embedding(word_embeddings.shape[0], word_embeddings.shape[1], padding_idx=0)
        self.word_embeddings.weight.data.copy_(torch.from_numpy(word_embeddings).float())
        self.word_embeddings.weight.requires_grad = False

        lstm_dropout = 0
        if args.num_layers > 1:
            lstm_dropout = args.dropout
        self.lstm1 = DynamicLSTM(args.embed_dim, args.hidden_dim, num_layers=args.num_layers,
                                 batch_first=True, bidirectional=True, dropout=lstm_dropout, rnn_type=args.rnn_type)
        self.lstm2 = DynamicLSTM(args.embed_dim, args.hidden_dim, num_layers=args.num_layers,
                                 batch_first=True, bidirectional=True, dropout=lstm_dropout, rnn_type=args.rnn_type)
        self.lstm3 = DynamicLSTM(2 * args.hidden_dim, args.hidden_dim, num_layers=args.num_layers,
                                 batch_first=True, bidirectional=True, dropout=lstm_dropout)

        self.linear1 = nn.ModuleList()
        self.linear2 = nn.ModuleList()
        for i in range(2):
            self.linear1.append(nn.Linear(4 * args.hidden_dim, 2 * args.hidden_dim))
            self.linear2.append(nn.Linear(2 * args.hidden_dim, 1))
        # self.fc1 = nn.Linear(4 * args.hidden_dim, 2 * args.hidden_dim)
        self.classifier = nn.Linear(2 * args.hidden_dim, num_clasees)
        self.dropout = nn.Dropout(args.dropout)
        # self.linear2 = nn.Linear(2 * args.hidden_dim, 1)

    def forward(self, inputs, initial):
        feature_ids, aspect_ids, feature_lens, aspect_lens, position_weight, masks, target, a_mask, a_value\
            , target_masks = get_features(inputs, self.device, initial)

        features = self.word_embeddings(feature_ids)
        aspects = self.word_embeddings(aspect_ids)
        v, (_, _) = self.lstm1(features, feature_lens)
        e, (_, _) = self.lstm2(aspects, aspect_lens)

        v = self.dropout(v)
        e = self.dropout(e)
        for i in range(2):
            a = torch.bmm(v, e.transpose(1, 2))
            a = a.masked_fill(torch.bmm(masks.unsqueeze(2), target_masks.unsqueeze(1)).eq(0), -1e9)
            a = F.softmax(a, 2)
            aspect_mid = torch.bmm(a, e)
            aspect_mid = torch.cat((aspect_mid, v), dim=2)
            aspect_mid = F.leaky_relu(self.linear1[i](aspect_mid))
            aspect_mid = self.dropout(aspect_mid)
            t = torch.sigmoid(self.linear2[i](v))
            v = (1 - t) * aspect_mid + t * v
            v = position_weight.unsqueeze(2) * v

        target_masks = target_masks.eq(0).unsqueeze(2).repeat(1, 1, e.shape[2])
        # z, (_, _) = self.lstm3(v, feature_lens)

        query = torch.max(e.masked_fill(target_masks, -1e9), dim=1)[0].unsqueeze(1)
        # hidden_fwd, hidden_bwd = e.chunk(2, 1)
        # query = torch.cat((hidden_fwd[:, -1, :], hidden_bwd[:, 0, :]), dim=2).unsqueeze(1)

        alpha = torch.bmm(v, query.transpose(1, 2))
        alpha.masked_fill_(masks.eq(0).unsqueeze(2), -1e9)
        alpha = F.softmax(alpha, 1)
        z = torch.bmm(alpha.transpose(1, 2), v)
        z = self.classifier(z.squeeze(1))
        return z, alpha.squeeze(2), target, a_mask, a_value
