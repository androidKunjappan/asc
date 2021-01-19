import torch
import torch.nn as nn
import torch.nn.functional as F

from RepWalk.rnn import DynamicLSTM


class WordEmbedLayer(nn.Module):
    ''' LSTM which can hold variable length sequence, use like TensorFlow's RNN(input, length...). '''

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0,
                 bidirectional=False, only_use_last_hidden_state=False, rnn_type='LSTM'):
        super(WordEmbedLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.only_use_last_hidden_state = only_use_last_hidden_state
        self.rnn_type = rnn_type

        if self.rnn_type == 'LSTM':
            self.RNN = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                               bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'GRU':
            self.RNN = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                              bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'RNN':
            self.RNN = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                              bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)

    def forward(self, x, x_len):
        '''
        sequence -> sort -> pad and pack -> process using RNN -> unpack -> unsort
        '''
        total_length = x.size(1) if self.batch_first else x.size(0)  # the length of sequence
        ''' sort '''
        x_sort_idx = torch.sort(x_len, descending=True)[1].long()
        x_unsort_idx = torch.sort(x_sort_idx)[1].long()
        x_len = x_len[x_sort_idx]
        x = x[x_sort_idx]
        ''' pack '''
        x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=self.batch_first)
        ''' process '''
        if self.rnn_type == 'LSTM':
            out_pack, (ht, ct) = self.RNN(x_emb_p, None)
        else:
            out_pack, ht = self.RNN(x_emb_p, None)
            ct = None
        ''' unsort '''
        ht = ht[:, x_unsort_idx]
        if self.only_use_last_hidden_state:
            return ht
        else:
            out, _ = torch.nn.utils.rnn.pad_packed_sequence(out_pack, batch_first=self.batch_first,
                                                            total_length=total_length)
            if self.batch_first:
                out = out[x_unsort_idx]
            else:
                out = out[:, x_unsort_idx]
            if self.rnn_type == 'LSTM':
                ct = ct[:, x_unsort_idx]
            return out, (ht, ct)


class RepWalk(nn.Module):
    ''' Neural Network Structure '''

    def __init__(self, embedding_matrix, args):
        super(RepWalk, self).__init__()
        ''' embedding layer '''
        self.word_embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.pos_embedding = nn.Embedding(len(args.tokenizer.vocab['pos']), args.pos_dim, padding_idx=0)
        self.deprel_embedding = nn.Embedding(len(args.tokenizer.vocab['deprel']), args.dep_dim, padding_idx=0)
        ''' other parameter '''
        self.pad_word = nn.Parameter(torch.zeros(args.hidden_dim * 2), requires_grad=False)
        self.pad_edge = nn.Parameter(torch.ones(1), requires_grad=False)
        self.ext_rel = nn.Parameter(torch.Tensor(args.dep_dim), requires_grad=True)
        ''' main layer '''
        self.rnn = WordEmbedLayer(args.word_dim + args.pos_dim, args.hidden_dim, num_layers=1, batch_first=True,
                                  bidirectional=True, rnn_type='GRU')  # bi-gru layer
        self.cpt = CPT(args, args.word_dim)
        self.linear = nn.Linear(2 * args.hidden_dim, 1)

        self.bilinear = nn.Bilinear(args.hidden_dim * 4, args.dep_dim, 1)
        self.fc_out = nn.Linear(args.hidden_dim * 2, 3)
        ''' dropout layer '''
        self.embed_dropout = nn.Dropout(args.embed_dropout)
        self.bilinear_dropout = nn.Dropout(args.bilinear_dropout)
        self.fc_dropout = nn.Dropout(args.fc_dropout)

    def forward(self, inputs, compressed_version):
        text, pos, deprel, aspect_head, aspect_mask, gather_idx, path, aspect_ids, position_weight = inputs
        '''Generate hidden representation for nodes as BiGRU(node_embeddings <+> pos-tag_embeddings)'''
        word_feature = self.embed_dropout(self.word_embedding(text))
        pos_feature = self.embed_dropout(self.pos_embedding(pos))
        text_len = torch.sum(text != 0, dim=-1)

        aspect_feature = self.word_embedding(aspect_ids)
        aspect_lens = torch.sum(aspect_ids != 0, dim=-1)

        node_feature, _ = self.rnn(torch.cat((word_feature, pos_feature), dim=-1), text_len.cpu())
        BS, SL, FD = node_feature.shape

        masks = (text != 0).float()
        target_masks = (aspect_ids != 0).float()
        word_feature1 = self.word_embedding(text)
        v = self.cpt(word_feature1, text_len, aspect_feature, aspect_lens, masks, target_masks, position_weight)

        '''add a padding word.. somehow this improves performance'''
        # padword_feature = self.pad_word.reshape(1, 1, -1).expand(BS, -1, -1)
        # padded_node_feature = torch.cat((padword_feature, node_feature), dim=1)
        #
        # '''Gather right words from sentence(by removing embeds of some token words like <p>, based on gather_idx'''
        # gather_idx = gather_idx.unsqueeze(0).expand(FD, -1, -1).permute(1, 2, 0)
        # node_feature = torch.cat((padword_feature, node_feature), dim=1)
        # node_feature = torch.gather(padded_node_feature, 1, gather_idx)
        #
        # '''Along with this node feature, concatenate features of head nodes and tail nodes'''
        # aspect_head = aspect_head.unsqueeze(0).expand(FD, -1, -1).permute(1, 2, 0)
        # deptext_feature = torch.gather(padded_node_feature, 1, aspect_head)
        # if compressed_version == False:
        #     head_text_feature = torch.cat((deptext_feature, node_feature), dim=1)
        #     exttext_feature = self.pad_word.reshape(1, 1, -1).expand(BS, SL, -1)
        #     tail_text_feature = torch.cat((node_feature, exttext_feature), dim=1)
        #     edge_feature = torch.cat((head_text_feature, tail_text_feature), dim=-1)
        # else:
        #     edge_feature = torch.cat((deptext_feature, node_feature), dim=-1)
        #
        # ''' score function by attention over the node features and edge features '''
        # if compressed_version == False:
        #     deprel_feature = self.embed_dropout(self.deprel_embedding(deprel))
        #     extrel_feature = self.embed_dropout(self.ext_rel).reshape(1, 1, -1).expand(BS, SL, -1)
        #     label_feature = torch.cat((deprel_feature, extrel_feature), dim=1)
        # else:
        #     label_feature = self.embed_dropout(self.deprel_embedding(deprel))
        # edge_score = torch.sigmoid(self.bilinear(self.bilinear_dropout(edge_feature), label_feature))
        # padedge_feature = self.pad_edge.reshape(1, 1, -1).expand(BS, -1, -1)
        # edge_score = torch.cat((padedge_feature, edge_score.transpose(1, 2)), dim=-1).expand(-1, SL, -1)
        #
        # ''' node weight '''
        # if compressed_version == True:
        #     path = path * (path <= SL)
        # x = torch.gather(edge_score, 2, path)
        # node_weight = torch.prod(x, dim=-1, keepdim=True)
        # text_mask = (text != 0).unsqueeze(-1)
        # node_weight = torch.where(text_mask != 0, node_weight, torch.zeros_like(node_weight))
        # aspect_mask = aspect_mask.unsqueeze(-1)
        # node_weight = torch.where(aspect_mask == 0, node_weight, torch.zeros_like(node_weight))
        #
        # # softmax = torch.nn.Softmax(dim=1)
        # # node_weight = softmax(torch.where(node_weight!=0, node_weight, torch.ones_like(node_weight)*-100000000000))
        # # node_weight = torch.where(aspect_mask==0, node_weight, torch.zeros_like(node_weight))
        # # node_weight = node_weight.squeeze(-1)
        #
        # node_weight = node_weight.squeeze(-1)
        # # print(node_weight.shape)
        # # print(torch.sum(node_weight, dim=1).expand(node_weight.shape[1], -1).T.shape)
        # node_weight = torch.div(node_weight, torch.sum(node_weight, dim=1).expand(node_weight.shape[1], -1).T)
        # # print(node_weight.shape)
        #
        # weight_norm = torch.sum(node_weight, dim=-1)
        # ''' sentence representation '''
        # sentence_feature = torch.sum(node_weight.unsqueeze(-1) * node_feature, dim=1)
        #
        # t = torch.sigmoid(self.linear(v))
        # t = 1
        # sentence_feature = (1 - t) * sentence_feature + t * v
        weight_norm = None
        # predicts = self.fc_out(self.fc_dropout(v))
        predicts = self.fc_out(v)
        return [predicts, weight_norm]


class CPT(nn.Module):

    def __init__(self, args, embed_dim):
        super(CPT, self).__init__()
        # self.lstm1 = WordEmbedLayer(args.embed_dim, args.hidden_dim, num_layers=args.num_layers,
        #                            batch_first=True, bidirectional=True, dropout=lstm_dropout, rnn_type=args.rnn_type)
        lstm_dropout = 0
        self.lstm1 = DynamicLSTM(embed_dim, args.hidden_dim, num_layers=1,
                                 batch_first=True, bidirectional=True, dropout=lstm_dropout, rnn_type=args.rnn_type)
        self.lstm2 = DynamicLSTM(embed_dim, args.hidden_dim, num_layers=1,
                                 batch_first=True, bidirectional=True, dropout=lstm_dropout, rnn_type=args.rnn_type)
        self.dropout = nn.Dropout(.3)
        self.linear1 = nn.ModuleList()
        self.linear2 = nn.ModuleList()
        for i in range(2):
            self.linear1.append(nn.Linear(4 * args.hidden_dim, 2 * args.hidden_dim))
            self.linear2.append(nn.Linear(2 * args.hidden_dim, 1))

        self.once = True

    def forward(self, word_feature, text_len, aspects, aspect_lens, masks, target_masks, position_weight):
        v, (_, _) = self.lstm1(word_feature, text_len.cpu())
        e, (_, _) = self.lstm2(aspects, aspect_lens.cpu())

        # if self.once:
        #     self.once = False
        #     torch.set_printoptions(profile="full")
        #     print(position_weight[:, :])
        #     print(masks[:])
        #     print(target_masks[:])
        #     torch.set_printoptions(profile="default")

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
        alpha = alpha.masked_fill(masks.eq(0).unsqueeze(2), -1e9)
        alpha = F.softmax(alpha, 1)
        z = torch.bmm(alpha.transpose(1, 2), v)

        return z.squeeze(1)
