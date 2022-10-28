import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *
from sklearn.metrics import roc_auc_score
from torch.nn.utils.rnn import pack_padded_sequence

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class new_encoder(torch.nn.Module):
    def __init__(self, word_dim, title_word_size, dropout_prob, query_vector_dim, num_filters, window_sizes):
        super(new_encoder, self).__init__()
        self.cnn = cnn(title_word_size, word_dim, dropout_prob, query_vector_dim, num_filters, window_sizes)
        self.norm1 = nn.LayerNorm(num_filters)
        self.dropout_prob = 0.2
    def forward(self, word_embedding):
        word_rep = self.cnn(word_embedding)
        word_rep = F.dropout(word_rep, p=self.dropout_prob, training=self.training)
        new_rep = self.norm1(word_rep)
        return new_rep

class user_encoder(torch.nn.Module):
    def __init__(self, word_dim, title_word_size,  dropout_prob, query_vector_dim, num_filters, window_sizes,  masking_probability):
        super(user_encoder, self).__init__()
        self.masking_probability = masking_probability
        self.new_encoder = new_encoder(word_dim, title_word_size,  dropout_prob, query_vector_dim, num_filters, window_sizes)
        self.gru = nn.GRU(num_filters , num_filters , num_layers = 1 , batch_first = True )
        self.norm2 = nn.LayerNorm(num_filters)
        self.dropout_prob = 0.2
    def forward(self, clicked_new_length, word_embedding):
        new_rep = self.new_encoder(word_embedding)
        packed_clicked_news_rep = pack_padded_sequence(new_rep.unsqueeze(0), lengths=clicked_new_length, batch_first=True, enforce_sorted=False)
        _, last_hidden = self.gru(packed_clicked_news_rep)
        return last_hidden.squeeze(dim=0)

class GRU(torch.nn.Module):
    def __init__(self, args):
        super(GRU, self).__init__()
        self.word_dim = args.word_embedding_dim
        self.title_word_size = args.title_word_size
        self.sample_num = args.sample_num
        self.user_clicked_new_num = args.user_clicked_new_num
        self.num_filters = args.lstur_num_filters
        self.window_sizes = args.lstur_window_sizes
        self.query_vector_dim = args.query_vector_dim
        self.drop_prob = args.drop_prob
        self.long_short_term_method = args.long_short_term_method
        self.masking_probability = args.masking_probability

        self.new_encoder = new_encoder(self.word_dim, self.title_word_size,  self.drop_prob, self.query_vector_dim, self.num_filters, self.window_sizes)
        self.user_encoder = user_encoder(self.word_dim, self.title_word_size, self.drop_prob, self.query_vector_dim, self.num_filters, self.window_sizes, self.masking_probability)

    def forward(self, candidate_new_word_embedding, user_clicked_new_word_embedding, user_clicked_new_length):
        candidate_new_word_embedding = candidate_new_word_embedding.to(device)
        user_clicked_new_word_embedding = user_clicked_new_word_embedding.to(device)
        user_clicked_new_length = user_clicked_new_length.to(device)
        # 新闻编码器
        new_rep = None
        for i in range(self.sample_num):
            new_word_embedding_one = candidate_new_word_embedding[:, i, :]
            new_rep_one = self.new_encoder(new_word_embedding_one).unsqueeze(1)
            if i == 0:
                new_rep = new_rep_one
            else:
                new_rep = torch.cat([new_rep, new_rep_one], dim = 1)
        # 用户编码器
        user_rep = None
        for i in range(self.user_clicked_new_num):
            # 点击新闻单词嵌入
            clicked_new_word_embedding_one = user_clicked_new_word_embedding[i, :, :, :]
            clicked_new_word_embedding_one = clicked_new_word_embedding_one.squeeze()
            # 点击新闻长度
            clicked_new_length = user_clicked_new_length[i]
            # 用户表征
            user_rep_one = self.user_encoder(clicked_new_length, clicked_new_word_embedding_one).unsqueeze(0)
            if i == 0:
                user_rep = user_rep_one
            else:
                user_rep = torch.cat([user_rep, user_rep_one], dim=0)
        sorce = torch.sum(new_rep * user_rep, dim=2)
        return sorce

    def loss(self, candidate_new_word_embedding, user_clicked_word_embedding, user_clicked_new_length, label):
        label = label.to(device)
        sorce = self.forward(candidate_new_word_embedding, user_clicked_word_embedding, user_clicked_new_length)
        loss = torch.nn.functional.cross_entropy(sorce, torch.argmax(label, dim =1))
        try:
            auc = roc_auc_score(label.cpu().numpy(), F.softmax(sorce.cpu(), dim=1).detach().numpy())
        except ValueError:
            auc = 0.5
        return loss, auc
