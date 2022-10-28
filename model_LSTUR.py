import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *
from sklearn.metrics import roc_auc_score
from torch.nn.utils.rnn import pack_padded_sequence

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class new_encoder(torch.nn.Module):
    def __init__(self, word_dim, title_word_size, category_dim, subcategory_dim,
                 dropout_prob, query_vector_dim, num_filters, window_sizes, category_size, subcategory_size):
        super(new_encoder, self).__init__()
        self.embedding_layer1 = nn.Embedding(category_size, embedding_dim=category_dim)
        self.embedding_layer2 = nn.Embedding(subcategory_size, embedding_dim=subcategory_dim)
        self.fc1 = nn.Linear(category_dim, num_filters, bias=True)
        self.fc2 = nn.Linear(subcategory_dim, num_filters, bias=True)
        self.norm1 = nn.LayerNorm(num_filters)
        self.norm2 = nn.LayerNorm(num_filters)
        self.cnn = cnn(title_word_size, word_dim, dropout_prob, query_vector_dim, num_filters, window_sizes)
        self.norm3 = nn.LayerNorm(num_filters)
        self.dropout_prob = 0.3
    def forward(self, word_embedding, category_index, subcategory_index):
        # 主题表征
        category_embedding = self.embedding_layer1(category_index.to(torch.int64))
        category_rep = self.fc1(category_embedding)
        category_rep = F.dropout(category_rep, p=self.dropout_prob, training=self.training)
        category_rep = self.norm1(category_rep)
        # 副主题表征
        subcategory_embedding = self.embedding_layer2(subcategory_index.to(torch.int64))
        subcategory_rep = self.fc2(subcategory_embedding)
        subcategory_rep = F.dropout(subcategory_rep, p=self.dropout_prob, training=self.training)
        subcategory_rep = self.norm2(subcategory_rep)
        # 单词表征
        # word_embedding = F.dropout(word_embedding, p=self.dropout_prob, training=self.training)
        word_rep = self.cnn(word_embedding)
        word_rep = F.dropout(word_rep, p=self.dropout_prob, training=self.training)
        word_rep = self.norm3(word_rep)
        # 附加注意力
        new_rep = torch.cat([word_rep, category_rep, subcategory_rep], dim=1)
        return new_rep

class user_encoder(torch.nn.Module):
    def __init__(self, word_dim, title_word_size, category_dim, subcategory_dim,
                 dropout_prob, query_vector_dim, num_filters, window_sizes, category_size, subcategory_size,
                 user_size, long_short_term_method, masking_probability):
        super(user_encoder, self).__init__()
        self.masking_probability = masking_probability
        self.long_short_term_method = long_short_term_method
        self.user_embedding = nn.Embedding(user_size,
                                           num_filters * 3 if long_short_term_method == 'ini'
                                           else int(num_filters * 1.5),
                                           padding_idx=0)
        self.new_encoder = new_encoder(word_dim, title_word_size, category_dim, subcategory_dim,
                                       dropout_prob, query_vector_dim, num_filters, window_sizes, category_size, subcategory_size)
        self.gru = nn.GRU(num_filters * 3,
                          num_filters * 3 if long_short_term_method == 'ini'
                          else int(num_filters * 1.5),
                          num_layers = 1 , batch_first = True )
        self.norm2 = nn.LayerNorm(num_filters)
        self.dropout_prob = 0.3
    def forward(self, user_index, clicked_new_length, word_embedding, category_index, subcategory_index):
        # 新闻编码
        new_rep = self.new_encoder(word_embedding, category_index, subcategory_index)
        # 用户编码
        user_embedding = F.dropout2d(self.user_embedding(user_index.to(device)).unsqueeze(dim=0),
                                     p=self.masking_probability,
                                     training=self.training).squeeze(dim=0)

        if self.long_short_term_method == 'ini':
            packed_clicked_news_rep = pack_padded_sequence(new_rep.unsqueeze(0), lengths=clicked_new_length, batch_first=True, enforce_sorted=False)
            _, last_hidden = self.gru(packed_clicked_news_rep, user_embedding.unsqueeze(0))
            return last_hidden.squeeze(dim=0)
        else:
            packed_clicked_news_rep = pack_padded_sequence(new_rep, clicked_new_length, batch_first=True, enforce_sorted=False)
            _, last_hidden = self.gru(packed_clicked_news_rep)
            return torch.cat((last_hidden.squeeze(dim=0), user_embedding), dim=1)

class LSTUR(torch.nn.Module):
    def __init__(self, args):
        super(LSTUR, self).__init__()
        self.word_dim = args.word_embedding_dim
        self.title_word_size = args.title_word_size
        self.sample_num = args.sample_num
        self.user_clicked_new_num = args.user_clicked_new_num
        self.category_dim = args.category_dim
        self.subcategory_dim = args.subcategory_dim
        self.category_size = args.category_size
        self.subcategory_size = args.subcategory_size
        self.num_filters = args.lstur_num_filters
        self.window_sizes = args.lstur_window_sizes
        self.query_vector_dim = args.query_vector_dim
        self.drop_prob = args.drop_prob
        self.user_size = args.user_size
        self.long_short_term_method = args.long_short_term_method
        self.masking_probability = args.masking_probability

        self.new_encoder = new_encoder(self.word_dim, self.title_word_size, self.category_dim, self.subcategory_dim,
                                       self.drop_prob, self.query_vector_dim, self.num_filters, self.window_sizes, self.category_size, self.subcategory_size)
        self.user_encoder = user_encoder(self.word_dim, self.title_word_size, self.category_dim, self.subcategory_dim,
                                         self.drop_prob, self.query_vector_dim, self.num_filters, self.window_sizes, self.category_size, self.subcategory_size,
                                         self.user_size,  self.long_short_term_method, self.masking_probability)

    def forward(self,  candidate_new_word_embedding, user_clicked_new_word_embedding,
                candidate_new_category_index, user_clicked_new_category_index,
                candidate_new_subcategory_index, user_clicked_new_subcategory_index,
                user_clicked_new_length, user_index):
        user_index = user_index.to(device)
        candidate_new_word_embedding = candidate_new_word_embedding.to(device)
        user_clicked_new_word_embedding = user_clicked_new_word_embedding.to(device)
        candidate_new_category_index = candidate_new_category_index.to(device)
        user_clicked_new_category_index = user_clicked_new_category_index.to(device)
        candidate_new_subcategory_index = candidate_new_subcategory_index.to(device)
        user_clicked_new_subcategory_index = user_clicked_new_subcategory_index.to(device)
        user_clicked_new_length = user_clicked_new_length.to(device)
        # 新闻编码器
        new_rep = None
        for i in range(self.sample_num):
            new_word_embedding_one = candidate_new_word_embedding[:, i, :]
            new_category_index = candidate_new_category_index[:, i]
            new_subcategory_index = candidate_new_subcategory_index[:, i]
            new_rep_one = self.new_encoder(new_word_embedding_one, new_category_index, new_subcategory_index).unsqueeze(1)
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
            # 点击新闻主题index
            clicked_new_category_index = user_clicked_new_category_index[i, :]
            # 点击新闻副主题index
            clicked_new_subcategory_index = user_clicked_new_subcategory_index[i, :]
            # 点击新闻长度
            clicked_new_length = user_clicked_new_length[i]
            # 用户嵌入
            user_index_one = user_index[i]
            # 用户表征
            user_rep_one = self.user_encoder(user_index_one, clicked_new_length, clicked_new_word_embedding_one, clicked_new_category_index, clicked_new_subcategory_index).unsqueeze(0)
            if i == 0:
                user_rep = user_rep_one
            else:
                user_rep = torch.cat([user_rep, user_rep_one], dim=0)
        sorce = torch.sum(new_rep * user_rep, dim=2)
        return sorce

    def loss(self, candidate_new_word_embedding, user_clicked_word_embedding,
             candidate_new_category_index, user_clicked_new_category_index,
             candidate_new_subcategory_index, user_clicked_new_subcategory_index,
             user_clicked_new_length, user_index, label):
        label = label.to(device)
        sorce = self.forward(candidate_new_word_embedding, user_clicked_word_embedding,
                             candidate_new_category_index, user_clicked_new_category_index,
                             candidate_new_subcategory_index, user_clicked_new_subcategory_index,
                             user_clicked_new_length, user_index)
        loss = torch.nn.functional.cross_entropy(sorce, torch.argmax(label, dim =1))
        try:
            auc = roc_auc_score(label.cpu().numpy(), F.softmax(sorce.cpu(), dim=1).detach().numpy())
        except ValueError:
            auc = 0.5
        return loss, auc
