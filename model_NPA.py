import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *
from sklearn.metrics import roc_auc_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class new_encoder(torch.nn.Module):
    def __init__(self, word_dim, title_word_size, dropout_prob, num_filters, window_sizes, query_vector_dim):
        super(new_encoder, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=num_filters,
                              kernel_size=(window_sizes, word_dim),
                              padding=(int((window_sizes - 1) / 2), 0))
        self.norm1 = nn.LayerNorm(num_filters)
        # self.new_attention = SimilarityAttention(query_vector_dim, num_filters)
        self.new_attention = QueryAttention(query_vector_dim, num_filters)
        self.norm2 = nn.LayerNorm(num_filters)
        self.dropout_prob = 0.2
    def forward(self, word_embedding, user_embedding):
        # 单词表征
        word_embedding = self.conv(word_embedding.unsqueeze(dim=1)).squeeze(dim=3)
        word_embedding = F.dropout(F.relu(word_embedding), p=self.dropout_prob, training=self.training)
        word_embedding = self.norm1(word_embedding.transpose(2,1))
        # 附加注意力
        new_rep = self.new_attention(user_embedding, word_embedding)
        #new_rep = F.dropout(new_rep, p=self.dropout_prob, training=self.training)
        new_rep = self.norm2(new_rep)
        return new_rep

class user_encoder(torch.nn.Module):
    def __init__(self, word_dim,  title_word_size, dropout_prob, num_filters, window_sizes, query_vector_dim):
        super(user_encoder, self).__init__()
        self.new_encoder = new_encoder(word_dim, title_word_size, dropout_prob, num_filters, window_sizes, query_vector_dim)
        self.norm1 = nn.LayerNorm(num_filters)
        self.user_attention = QueryAttention(query_vector_dim, num_filters)
        #self.user_attention = SimilarityAttention(query_vector_dim, num_filters)
        self.norm2 = nn.LayerNorm(num_filters)
        self.dropout_prob = 0.2

    def forward(self, word_embedding, user_embedding, user_embeding_two):
        # 新闻编码
        new_rep = self.new_encoder(word_embedding, user_embedding.unsqueeze(1).repeat(50,1,1))
        #new_rep = F.dropout(new_rep, p=self.dropout_prob, training=self.training)
        new_rep = self.norm1(new_rep)
        # 用户编码
        user_rep = self.user_attention(user_embeding_two.unsqueeze(0), new_rep.unsqueeze(0))
        #user_rep = F.dropout(user_rep, p=self.dropout_prob, training=self.training)
        user_rep = self.norm2(user_rep)
        return user_rep

class NPA(torch.nn.Module):
    def __init__(self, args):
        super(NPA, self).__init__()
        self.word_dim = args.word_embedding_dim
        self.title_word_size = args.title_word_size
        self.sample_num = args.sample_num
        self.user_clicked_new_num = args.user_clicked_new_num
        self.num_filters = args.cnn_num_filters
        self.window_sizes = args.cnn_window_sizes
        self.query_vector_dim = args.query_vector_dim
        self.drop_prob = args.drop_prob
        self.user_size = args.user_size
        self.user_embedding_size = 50

        self.user_embedding = nn.Embedding(args.user_size, self.user_embedding_size)
        self.fc1 = nn.Linear(self.user_embedding_size, self.query_vector_dim)
        self.norm1 = nn.LayerNorm(self.query_vector_dim)
        self.fc2 = nn.Linear(self.user_embedding_size, self.query_vector_dim)
        self.norm2 = nn.LayerNorm(self.query_vector_dim)

        self.new_encoder = new_encoder(self.word_dim, self.title_word_size, self.drop_prob, self.num_filters, self.window_sizes, self.query_vector_dim)
        self.user_encoder = user_encoder(self.word_dim, self.title_word_size, self.drop_prob,  self.num_filters, self.window_sizes, self.query_vector_dim)

    def forward(self,  candidate_new_word_embedding, user_clicked_new_word_embedding, user_index):
        user_index = user_index.to(device)
        candidate_new_word_embedding = candidate_new_word_embedding.to(device)
        user_clicked_new_word_embedding = user_clicked_new_word_embedding.to(device)
        # 获取用户嵌入
        user_embedding = self.user_embedding(user_index)
        user_vector = self.fc1(user_embedding)
        user_vector = self.norm1(user_vector)
        user_vector_2 = self.fc2(user_embedding)
        user_vector_2 = self.norm2(user_vector_2)
        # 新闻编码器
        new_rep = None
        for i in range(self.sample_num):
            new_word_embedding_one = candidate_new_word_embedding[:, i, :]
            new_rep_one = self.new_encoder(new_word_embedding_one, user_vector).unsqueeze(1)
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
            # 用户嵌入
            user_vector_one = user_vector[i, :]
            user_vector_one_2 = user_vector_2[i, :]
            # 用户表征
            user_rep_one = self.user_encoder(clicked_new_word_embedding_one, user_vector_one, user_vector_one_2).unsqueeze(0)
            if i == 0:
                user_rep = user_rep_one
            else:
                user_rep = torch.cat([user_rep, user_rep_one], dim=0)
        sorce = torch.sum(new_rep * user_rep, dim=2)
        return sorce
    def loss(self, candidate_new_word_embedding, user_clicked_word_embedding, user_index, label):
        label = label.to(device)
        sorce = self.forward(candidate_new_word_embedding, user_clicked_word_embedding, user_index)
        loss = torch.nn.functional.cross_entropy(sorce, torch.argmax(label, dim =1))
        try:
            auc = roc_auc_score(label.cpu().numpy(), F.softmax(sorce.cpu(), dim=1).detach().numpy())
        except ValueError:
            auc = 0.5
        return loss, auc
