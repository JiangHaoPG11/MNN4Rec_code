import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *
from sklearn.metrics import roc_auc_score

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
        self.new_attention = Additive_Attention(query_vector_dim, num_filters)
        self.norm4 = nn.LayerNorm(num_filters)
        self.dropout_prob = 0.2

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
        word_rep = self.cnn(word_embedding)
        word_rep = F.dropout(word_rep, p=self.dropout_prob, training=self.training)
        word_rep = self.norm3(word_rep)
        # 附加注意力
        new_rep = torch.cat([word_rep.unsqueeze(1), category_rep.unsqueeze(1), subcategory_rep.unsqueeze(1)], dim=1)
        new_rep = self.new_attention(new_rep)
        new_rep = F.dropout(new_rep, p=self.dropout_prob, training=self.training)
        new_rep = self.norm4(new_rep)
        return new_rep

class user_encoder(torch.nn.Module):
    def __init__(self, word_dim, title_word_size, category_dim, subcategory_dim,
                 dropout_prob, query_vector_dim, num_filters, window_sizes, category_size,subcategory_size):
        super(user_encoder, self).__init__()
        self.new_encoder = new_encoder(word_dim, title_word_size, category_dim, subcategory_dim,
                                       dropout_prob, query_vector_dim, num_filters, window_sizes, category_size, subcategory_size)
        self.user_attention = Additive_Attention(query_vector_dim, num_filters)
        self.norm2 = nn.LayerNorm(num_filters)
        self.dropout_prob = 0.2

    def forward(self, word_embedding, category_index, subcategory_index):
        new_rep = self.new_encoder(word_embedding, category_index, subcategory_index).unsqueeze(0)
        user_rep = self.user_attention(new_rep)
        user_rep = F.dropout(user_rep, p=self.dropout_prob, training=self.training)
        user_rep = self.norm2(user_rep)
        return user_rep

class NAML(torch.nn.Module):
    def __init__(self, args):
        super(NAML, self).__init__()
        self.word_dim = args.word_embedding_dim
        self.title_word_size = args.title_word_size
        self.sample_num = args.sample_num
        self.user_clicked_new_num = args.user_clicked_new_num
        self.category_dim = args.category_dim
        self.subcategory_dim = args.subcategory_dim
        self.category_size = args.category_size
        self.subcategory_size = args.subcategory_size
        self.num_filters = args.cnn_num_filters
        self.window_sizes = args.cnn_window_sizes
        self.query_vector_dim = args.query_vector_dim
        self.drop_prob = args.drop_prob

        self.new_encoder = new_encoder(self.word_dim, self.title_word_size, self.category_dim, self.subcategory_dim,
                                       self.drop_prob, self.query_vector_dim, self.num_filters, self.window_sizes, self.category_size, self.subcategory_size)
        self.user_encoder = user_encoder(self.word_dim, self.title_word_size, self.category_dim, self.subcategory_dim,
                                         self.drop_prob, self.query_vector_dim, self.num_filters, self.window_sizes, self.category_size, self.subcategory_size)

    def forward(self, candidate_new_word_embedding, user_clicked_new_word_embedding,
                candidate_new_category_index, user_clicked_new_category_index,
                candidate_new_subcategory_index, user_clicked_new_subcategory_index):
        candidate_new_word_embedding = candidate_new_word_embedding.to(device)
        user_clicked_new_word_embedding = user_clicked_new_word_embedding.to(device)
        candidate_new_category_index = candidate_new_category_index.to(device)
        user_clicked_new_category_index = user_clicked_new_category_index.to(device)
        candidate_new_subcategory_index = candidate_new_subcategory_index.to(device)
        user_clicked_new_subcategory_index = user_clicked_new_subcategory_index.to(device)
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
            # 用户表征
            user_rep_one = self.user_encoder(clicked_new_word_embedding_one, clicked_new_category_index, clicked_new_subcategory_index).unsqueeze(0)
            if i == 0:
                user_rep = user_rep_one
            else:
                user_rep = torch.cat([user_rep, user_rep_one], dim=0)
        sorce = torch.sum(new_rep * user_rep, dim=2)
        return sorce

    def loss(self, candidate_new_word_embedding,  user_clicked_word_embedding,
             candidate_new_category_index, user_clicked_new_category_index,
             candidate_new_subcategory_index, user_clicked_new_subcategory_index, label):
        label = label.to(device)
        sorce = self.forward(candidate_new_word_embedding, user_clicked_word_embedding,
                             candidate_new_category_index, user_clicked_new_category_index,
                             candidate_new_subcategory_index, user_clicked_new_subcategory_index)
        loss = torch.nn.functional.cross_entropy(sorce, torch.argmax(label, dim =1))
        try:
            auc = roc_auc_score(label.cpu().numpy(), F.softmax(sorce.cpu(), dim=1).detach().numpy())
        except ValueError:
            auc = 0.5
        return loss, auc
