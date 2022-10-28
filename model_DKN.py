import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *
from sklearn.metrics import roc_auc_score
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class new_encoder(torch.nn.Module):
    def __init__(self, batch_size, word_dim, entity_dim, title_word_size, num_filters, window_sizes, use_context):
        super(new_encoder, self).__init__()
        self.new_dim = num_filters * len(window_sizes)
        self.KCNN = KCNN( batch_size, title_word_size, word_dim, entity_dim, num_filters, window_sizes, use_context)
        self.norm1 = nn.LayerNorm(self.new_dim)
        self.dropout_prob = 0.2

    def forward(self, word_embedding, entity_embedding, context_embedding = None):
        new_rep = self.KCNN(word_embedding, entity_embedding, context_embedding)
        new_rep = F.dropout(new_rep, p=self.dropout_prob, training=self.training)
        new_rep = self.norm1(new_rep)
        return new_rep

class user_encoder(torch.nn.Module):
    def __init__(self, user_clicked_new_num, word_dim, entity_dim, title_word_size, num_filters, window_sizes, use_context):
        super(user_encoder, self).__init__()
        self.new_dim = num_filters * len(window_sizes)
        self.new_encoder = new_encoder(user_clicked_new_num, word_dim, entity_dim, title_word_size, num_filters, window_sizes, use_context)
        self.norm1 = nn.LayerNorm(self.new_dim)
        self.user_attention = dkn_attention(window_sizes, num_filters, user_clicked_new_num)
        self.norm2 = nn.LayerNorm(self.new_dim)
        self.dropout_prob = 0.2
    def forward(self, word_embedding, entity_embedding, candidate_new_rep, context_embedding=None):
        new_rep = self.new_encoder(word_embedding, entity_embedding, context_embedding)
        new_rep = F.dropout(new_rep, p=self.dropout_prob, training=self.training)
        new_rep = self.norm1(new_rep)
        user_rep = self.user_attention(candidate_new_rep, new_rep)
        user_rep = F.dropout(user_rep, p=self.dropout_prob, training=self.training)
        user_rep = self.norm2(user_rep)
        return user_rep

class DKN(torch.nn.Module):
    def __init__(self, args):
        super(DKN, self).__init__()
        self.batch_size = args.batch_size
        self.word_dim = args.word_embedding_dim
        self.entity_dim = args.entity_embedding_dim
        self.title_word_size = args.title_word_size
        self.sample_num = args.sample_num
        self.user_clicked_new_num = args.user_clicked_new_num
        self.num_filters = args.kcnn_num_filters
        self.window_sizes = args.kcnn_window_sizes
        self.use_context = False
        self.new_dim = self.num_filters * len(self.window_sizes)
        self.new_encoder = new_encoder(self.batch_size, self.word_dim, self.entity_dim, self.title_word_size, self.num_filters, self.window_sizes,  self.use_context)
        self.user_encoder = user_encoder(self.user_clicked_new_num, self.word_dim, self.entity_dim, self.title_word_size,
                                         self.num_filters, self.window_sizes,  self.use_context)
        self.dnn = nn.Sequential(nn.Linear(self.new_dim * 2, int(math.sqrt(self.new_dim))),
                                 nn.ReLU(),
                                 nn.Linear(int(math.sqrt(self.new_dim)), 1))

    def forward(self, candidate_new_word_embedding, user_clicked_new_word_embedding,
                candidate_new_entity_embedding, user_clicked_new_entity_embedding):
        candidate_new_word_embedding = candidate_new_word_embedding.to(device)
        user_clicked_new_word_embedding = user_clicked_new_word_embedding.to(device)
        candidate_new_entity_embedding = candidate_new_entity_embedding.to(device)
        user_clicked_new_entity_embedding = user_clicked_new_entity_embedding.to(device)
        ## 新闻编码器
        new_rep = None
        for i in range(self.sample_num):
            new_word_embedding_one = candidate_new_word_embedding[:, i, :]
            new_entity_embedding_one = candidate_new_entity_embedding[:, i, :]
            new_rep_one = self.new_encoder(new_word_embedding_one, new_entity_embedding_one)
            new_rep_one = new_rep_one.unsqueeze(1)
            if i==0:
                new_rep = new_rep_one
            else:
                new_rep = torch.cat([new_rep, new_rep_one], dim = 1)
        # 用户编码器
        user_rep = None
        for i in range(self.batch_size):
            clicked_word_embedding_one = user_clicked_new_word_embedding[i, :, :, :]
            clicked_word_embedding_one = clicked_word_embedding_one.squeeze()
            clicked_entity_embedding_one = user_clicked_new_entity_embedding[i, :, :, :]
            clicked_entity_embedding_one = clicked_entity_embedding_one.squeeze()
            candidate_new_rep = new_rep[i, :]
            user_rep_one = self.user_encoder(clicked_word_embedding_one, clicked_entity_embedding_one, candidate_new_rep)
            user_rep_one = user_rep_one.unsqueeze(0)
            if i == 0:
                user_rep = user_rep_one
            else:
                user_rep = torch.cat([user_rep, user_rep_one], dim=0)
        sorce = self.dnn(torch.cat([new_rep, user_rep], dim=2)).squeeze()
        return sorce

    def loss(self, candidate_new_word_embedding, user_clicked_word_embedding,
             candidate_new_entity_embedding, user_clicked_entity_embedding, label):
        label = label.to(device)
        sorce = self.forward(candidate_new_word_embedding, user_clicked_word_embedding, candidate_new_entity_embedding, user_clicked_entity_embedding)
        loss = torch.nn.functional.cross_entropy(sorce, torch.argmax(label, dim =1))
        try:
            auc = roc_auc_score(label.cpu().numpy(), F.softmax(sorce.cpu(), dim=1).detach().numpy())
        except ValueError:
            auc = 0.5
        return loss, auc
