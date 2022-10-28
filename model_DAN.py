import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *
from sklearn.metrics import roc_auc_score
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class new_encoder(torch.nn.Module):
    def __init__(self, word_dim, entity_dim, title_word_size, entity_cate_size):
        super(new_encoder, self).__init__()
        self.PCNN = PCNN(title_word_size, word_dim, entity_dim, entity_cate_size)
        self.norm1 = nn.LayerNorm(160)
        self.dropout_prob = 0.2

    def forward(self, word_embedding, entity_embedding, new_entity_cate_one):
        new_rep = self.PCNN(word_embedding, entity_embedding, new_entity_cate_one)
        new_rep = F.dropout(new_rep, p=self.dropout_prob, training=self.training)
        new_rep = self.norm1(new_rep)
        return new_rep

class user_encoder(torch.nn.Module):
    def __init__(self, user_clicked_new_num, word_dim, entity_dim, title_word_size, entity_cate_size, sample_num):
        super(user_encoder, self).__init__()
        self.title_dim = 160
        self.hidden_size = 100
        self.new_encoder = new_encoder( word_dim, entity_dim, title_word_size, entity_cate_size )
        self.norm1 = nn.LayerNorm(self.title_dim)
        self.ANN = ANN(user_clicked_new_num, sample_num, self.title_dim, self.hidden_size, mode='interest')
        self.ARNN = ARNN(user_clicked_new_num, word_dim, self.title_dim, sample_num)
        self.norm2 = nn.LayerNorm(self.title_dim)
        self.dropout_prob = 0.2
    def forward(self, word_embedding, entity_embedding, entity_cate, candidate_new_rep):
        new_rep = self.new_encoder(word_embedding, entity_embedding, entity_cate)
        new_rep = F.dropout(new_rep, p=self.dropout_prob, training=self.training)
        new_rep = self.norm1(new_rep)
        user_rep_interest = self.ANN(candidate_new_rep, new_rep)
        user_rep_hist = self.ARNN(user_rep_interest,new_rep)
        user_rep = F.dropout(user_rep_hist, p=self.dropout_prob, training=self.training)
        user_rep = self.norm2(user_rep)
        return user_rep

class DAN(torch.nn.Module):
    def __init__(self, args):
        super(DAN, self).__init__()
        self.batch_size = args.batch_size
        self.word_dim = args.word_embedding_dim
        self.entity_dim = args.entity_embedding_dim
        self.title_word_size = args.title_word_size
        self.sample_num = args.sample_num
        self.user_clicked_new_num = args.user_clicked_new_num
        self.num_filters = args.kcnn_num_filters
        self.entity_cate_size = args.entity_cate_size

        self.new_encoder = new_encoder( self.word_dim, self.entity_dim, self.title_word_size, self.entity_cate_size)
        self.user_encoder = user_encoder(self.user_clicked_new_num, self.word_dim, self.entity_dim, self.title_word_size, self.entity_cate_size, self.sample_num)
        # self.dnn = nn.Sequential(nn.Linear(self.new_dim * 2, int(math.sqrt(self.new_dim))),
        #                          nn.ReLU(),
        #                          nn.Linear(int(math.sqrt(self.new_dim)), 1))

    def forward(self, candidate_new_word_embedding, user_clicked_new_word_embedding,
                candidate_new_entity_embedding, user_clicked_new_entity_embedding,
                candidate_new_entity_cate, user_clicked_new_entity_cate):
        candidate_new_word_embedding = candidate_new_word_embedding.to(device)
        user_clicked_new_word_embedding = user_clicked_new_word_embedding.to(device)
        candidate_new_entity_embedding = candidate_new_entity_embedding.to(device)
        user_clicked_new_entity_embedding = user_clicked_new_entity_embedding.to(device)
        candidate_new_entity_cate = candidate_new_entity_cate.to(device)
        user_clicked_new_entity_cate = user_clicked_new_entity_cate.to(device)

        ## 新闻编码器
        new_rep = None
        for i in range(self.sample_num):
            new_word_embedding_one = candidate_new_word_embedding[:, i, :]
            new_entity_embedding_one = candidate_new_entity_embedding[:, i, :]
            new_entity_cate_one = candidate_new_entity_cate[:, i]
            new_rep_one = self.new_encoder(new_word_embedding_one, new_entity_embedding_one, new_entity_cate_one)
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
            user_clicked_new_entity_cate_one = user_clicked_new_entity_cate[i, :, :]
            user_clicked_new_entity_cate_one = user_clicked_new_entity_cate_one.squeeze()
            candidate_new_rep = new_rep[i, :]
            user_rep_one = self.user_encoder(clicked_word_embedding_one, clicked_entity_embedding_one, user_clicked_new_entity_cate_one, candidate_new_rep)
            if i == 0:
                user_rep = user_rep_one
            else:
                user_rep = torch.cat([user_rep, user_rep_one], dim=0)
            print(new_rep)
            print(user_rep)
        sorce = torch.cosine_similarity(new_rep, user_rep, dim = -1)
        return sorce

    def loss(self, candidate_new_word_embedding, user_clicked_word_embedding,
             candidate_new_entity_embedding, user_clicked_entity_embedding,
             candidate_new_entity_cate, user_clicked_new_entity_cate, label):
        label = label.to(device)
        sorce = self.forward(candidate_new_word_embedding, user_clicked_word_embedding, candidate_new_entity_embedding,
                             user_clicked_entity_embedding, candidate_new_entity_cate, user_clicked_new_entity_cate)
        loss = torch.nn.functional.cross_entropy(sorce, torch.argmax(label, dim =1))
        try:
            auc = roc_auc_score(label.cpu().numpy(), F.softmax(sorce.cpu(), dim=1).detach().numpy())
        except ValueError:
            auc = 0.5
        return loss, auc
