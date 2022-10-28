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

class user_shortterm_encoder(torch.nn.Module):
    def __init__(self, user_clicked_new_num, word_dim, entity_dim, title_word_size, entity_cate_size, sample_num, max_two_hop):
        super(user_shortterm_encoder, self).__init__()
        self.title_dim = 160
        self.hidden_size = 100
        self.max_two_hop = max_two_hop
        self.new_encoder = new_encoder( word_dim, entity_dim, title_word_size, entity_cate_size )
        self.norm1 = nn.LayerNorm(self.title_dim)
        self.fc = nn.Linear(self.title_dim, self.title_dim, bias = True)
        self.norm2 = nn.LayerNorm(self.title_dim)
        self.dropout_prob = 0.2
    def forward(self, neighbor_user_embedding, word_embedding, entity_embedding, entity_cate, category_embedding, new_rep):
        two_new_rep = self.new_encoder(torch.flatten(word_embedding,0,1), torch.flatten(entity_embedding,0,1), torch.flatten(entity_cate,0,1))
        two_new_rep = two_new_rep.view(50, self.max_two_hop, -1)
        two_new_rep = F.dropout(two_new_rep, p=self.dropout_prob, training=self.training)
        two_new_rep = self.norm1(two_new_rep)
        user_shortterm_rep = torch.sum(neighbor_user_embedding, dim=1) * (1 / neighbor_user_embedding.shape[1])
        new_rep = torch.cat([two_new_rep, neighbor_user_embedding, category_embedding.unsqueeze(1), new_rep.unsqueeze(1)], dim = 1)
        total_num = neighbor_user_embedding.shape[1] + two_new_rep.shape[1] + 2
        new_rep = torch.sum(new_rep, dim = 1) * (1 / total_num)
        new_rep = self.fc(new_rep)
        new_rep = self.norm2(new_rep)
        return user_shortterm_rep, new_rep

class user_longterm_encoder(torch.nn.Module):
    def __init__(self, user_clicked_new_num, word_dim, entity_dim, title_word_size, entity_cate_size, sample_num):
        super(user_longterm_encoder, self).__init__()
        self.title_dim = 160
        self.hidden_size = 100
        self.new_encoder = new_encoder( word_dim, entity_dim, title_word_size, entity_cate_size )
        self.norm1 = nn.LayerNorm(self.title_dim)
        self.ANN = ANN(user_clicked_new_num, sample_num, self.title_dim, self.hidden_size, mode='interest')
        self.ARNN = ARNN(user_clicked_new_num, word_dim, self.title_dim, sample_num)
        self.norm2 = nn.LayerNorm(self.title_dim)
        self.fc = nn.Linear(2*self.title_dim, self.title_dim, bias=True)
        self.dropout_prob = 0.2
    def forward(self, word_embedding, entity_embedding, entity_cate, candidate_new_rep):
        new_rep = self.new_encoder(word_embedding, entity_embedding, entity_cate)
        new_rep = F.dropout(new_rep, p=self.dropout_prob, training=self.training)
        new_rep = self.norm1(new_rep)
        user_rep_interest = self.ANN(candidate_new_rep, new_rep)
        user_rep_hist = self.ARNN(user_rep_interest, new_rep)
        user_longterm_rep = self.fc(torch.cat([user_rep_hist, user_rep_interest], dim = -1))
        user_longterm_rep = F.dropout(user_longterm_rep, p=self.dropout_prob, training=self.training)
        user_longterm_rep = self.norm2 (user_longterm_rep)
        return user_longterm_rep

class GNewsRec(torch.nn.Module):
    def __init__(self, args):
        super(GNewsRec, self).__init__()
        self.batch_size = args.batch_size
        self.word_dim = args.word_embedding_dim
        self.entity_dim = args.entity_embedding_dim
        self.title_word_size = args.title_word_size
        self.sample_num = args.sample_num
        self.user_clicked_new_num = args.user_clicked_new_num
        self.num_filters = args.kcnn_num_filters
        self.entity_cate_size = args.entity_cate_size
        self.max_two_hop = args.max_two_hop
        self.user_num = args.user_size
        self.category_size = args.category_size
        self.title_dim = 160
        self.user_embedding = nn.Embedding(self.user_num+1, self.title_dim)
        self.category_embedding = nn.Embedding(self.category_size, self.title_dim)
        self.new_encoder = new_encoder( self.word_dim, self.entity_dim, self.title_word_size, self.entity_cate_size)
        self.user_shortterm_encoder = user_shortterm_encoder(self.user_clicked_new_num, self.word_dim, self.entity_dim,
                                                             self.title_word_size, self.entity_cate_size, self.sample_num,
                                                             self.max_two_hop)
        self.user_longterm_encoder = user_longterm_encoder(self.user_clicked_new_num, self.word_dim, self.entity_dim,
                                                           self.title_word_size, self.entity_cate_size,self.sample_num)


        self.user_final_fc = nn.Linear(self.title_dim * 2, self.title_dim, bias=True)
        self.predictor = nn.Sequential(nn.Linear(self.title_dim * 2, self.title_dim),
                                       nn.ReLU(),
                                       nn.Linear(self.title_dim, 1))


    def forward(self, candidate_new_word_embedding, user_clicked_new_word_embedding,
                candidate_new_entity_embedding, user_clicked_new_entity_embedding,
                candidate_new_entity_cate, user_clicked_new_entity_cate,
                new_one_hop_neighbor, new_two_hop_neighbor_new_word_embedding,
                new_two_hop_neighbor_new_entity_embedding, new_two_hop_neighbor_new_entity_cate,
                candidate_new_category_index):
        candidate_new_word_embedding = candidate_new_word_embedding.to(device)
        user_clicked_new_word_embedding = user_clicked_new_word_embedding.to(device)
        candidate_new_entity_embedding = candidate_new_entity_embedding.to(device)
        user_clicked_new_entity_embedding = user_clicked_new_entity_embedding.to(device)
        candidate_new_entity_cate = candidate_new_entity_cate.to(device)
        user_clicked_new_entity_cate = user_clicked_new_entity_cate.to(device)
        new_one_hop_neighbor = new_one_hop_neighbor.to(device) # bz, sample_num, max_one_hop
        new_two_hop_neighbor_new_word_embedding = new_two_hop_neighbor_new_word_embedding.to(device) # bz, sample_num, max_two_hop, max_seq, word_dim
        new_two_hop_neighbor_new_entity_embedding = new_two_hop_neighbor_new_entity_embedding.to(device) # bz, sample_num, max_two_hop, max_seq, enntiy_dim
        new_two_hop_neighbor_new_entity_cate = new_two_hop_neighbor_new_entity_cate.to(device) # bz, sample_num, max_two_hop, max_seq,
        candidate_new_category_index = candidate_new_category_index.to(device) # bz, sample_num,

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

        # 用户短期编码器
        update_new_rep = None
        user_shortterm_rep = None
        new_one_hop_user_embedding = self.user_embedding(new_one_hop_neighbor.to(torch.int64))
        new_category_embedding = self.category_embedding(candidate_new_category_index.to(torch.int64))
        for i in range(self.sample_num):
            new_one_hop_user_embedding_one = new_one_hop_user_embedding[:, i, :, :]
            new_two_hop_neighbor_new_word_embedding_one = new_two_hop_neighbor_new_word_embedding[:, i, :, :, :]
            new_two_hop_neighbor_new_entity_embedding_one = new_two_hop_neighbor_new_entity_embedding[:, i, :, :, :]
            new_two_hop_neighbor_new_entity_cate_one = new_two_hop_neighbor_new_entity_cate[:, i, :, :]
            new_category_embedding_one = new_category_embedding[:, i, :]
            candidate_new_rep = new_rep[:, i, :]
            user_shortterm_rep_one, update_new_rep_one = self.user_shortterm_encoder(new_one_hop_user_embedding_one,
                                                                                     new_two_hop_neighbor_new_word_embedding_one,
                                                                                     new_two_hop_neighbor_new_entity_embedding_one,
                                                                                     new_two_hop_neighbor_new_entity_cate_one,
                                                                                     new_category_embedding_one,
                                                                                     candidate_new_rep)
            user_shortterm_rep_one = user_shortterm_rep_one.unsqueeze(1)
            update_new_rep_one = update_new_rep_one.unsqueeze(1)
            if i == 0:
                user_shortterm_rep = user_shortterm_rep_one
                update_new_rep = update_new_rep_one
            else:
                user_shortterm_rep = torch.cat([user_shortterm_rep, user_shortterm_rep_one], dim=1)
                update_new_rep = torch.cat([update_new_rep, update_new_rep_one], dim=1)

        # 用户长期编码器
        user_longterm_rep = None
        for i in range(self.batch_size):
            clicked_word_embedding_one = user_clicked_new_word_embedding[i, :, :, :]
            clicked_word_embedding_one = clicked_word_embedding_one.squeeze()
            clicked_entity_embedding_one = user_clicked_new_entity_embedding[i, :, :, :]
            clicked_entity_embedding_one = clicked_entity_embedding_one.squeeze()
            user_clicked_new_entity_cate_one = user_clicked_new_entity_cate[i, :, :]
            user_clicked_new_entity_cate_one = user_clicked_new_entity_cate_one.squeeze()
            candidate_new_rep = new_rep[i, :]
            user_longterm_rep_one = self.user_longterm_encoder(clicked_word_embedding_one, clicked_entity_embedding_one,
                                                               user_clicked_new_entity_cate_one, candidate_new_rep)
            if i == 0:
                user_longterm_rep = user_longterm_rep_one
            else:
                user_longterm_rep = torch.cat([user_longterm_rep, user_longterm_rep_one], dim=0)
        user_rep = torch.cat([user_shortterm_rep, user_longterm_rep], dim = -1)
        user_rep = self.user_final_fc(user_rep)
        score = self.predictor(torch.cat([user_rep, new_rep], dim = -1)).squeeze()
        return score

    def loss(self, candidate_new_word_embedding, user_clicked_word_embedding,
             candidate_new_entity_embedding, user_clicked_entity_embedding,
             candidate_new_entity_cate, user_clicked_new_entity_cate,
             new_one_hop_neighbor, new_two_hop_neighbor_new_word_embedding,
             new_two_hop_neighbor_new_entity_embedding, new_two_hop_neighbor_new_entity_cate,
             candidate_new_category_index, label):
        label = label.to(device)
        sorce = self.forward(candidate_new_word_embedding, user_clicked_word_embedding, candidate_new_entity_embedding,
                             user_clicked_entity_embedding, candidate_new_entity_cate, user_clicked_new_entity_cate,
                             new_one_hop_neighbor, new_two_hop_neighbor_new_word_embedding,
                             new_two_hop_neighbor_new_entity_embedding, new_two_hop_neighbor_new_entity_cate,
                             candidate_new_category_index )
        loss = torch.nn.functional.cross_entropy(sorce, torch.argmax(label, dim =1))
        try:
            auc = roc_auc_score(label.cpu().numpy(), F.softmax(sorce.cpu(), dim=1).detach().numpy())
        except ValueError:
            auc = 0.5
        return loss, auc
