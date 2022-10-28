import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *
from sklearn.metrics import roc_auc_score
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class neighborhood_routing_algorithm(torch.nn.Module):
    def __init__(self, mode):
        super(neighborhood_routing_algorithm, self).__init__()
        self.iterations = 2
        self.norm1 = nn.LayerNorm(10)
        self.mode = mode
    def forward(self, s_stack, wrt_s_stack): # 50, k, 10  # k, 10
        if self.mode == 'user':
            z = 0
            for i in range(self.iterations):
                weight = None
                for j in range(s_stack.size(0)):
                    s_stack_one = s_stack[i,:,:].squeeze()  # k, 10
                    weight_one = F.softmax(torch.matmul(wrt_s_stack, s_stack_one.transpose(0,1)),dim = -1).unsqueeze(0)
                    if j == 0:
                        weight = weight_one
                    else:
                        weight = torch.cat([weight, weight_one], dim = 0)   # 50, k, k
                temp = torch.sum(torch.matmul(weight, s_stack), dim = 0)
                z = wrt_s_stack + temp
                z = self.norm1(z)
            return z
        if self.mode == 'news':
            z = 0
            for i in range(self.iterations):
                weight = None
                for j in range(s_stack.size(1)):
                    s_stack_one = s_stack[:, i, :, :].squeeze()  # 50, k, 10
                    weight_one = F.softmax(torch.matmul(wrt_s_stack, s_stack_one.transpose(1, 2)), dim=-1).unsqueeze(1)
                    if j == 0:
                        weight = weight_one
                    else:
                        weight = torch.cat([weight, weight_one], dim=1)  # 50, 50, k, k
                temp = torch.sum(torch.matmul(weight, s_stack), dim=1)
                z = wrt_s_stack + temp
                z = self.norm1(z)
            return z

class user_encoder(torch.nn.Module):
    def __init__(self, user_clicked_new_num, word_dim, entity_dim, title_word_size, entity_cate_size, sample_num, user_size):
        super(user_encoder, self).__init__()
        self.sample_num = sample_num
        self.title_dim = 160
        self.num_filters = 8
        self.user_size = user_size
        self.PCNN = PCNN(title_word_size, word_dim, entity_dim, entity_cate_size)
        self.user_id_embedding = nn.Embedding(self.user_size, self.title_dim)
        self.norm1 = nn.LayerNorm(self.title_dim)
        self.user_conv_filters = nn.ModuleDict(
            {
                str(x): nn.Conv2d(1, 10, kernel_size=(user_clicked_new_num+1, self.title_dim), stride=(2, 2))
                for x in range(self.num_filters)
            })
        self.news_conv_filters = nn.ModuleDict(
            {
                str(x): nn.Conv2d(1, 10, kernel_size=(1, self.title_dim), stride=(2, 2))
                for x in range(self.num_filters)
            })
        self.norm2 = nn.LayerNorm(10)
        self.norm3 = nn.LayerNorm(10)
        self.dropout_prob = 0.2
        self.user_neighborhood_routing = neighborhood_routing_algorithm(mode = 'user')

    def forward(self, word_embedding, entity_embedding, entity_cate, user_index):
        new_rep = self.PCNN(word_embedding, entity_embedding, entity_cate)
        new_rep = F.dropout(new_rep, p=self.dropout_prob, training=self.training)
        new_rep = self.norm1(new_rep) # max_clicked_num, title_dim
        user_id_embedding = self.user_id_embedding(user_index.to(torch.int64))
        user_id_embedding = F.dropout(user_id_embedding, p=self.dropout_prob, training=self.training) # 1, title_dim
        h_stack = torch.cat([user_id_embedding, new_rep], dim = 0) # max_clicked_num+1 , title_dim
        h_stack = h_stack.unsqueeze(0).unsqueeze(0) # 1, 1, max_clicked_num+1 , title_dim
        # 用户池化&卷积
        user_pooled_embedding = []
        for x in range(self.num_filters):
            convoluted = self.user_conv_filters[str(x)](h_stack).squeeze(-1)
            activated = F.relu(convoluted)
            pooled = activated.max(dim=-1)[0]
            user_pooled_embedding.append(pooled.unsqueeze(1))
        user_s_stack = torch.cat(user_pooled_embedding, dim=1)
        user_s_stack = F.dropout(user_s_stack, p=self.dropout_prob, training=self.training)
        user_s_stack = self.norm2(user_s_stack) # 1, K, 10
        # 新闻池化&卷积
        new_rep = self.norm1(new_rep) # max_clicked_num, title_dim
        new_rep = new_rep.unsqueeze(1).unsqueeze(1)  # max_clicked_num, 1, 1, title_dim
        news_pooled_embedding = []
        for x in range(self.num_filters):
            convoluted = self.news_conv_filters[str(x)](new_rep).squeeze(-1)
            activated = F.relu(convoluted)
            pooled = activated.max(dim=-1)[0]
            news_pooled_embedding.append(pooled.unsqueeze(1))
        news_s_stack = torch.cat(news_pooled_embedding, dim=1)
        news_s_stack = F.dropout(news_s_stack, p=self.dropout_prob, training=self.training)
        news_s_stack = self.norm3(news_s_stack)# max_clicked_num, K，10
        z_user = self.user_neighborhood_routing(news_s_stack, user_s_stack.squeeze())  # K，10
        return z_user

class new_encoder(torch.nn.Module):
    def __init__(self, user_size, sample_num, word_dim, entity_dim, title_word_size, entity_cate_size, max_one_hop):
        super(new_encoder, self).__init__()
        self.max_one_hop = max_one_hop
        self.title_dim = 160
        self.user_size = user_size
        self.sample_num = sample_num
        self.num_filters = 8
        self.PCNN = PCNN(title_word_size, word_dim, entity_dim, entity_cate_size)
        self.user_id_embedding = nn.Embedding(self.user_size+1, self.title_dim)
        self.norm1 = nn.LayerNorm(160)
        self.dropout_prob = 0.2
        self.news_conv_filters = nn.ModuleDict(
            {
                str(x): nn.Conv2d(1, 10, kernel_size=(self.max_one_hop + 1, self.title_dim), stride=(2, 2))
                for x in range(self.num_filters)
            })
        self.user_conv_filters = nn.ModuleDict(
            {
                str(x): nn.Conv2d(1, 10, kernel_size=(1, self.title_dim), stride=(2, 2))
                for x in range(self.num_filters)
            })
        self.norm2 = nn.LayerNorm(10)
        self.norm3 = nn.LayerNorm(10)
        self.news_neighborhood_routing = neighborhood_routing_algorithm( mode = 'news')
    def forward(self, word_embedding, entity_embedding, new_entity_cate_one, new_one_hop_neighbor):
        new_rep = self.PCNN(word_embedding, entity_embedding, new_entity_cate_one).unsqueeze(1)
        new_rep = F.dropout(new_rep, p=self.dropout_prob, training=self.training)
        new_rep = self.norm1(new_rep) # bz, 1, title_dim
        user_id_embedding = self.user_id_embedding(new_one_hop_neighbor.to(torch.int64))
        user_id_embedding = F.dropout(user_id_embedding, p=self.dropout_prob, training=self.training) # bz, max_one_hop, title_dim
        h_stack = torch.cat([new_rep, user_id_embedding], dim = 1).unsqueeze(1) # bz, max_one_hop+1, title_dim
        # 新闻池化&卷积
        news_pooled_embedding = []
        for x in range(self.num_filters):
            convoluted = self.news_conv_filters[str(x)](h_stack).squeeze(-1)
            activated = F.relu(convoluted)
            pooled = activated.max(dim=-1)[0]
            news_pooled_embedding.append(pooled.unsqueeze(1))
        news_s_stack = torch.cat(news_pooled_embedding, dim=1)
        news_s_stack = F.dropout(news_s_stack, p=self.dropout_prob, training=self.training)
        news_s_stack = self.norm2(news_s_stack) # bz, k, 10
        user_id_embedding = torch.flatten(user_id_embedding, 0, 1) # bz * max_one_hop, title
        user_id_embedding = user_id_embedding.unsqueeze(1).unsqueeze(1)
        # 用户池化&卷积
        user_pooled_embedding = []
        for x in range(self.num_filters):
            convoluted = self.user_conv_filters[str(x)](user_id_embedding).squeeze(-1)
            activated = F.relu(convoluted)
            pooled = activated.max(dim=-1)[0].view(-1, self.max_one_hop, 10)
            user_pooled_embedding.append(pooled.unsqueeze(2))
        user_s_stack = torch.cat(user_pooled_embedding, dim=2)
        user_s_stack = F.dropout(user_s_stack, p=self.dropout_prob, training=self.training)
        user_s_stack = self.norm2(user_s_stack) # bz, max_one_hop, k, 10
        z_news = self.news_neighborhood_routing(user_s_stack, news_s_stack)
        return z_news

class GNUD(torch.nn.Module):
    def __init__(self, args):
        super(GNUD, self).__init__()
        # self.max_one_hop = args.max_one_hop
        self.max_one_hop = 5
        self.batch_size = args.batch_size
        self.word_dim = args.word_embedding_dim
        self.entity_dim = args.entity_embedding_dim
        self.title_word_size = args.title_word_size
        self.sample_num = args.sample_num
        self.user_clicked_new_num = args.user_clicked_new_num
        self.num_filters = 8
        self.derta_n = 10
        self.entity_cate_size = args.entity_cate_size
        self.user_size = args.user_size
        self.new_encoder = new_encoder(self.user_size, self.sample_num, self.word_dim, self.entity_dim, self.title_word_size, self.entity_cate_size, self.max_one_hop)
        self.user_encoder = user_encoder(self.user_clicked_new_num, self.word_dim, self.entity_dim, self.title_word_size, self.entity_cate_size, self.sample_num, self.user_size)
        self.distinguish = nn.Linear(self.derta_n, self.num_filters, bias = True)

    def forward(self, user_index, user_clicked_new_word_embedding,
                user_clicked_new_entity_embedding, user_clicked_new_entity_cate,
                candidate_new_word_embedding, candidate_new_entity_embedding,
                candidate_new_entity_cate, new_one_hop_neighbor):
        user_index = user_index.to(device)
        user_clicked_new_word_embedding = user_clicked_new_word_embedding.to(device)
        user_clicked_new_entity_embedding = user_clicked_new_entity_embedding.to(device)
        user_clicked_new_entity_cate = user_clicked_new_entity_cate.to(device)
        candidate_new_word_embedding = candidate_new_word_embedding.to(device)
        candidate_new_entity_embedding = candidate_new_entity_embedding.to(device)
        candidate_new_entity_cate = candidate_new_entity_cate.to(device)
        new_one_hop_neighbor = new_one_hop_neighbor.to(device)
        # 用户编码器
        user_z_stack = None
        for i in range(self.user_clicked_new_num):
            user_index_one = user_index[i, :]
            clicked_word_embedding_one = user_clicked_new_word_embedding[i, :, :, :]
            clicked_word_embedding_one = clicked_word_embedding_one.squeeze()
            clicked_entity_embedding_one = user_clicked_new_entity_embedding[i, :, :, :]
            clicked_entity_embedding_one = clicked_entity_embedding_one.squeeze()
            user_clicked_new_entity_cate_one = user_clicked_new_entity_cate[i, :, :]
            user_clicked_new_entity_cate_one = user_clicked_new_entity_cate_one.squeeze()
            user_z_stack_one = self.user_encoder(clicked_word_embedding_one, clicked_entity_embedding_one,
                                                 user_clicked_new_entity_cate_one, user_index_one).unsqueeze(0)
            if i == 0:
                user_z_stack = user_z_stack_one
            else:
                user_z_stack = torch.cat([user_z_stack, user_z_stack_one], dim=0)
        ## 新闻编码器
        news_z_stack = None
        for i in range(self.sample_num):
            new_one_hop_neighbor_one = new_one_hop_neighbor[:, i, :]
            new_word_embedding_one = candidate_new_word_embedding[:, i, :]
            new_entity_embedding_one = candidate_new_entity_embedding[:, i, :]
            new_entity_cate_one = candidate_new_entity_cate[:, i]
            news_z_stack_one = self.new_encoder(new_word_embedding_one, new_entity_embedding_one, new_entity_cate_one, new_one_hop_neighbor_one)
            news_z_stack_one = news_z_stack_one.unsqueeze(1)
            if i==0:
                news_z_stack = news_z_stack_one
            else:
                news_z_stack = torch.cat([news_z_stack, news_z_stack_one], dim = 1)

        self.news_z_stack = news_z_stack
        self.user_z_stack = user_z_stack
        news_z_stack = torch.flatten(news_z_stack, -2, -1)
        user_z_stack = torch.flatten(user_z_stack, -2, -1).unsqueeze(1).repeat(1, 5,1)
        score = torch.sum(news_z_stack * user_z_stack, dim = -1)

        return score

    def loss_regular(self):
        loss = None
        for j in range(self.num_filters):
            news_z_stack_one = self.news_z_stack[:,:,j,:].squeeze()
            user_z_stack_one = self.user_z_stack[:,j,:].squeeze()
            u_loss_one = F.softmax(self.distinguish(user_z_stack_one), dim = -1)
            u_loss = u_loss_one[:,j]
            d_loss = None
            for i in range(self.sample_num):
                news_z_stack_one_j = news_z_stack_one[:,i,:].squeeze()
                d_loss_one = F.softmax(self.distinguish(news_z_stack_one_j), dim = -1)
                d_loss_one = d_loss_one[:,j]
                if i == 0:
                    d_loss = d_loss_one
                else:
                    d_loss = d_loss + d_loss_one
            loss_one = u_loss + d_loss
            if j == 0:
                loss = loss_one
            else:
                loss = loss + loss_one
        loss = torch.mean(loss / self.num_filters)
        return loss


    def loss(self, ser_index, user_clicked_new_word_embedding,
                user_clicked_new_entity_embedding, user_clicked_new_entity_cate,
                candidate_new_word_embedding, candidate_new_entity_embedding,
                candidate_new_entity_cate, new_one_hop_neighbor, label):
        label = label.to(device)
        sorce = self.forward(ser_index, user_clicked_new_word_embedding,
                             user_clicked_new_entity_embedding, user_clicked_new_entity_cate,
                             candidate_new_word_embedding, candidate_new_entity_embedding,
                             candidate_new_entity_cate, new_one_hop_neighbor)
        loss_regular = self.loss_regular()
        loss = torch.nn.functional.cross_entropy(sorce, torch.argmax(label, dim =1))
        loss = (1-0.2) * loss - 0.2 * loss_regular
        try:
            auc = roc_auc_score(label.cpu().numpy(), F.softmax(sorce.cpu(), dim=1).detach().numpy())
        except ValueError:
            auc = 0.5
        return loss, auc
