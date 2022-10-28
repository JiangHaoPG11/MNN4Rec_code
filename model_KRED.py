import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *
from sklearn.metrics import roc_auc_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class new_encoder(torch.nn.Module):
    def __init__(self, word_dim, attention_dim, attention_heads, query_vector_dim, entity_size, entity_embedding_dim,
                 pos_size, freq_size, cate_size):
        super(new_encoder, self).__init__()
        self.multi_dim = attention_dim * attention_heads
        self.entity_size = entity_size
        # 单词级表征表征网络
        self.multiheadatt = MultiHeadSelfAttention_2(word_dim, attention_dim * attention_heads, attention_heads)
        self.norm1 = nn.LayerNorm(self.multi_dim)
        self.word_attention = Additive_Attention(query_vector_dim, self.multi_dim)
        # 实体级表征网络
        self.KGAT = KGAT(self.multi_dim, entity_embedding_dim)
        self.pos_embed = nn.Embedding(pos_size, 2 * entity_embedding_dim)
        self.freq_embed = nn.Embedding(freq_size, 2 * entity_embedding_dim)
        self.cate_embed = nn.Embedding(cate_size, 2 * entity_embedding_dim)
        # 融合表征网络
        self.attention_layer_1 = nn.Linear(self.multi_dim + 2 * entity_embedding_dim, 200, bias = True)
        self.attention_layer_2 = nn.Linear(200, 1, bias = True)
        self.attention = nn.Linear(2 * entity_embedding_dim + self.multi_dim, self.multi_dim, bias = True)
        self.norm2 = nn.LayerNorm(self.multi_dim)
        self.dropout_prob = 0.2

    def forward(self, word_embedding, entity_embedding,
                neigh_entity_embedding, neigh_relation_embedding,
                pos_index, freq_index, cate_index):
        # 单词级新闻表征
        word_embedding = self.multiheadatt(word_embedding)
        word_embedding = self.norm1(word_embedding)
        word_embedding = F.dropout(word_embedding, p=self.dropout_prob, training=self.training)
        word_rep = self.word_attention(word_embedding)

        #  Entity Representation Layer
        entity_agg = self.KGAT(entity_embedding, neigh_entity_embedding, neigh_relation_embedding)
        # Context Embedding Layer
        pos_rep = self.pos_embed(pos_index.to(torch.int64))
        freq_rep = self.freq_embed(freq_index.to(torch.int64))
        cate_rep = self.cate_embed(cate_index.to(torch.int64))
        entity_rep = entity_agg + pos_rep + freq_rep + cate_rep  # b, max_entity, 2*dim
        # Information Distillation Layer
        word_entity_cat = torch.cat([entity_rep, word_rep.unsqueeze(1).repeat(1,self.entity_size,1)], dim = -1)
        att_weight = self.attention_layer_2(torch.relu(self.attention_layer_1(word_entity_cat))) # b, max_entity, 1
        entity_rep_weight = torch.matmul(torch.transpose(att_weight, -1, -2), entity_rep).squeeze()
        news_rep = torch.cat([entity_rep_weight, word_rep], dim = -1)
        news_rep = self.attention(news_rep)
        news_rep = self.norm2(news_rep)
        return news_rep

class user_encoder(torch.nn.Module):
    def __init__(self, word_dim, attention_dim, attention_heads, query_vector_dim, entity_size, entity_embedding_dim,
                 pos_size, freq_size, cate_size):
        super(user_encoder, self).__init__()
        self.new_encoder = new_encoder(word_dim, attention_dim, attention_heads, query_vector_dim, entity_size,
                                       entity_embedding_dim, pos_size, freq_size, cate_size)
        self.multi_dim = attention_dim * attention_heads
        self.multiheadatt = MultiHeadSelfAttention_2(self.multi_dim, self.multi_dim, attention_heads)
        self.norm1 = nn.LayerNorm(self.multi_dim)
        self.user_attention = Additive_Attention(query_vector_dim, self.multi_dim)
        self.norm2 = nn.LayerNorm(self.multi_dim)
        self.dropout_prob = 0.2

    def forward(self, word_embedding, entity_embedding,
                neigh_entity_embedding, neigh_relation_embedding,
                pos_index, freq_index, cate_index):
        # 点击新闻表征
        new_rep = self.new_encoder(word_embedding, entity_embedding,
                                   neigh_entity_embedding, neigh_relation_embedding,
                                   pos_index, freq_index, cate_index).unsqueeze(0)
        new_rep = F.dropout(new_rep, p=self.dropout_prob, training=self.training)
        new_rep = self.multiheadatt(new_rep)
        new_rep = self.norm1(new_rep)
        user_rep = self.user_attention(new_rep)
        user_rep = self.norm2(user_rep)
        return user_rep

class KRED(torch.nn.Module):
    def __init__(self, args):
        super(KRED, self).__init__()
        self.word_dim = args.word_embedding_dim
        self.attention_dim = args.attention_dim
        self.attention_heads = args.attention_heads
        self.sample_num = args.sample_num
        self.user_clicked_new_num = args.user_clicked_new_num
        self.entity_size = args.new_entity_size
        self.entity_embedding_dim = args.entity_embedding_dim
        self.query_vector_dim = args.query_vector_dim
        self.pos_size = args.entity_pos_size
        self.freq_size = args.entity_freq_size
        self.cate_size = args.entity_cate_size

        self.new_encoder = new_encoder(self.word_dim, self.attention_dim, self.attention_heads, self.query_vector_dim, self.entity_size, self.entity_embedding_dim,
                                       self.pos_size, self.freq_size, self.cate_size)
        self.user_encoder = user_encoder(self.word_dim, self.attention_dim, self.attention_heads, self.query_vector_dim, self.entity_size, self.entity_embedding_dim,
                                         self.pos_size, self.freq_size, self.cate_size)
    def forward(self,
                candidate_new_word_embedding, user_clicked_new_word_embedding,
                candidate_new_entity_embedding,  user_clicked_new_entity_embedding,
                candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding,
                candidate_new_neigh_relation_embedding,user_clicked_new_neigh_relation_embedding,
                candidate_new_entity_pos_index, user_clicked_new_entity_pos_index,
                candidate_new_entity_freq_index, user_clicked_new_entity_freq_index,
                candidate_new_entity_cate_index, user_clicked_new_entity_cate_index):

        candidate_new_word_embedding = candidate_new_word_embedding.to(device)
        user_clicked_new_word_embedding = user_clicked_new_word_embedding.to(device)
        candidate_new_entity_embedding = candidate_new_entity_embedding.to(device)
        user_clicked_new_entity_embedding = user_clicked_new_entity_embedding.to(device)
        candidate_new_neigh_entity_embedding = candidate_new_neigh_entity_embedding.to(device)
        user_clicked_new_neigh_entity_embedding = user_clicked_new_neigh_entity_embedding.to(device)
        candidate_new_neigh_relation_embedding = candidate_new_neigh_relation_embedding.to(device)
        user_clicked_new_neigh_relation_embedding = user_clicked_new_neigh_relation_embedding.to(device)
        candidate_new_entity_pos_index = candidate_new_entity_pos_index.to(device)
        user_clicked_new_entity_pos_index = user_clicked_new_entity_pos_index.to(device)
        candidate_new_entity_freq_index = candidate_new_entity_freq_index.to(device)
        user_clicked_new_entity_freq_index = user_clicked_new_entity_freq_index.to(device)
        candidate_new_entity_cate_index = candidate_new_entity_cate_index.to(device)
        user_clicked_new_entity_cate_index = user_clicked_new_entity_cate_index.to(device)

        ## 新闻编码器
        new_rep = None
        for i in range(self.sample_num):
            new_word_embedding_one = candidate_new_word_embedding[:, i, :]
            new_entity_embedding_one = candidate_new_entity_embedding[:, i, :]
            new_neigh_entity_embedding_one = candidate_new_neigh_entity_embedding[:, i, :, :, :]
            new_neigh_relation_embedding_one = candidate_new_neigh_relation_embedding[:, i, :, :, :]
            new_entity_pos_index = candidate_new_entity_pos_index[:, i]
            new_entity_freq_index = candidate_new_entity_freq_index[:, i]
            new_entity_cate_index = candidate_new_entity_cate_index[:, i]
            new_rep_one = self.new_encoder(new_word_embedding_one, new_entity_embedding_one,
                                           new_neigh_entity_embedding_one, new_neigh_relation_embedding_one,
                                           new_entity_pos_index, new_entity_freq_index, new_entity_cate_index).unsqueeze(1)
            if i == 0:
                new_rep = new_rep_one
            else:
                new_rep = torch.cat([new_rep, new_rep_one], dim=1)
        # 用户编码器
        user_rep = None
        for i in range(self.user_clicked_new_num):
            # 点击新闻单词嵌入
            clicked_new_word_embedding_one = user_clicked_new_word_embedding[i, :, :, :]
            clicked_new_word_embedding_one = clicked_new_word_embedding_one.squeeze()
            # 点击新闻实体嵌入
            clicked_new_entity_embedding_one = user_clicked_new_entity_embedding[i, :, :, :]
            clicked_new_entity_embedding_one = clicked_new_entity_embedding_one.squeeze()
            # 点击新闻实体邻居实体嵌入
            clicked_new_neigh_entity_embedding_one = user_clicked_new_neigh_entity_embedding[i, :, :, :]
            clicked_new_neigh_entity_embedding_one = clicked_new_neigh_entity_embedding_one.squeeze()
            # 点击新闻实体邻居实体关系嵌入
            clicked_new_neigh_relation_embedding_one = user_clicked_new_neigh_relation_embedding[i, :, :, :]
            clicked_new_neigh_relation_embedding_one = clicked_new_neigh_relation_embedding_one.squeeze()
            # 点击新闻位置，频率，类别
            clicked_new_entity_pos_index = user_clicked_new_entity_pos_index[i, :]
            clicked_new_entity_freq_index = user_clicked_new_entity_freq_index[i, :]
            clicked_new_entity_cate_index = user_clicked_new_entity_cate_index[i, :]
            # 用户表征
            user_rep_one = self.user_encoder(clicked_new_word_embedding_one, clicked_new_entity_embedding_one,
                                             clicked_new_neigh_entity_embedding_one, clicked_new_neigh_relation_embedding_one,
                                             clicked_new_entity_pos_index, clicked_new_entity_freq_index, clicked_new_entity_cate_index).unsqueeze(0)
            if i == 0:
                user_rep = user_rep_one
            else:
                user_rep = torch.cat([user_rep, user_rep_one], dim=0)
        sorce = torch.sum(new_rep * user_rep, dim=2)

        return sorce

    def loss(self, candidate_new_word_embedding, user_clicked_new_word_embedding,
             candidate_new_entity_embedding,  user_clicked_new_entity_embedding,
             candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding,
             candidate_new_neigh_relation_embedding,user_clicked_new_neigh_relation_embedding,
             candidate_new_entity_pos_index, user_clicked_new_entity_pos_index,
             candidate_new_entity_freq_index, user_clicked_new_entity_freq_index,
             candidate_new_entity_cate_index, user_clicked_new_entity_cate_index,
             label):
        label = label.to(device)
        sorce = self.forward(candidate_new_word_embedding, user_clicked_new_word_embedding,
                             candidate_new_entity_embedding, user_clicked_new_entity_embedding,
                             candidate_new_neigh_entity_embedding,user_clicked_new_neigh_entity_embedding,
                             candidate_new_neigh_relation_embedding, user_clicked_new_neigh_relation_embedding,
                             candidate_new_entity_pos_index, user_clicked_new_entity_pos_index,
                             candidate_new_entity_freq_index, user_clicked_new_entity_freq_index,
                             candidate_new_entity_cate_index, user_clicked_new_entity_cate_index)
        loss = torch.nn.functional.cross_entropy(sorce, torch.argmax(label, dim =1))
        try:
            auc = roc_auc_score(label.cpu().numpy(), F.softmax(sorce.cpu(), dim=1).detach().numpy())
        except ValueError:
            auc = 0.5
        return loss, auc
