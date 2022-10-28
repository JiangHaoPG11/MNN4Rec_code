import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *
from sklearn.metrics import roc_auc_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class new_encoder(torch.nn.Module):
    def __init__(self, attention_dim, query_vector_dim, entity_size, entity_embedding_dim):
        super(new_encoder, self).__init__()
        self.GCN = gcn(entity_size, 2 * entity_embedding_dim, attention_dim)
        self.KGAT = KGAT(attention_dim, entity_embedding_dim)
        self.norm3 = nn.LayerNorm(attention_dim)
        self.entity_attention = Additive_Attention(query_vector_dim, attention_dim)
        self.norm4 = nn.LayerNorm(attention_dim)
        self.dropout_prob = 0.2
    def forward(self, entity_embedding, neigh_entity_embedding, neigh_relation_embedding):
        entity_agg = self.KGAT(entity_embedding, neigh_entity_embedding, neigh_relation_embedding)
        entity_inter = self.GCN(entity_agg)
        entity_inter = F.dropout(entity_inter, p=self.dropout_prob, training=self.training)
        entity_inter = self.norm3(entity_inter)
        entity_rep = self.entity_attention(entity_inter)
        entity_rep = F.dropout(entity_rep, p=self.dropout_prob, training=self.training)
        new_rep = self.norm4(entity_rep)
        return new_rep

class user_encoder(torch.nn.Module):
    def __init__(self,attention_dim, attention_heads, query_vector_dim, entity_size, entity_embedding_dim):
        super(user_encoder, self).__init__()
        self.new_encoder = new_encoder(attention_dim, query_vector_dim, entity_size, entity_embedding_dim)
        self.mutiheadatt = MultiHeadSelfAttention(attention_dim, attention_dim, attention_heads)
        self.norm1 = nn.LayerNorm(attention_dim)
        self.user_attention = Additive_Attention(query_vector_dim, attention_dim)
        self.norm2 = nn.LayerNorm(attention_dim)
        self.attention_dim = attention_dim

    def forward(self,entity_embedding, neigh_entity_embedding, neigh_relation_embedding):
        new_rep = self.new_encoder(entity_embedding, neigh_entity_embedding, neigh_relation_embedding).unsqueeze(0)
        new_rep = self.mutiheadatt(new_rep, self.attention_dim)
        new_rep = F.dropout(new_rep, p=self.dropout_prob, training=self.training)
        new_rep = self.norm1(new_rep)
        user_rep = self.user_attention(new_rep)
        user_rep = F.dropout(user_rep, p=self.dropout_prob, training=self.training)
        user_rep = self.norm2(user_rep)
        return user_rep

class GCN_KGAT(torch.nn.Module):
    def __init__(self, args):
        super(GCN_KGAT, self).__init__()
        self.word_dim = args.word_embedding_dim
        self.attention_dim = args.attention_dim
        self.attention_heads = args.attention_heads
        self.title_word_size = args.title_word_size
        self.sample_num = args.sample_num
        self.user_clicked_new_num = args.user_clicked_new_num
        self.entity_size = args.new_entity_size
        self.entity_embedding_dim = args.entity_embedding_dim
        self.query_vector_dim = args.query_vector_dim

        self.new_encoder = new_encoder( self.attention_dim, self.query_vector_dim, self.entity_size, self.entity_embedding_dim)
        self.user_encoder = user_encoder(self.attention_dim, self.attention_heads, self.query_vector_dim, self.entity_size, self.entity_embedding_dim)

    def forward(self, candidate_new_entity_embedding, user_clicked_new_entity_embedding,
                candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding,
                candidate_new_neigh_relation_embedding, user_clicked_new_neigh_relation_embedding):
        candidate_new_entity_embedding = candidate_new_entity_embedding.to(device)
        user_clicked_new_entity_embedding = user_clicked_new_entity_embedding.to(device)
        candidate_new_neigh_entity_embedding = candidate_new_neigh_entity_embedding.to(device)
        user_clicked_new_neigh_entity_embedding = user_clicked_new_neigh_entity_embedding.to(device)
        candidate_new_neigh_relation_embedding = candidate_new_neigh_relation_embedding.to(device)
        user_clicked_new_neigh_relation_embedding = user_clicked_new_neigh_relation_embedding.to(device)
        ## 新闻编码器
        new_rep = None
        for i in range(self.sample_num):
            new_entity_embedding_one = candidate_new_entity_embedding[:, i, :]
            new_neigh_entity_embedding_one = candidate_new_neigh_entity_embedding[:, i, :, :, :]
            new_neigh_relation_embedding_one = candidate_new_neigh_relation_embedding[:, i, :, :, :]
            new_rep_one = self.new_encoder(new_entity_embedding_one, new_neigh_entity_embedding_one, new_neigh_relation_embedding_one).unsqueeze(1)
            if i == 0:
                new_rep = new_rep_one
            else:
                new_rep = torch.cat([new_rep, new_rep_one], dim=1)
        # 用户编码器
        user_rep = None
        for i in range(self.user_clicked_new_num):
            # 点击新闻实体嵌入
            clicked_new_entity_embedding_one = user_clicked_new_entity_embedding[i, :, :, :]
            clicked_new_entity_embedding_one = clicked_new_entity_embedding_one.squeeze()
            # 点击新闻实体邻居实体嵌入
            clicked_new_neigh_entity_embedding_one = user_clicked_new_neigh_entity_embedding[i, :, :, :]
            clicked_new_neigh_entity_embedding_one = clicked_new_neigh_entity_embedding_one.squeeze()
            # 点击新闻实体邻居实体关系嵌入
            clicked_new_neigh_relation_embedding_one = user_clicked_new_neigh_relation_embedding[i, :, :, :]
            clicked_new_neigh_relation_embedding_one = clicked_new_neigh_relation_embedding_one.squeeze()
            # 用户表征
            user_rep_one = self.user_encoder(clicked_new_entity_embedding_one, clicked_new_neigh_entity_embedding_one, clicked_new_neigh_relation_embedding_one).unsqueeze(0)
            if i == 0:
                user_rep = user_rep_one
            else:
                user_rep = torch.cat([user_rep, user_rep_one], dim=0)
        sorce = torch.sum(new_rep * user_rep, dim=2)
        return sorce

    def loss(self, candidate_new_entity_embedding, user_clicked_new_entity_embedding,
             candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding,
             candidate_new_neigh_relation_embedding, user_clicked_new_neigh_relation_embedding,label):
        label = label.to(device)
        sorce = self.forward(candidate_new_entity_embedding, user_clicked_new_entity_embedding,
                             candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding,
                             candidate_new_neigh_relation_embedding,  user_clicked_new_neigh_relation_embedding)
        loss = torch.nn.functional.cross_entropy(sorce, torch.argmax(label, dim =1))
        try:
            auc = roc_auc_score(label.cpu().numpy(), F.softmax(sorce.cpu(), dim=1).detach().numpy())
        except ValueError:
            auc = 0.5
        return loss, auc
