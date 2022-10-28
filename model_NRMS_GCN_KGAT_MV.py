import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *
from sklearn.metrics import roc_auc_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class new_encoder(torch.nn.Module):
    def __init__(self, word_dim, attention_dim, attention_heads, query_vector_dim, entity_size, entity_embedding_dim,
                 category_dim, subcategory_dim, category_size, subcategory_size):
        print(entity_size)
        super(new_encoder, self).__init__()
        self.multi_dim = attention_dim * attention_heads
        self.dropout_prob = 0.2
        # 主题级表征网络
        self.category_net = torch.nn.Sequential(
            nn.Embedding(category_size, embedding_dim=category_dim),
            nn.Linear(category_dim, 100, bias=True ),
            nn.Linear(100, self.multi_dim, bias=True),
            #nn.BatchNorm1d(self.multi_dim),
            nn.Tanh(),
            nn.Dropout(self.dropout_prob),
            )
        # 副主题级表征网络
        self.subcategory_net = torch.nn.Sequential(
            nn.Embedding(subcategory_size, embedding_dim=category_dim),
            nn.Linear(subcategory_dim, 100, bias=True),
            nn.Linear(100, self.multi_dim, bias=True),
            #nn.BatchNorm1d(self.multi_dim),
            nn.Tanh(),
            nn.Dropout(self.dropout_prob)
            )
        # 单词级表征网络
        self.word_net = torch.nn.Sequential(
            nn.Dropout(self.dropout_prob),
            nn.Linear(word_dim, 100, bias=True),
            nn.Tanh(),
            MultiHeadSelfAttention_2(100, attention_dim * attention_heads, attention_heads),
            nn.Tanh(),
            nn.Dropout(self.dropout_prob),
            Additive_Attention(query_vector_dim, self.multi_dim),
            nn.Dropout(self.dropout_prob)
           )
        self.KGAT = KGAT(self.multi_dim, entity_embedding_dim)
        # 实体级表征网络
        self.entity_net = torch.nn.Sequential(
            #nn.Dropout(self.dropout_prob),
            nn.Linear(2 * entity_embedding_dim, 100, bias=True),
            nn.Tanh(),
            gcn(entity_size, 100, self.multi_dim),
            nn.Dropout(self.dropout_prob),
            Additive_Attention(query_vector_dim, self.multi_dim),
            nn.Tanh(),
            nn.Dropout(self.dropout_prob)
        )
        # 附加注意力网络
        self.add_attention_net = torch.nn.Sequential(
            Additive_Attention(query_vector_dim, self.multi_dim),
            nn.BatchNorm1d(self.multi_dim)
        )


    def forward(self, word_embedding, entity_embedding,
                neigh_entity_embedding, neigh_relation_embedding,
                category_index, subcategory_index):
        category_rep = self.category_net(category_index.to(torch.int64))
        subcategory_rep = self.subcategory_net(subcategory_index.to(torch.int64))
        word_rep = self.word_net(word_embedding)
        entity_agg = self.KGAT(entity_embedding, neigh_entity_embedding, neigh_relation_embedding)
        entity_rep = self.entity_net(entity_agg)
        # 新闻附加注意力
        new_rep = torch.cat(
            [word_rep.unsqueeze(1), entity_rep.unsqueeze(1), category_rep.unsqueeze(1), subcategory_rep.unsqueeze(1)],
            dim=1)
        new_rep = self.add_attention_net(new_rep)
        return new_rep

class user_encoder(torch.nn.Module):
    def __init__(self, word_dim, attention_dim, attention_heads, query_vector_dim, entity_size, entity_embedding_dim,
                 category_dim, subcategory_dim, category_size, subcategory_size):
        super(user_encoder, self).__init__()
        self.new_encoder = new_encoder(word_dim, attention_dim, attention_heads, query_vector_dim, entity_size,
                                       entity_embedding_dim,
                                       category_dim, subcategory_dim, category_size, subcategory_size)
        self.multiheadatt = MultiHeadSelfAttention_2(attention_dim * attention_heads, attention_dim * attention_heads,
                                                     attention_heads)
        self.multi_dim = attention_dim * attention_heads
        self.norm1 = nn.LayerNorm(self.multi_dim)
        self.user_attention = Additive_Attention(query_vector_dim, self.multi_dim)
        self.norm2 = nn.LayerNorm(self.multi_dim)
        self.dropout_prob = 0.2

    def forward(self, word_embedding, entity_embedding,
                neigh_entity_embedding, neigh_relation_embedding,
                category_index, subcategory_index):
        # 点击新闻表征
        new_rep = self.new_encoder(word_embedding, entity_embedding,
                                   neigh_entity_embedding, neigh_relation_embedding,
                                   category_index, subcategory_index).unsqueeze(0)
        new_rep = F.dropout(new_rep, p=self.dropout_prob, training=self.training)
        new_rep = self.multiheadatt(new_rep)
        new_rep = F.dropout(new_rep, p=self.dropout_prob, training=self.training)
        new_rep = self.norm1(new_rep)
        # 用户表征
        user_rep = self.user_attention(new_rep)
        user_rep = self.norm2(user_rep)
        return user_rep

class NRMS_GCN_KGAT_MV(torch.nn.Module):
    def __init__(self, args):
        super(NRMS_GCN_KGAT_MV, self).__init__()
        self.word_dim = args.word_embedding_dim
        self.attention_dim = args.attention_dim
        self.attention_heads = args.attention_heads
        self.sample_num = args.sample_num
        self.user_clicked_new_num = args.user_clicked_new_num
        self.entity_size = args.new_entity_size
        self.entity_embedding_dim = args.entity_embedding_dim
        self.query_vector_dim = args.query_vector_dim
        self.category_dim = args.category_dim
        self.subcategory_dim = args.subcategory_dim
        self.category_size = args.category_size
        self.subcategory_size = args.subcategory_size

        self.new_encoder = new_encoder(self.word_dim, self.attention_dim, self.attention_heads, self.query_vector_dim, self.entity_size, self.entity_embedding_dim,
                                       self.category_dim, self.subcategory_dim, self.category_size, self.subcategory_size)
        self.user_encoder = user_encoder(self.word_dim, self.attention_dim, self.attention_heads, self.query_vector_dim, self.entity_size, self.entity_embedding_dim,
                                         self.category_dim, self.subcategory_dim, self.category_size, self.subcategory_size)

    def forward(self,candidate_new_word_embedding, user_clicked_new_word_embedding,
                candidate_new_entity_embedding,  user_clicked_new_entity_embedding,
                candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding,
                candidate_new_neigh_relation_embedding,user_clicked_new_neigh_relation_embedding,
                candidate_new_category_index, user_clicked_new_category_index,
                candidate_new_subcategory_index, user_clicked_new_subcategory_index
                ):
        candidate_new_word_embedding = candidate_new_word_embedding.to(device)
        user_clicked_new_word_embedding = user_clicked_new_word_embedding.to(device)
        candidate_new_entity_embedding = candidate_new_entity_embedding.to(device)
        user_clicked_new_entity_embedding = user_clicked_new_entity_embedding.to(device)
        candidate_new_neigh_entity_embedding = candidate_new_neigh_entity_embedding.to(device)
        user_clicked_new_neigh_entity_embedding = user_clicked_new_neigh_entity_embedding.to(device)
        candidate_new_neigh_relation_embedding = candidate_new_neigh_relation_embedding.to(device)
        user_clicked_new_neigh_relation_embedding = user_clicked_new_neigh_relation_embedding.to(device)
        candidate_new_category_index = candidate_new_category_index.to(device)
        user_clicked_new_category_index = user_clicked_new_category_index.to(device)
        candidate_new_subcategory_index = candidate_new_subcategory_index.to(device)
        user_clicked_new_subcategory_index = user_clicked_new_subcategory_index.to(device)

        ## 新闻编码器
        new_rep = None
        for i in range(self.sample_num):
            new_word_embedding_one = candidate_new_word_embedding[:, i, :]
            new_entity_embedding_one = candidate_new_entity_embedding[:, i, :]
            new_neigh_entity_embedding_one = candidate_new_neigh_entity_embedding[:, i, :, :, :]
            new_neigh_relation_embedding_one = candidate_new_neigh_relation_embedding[:, i, :, :, :]
            new_category_index = candidate_new_category_index[:, i]
            new_subcategory_index = candidate_new_subcategory_index[:, i]
            new_rep_one = self.new_encoder(new_word_embedding_one, new_entity_embedding_one,
                                           new_neigh_entity_embedding_one, new_neigh_relation_embedding_one,
                                           new_category_index, new_subcategory_index).unsqueeze(1)
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
            # 点击新闻实体嵌入
            clicked_new_entity_embedding_one = user_clicked_new_entity_embedding[i, :, :, :]
            clicked_new_entity_embedding_one = clicked_new_entity_embedding_one.squeeze()
            # 点击新闻实体邻居实体嵌入
            clicked_new_neigh_entity_embedding_one = user_clicked_new_neigh_entity_embedding[i, :, :, :]
            clicked_new_neigh_entity_embedding_one = clicked_new_neigh_entity_embedding_one.squeeze()
            # 点击新闻实体邻居实体关系嵌入
            clicked_new_neigh_relation_embedding_one = user_clicked_new_neigh_relation_embedding[i, :, :, :]
            clicked_new_neigh_relation_embedding_one = clicked_new_neigh_relation_embedding_one.squeeze()
            # 点击新闻主题index
            clicked_new_category_index = user_clicked_new_category_index[i, :]
            # 点击新闻副主题index
            clicked_new_subcategory_index = user_clicked_new_subcategory_index[i, :]
            # 用户表征
            user_rep_one = self.user_encoder(clicked_new_word_embedding_one, clicked_new_entity_embedding_one,
                                             clicked_new_neigh_entity_embedding_one,clicked_new_neigh_relation_embedding_one,
                                             clicked_new_category_index, clicked_new_subcategory_index).unsqueeze(0)
            if i == 0:
                user_rep = user_rep_one
            else:
                user_rep = torch.cat([user_rep, user_rep_one], dim=0)
        sorce = torch.sum(new_rep * user_rep, dim = 2)
        return sorce

    def loss(self, candidate_new_word_embedding,  user_clicked_word_embedding,
             candidate_new_entity_embedding, user_clicked_new_entity_embedding,
             candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding,
             candidate_new_neigh_relation_embedding, user_clicked_new_neigh_relation_embedding,
             candidate_new_category_index, user_clicked_new_category_index,
             candidate_new_subcategory_index, user_clicked_new_subcategory_index,
             label):
        label = label.to(device)
        sorce = self.forward(candidate_new_word_embedding, user_clicked_word_embedding,
                             candidate_new_entity_embedding, user_clicked_new_entity_embedding,
                             candidate_new_neigh_entity_embedding,user_clicked_new_neigh_entity_embedding,
                             candidate_new_neigh_relation_embedding, user_clicked_new_neigh_relation_embedding,
                             candidate_new_category_index, user_clicked_new_category_index,
                             candidate_new_subcategory_index, user_clicked_new_subcategory_index)
        loss = torch.nn.functional.cross_entropy(sorce, torch.argmax(label, dim =1))
        try:
            auc = roc_auc_score(label.cpu().numpy(), F.softmax(sorce.cpu(), dim=1).detach().numpy())
        except ValueError:
            auc = 0.5
        return loss, auc
