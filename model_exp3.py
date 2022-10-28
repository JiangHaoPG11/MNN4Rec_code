import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *
from sklearn.metrics import roc_auc_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class new_encoder(torch.nn.Module):
    def __init__(self, word_dim, attention_dim, attention_heads, query_vector_dim, entity_size, entity_embedding_dim,
                 category_dim, category_size ):
        super(new_encoder, self).__init__()
        self.multi_dim = attention_dim * attention_heads
        # 主题级表征网络
        self.embedding_layer1 = nn.Embedding(category_size, embedding_dim=category_dim)
        self.fc1 = nn.Linear(category_dim, self.multi_dim, bias=True)
        self.norm1 = nn.LayerNorm(self.multi_dim)
        # 单词级表征网络
        self.fc3 = nn.Linear(word_dim, 100, bias=True)
        self.multiheadatt = MultiHeadSelfAttention_2(100, attention_dim * attention_heads, attention_heads)
        self.norm2 = nn.LayerNorm(self.multi_dim)
        self.word_attention_c = SimilarityAttention()
        self.word_attention = QueryAttention(query_vector_dim, self.multi_dim)
        self.norm3 = nn.LayerNorm(self.multi_dim)
        self.norm4 = nn.LayerNorm(self.multi_dim)
        # 实体级表征网络
        self.fc4 = nn.Linear(2 * entity_embedding_dim, 100 , bias = True)
        self.GCN = gcn(entity_size, 100, self.multi_dim)
        self.KGAT = KGAT(self.multi_dim, entity_embedding_dim)
        self.norm5 = nn.LayerNorm(self.multi_dim)
        self.entity_attention_c = SimilarityAttention()
        self.entity_attention = QueryAttention(query_vector_dim, self.multi_dim)
        # self.entity_attention = Additive_Attention(query_vector_dim, self.multi_dim)
        self.norm6 = nn.LayerNorm(self.multi_dim)
        self.norm7 = nn.LayerNorm(self.multi_dim)
        # 新闻注意力
        self.new_attention = Additive_Attention(query_vector_dim, self.multi_dim)
        self.norm8 = nn.LayerNorm(self.multi_dim)
        self.dropout_prob = 0.2

    def forward(self, word_embedding, entity_embedding,
                neigh_entity_embedding, neigh_relation_embedding,
                category_index, cand_title_length, user_vector):
        # 主题级表征
        category_embedding = self.embedding_layer1(category_index.to(torch.int64))
        category_rep = torch.tanh(self.fc1(category_embedding))
        category_rep = F.dropout(category_rep, p=self.dropout_prob, training=self.training)
        category_rep = self.norm1(category_rep)
        # 单词级新闻表征
        word_embedding = torch.tanh(self.fc3(word_embedding))
        word_embedding = F.dropout(word_embedding, p=self.dropout_prob, training=self.training)
        word_embedding = self.multiheadatt(word_embedding, length=cand_title_length)
        word_embedding = self.norm2(word_embedding)
        # 类别感知
        word_rep_c = self.word_attention_c(category_rep, word_embedding)
        word_rep_c = F.dropout(word_rep_c, p=self.dropout_prob, training=self.training)
        word_rep_c = self.norm3(word_rep_c)
        # 非类别感知
        word_rep = self.word_attention(user_vector, word_embedding)
        word_rep = F.dropout(word_rep, p=self.dropout_prob, training=self.training)
        word_rep = self.norm4(word_rep)
        # 实体级新闻表征(类别感知)
        entity_embedding = F.dropout(entity_embedding, p=self.dropout_prob, training=self.training)
        entity_agg = self.KGAT(entity_embedding, neigh_entity_embedding, neigh_relation_embedding)
        entity_agg = torch.tanh(self.fc4(entity_agg))
        entity_inter = self.GCN(entity_agg)
        entity_inter = self.norm5(entity_inter)
        # 类别感知
        entity_rep_c = self.entity_attention_c(category_rep, entity_inter)
        entity_rep_c = F.dropout(entity_rep_c, p=self.dropout_prob, training=self.training)
        entity_rep_c = self.norm6(entity_rep_c)
        # 非类别感知
        entity_rep = self.entity_attention(user_vector, entity_inter)
        entity_rep = F.dropout(entity_rep, p=self.dropout_prob, training=self.training)
        entity_rep = self.norm7(entity_rep)
        # 新闻附加注意力
        new_rep = torch.cat([word_rep.unsqueeze(1), word_rep_c.unsqueeze(1), entity_rep.unsqueeze(1), entity_rep_c.unsqueeze(1)], dim=1)
        new_rep = self.new_attention(new_rep)
        new_rep = self.norm8(new_rep)
        return new_rep

class user_encoder(torch.nn.Module):
    def __init__(self, word_dim, attention_dim, attention_heads, query_vector_dim, entity_size, entity_embedding_dim,
                 category_dim, category_size):
        super(user_encoder, self).__init__()
        self.new_encoder = new_encoder(word_dim, attention_dim, attention_heads, query_vector_dim, entity_size, entity_embedding_dim,
                                       category_dim, category_size)
        self.multiheadatt = MultiHeadSelfAttention_2(attention_dim * attention_heads, attention_dim * attention_heads, attention_heads)
        self.multi_dim = attention_dim * attention_heads
        self.norm1 = nn.LayerNorm(self.multi_dim)
        self.user_attention = Additive_Attention(query_vector_dim, self.multi_dim)
        self.norm2 = nn.LayerNorm(self.multi_dim)
        self.dropout_prob = 0.2

    def forward(self, word_embedding, entity_embedding,
                neigh_entity_embedding, neigh_relation_embedding,
                category_index, user_clicked_new_title_length, user_clicked_new_length, user_vector_1, user_vector_2):
        # 点击新闻表征
        new_rep = self.new_encoder(word_embedding, entity_embedding,
                                   neigh_entity_embedding, neigh_relation_embedding,
                                   category_index, user_clicked_new_title_length, user_vector_1.unsqueeze(1).repeat(50,1,1)).unsqueeze(0)
        new_rep = F.dropout(new_rep, p=self.dropout_prob, training=self.training)
        new_rep = self.multiheadatt(new_rep, length=user_clicked_new_length)
        new_rep = F.dropout(new_rep, p=self.dropout_prob, training=self.training)
        new_rep = self.norm1(new_rep)
        # 用户表征
        user_rep = self.user_attention(new_rep)
        user_rep = F.dropout(user_rep, p=self.dropout_prob, training=self.training)
        user_rep = self.norm2(user_rep)
        return user_rep

class exp3(torch.nn.Module):
    '''
        将主题信息作为注意力的询问向量，分别选择单词和实体，构建主题感知
        将用户向量作为注意力的询问向量，分别选择单词和实体，构建非主题感知
        并用户加入mask的多头自我注意力机制来处理单词交互信息
    '''
    def __init__(self, args):
        super(exp3, self).__init__()
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
        self.user_embedding_size = args.user_embedding_size
        # 用户嵌入
        self.user_embedding = nn.Embedding(args.user_size, self.user_embedding_size)
        self.fc1 = nn.Linear(self.user_embedding_size, self.query_vector_dim)
        self.norm1 = nn.LayerNorm(self.query_vector_dim)
        self.fc2 = nn.Linear(self.user_embedding_size, self.query_vector_dim)
        self.norm2 = nn.LayerNorm(self.query_vector_dim)
        # 编码器初始化
        self.new_encoder = new_encoder(self.word_dim, self.attention_dim, self.attention_heads, self.query_vector_dim, self.entity_size, self.entity_embedding_dim,
                                       self.category_dim, self.category_size)
        self.user_encoder = user_encoder(self.word_dim, self.attention_dim, self.attention_heads, self.query_vector_dim, self.entity_size, self.entity_embedding_dim,
                                         self.category_dim, self.category_size)

    def forward(self,candidate_new_word_embedding, user_clicked_new_word_embedding,
                candidate_new_entity_embedding,  user_clicked_new_entity_embedding,
                candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding,
                candidate_new_neigh_relation_embedding,user_clicked_new_neigh_relation_embedding,
                candidate_new_category_index, user_clicked_new_category_index,
                cand_title_length, user_clicked_new_title_length, user_clicked_new_length, user_index):
        # 传入gpu
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
        cand_title_length = cand_title_length.to(device)
        user_clicked_new_title_length = user_clicked_new_title_length.to(device)
        user_clicked_new_length = user_clicked_new_length.to(device)
        user_index = user_index.to(device)
        # 获取用户嵌入
        user_embedding = self.user_embedding(user_index)
        user_vector_1 = self.fc1(user_embedding)
        user_vector_1 = self.norm1(user_vector_1)
        user_vector_2 = self.fc2(user_embedding)
        user_vector_2 = self.norm2(user_vector_2)
        ## 新闻编码器
        new_rep = None
        for i in range(self.sample_num):
            new_word_embedding_one = candidate_new_word_embedding[:, i, :]
            new_entity_embedding_one = candidate_new_entity_embedding[:, i, :]
            new_neigh_entity_embedding_one = candidate_new_neigh_entity_embedding[:, i, :, :, :]
            new_neigh_relation_embedding_one = candidate_new_neigh_relation_embedding[:, i, :, :, :]
            new_category_index = candidate_new_category_index[:, i]
            cand_title_length_one = cand_title_length[:, i]
            new_rep_one = self.new_encoder(new_word_embedding_one, new_entity_embedding_one,
                                           new_neigh_entity_embedding_one, new_neigh_relation_embedding_one,
                                           new_category_index,cand_title_length_one, user_vector_1).unsqueeze(1)
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
            # 点击新闻的标题长度
            user_clicked_new_title_length_one = user_clicked_new_title_length[i, :]
            user_clicked_new_title_length_one = user_clicked_new_title_length_one.squeeze()
            # 点击新闻长度
            user_clicked_new_length_one = user_clicked_new_length[i, :]
            user_clicked_new_length_one = user_clicked_new_length_one.squeeze()
            # 选择用户嵌入
            user_vector_1_one = user_vector_1[i, :]
            user_vector_2_one = user_vector_2[i, :]
            # 用户表征
            user_rep_one = self.user_encoder(clicked_new_word_embedding_one, clicked_new_entity_embedding_one,
                                             clicked_new_neigh_entity_embedding_one,clicked_new_neigh_relation_embedding_one,
                                             clicked_new_category_index, user_clicked_new_title_length_one, user_clicked_new_length_one,
                                             user_vector_1_one, user_vector_2_one).unsqueeze(0)
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
             cand_title_length, user_clicked_new_title_length, user_clicked_new_length, user_index, label):
        label = label.to(device)
        sorce = self.forward(candidate_new_word_embedding, user_clicked_word_embedding,
                             candidate_new_entity_embedding, user_clicked_new_entity_embedding,
                             candidate_new_neigh_entity_embedding,user_clicked_new_neigh_entity_embedding,
                             candidate_new_neigh_relation_embedding, user_clicked_new_neigh_relation_embedding,
                             candidate_new_category_index, user_clicked_new_category_index,
                             cand_title_length, user_clicked_new_title_length, user_clicked_new_length, user_index)
        loss = torch.nn.functional.cross_entropy(sorce, torch.argmax(label, dim =1))
        auc = roc_auc_score(label.cpu().numpy(), F.softmax(sorce, dim=1).detach().numpy())
        return loss, auc
