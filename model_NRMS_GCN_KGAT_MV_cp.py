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
        super(new_encoder, self).__init__()
        self.multi_dim = attention_dim * attention_heads
        self.tanh = nn.Tanh()
        self.dropout_prob = 0.2

        # 主题级表征网络
        self.embedding_layer1 = nn.Embedding(category_size, embedding_dim=category_dim)
        self.fc1 = nn.Linear(category_dim, self.multi_dim, bias=True)

        # 副主题节点学习网络
        self.embedding_layer2 = nn.Embedding(subcategory_size, embedding_dim=subcategory_dim)
        self.fc2 = nn.Linear(subcategory_dim, self.multi_dim, bias=True)

        # 标题节点学习网络
        self.layernorm_word = nn.LayerNorm(word_dim)
        self.multiheadatt = MultiHeadSelfAttention_2(word_dim, attention_dim * attention_heads, attention_heads)
        self.word_attention = Additive_Attention(query_vector_dim, self.multi_dim)
        self.layernorm1 = nn.LayerNorm(self.multi_dim)
        #self.norm2 = nn.LayerNorm(self.multi_dim)

        # 实体节点学习网络
        self.layernorm_entity = nn.LayerNorm(entity_embedding_dim)
        self.fc4 = nn.Linear(2 * entity_embedding_dim, 100, bias=True)
        self.layernorm2 = nn.LayerNorm(self.multi_dim)
        self.GCN = gcn(entity_size, 100, self.multi_dim)
        self.KGAT = KGAT(self.multi_dim, entity_embedding_dim)
        self.entity_attention = Additive_Attention(query_vector_dim, self.multi_dim)

        # 新闻融合
        self.new_attention = Additive_Attention(query_vector_dim, self.multi_dim)

        # 新闻显式关系构建网络
        self.fc5 = nn.Linear(2 * entity_embedding_dim, self.multi_dim, bias = True)
        self.relation_attention = SimilarityAttention()
        self.layernorm3 = nn.LayerNorm(self.multi_dim)

        # 最终归一化
        self.layernorm4 = nn.LayerNorm(self.multi_dim)


    def forward(self, word_embedding, entity_embedding,
                neigh_entity_embedding, neigh_relation_embedding,
                category_index, subcategory_index):
        # 主题级表征
        category_embedding = self.embedding_layer1(category_index.to(torch.int64))
        category_rep = self.tanh(self.fc1(category_embedding))
        category_rep = F.dropout(category_rep, p=self.dropout_prob, training=self.training)

        # 副主题级表征
        subcategory_embedding = self.embedding_layer2(subcategory_index.to(torch.int64))
        subcategory_rep = self.tanh(self.fc2(subcategory_embedding))
        subcategory_rep = F.dropout(subcategory_rep, p=self.dropout_prob, training=self.training)

        # 单词级新闻表征
        word_embedding = self.layernorm_word(word_embedding)
        # word_embedding = F.dropout(word_embedding, p=self.dropout_prob, training=self.training)
        word_embedding = self.multiheadatt(word_embedding)
        word_embedding = self.layernorm1(word_embedding)
        word_embedding = F.dropout(word_embedding, p=self.dropout_prob, training=self.training)
        word_rep = self.tanh(self.word_attention(word_embedding))    
        word_rep = F.dropout(word_rep, p=self.dropout_prob, training=self.training)
    
        # 实体级新闻表征
        entity_embedding = self.layernorm_entity(entity_embedding)
        neigh_entity_embedding = self.layernorm_entity(neigh_entity_embedding)
        neigh_relation_embedding = self.layernorm_entity(neigh_relation_embedding)
        # entity_embedding = F.dropout(entity_embedding, p=self.dropout_prob, training=self.training)
        entity_agg = self.KGAT(entity_embedding, neigh_entity_embedding, neigh_relation_embedding)
        entity_inter = self.tanh(self.layernorm_entity(self.fc4(entity_agg)))
        entity_inter = self.GCN(entity_inter)
        entity_inter = F.dropout(entity_inter, p=self.dropout_prob, training=self.training)
        entity_rep = self.tanh(self.entity_attention(entity_inter))
        entity_rep = F.dropout(entity_rep, p=self.dropout_prob, training=self.training)

        # 语义级-新闻表征
        new_rep = torch.cat([word_rep.unsqueeze(1), entity_rep.unsqueeze(1),
                             category_rep.unsqueeze(1), subcategory_rep.unsqueeze(1)], dim=1)
        new_rep = self.new_attention(new_rep)
        new_rep = self.layernorm4(new_rep)

        # 通道级-新闻表征
        relation = torch.tanh(self.layernorm3(self.fc5(entity_agg)))
        # relation = F.dropout(relation, p=self.dropout_prob, training=self.training)
        relation = self.relation_attention(new_rep, relation)
        relation_rep = F.dropout(relation, p=self.dropout_prob, training=self.training)
        new_rep = self.layernorm4(relation_rep)

        # 融合
        new_rep = torch.stack([new_rep, relation_rep], dim = 1)
        return new_rep

class user_encoder(torch.nn.Module):
    def __init__(self, word_dim, attention_dim, attention_heads, query_vector_dim, entity_size, entity_embedding_dim,
                 category_dim, subcategory_dim, category_size, subcategory_size):
        super(user_encoder, self).__init__()
        self.multi_dim = attention_dim * attention_heads
        self.type_encoder = 'userencoder'
        self.new_encoder = new_encoder(word_dim, attention_dim, attention_heads, query_vector_dim, entity_size,
                                       entity_embedding_dim,
                                       category_dim, subcategory_dim, category_size, subcategory_size)
        self.multiheadatt = MultiHeadSelfAttention_2(attention_dim * attention_heads, attention_dim * attention_heads,
                                                     attention_heads)
        self.multiheadatt2 = MultiHeadSelfAttention_2(attention_dim * attention_heads, attention_dim * attention_heads,
                                                     attention_heads)
        self.user_attention = Additive_Attention(query_vector_dim, self.multi_dim)
        self.layernorm1 = nn.LayerNorm(self.multi_dim)
        self.layernorm2 = nn.LayerNorm(self.multi_dim)
        self.layernorm3 = nn.LayerNorm(self.multi_dim)
        self.dropout_prob = 0.2

    def forward(self, word_embedding, entity_embedding,
                neigh_entity_embedding, neigh_relation_embedding,
                category_index, subcategory_index):
        # 点击新闻表征
        new_rep = self.new_encoder(word_embedding, entity_embedding,
                                   neigh_entity_embedding, neigh_relation_embedding,
                                   category_index, subcategory_index).unsqueeze(0)
        new_rep = F.dropout(new_rep, p=self.dropout_prob, training=self.training)

        # 点击新闻语义
        new_embedding = new_rep[:, :, 0, :]
        # 点击关系语义
        relation_embedding = new_rep[:, :, 1, :]

        # 语义级交互
        new_rep = self.multiheadatt(new_embedding)
        new_rep = F.dropout(new_rep, p=self.dropout_prob, training=self.training)
        new_rep = self.layernorm1(new_rep)

        # 关系级交互
        relation_rep = self.multiheadatt2(relation_embedding)
        relation_rep = F.dropout(relation_rep, p=self.dropout_prob, training=self.training)
        relation_rep = self.layernorm2(relation_rep)

        # 用户表征
        new_rep = torch.cat([new_rep, relation_rep], dim=0)
        user_rep = self.user_attention(new_rep)
        user_rep = F.dropout(user_rep, p=self.dropout_prob, training=self.training)
        user_rep = self.layernorm3(user_rep)
        #print(user_rep.shape)
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
                candidate_new_subcategory_index, user_clicked_new_subcategory_index, user_index
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
                                           new_category_index, new_subcategory_index
                                           ).unsqueeze(1)
            if i == 0:
                new_rep = new_rep_one
            else:
                new_rep = torch.cat([new_rep, new_rep_one], dim = 1)
        #print(new_rep.shape)
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
                                             clicked_new_category_index, clicked_new_subcategory_index
                                             ).unsqueeze(0)
            if i == 0:
                user_rep = user_rep_one
            else:
                user_rep = torch.cat([user_rep, user_rep_one], dim=0)
        sorce = torch.sum(torch.sum(new_rep * user_rep.unsqueeze(1), dim = -1), dim = -1)
        return sorce

    def loss(self, candidate_new_word_embedding,  user_clicked_word_embedding,
             candidate_new_entity_embedding, user_clicked_new_entity_embedding,
             candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding,
             candidate_new_neigh_relation_embedding, user_clicked_new_neigh_relation_embedding,
             candidate_new_category_index, user_clicked_new_category_index,
             candidate_new_subcategory_index, user_clicked_new_subcategory_index,
             label,user_index = None):
        label = label.to(device)
        sorce = self.forward(candidate_new_word_embedding, user_clicked_word_embedding,
                             candidate_new_entity_embedding, user_clicked_new_entity_embedding,
                             candidate_new_neigh_entity_embedding,user_clicked_new_neigh_entity_embedding,
                             candidate_new_neigh_relation_embedding, user_clicked_new_neigh_relation_embedding,
                             candidate_new_category_index, user_clicked_new_category_index,
                             candidate_new_subcategory_index, user_clicked_new_subcategory_index, user_index)
        loss = torch.nn.functional.cross_entropy(sorce, torch.argmax(label, dim =1))
        try:
            auc = roc_auc_score(label.cpu().numpy(), F.softmax(sorce.cpu(), dim=1).detach().numpy())
        except ValueError:
            auc = 0.5
        return loss, auc