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
        # # 主题级表征网络
        # self.category_net = torch.nn.Sequential(
        #     nn.Embedding(category_size, embedding_dim=category_dim),
        #     nn.Linear(category_dim, 100, bias=True ),
        #     nn.Tanh(),
        #     nn.Dropout(self.dropout_prob),
        #     )
        # # 副主题级表征网络
        # self.subcategory_net = torch.nn.Sequential(
        #     nn.Embedding(subcategory_size, embedding_dim=category_dim),
        #     nn.Linear(subcategory_dim, 100, bias=True),
        #     nn.Tanh(),
        #     nn.Dropout(self.dropout_prob)
        #     )
        # 单词级表征网络
        self.title_net = torch.nn.Sequential(
            nn.Dropout(self.dropout_prob),
            nn.Linear(768, 400, bias=True),
            nn.Tanh(),
            nn.Dropout(self.dropout_prob)
           )
        self.KGAT = KGAT(self.multi_dim, entity_embedding_dim)
        # # 实体级表征网络
        # self.entity_net = torch.nn.Sequential(
        #     nn.Linear(2 * entity_embedding_dim, 100, bias=True),
        #     nn.Tanh(),
        #     gcn(entity_size, 100, 100),
        #     nn.Dropout(self.dropout_prob),
        #     Additive_Attention(query_vector_dim, 100),
        #     nn.Tanh(),
        #     nn.Dropout(self.dropout_prob)
        # )
        # # 附加注意力网络
        # self.add_attention_net = torch.nn.Sequential(
        #     Additive_Attention(query_vector_dim, 100),
        # )


    def forward(self, title_embedding, entity_embedding,
                neigh_entity_embedding, neigh_relation_embedding,
                category_index, subcategory_index):
        # category_rep = self.category_net(category_index.to(torch.int64))
        # subcategory_rep = self.subcategory_net(subcategory_index.to(torch.int64))
        title_rep = self.title_net(title_embedding)
        # entity_agg = self.KGAT(entity_embedding, neigh_entity_embedding, neigh_relation_embedding)
        # entity_rep = self.entity_net(entity_agg)
        # 新闻附加注意力
        # new_rep = torch.cat(
        #     [title_rep.unsqueeze(1), entity_rep.unsqueeze(1), category_rep.unsqueeze(1), subcategory_rep.unsqueeze(1)],
        #     dim=1)
        # new_rep = self.add_attention_net(title_rep)
        new_rep  = title_rep
        return new_rep

class user_encoder(torch.nn.Module):
    def __init__(self, word_dim, attention_dim, attention_heads, query_vector_dim, entity_size, entity_embedding_dim,
                 category_dim, subcategory_dim, category_size, subcategory_size, A_size):
        super(user_encoder, self).__init__()
        self.new_encoder = new_encoder(word_dim, attention_dim, attention_heads, query_vector_dim, entity_size,
                                       entity_embedding_dim, category_dim, subcategory_dim, category_size, subcategory_size)
        self.multi_dim = attention_dim * attention_heads
        self.attention_heads = attention_heads
        self.dropout_prob = 0.2
        self.A_size = A_size
        # 异构图主题网络
        self.category_net = torch.nn.Sequential(
            nn.Embedding(category_size, embedding_dim=category_dim),
            nn.Linear(category_dim, 100, bias=True),
            nn.Linear(100, self.multi_dim, bias=True),
            nn.Tanh(),
            nn.Dropout(self.dropout_prob),
        )
        # 异构图实体网络
        self.entity_net = torch.nn.Sequential(
            nn.Linear(entity_embedding_dim, 100, bias=True),
            nn.Linear(100, self.multi_dim, bias=True),
            nn.Tanh(),
            nn.Dropout(self.dropout_prob),
        )
        attn_heads_reduction = 'mean'
        self.GraphAttention_basis = graphattention(self.multi_dim, attention_dim, attention_heads, attn_heads_reduction, self.dropout_prob)
        ################################### 第一次 graph_network ########################################
        # graphattention_n1
        self.GraphAttention_n1 = graphattention(attention_dim, 16, attention_heads, attn_heads_reduction, self.dropout_prob)
        self.fc_n1 = nn.Linear(self.A_size, 50, bias=True)
        # graphattention_t1
        self.GraphAttention_t1 = graphattention(attention_dim, 3, attention_heads, attn_heads_reduction, self.dropout_prob)
        self.fc_t1 = nn.Linear(self.A_size, category_size, bias=True)
        # graphattention_e1
        self.GraphAttention_e1 = graphattention(attention_dim, 9, attention_heads, attn_heads_reduction, self.dropout_prob)
        self.fc_e1 = nn.Linear(self.A_size, 50, bias=True)
        ################################### 第二次 graph_network ########################################
        # graphattention_n2
        self.GraphAttention_n2 = graphattention(attention_dim, 1, attention_heads, attn_heads_reduction, self.dropout_prob)
        self.fc_n2 = nn.Linear(16 + 3 + 9, 16, bias=True)
        # graphattention_t1
        self.GraphAttention_t2 = graphattention(attention_dim, 1, attention_heads, attn_heads_reduction, self.dropout_prob)
        self.fc_t2 = nn.Linear(16 + 3 + 9, 3, bias=True)
        # graphattention_e1
        self.GraphAttention_e2 = graphattention(attention_dim, 1, attention_heads, attn_heads_reduction, self.dropout_prob)
        self.fc_e2 = nn.Linear(16 + 3 + 9, 9, bias=True)

        # 输出最终
        self.diff_pool = graphattention(attention_dim, 1, attention_heads, attn_heads_reduction = 'concat', dropout_prob = self.dropout_prob)


    def forward(self, title_embedding, entity_embedding,
                neigh_entity_embedding, neigh_relation_embedding,
                category_index, subcategory_index,
                heterogeneous_user_graph_A, heterogeneous_user_graph_categoryindex,
                heterogeneous_user_graph_entity_embedding ):
        # 点击新闻表征
        new_rep = self.new_encoder(title_embedding, entity_embedding,
                                   neigh_entity_embedding, neigh_relation_embedding,
                                   category_index, subcategory_index).unsqueeze(0)
        new_rep = F.dropout(new_rep, p=self.dropout_prob, training=self.training)

        # 点击新闻主题表征
        topic_rep = self.category_net(heterogeneous_user_graph_categoryindex.to(torch.int64)).unsqueeze(0)
        # 点击新闻实体表征
        entity_rep = self.entity_net(heterogeneous_user_graph_entity_embedding).unsqueeze(0)
        feature = torch.cat([new_rep, topic_rep, entity_rep], dim=1)
        graph_attention_basis = self.GraphAttention_basis(feature, heterogeneous_user_graph_A)
        ############################### 第一次池化 ###############################################
        # 点击新闻
        graph_pool_n1 = self.GraphAttention_n1(graph_attention_basis, heterogeneous_user_graph_A)
        graph_pool_n1 = F.softmax(torch.transpose(self.fc_n1(torch.transpose(graph_pool_n1, 2, 1)), 2, 1), dim=1)
        graph_pool_n1 = torch.cat(
            [graph_pool_n1, torch.zeros_like(topic_rep[:, :, : 16]), torch.zeros_like(entity_rep[:, :, :16])], dim=1)
        Apool_nn1 = torch.bmm(torch.transpose(graph_pool_n1, 2, 1),
                              torch.bmm(heterogeneous_user_graph_A, graph_pool_n1))
        xpool_n1 = torch.bmm(torch.transpose(graph_pool_n1, 2, 1), graph_attention_basis)
        # 主题
        graph_pool_t1 = self.GraphAttention_t1(graph_attention_basis, heterogeneous_user_graph_A)
        graph_pool_t1 = F.softmax(torch.transpose(self.fc_t1(torch.transpose(graph_pool_t1, 2, 1)), 2, 1), dim=1)
        graph_pool_t1 = torch.cat(
            [torch.zeros_like(new_rep[:, :, :3]), graph_pool_t1, torch.zeros_like(entity_rep[:, :, :3])], dim=1)
        Apool_tt1 = torch.bmm(torch.transpose(graph_pool_t1, 2, 1),
                              torch.bmm(heterogeneous_user_graph_A, graph_pool_t1))
        xpool_t1 = torch.bmm(torch.transpose(graph_pool_t1, 2, 1), graph_attention_basis)
        # 实体
        graph_pool_e1 = self.GraphAttention_e1(graph_attention_basis, heterogeneous_user_graph_A)
        graph_pool_e1 = F.softmax(torch.transpose(self.fc_e1(torch.transpose(graph_pool_e1, 2, 1)), 2, 1), dim=1)
        graph_pool_e1 = torch.cat(
            [torch.zeros_like(new_rep[:, :, :9]), torch.zeros_like(topic_rep[:, :, :9]), graph_pool_e1], dim=1)
        Apool_ee1 = torch.bmm(torch.transpose(graph_pool_e1, 2, 1),
                              torch.bmm(heterogeneous_user_graph_A, graph_pool_e1))
        xpool_e1 = torch.bmm(torch.transpose(graph_pool_e1, 2, 1), graph_attention_basis)
        # 构建池化后连接矩阵和特征向量矩阵
        Apool_tn1 = torch.bmm(torch.transpose(graph_pool_t1, 2, 1),
                              torch.bmm(heterogeneous_user_graph_A, graph_pool_n1))
        Apool_en1 = torch.bmm(torch.transpose(graph_pool_e1, 2, 1),
                              torch.bmm(heterogeneous_user_graph_A, graph_pool_n1))
        Apool_nt1 = torch.bmm(torch.transpose(graph_pool_n1, 2, 1),
                              torch.bmm(heterogeneous_user_graph_A, graph_pool_t1))
        Apool_et1 = torch.bmm(torch.transpose(graph_pool_e1, 2, 1),
                              torch.bmm(heterogeneous_user_graph_A, graph_pool_t1))
        Apool_ne1 = torch.bmm(torch.transpose(graph_pool_n1, 2, 1),
                              torch.bmm(heterogeneous_user_graph_A, graph_pool_e1))
        Apool_te1 = torch.bmm(torch.transpose(graph_pool_t1, 2, 1),
                              torch.bmm(heterogeneous_user_graph_A, graph_pool_e1))
        Ap1 = torch.cat([torch.cat([Apool_nn1, Apool_tn1, Apool_en1], dim=1),
                         torch.cat([Apool_nt1, Apool_tt1, Apool_et1], dim=1),
                         torch.cat([Apool_ne1, Apool_te1, Apool_ee1], dim=1)], dim=2)
        xp1 = torch.cat([xpool_n1, xpool_t1, xpool_e1], dim=1)
        ############################### 第二次池化 ###############################################
        # 点击新闻
        graph_pool_n2 = self.GraphAttention_n2(xp1, Ap1)
        graph_pool_n2 = F.softmax(torch.transpose(self.fc_n2(torch.transpose(graph_pool_n2, 2, 1)), 2, 1), dim=1)
        graph_pool_n2 = torch.cat(
            [graph_pool_n2, torch.zeros_like(xpool_t1[:, :, : 1]), torch.zeros_like(xpool_e1[:, :, :1])], dim=1)
        Apool_nn2 = torch.bmm(torch.transpose(graph_pool_n2, 2, 1), torch.bmm(Ap1, graph_pool_n2))
        xpool_n2 = torch.bmm(torch.transpose(graph_pool_n2, 2, 1), xp1)
        # 主题
        graph_pool_t2 = self.GraphAttention_t2(xp1, Ap1)
        graph_pool_t2 = F.softmax(torch.transpose(self.fc_t2(torch.transpose(graph_pool_t2, 2, 1)), 2, 1), dim=1)
        graph_pool_t2 = torch.cat(
            [torch.zeros_like(xpool_n1[:, :, :1]), graph_pool_t2, torch.zeros_like(xpool_e1[:, :, :1])], dim=1)
        Apool_tt2 = torch.bmm(torch.transpose(graph_pool_t2, 2, 1), torch.bmm(Ap1, graph_pool_t2))
        xpool_t2 = torch.bmm(torch.transpose(graph_pool_t2, 2, 1), xp1)
        # 实体
        graph_pool_e2 = self.GraphAttention_e2(xp1, Ap1)
        graph_pool_e2 = F.softmax(torch.transpose(self.fc_e2(torch.transpose(graph_pool_e2, 2, 1)), 2, 1), dim=1)
        graph_pool_e2 = torch.cat(
            [torch.zeros_like(xpool_n1[:, :, :1]), torch.zeros_like(xpool_t1[:, :, :1]), graph_pool_e2], dim=1)
        Apool_ee2 = torch.bmm(torch.transpose(graph_pool_e2, 2, 1), torch.bmm(Ap1, graph_pool_e2))
        xpool_e2 = torch.bmm(torch.transpose(graph_pool_e2, 2, 1), xp1)
        # 构建池化后连接矩阵和特征向量矩阵
        Apool_tn2 = torch.bmm(torch.transpose(graph_pool_t2, 2, 1), torch.bmm(Ap1, graph_pool_n2))
        Apool_en2 = torch.bmm(torch.transpose(graph_pool_e2, 2, 1), torch.bmm(Ap1, graph_pool_n2))
        Apool_nt2 = torch.bmm(torch.transpose(graph_pool_n2, 2, 1), torch.bmm(Ap1, graph_pool_t2))
        Apool_et2 = torch.bmm(torch.transpose(graph_pool_e2, 2, 1), torch.bmm(Ap1, graph_pool_t2))
        Apool_ne2 = torch.bmm(torch.transpose(graph_pool_n2, 2, 1), torch.bmm(Ap1, graph_pool_e2))
        Apool_te2 = torch.bmm(torch.transpose(graph_pool_t2, 2, 1), torch.bmm(Ap1, graph_pool_e2))
        Ap2 = torch.cat([torch.cat([Apool_nn2, Apool_tn2, Apool_en2], dim=1),
                         torch.cat([Apool_nt2, Apool_tt2, Apool_et2], dim=1),
                         torch.cat([Apool_ne2, Apool_te2, Apool_ee2], dim=1)], dim=2)
        xp2 = torch.cat([xpool_n2, xpool_t2, xpool_e2], dim=1)
        # 输出用户表示
        diff_pool = torch.transpose(F.softmax(self.diff_pool(xp2, Ap2), dim=1), 2, 1)
        user_rep = torch.flatten(torch.bmm(diff_pool, xp2), 1)
        return user_rep


class heter_graph_bert(torch.nn.Module):
    def __init__(self, args):
        super(heter_graph_bert, self).__init__()
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
                                         self.category_dim, self.subcategory_dim, self.category_size, self.subcategory_size, A_size = 115)

    def forward(self,candidate_new_title_bert, user_clicked_new_title_bert,
                candidate_new_entity_embedding,  user_clicked_new_entity_embedding,
                candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding,
                candidate_new_neigh_relation_embedding,user_clicked_new_neigh_relation_embedding,
                candidate_new_category_index, user_clicked_new_category_index,
                candidate_new_subcategory_index, user_clicked_new_subcategory_index,
                heterogeneous_user_graph_A, heterogeneous_user_graph_entity_embedding,
                heterogeneous_user_graph_categoryindex ):

        candidate_new_title_bert = candidate_new_title_bert.to(device)
        user_clicked_new_title_bert = user_clicked_new_title_bert.to(device)
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
        heterogeneous_user_graph_A = heterogeneous_user_graph_A.to(device)
        heterogeneous_user_graph_entity_embedding = heterogeneous_user_graph_entity_embedding.to(device)
        heterogeneous_user_graph_categoryindex = heterogeneous_user_graph_categoryindex.to(device)

        ## 新闻编码器
        new_rep = None
        for i in range(self.sample_num):
            new_title_embedding_one = candidate_new_title_bert[:, i]
            new_entity_embedding_one = candidate_new_entity_embedding[:, i]
            new_neigh_entity_embedding_one = candidate_new_neigh_entity_embedding[:, i, :, :, :]
            new_neigh_relation_embedding_one = candidate_new_neigh_relation_embedding[:, i, :, :, :]
            new_category_index = candidate_new_category_index[:, i]
            new_subcategory_index = candidate_new_subcategory_index[:, i]
            new_rep_one = self.new_encoder(new_title_embedding_one, new_entity_embedding_one,
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
            clicked_new_title_embedding_one = user_clicked_new_title_bert[i, :, :]
            # clicked_new_word_embedding_one = clicked_new_word_embedding_one.squeeze()
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
            ## 用于构建异构图
            user_graph_A = heterogeneous_user_graph_A[i, :, :].unsqueeze(0)
            user_graph_entity_embedding = heterogeneous_user_graph_entity_embedding[i, :, :]
            user_graph_categoryindex = heterogeneous_user_graph_categoryindex[i, :]
            # 用户表征
            user_rep_one = self.user_encoder(clicked_new_title_embedding_one, clicked_new_entity_embedding_one,
                                             clicked_new_neigh_entity_embedding_one,clicked_new_neigh_relation_embedding_one,
                                             clicked_new_category_index, clicked_new_subcategory_index, user_graph_A,
                                             user_graph_categoryindex, user_graph_entity_embedding).unsqueeze(0)
            if i == 0:
                user_rep = user_rep_one
            else:
                user_rep = torch.cat([user_rep, user_rep_one], dim=0)
        sorce = torch.sum(new_rep * user_rep, dim = 2)
        return sorce

    def loss(self, candidate_new_title_bert, user_clicked_new_title_bert,
             candidate_new_entity_embedding, user_clicked_new_entity_embedding,
             candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding,
             candidate_new_neigh_relation_embedding, user_clicked_new_neigh_relation_embedding,
             candidate_new_category_index, user_clicked_new_category_index,
             candidate_new_subcategory_index, user_clicked_new_subcategory_index,
             heterogeneous_user_graph_A, heterogeneous_user_graph_entity_embedding,
             heterogeneous_user_graph_categoryindex, label):
        label = label.to(device)
        sorce = self.forward(candidate_new_title_bert, user_clicked_new_title_bert,
                             candidate_new_entity_embedding, user_clicked_new_entity_embedding,
                             candidate_new_neigh_entity_embedding,user_clicked_new_neigh_entity_embedding,
                             candidate_new_neigh_relation_embedding, user_clicked_new_neigh_relation_embedding,
                             candidate_new_category_index, user_clicked_new_category_index,
                             candidate_new_subcategory_index, user_clicked_new_subcategory_index,
                             heterogeneous_user_graph_A, heterogeneous_user_graph_entity_embedding,
                             heterogeneous_user_graph_categoryindex)
        loss = torch.nn.functional.cross_entropy(sorce, torch.argmax(label, dim =1))
        try:
            auc = roc_auc_score(label.cpu().numpy(), F.softmax(sorce.cpu(), dim=1).detach().numpy())
        except ValueError:
            auc = 0.5
        return loss, auc
