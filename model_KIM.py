import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *
from sklearn.metrics import roc_auc_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Knowledge_co_encoder(torch.nn.Module):
    def __init__(self, entity_num, neigh_num, entity_dim):
        super(Knowledge_co_encoder, self).__init__()
        self.entity_num = entity_num
        self.neigh_num = neigh_num
        self.entity_dim = entity_dim
        # gat
        self.gat = GAT(self.entity_dim, self.entity_dim)
        self.gat_fc = nn.Linear(2 * self.entity_dim, self.entity_dim, bias = True)
        # GCAT
        self.gcat = GraphCoAttNet(self.entity_dim, self.entity_dim)
        self.gcat_fc = nn.Linear(2 * self.entity_dim, self.entity_dim, bias = True)
        # co-attention
        self.co_att = GraphCoAttNet(self.entity_dim, self.entity_dim)
    def forward(self, clicked_entity, clicked_onehop, new_entity, new_onehop):
        # 模型中第一部分：gat
        new_can = self.gat(new_onehop) # b, 50, max_num, 100
        new_can = torch.cat([new_can,new_entity], dim = -1)
        new_can = self.gat_fc(new_can).unsqueeze(-2).repeat(1, 1, 1,self.neigh_num, 1)
        user_can = self.gat(clicked_onehop) # b, 50, max_num, 100
        user_can = torch.cat([user_can, clicked_entity], dim=-1)
        user_can = self.gat_fc(user_can).unsqueeze(-2).repeat(1, 1, 1,self.neigh_num, 1)
        # 模型中第二部分：GCAT
        news_entity_onehop = self.gcat(new_onehop, user_can)  # b, 50, max_num, 100
        user_entity_onehop = self.gcat(clicked_onehop, new_can)  # b, 50, max_num, 100
        new_entity_vecs = self.gcat_fc(torch.cat([new_entity, news_entity_onehop], dim = -1)) # b, 50, max_num, 100
        user_entity_vecs =self.gcat_fc(torch.cat([clicked_entity, user_entity_onehop], dim=-1)) # b, 50, max_num, 100
        # 模型中第三部分： entity co-attention network
        new_entity_vecs = self.co_att(new_entity_vecs, clicked_entity)
        user_entity_vec = self.co_att(user_entity_vecs, new_entity)
        return new_entity_vecs, user_entity_vec

class Semantic_Co_Encoder(torch.nn.Module):
    def __init__(self, word_dim, word_num, attention_dim, attention_heads, query_vector_dim):
        super(Semantic_Co_Encoder, self).__init__()
        self.word_num = word_num
        self.attention_dim = attention_dim * attention_heads
        self.multiheadatt_new = MultiHeadSelfAttention_2(word_dim, self.attention_dim, attention_heads)
        self.multiheadatt_clicked = MultiHeadSelfAttention_3(word_dim, self.attention_dim, attention_heads)
        # self.new_att = Additive_Attention(query_vector_dim, self.attention_dim)
        # self.clicked_att = Additive_Attention(query_vector_dim, self.attention_dim)
        self.new_att_1 = nn.Linear(self.attention_dim, 200, bias = True)
        self.new_att_2 = nn.Linear(200, 1, bias = True)
        self.clicked_att_1 = nn.Linear(self.attention_dim, 200, bias=True)
        self.clicked_att_2 = nn.Linear(200, 1, bias=True)
        self.get_agg = get_agg(self.word_num)
        self.get_context_aggergator = get_context_aggergator(self.attention_dim)

    def forward(self, new_title, clicked_title):
        batch_size = new_title.size(0)
        # 新闻编码器
        new_title = self.multiheadatt_new(new_title) #b, max_word, 300
        clicked_title = self.multiheadatt_clicked(clicked_title)  # b, 50, max_word, 400

        # 计算候选新闻自身注意力
        new_title_att_vecs = torch.tanh(self.new_att_1(new_title)) #b, max_word, 200
        new_title_att0 = self.new_att_2(new_title_att_vecs) #b, max_word, 1
        new_title_att = new_title_att0.squeeze().unsqueeze(1).repeat(1,50,1) #b, 50, max_word

        # 计算点击新闻自身注意力
        clicked_title_att_vecs = torch.tanh(self.clicked_att_1(clicked_title)) #b, 50, max_word, 200
        clicked_title_att = self.clicked_att_2(clicked_title_att_vecs).squeeze()   #b, 50, max_word

        # 计算候选交叉注意力
        clicked_title_att_vecs = torch.flatten(clicked_title_att_vecs, 1, 2)  #b, 50*max_word, 200
        new_title_att_vecs = torch.transpose(new_title_att_vecs, 2, 1)  #b, 200, max_word
        cross_att = torch.matmul(clicked_title_att_vecs, new_title_att_vecs)  #b, 50*max_word, max_word
        cross_att_candi = F.softmax(cross_att, dim=-1)  #b, 50*max_word, max_word
        cross_att_candi = 0.001 * torch.reshape(torch.matmul(cross_att_candi, new_title_att0), (-1, 50, self.word_num))#b, 50, max_word

        # 计算点击注意力（自身注意力加交叉注意力）
        clicked_title_att = torch.add(clicked_title_att, cross_att_candi)
        clicked_title_att = F.softmax(clicked_title_att, dim=-1) #b, 50, max_word

        # 计算点击交叉注意力
        cross_att_click = torch.transpose(torch.reshape(cross_att, (batch_size, 50, self.word_num, self.word_num )), -1, -2) #(bz,#click,#candi_word,#click_word,)
        clicked_title_att_re = clicked_title_att.unsqueeze(2) #(bz,#click,1,#click_word,)
        cross_att_click_vecs = torch.cat([cross_att_click, clicked_title_att_re] , dim = -2) #(bz,#click,#candi_word+1,#click_word,)
        cross_att_click = self.get_agg(cross_att_click_vecs)

        # 计算候选注意力（自身注意力加交叉注意力）
        new_title_att = torch.add(new_title_att, cross_att_click)
        new_title_att = F.softmax(new_title_att, dim = -1) #b, 50, max_word
        new_title_vecs = torch.matmul(new_title_att, new_title) #b, 50, 400

        clicked_title_att = clicked_title_att.unsqueeze(-1)
        clicked_title_word_vecs_att = torch.cat([clicked_title,clicked_title_att], dim =-1)
        clicked_title_vecs = self.get_context_aggergator(clicked_title_word_vecs_att) #b, 50, 400

        return new_title_vecs, clicked_title_vecs

class News_User_Co_Encoder(torch.nn.Module):
    def __init__(self, input_dim):
        super(News_User_Co_Encoder, self).__init__()
        self.news_att_1 = nn.Linear(input_dim, 100, bias = True)
        self.news_att_2 = nn.Linear(100, 1, bias = True)
        self.user_att_1 = nn.Linear(input_dim, 100, bias = True)
        self.user_att_2 = nn.Linear(100, 1, bias = True)
        self.co_att = nn.Linear(input_dim, 100, bias = True)
        self.norm1 = nn.LayerNorm(400)
        self.norm2 = nn.LayerNorm(400)
    def forward(self, news_vecs, user_vecs):
        # 计算自身注意力
        news_att = self.news_att_2(torch.tanh(self.news_att_1(news_vecs)))
        user_att = self.user_att_2(torch.tanh(self.user_att_1(user_vecs))) #(bz,50,1)
        # 计算交叉注意力
        cross_news_vecs = self.co_att(news_vecs) #(bz,50,100)
        cross_user_vecs = self.co_att(user_vecs) #(bz,50,100)
        cross_att = torch.matmul(cross_news_vecs, torch.transpose(cross_user_vecs, -1, -2))
        # 计算用户交叉注意力（自身注意力加交叉注意力）
        cross_user_att = F.softmax(cross_att, dim = -1) #(bz,50,50)
        cross_user_att = 0.01 * torch.matmul(cross_user_att, news_att) #(bz,50,1)
        user_att = F.softmax(torch.add(user_att,cross_user_att), dim = -1) #(bz,50,1)
        # 计算新闻交叉注意力（自身注意力加交叉注意力）
        cross_news_att = F.softmax(torch.transpose(cross_att,-1,-2) , dim = -1)  # (bz,50,50)
        cross_news_att = 0.01 * torch.matmul(cross_news_att, user_att)  # (bz,50,1)
        news_att = F.softmax(torch.add(news_att, cross_news_att), dim = -1)  # (bz,50,1)
        # 计算新闻向量和用户向量
        news_vec = torch.matmul(torch.transpose(news_att, -1, -2), news_vecs)
        news_vec = self.norm1(news_vec)
        user_vec = torch.matmul(torch.transpose(user_att, -1, -2), user_vecs)
        user_vec = self.norm2(user_vec)

        # 计算得分
        sorce = torch.sum(news_vec * user_vec, dim = 2) # (bz,50,1)
        return sorce

class KIM(torch.nn.Module):
    def __init__(self, args):
        super(KIM, self).__init__()
        self.max_clicked = 50
        self.sample_num = 5
        self.entity_num = 5
        self.word_num = 23
        self.neigh_num = 5
        self.entity_dim = 100
        self.word_dim = 300
        self.attention_dim = 20
        self.attention_heads = 20
        self.query_vector_dim = 200

        self.knowledge_co_encoder = Knowledge_co_encoder(self.entity_num, self.neigh_num,  self.entity_dim)
        self.semantic_co_encoder = Semantic_Co_Encoder(self.word_dim, self.word_num, self.attention_dim, self.attention_heads, self.query_vector_dim)
        self.user_new_co_encoder = News_User_Co_Encoder(self.attention_dim * self.attention_heads )
        self.MergeLayer = nn.Linear(500, 400, bias=True)
    def forward(self,candidate_new_word_embedding, user_clicked_new_word_embedding,
                candidate_new_entity_embedding,  user_clicked_new_entity_embedding,
                candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding,):
        candidate_new_word_embedding = candidate_new_word_embedding.to(device)
        user_clicked_new_word_embedding = user_clicked_new_word_embedding.to(device)
        candidate_new_entity_embedding = candidate_new_entity_embedding.to(device)
        user_clicked_new_entity_embedding = user_clicked_new_entity_embedding.to(device)
        print(candidate_new_entity_embedding.shape)
        # 处理新闻单词
        score = None
        # 处理新闻实体
        for i in range(self.sample_num):
            new_title = candidate_new_word_embedding[:,i,:,:]
            clicked_title = user_clicked_new_word_embedding
            new_title_vecs, clicked_title_vecs = self.semantic_co_encoder(new_title, clicked_title)
            new_entity = candidate_new_entity_embedding[:,i,:,:].unsqueeze(1).repeat(1,50,1,1)
            new_onehop = candidate_new_neigh_entity_embedding[:,i,:,:,:].unsqueeze(1).repeat(1,50,1, 1,1)
            clicked_onehop = user_clicked_new_neigh_entity_embedding
            clicked_entity = user_clicked_new_entity_embedding
            news_entity_vecs, user_entity_vecs = self.knowledge_co_encoder(clicked_entity, clicked_onehop, new_entity, new_onehop)
            news_vecs = torch.cat([new_title_vecs,news_entity_vecs], dim = -1)
            news_vecs = self.MergeLayer(news_vecs)
            user_vecs = torch.cat([clicked_title_vecs, user_entity_vecs], dim=-1)
            user_vecs = self.MergeLayer(user_vecs)
            score_one = self.user_new_co_encoder(news_vecs, user_vecs)
            print(score_one.shape)
            if i == 0:
                score = score_one
            else:
                score = torch.cat([score, score_one], dim = -1)
        # # 语义协同编码器
        # new_title = torch.flatten(candidate_new_word_embedding, 0, 1)
        # clicked_title = user_clicked_new_word_embedding.repeat(5, 1, 1, 1)
        # new_title_vecs, clicked_title_vecs = self.semantic_co_encoder(new_title, clicked_title)
        # # 实体协同编码器
        # candidate_new_entity_embedding = torch.flatten(candidate_new_entity_embedding, 0, 1)
        # new_entity = candidate_new_entity_embedding.unsqueeze(1).repeat(1, 50, 1, 1)
        # candidate_new_neigh_entity_embedding = torch.flatten(candidate_new_neigh_entity_embedding, 0, 1)
        # new_onehop = candidate_new_neigh_entity_embedding.unsqueeze(1).repeat(1, 50, 1, 1, 1)
        # clicked_onehop = user_clicked_new_neigh_entity_embedding.repeat(5, 1, 1, 1, 1)
        # clicked_entity = user_clicked_new_entity_embedding.repeat(5, 1, 1, 1)
        # news_entity_vecs, user_entity_vecs = self.knowledge_co_encoder(clicked_entity, clicked_onehop, new_entity, new_onehop)
        # # 用户新闻协同编码器
        # news_vecs = torch.cat([new_title_vecs,news_entity_vecs], dim = -1)
        # news_vecs = self.MergeLayer(news_vecs)
        # user_vecs = torch.cat([clicked_title_vecs, user_entity_vecs], dim=-1)
        # user_vecs = self.MergeLayer(user_vecs)
        # score = torch.reshape(self.user_new_co_encoder(news_vecs, user_vecs),(50, 5))
        # print(score.shape)
        return score

    def loss(self, candidate_new_word_embedding,  user_clicked_word_embedding,
             candidate_new_entity_embedding, user_clicked_new_entity_embedding,
             candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding, label):
        label = label.to(device)

        sorce = self.forward(candidate_new_word_embedding, user_clicked_word_embedding,
                             candidate_new_entity_embedding, user_clicked_new_entity_embedding,
                             candidate_new_neigh_entity_embedding,user_clicked_new_neigh_entity_embedding)
        loss = torch.nn.functional.cross_entropy(sorce, torch.argmax(label, dim =1))
        try:
            auc = roc_auc_score(label.cpu().numpy(), F.softmax(sorce.cpu(), dim = 1).detach().numpy())
        except ValueError:
            auc = 0.5
        return loss, auc
