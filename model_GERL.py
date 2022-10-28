import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *
from sklearn.metrics import roc_auc_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class GERL(torch.nn.Module):
    def __init__(self, args):
        super(GERL, self).__init__()
        self.word_dim = args.word_embedding_dim
        self.user_num = args.user_size + 1
        self.news_num = args.title_size
        self.attention_dim = args.attention_dim
        self.attention_heads = args.attention_heads
        self.sample_num = args.sample_num
        self.max_one_hop = args.max_one_hop
        self.max_two_hop = args.max_two_hop
        self.dropout_prob = 0.2

        self.title_dim = self.attention_dim * self.attention_heads
        self.embedding_dim = self.title_dim
        self.user_embedding = nn.Embedding(self.user_num, self.embedding_dim)
        self.news_embedding = nn.Embedding(self.news_num, self.embedding_dim)
        # 用户layer
        self.user_two_hop_attend_id = Self_Attention(self.embedding_dim, self.embedding_dim)
        self.multiheadatt_user = MultiHeadSelfAttention_3(self.word_dim, self.title_dim, self.attention_heads)
        self.att_user = Self_Attention(self.title_dim, self.title_dim)
        self.user_one_hop_attend = Self_Attention(self.embedding_dim, self.embedding_dim)
        # 新闻layer
        self.multiheadatt_news = MultiHeadSelfAttention_3(self.word_dim, self.title_dim, self.attention_heads)
        self.att_news = Self_Attention(self.title_dim, self.title_dim)
        self.news_title_attend = Self_Attention(self.embedding_dim, self.embedding_dim)
        self.news_two_hop_id_attend = Self_Attention(self.embedding_dim, self.embedding_dim)
        self.multiheadatt_news_twohop = MultiHeadSelfAttention_3(self.word_dim, self.title_dim, self.attention_heads)
        self.att_news_twohop = Self_Attention(self.title_dim, self.title_dim)
        self.news_two_hop_title_attend = Self_Attention(self.embedding_dim, self.embedding_dim)

        self.norm1 = nn.LayerNorm(self.title_dim)
        self.norm2 = nn.LayerNorm(self.title_dim)
        self.norm3 = nn.LayerNorm(self.title_dim)

    def forward(self, user_index, user_one_hop_new_word_embedding, user_two_hop_neighbor,
             candidate_new_word_embedding, new_two_hop_neighbor, new_two_hop_neighbor_new_word_embedding,):
        user_index = user_index.to(device) # bz, 1
        user_one_hop_new_word_embedding = user_one_hop_new_word_embedding.to(device) # bz, max_one_hop, max_seq, word_dim
        user_two_hop_neighbor = user_two_hop_neighbor.to(device)  # bz, max_two_hop
        candidate_new_word_embedding = candidate_new_word_embedding.to(device) # bz, sample_num, max_seq, word_dim
        new_two_hop_neighbor = new_two_hop_neighbor.to(device)   # bz, sample_num, max_two_hop
        new_two_hop_neighbor_new_word_embedding = new_two_hop_neighbor_new_word_embedding.to(device)  # bz, sample_num, max_two_hop, max_seq, word_dim

        # 用户嵌入层
        user_embedding = self.user_embedding(user_index.to(torch.int64)).squeeze()
        user_embedding = F.dropout(user_embedding, p = self.dropout_prob, training = self.training) # bz, 1 , embedding_dim
        # 用户两跳嵌入
        neighbor_user_embedding = self.user_embedding(user_two_hop_neighbor.to(torch.int64))
        neighbor_user_embedding = F.dropout(neighbor_user_embedding, p=self.dropout_prob, training=self.training)  # bz, max_two_hop , embedding_dim
        # 新闻两跳嵌入
        neighbor_news_embedding = self.news_embedding(new_two_hop_neighbor.to(torch.int64))
        neighbor_news_embedding = F.dropout(neighbor_news_embedding, p=self.dropout_prob, training=self.training)  # bz, sample_num, max_two_hop , embedding_dim
        # 用户
        # 用户两跳id
        user_two_hop_rep = self.user_two_hop_attend_id(neighbor_user_embedding) # bz, embedding_dim
        # 用户一跳title
        user_one_hop_new = self.multiheadatt_user(user_one_hop_new_word_embedding)  # bz, max_seq, embedding_dim
        user_one_hop_new = self.norm1(user_one_hop_new)
        user_one_hop_new = self.att_user(torch.flatten(user_one_hop_new, 0, 1)).view(-1, self.max_one_hop, self.title_dim)
        user_one_hop_rep = self.user_one_hop_attend(user_one_hop_new)
        user_one_hop_rep = F.dropout(user_one_hop_rep, p=self.dropout_prob, training=self.training)
        # 最终用户表示
        final_user_rep = user_embedding + user_two_hop_rep + user_one_hop_rep # bz, embedding_dim
        # 新闻
        # 候选新闻
        candidate_new_word = self.multiheadatt_news(candidate_new_word_embedding)
        candidate_new_word = self.norm2(candidate_new_word)
        candidate_new_word = self.att_news(torch.flatten(candidate_new_word, 0, 1)).view(-1, self.sample_num, self.title_dim) # bz, sample_num, embedding_dim
        candidate_new_word = F.dropout(candidate_new_word, p=self.dropout_prob, training=self.training)
        # 两跳新闻id
        neighbor_news_embedding = neighbor_news_embedding.view(-1, self.max_two_hop, self.embedding_dim)
        news_two_hop_id_rep = self.news_two_hop_id_attend(neighbor_news_embedding)
        news_two_hop_id_rep = news_two_hop_id_rep.view(-1, self.sample_num, self.embedding_dim)  # bz, sample_num, embedding_dim
        # 两跳新闻title
        new_two_hop_neighbor_new_word_embedding = torch.flatten(new_two_hop_neighbor_new_word_embedding, 0, 1)
        news_two_hop_rep = self.multiheadatt_news_twohop(new_two_hop_neighbor_new_word_embedding)
        news_two_hop_rep = self.norm3(news_two_hop_rep)
        news_two_hop_rep = self.att_news_twohop(torch.flatten(news_two_hop_rep, 0, 1)).view(-1, self.max_two_hop, self.title_dim)
        news_two_hop_rep = self.news_two_hop_title_attend(news_two_hop_rep).view(-1, self.sample_num, self.title_dim)
        news_two_hop_rep = F.dropout(news_two_hop_rep, p=self.dropout_prob, training=self.training)
        # 最终新闻表示
        final_new_rep = candidate_new_word + news_two_hop_id_rep + news_two_hop_rep
        final_user_rep = final_user_rep.unsqueeze(1).repeat(1,self.sample_num,1)
        sorce = torch.sum(final_new_rep * final_user_rep, dim=2)
        return sorce

    def loss(self, user_index, user_one_hop_new_word_embedding, user_two_hop_neighbor,
             candidate_new_word_embedding, new_one_hop_neighbor, new_two_hop_neighbor_new_word_embedding, label):
        label = label.to(device)
        sorce = self.forward(user_index, user_one_hop_new_word_embedding, user_two_hop_neighbor,
                             candidate_new_word_embedding, new_one_hop_neighbor, new_two_hop_neighbor_new_word_embedding,)
        loss = torch.nn.functional.cross_entropy(sorce, torch.argmax(label, dim =1))
        try:
            auc = roc_auc_score(label.cpu().numpy(), F.softmax(sorce.cpu(), dim=1).detach().numpy())
        except ValueError:
            auc = 0.5
        return loss, auc
