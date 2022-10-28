import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *
from sklearn.metrics import roc_auc_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class new_encoder(torch.nn.Module):
    def __init__(self, word_dim, attention_dim, attention_heads, query_vector_dim):
        super(new_encoder, self).__init__()
        self.multiheadatt = MultiHeadSelfAttention_2(word_dim, attention_dim * attention_heads, attention_heads)
        self.multi_dim = attention_dim * attention_heads
        self.norm1 = nn.LayerNorm(self.multi_dim)
        # self.word_attention = Attention(self.multi_dim)
        self.word_attention = Additive_Attention(query_vector_dim, self.multi_dim)
        self.norm2 = nn.LayerNorm(self.multi_dim)
        self.dropout_prob = 0.2

    def forward(self, word_embedding):
        #word_embedding = F.dropout(word_embedding, p=self.dropout_prob, training=self.training)
        word_embedding = self.multiheadatt(word_embedding)
        word_embedding = self.norm1(word_embedding)
        word_embedding = F.dropout(word_embedding, p=self.dropout_prob, training=self.training)
        new_rep = self.word_attention(word_embedding)
        new_rep = F.dropout(new_rep, p=self.dropout_prob, training=self.training)
        new_rep = self.norm2(new_rep)
        return new_rep

class user_encoder(torch.nn.Module):
    def __init__(self,  word_dim, attention_dim, attention_heads,query_vector_dim):
        super(user_encoder, self).__init__()
        self.new_encoder = new_encoder(word_dim, attention_dim, attention_heads, query_vector_dim)
        self.multiheadatt = MultiHeadSelfAttention_2(attention_dim * attention_heads, attention_dim * attention_heads,
                                                     attention_heads)
        self.multi_dim = attention_dim * attention_heads
        self.user_attention = Additive_Attention(query_vector_dim, self.multi_dim)
        self.norm = nn.LayerNorm(self.multi_dim)
        self.dropout_prob = 0.2
    def forward(self, word_embedding):
        new_rep = self.new_encoder(word_embedding).unsqueeze(0)
        new_rep = F.dropout(new_rep, p=self.dropout_prob, training=self.training)
        new_rep = self.multiheadatt(new_rep)
        new_rep = F.dropout(new_rep, p=self.dropout_prob, training=self.training)
        user_rep = self.user_attention(new_rep)
        user_rep = self.norm(user_rep)
        return user_rep

class NRMS(torch.nn.Module):
    def __init__(self, args):
        super(NRMS, self).__init__()
        self.word_dim = args.word_embedding_dim
        self.attention_dim = args.attention_dim
        self.attention_heads = args.attention_heads
        self.sample_num = args.sample_num
        self.user_clicked_new_num = args.user_clicked_new_num
        self.query_vector_dim = args.query_vector_dim
        self.dropout_prob = 0.2
        self.new_encoder = new_encoder(self.word_dim, self.attention_dim, self.attention_heads, self.query_vector_dim)
        self.user_encoder = user_encoder(self.word_dim, self.attention_dim, self.attention_heads, self.query_vector_dim)

    def forward(self, new_word_embedding, clicked_word_embedding):
        new_word_embedding = new_word_embedding.to(device)
        clicked_word_embedding = clicked_word_embedding.to(device)
        ## 新闻编码器
        new_rep = None
        for i in range(self.sample_num):
            title_word_embedding_one = new_word_embedding[:, i, :]
            new_rep_one = self.new_encoder(title_word_embedding_one)
            new_rep_one = new_rep_one.unsqueeze(1)
            if i==0:
                new_rep = new_rep_one
            else:
                new_rep = torch.cat([new_rep, new_rep_one], dim = 1)
        # 用户编码器
        user_rep = None
        for i in range(self.user_clicked_new_num):
            clicked_word_embedding_one = clicked_word_embedding[i, :, :, :]
            clicked_word_embedding_one = clicked_word_embedding_one.squeeze()
            user_rep_one = self.user_encoder(clicked_word_embedding_one).unsqueeze(0)
            if i == 0:
                user_rep = user_rep_one
            else:
                user_rep = torch.cat([user_rep, user_rep_one], dim=0)
        sorce = torch.sum(new_rep * user_rep, dim=2)
        return sorce

    def loss(self, new_word_embedding, clicked_word_embedding, label):
        label = label.to(device)
        sorce = self.forward(new_word_embedding, clicked_word_embedding)
        loss = torch.nn.functional.cross_entropy(sorce, torch.argmax(label, dim =1))
        try:
            auc = roc_auc_score(label.cpu().numpy(), F.softmax(sorce.cpu(), dim=1).detach().numpy())
        except ValueError:
            auc = 0.5
        return loss, auc
