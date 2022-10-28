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
        self.word_attention = Additive_Attention(query_vector_dim, self.multi_dim)
        self.norm2 = nn.LayerNorm(self.multi_dim)
        self.dropout_prob = 0.2
    def forward(self, word_embedding):
        word_embedding = F.dropout(word_embedding, p=self.dropout_prob, training=self.training)
        word_embedding = self.multiheadatt(word_embedding)
        word_embedding = self.norm1(word_embedding)
        word_embedding = F.dropout(word_embedding, p=self.dropout_prob, training=self.training)
        new_rep = self.word_attention(word_embedding)
        new_rep = self.norm2(new_rep)
        return new_rep

class user_encoder(torch.nn.Module):
    def __init__(self,  word_dim, attention_dim, attention_heads,query_vector_dim):
        super(user_encoder, self).__init__()
        self.new_encoder = new_encoder(word_dim, attention_dim, attention_heads, query_vector_dim)
        self.multiheadatt = MultiHeadSelfAttention_2(attention_dim * attention_heads, attention_dim * attention_heads, attention_heads)
        self.multi_dim = attention_dim * attention_heads
        self.user_attention = Additive_Attention_PENR(query_vector_dim, self.multi_dim)
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

class popularity_predictor(torch.nn.Module):
    def __init__(self, news_dim, ):
        super(popularity_predictor, self).__init__()
        self.predictor = nn.Linear(news_dim, 1, bias=True)
        self.loss_pop = nn.MSELoss()
    def forward(self, new_rep):
        popularity = torch.relu(self.predictor(new_rep))
        return popularity
    def loss(self, popularity, true_popularity):
        loss_pop = self.loss_pop(popularity.squeeze(), true_popularity)
        return loss_pop

class PENR(torch.nn.Module):
    def __init__(self, args):
        super(PENR, self).__init__()
        self.word_dim = args.word_embedding_dim
        self.attention_dim = args.attention_dim
        self.attention_heads = args.attention_heads
        self.sample_num = args.sample_num
        self.user_clicked_new_num = args.user_clicked_new_num
        self.query_vector_dim = args.query_vector_dim
        self.dropout_prob = 0.2
        self.u = 0.2
        self.grama = 0.3
        self.lamda = 0.2
        self.beta = 0.2
        self.biaffine_matrix = nn.Parameter(torch.Tensor(50, self.sample_num, self.attention_dim * self.attention_heads,self.attention_dim * self.attention_heads))
        self.biaffine_matrix_bias = nn.Parameter(torch.Tensor(50, self.sample_num, self.user_clicked_new_num, 1))
        nn.init.normal_(self.biaffine_matrix)
        nn.init.normal_(self.biaffine_matrix_bias)
        self.new_encoder = new_encoder(self.word_dim, self.attention_dim, self.attention_heads, self.query_vector_dim)
        self.user_encoder = user_encoder(self.word_dim, self.attention_dim, self.attention_heads, self.query_vector_dim)
        self.popularity_predictor = popularity_predictor(self.attention_dim * self.attention_heads )
        self.final_score = nn.Linear(self.user_clicked_new_num, 1, bias = True)
        self.final_norm = nn.LayerNorm(self.sample_num)
        self.multi_view_interest = nn.Linear(self.attention_dim * self.attention_heads, 1, bias = True )
        self.loss_aux = nn.CrossEntropyLoss()

    def forward(self, new_word_embedding, clicked_word_embedding,
                candidate_new_popularity, clicked_new_popularity,):
        new_word_embedding = new_word_embedding.to(device)
        clicked_word_embedding = clicked_word_embedding.to(device)
        candidate_new_popularity = candidate_new_popularity.to(device)
        clicked_new_popularity = clicked_new_popularity.to(device)
        self.truth_new_popularity = torch.cat([candidate_new_popularity, clicked_new_popularity], dim =1 )
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
            user_rep_one = self.user_encoder(clicked_word_embedding_one)
            if i == 0:
                user_rep = user_rep_one
            else:
                user_rep = torch.cat([user_rep, user_rep_one], dim=0)

        self.subspace_prob = user_rep
        self.total_new_rep = torch.cat([new_rep, user_rep], dim =1)
        popularity = self.popularity_predictor.forward(self.total_new_rep)
        self.popularity = popularity
        candidate_popularity = popularity
        candidate_popularity = candidate_popularity[:,:5,:].squeeze()

        user_hot_attention = popularity
        user_hot_attention = torch.mean(user_hot_attention, dim = 1).unsqueeze(1)
        user_hot_attention = user_hot_attention.repeat(1, 5, 1).squeeze()

        new_rep = new_rep.unsqueeze(-1)
        user_rep = user_rep.unsqueeze(1).repeat(1,5,1,1)
        score = torch.matmul(user_rep, self.biaffine_matrix)
        score = torch.matmul(score, new_rep).squeeze()
        score = self.final_score(score).squeeze()
        score = self.final_norm(score)
        # 最终ctr得分
        ctr_score = (1 - self.u * user_hot_attention) * score + self.u * user_hot_attention * self.grama * candidate_popularity
        ctr_score = ctr_score.squeeze()
        return ctr_score

    def cal_loss_aux(self):
        subspace_prob = torch.flatten(self.subspace_prob, 0 ,1)
        truth_labe = torch.Tensor(np.linspace(0, 50, 50)).to(torch.int64)
        truth_labe = truth_labe.unsqueeze(0).repeat(50, 1)
        truth_labe = torch.flatten(truth_labe, 0, 1)
        loss_aux = self.loss_aux(subspace_prob, truth_labe)
        return loss_aux

    def loss(self, new_word_embedding, clicked_word_embedding,
             candidate_new_popularity, clicked_new_popularity,
             label):
        label = label.to(device)
        sorce = self.forward(new_word_embedding, clicked_word_embedding,
                             candidate_new_popularity, clicked_new_popularity)
        loss_ctr = torch.nn.functional.cross_entropy(sorce, torch.argmax(label, dim=1))
        loss_aux = self.cal_loss_aux()
        loss_pop = self.popularity_predictor.loss(self.popularity, self.truth_new_popularity)
        loss = loss_ctr + self.lamda * loss_pop + self.beta * loss_aux
        try:
            auc = roc_auc_score(label.cpu().numpy(), F.softmax(sorce.cpu(), dim=1).detach().numpy())
        except ValueError:
            auc = 0.5
        return loss, auc
