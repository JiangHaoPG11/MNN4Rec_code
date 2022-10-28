import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *
from sklearn.metrics import roc_auc_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class new_encoder(torch.nn.Module):
    def __init__(self, title_word_size, category_size, subcategory_size):
        super(new_encoder, self).__init__()
        self.news_feature_size = title_word_size + 2
        self.category_embedding_layer = nn.Embedding(category_size, embedding_dim=300)
        self.subcategory_embedding_layer = nn.Embedding(subcategory_size, embedding_dim=300)
        self.HDC_cnn = HDC_CNN_extractor(self.news_feature_size)
        self.dropout_prob = 0.2

    def forward(self, word_embedding, category_index, subcategory_index):
        # 主题嵌入
        category_embedding = self.category_embedding_layer(category_index.to(torch.int64))
        category_embedding = category_embedding.unsqueeze(1)
        # 副主题嵌入
        subcategory_embedding = self.subcategory_embedding_layer(subcategory_index.to(torch.int64))
        subcategory_embedding = subcategory_embedding.unsqueeze(1)
        # 空洞卷积
        news_feature_embedding = torch.cat([word_embedding, category_embedding, subcategory_embedding], dim = 1)
        news_vec = self.HDC_cnn(news_feature_embedding)
        return news_vec

class user_encoder(torch.nn.Module):
    def __init__(self, title_word_size, category_size, subcategory_size):
        super(user_encoder, self).__init__()
        self.new_encoder = new_encoder(title_word_size, category_size, subcategory_size)
        self.dropout_prob = 0.2

    def forward(self, word_embedding, category_index, subcategory_index):
        user_rep = self.new_encoder(word_embedding, category_index, subcategory_index).unsqueeze(0)
        return user_rep

class FIM(torch.nn.Module):
    def __init__(self, args):
        super(FIM, self).__init__()
        self.word_dim = args.word_embedding_dim
        self.title_word_size = args.title_word_size
        self.sample_num = args.sample_num
        self.user_clicked_new_num = args.user_clicked_new_num
        self.category_dim = args.category_dim
        self.subcategory_dim = args.subcategory_dim
        self.category_size = args.category_size
        self.subcategory_size = args.subcategory_size
        self.num_filters = args.cnn_num_filters
        self.window_sizes = args.cnn_window_sizes
        self.query_vector_dim = args.query_vector_dim
        self.drop_prob = args.drop_prob

        self.new_encoder = new_encoder(self.title_word_size, self.category_size, self.subcategory_size)
        self.user_encoder = user_encoder(self.title_word_size, self.category_size, self.subcategory_size)

        self.feature_dim = args.feature_dim
        self.news_feature_size = self.title_word_size + 2
        self.interaction_layer = FIM_interaction_layer(self.feature_dim)

    def forward(self, candidate_new_word_embedding, user_clicked_new_word_embedding,
                candidate_new_category_index, user_clicked_new_category_index,
                candidate_new_subcategory_index, user_clicked_new_subcategory_index):
        candidate_new_word_embedding = candidate_new_word_embedding.to(device)
        user_clicked_new_word_embedding = user_clicked_new_word_embedding.to(device)
        candidate_new_category_index = candidate_new_category_index.to(device)
        user_clicked_new_category_index = user_clicked_new_category_index.to(device)
        candidate_new_subcategory_index = candidate_new_subcategory_index.to(device)
        user_clicked_new_subcategory_index = user_clicked_new_subcategory_index.to(device)

        # 新闻编码器
        new_rep = None
        for i in range(self.sample_num):
            new_word_embedding_one = candidate_new_word_embedding[:, i, :]
            new_category_index = candidate_new_category_index[:, i]
            new_subcategory_index = candidate_new_subcategory_index[:, i]
            new_rep_one = self.new_encoder(new_word_embedding_one, new_category_index, new_subcategory_index).unsqueeze(1)
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
            # 点击新闻主题index
            clicked_new_category_index = user_clicked_new_category_index[i, :]
            # 点击新闻副主题index
            clicked_new_subcategory_index = user_clicked_new_subcategory_index[i, :]
            # 用户表征
            user_rep_one = self.user_encoder(clicked_new_word_embedding_one, clicked_new_category_index, clicked_new_subcategory_index)
            if i == 0:
                user_rep = user_rep_one
            else:
                user_rep = torch.cat([user_rep, user_rep_one], dim=0)
        score = self.interaction_layer(user_rep, new_rep)
        return score

    def loss(self, candidate_new_word_embedding,  user_clicked_word_embedding,
             candidate_new_category_index, user_clicked_new_category_index,
             candidate_new_subcategory_index, user_clicked_new_subcategory_index, label):
        label = label.to(device)
        score = self.forward(candidate_new_word_embedding, user_clicked_word_embedding,
                             candidate_new_category_index, user_clicked_new_category_index,
                             candidate_new_subcategory_index, user_clicked_new_subcategory_index)
        loss = torch.nn.functional.cross_entropy(score, torch.argmax(label, dim =1))
        try:
            auc = roc_auc_score(label.cpu().numpy(), F.softmax(score.cpu(), dim=1).detach().numpy())
        except ValueError:
            auc = 0.5
        return loss, auc
