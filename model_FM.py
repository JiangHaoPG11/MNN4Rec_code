import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *
from sklearn.metrics import roc_auc_score
from torchfm.layer import FactorizationMachine, FeaturesEmbedding, FeaturesLinear

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FM(torch.nn.Module):
    def __init__(self, args):
        super(FM, self).__init__()
        self.batch_size = args.batch_size
        self.word_dim = args.word_embedding_dim
        self.title_word_size = args.title_word_size
        self.sample_num = args.sample_num
        self.embed_dim = 100
        self.dropout_prob = 0.2
        # self.embedding = feature_embedding(self.batch_size * self.title_word_size, self.embed_dim)
        self.linear = nn.Linear(args.user_clicked_new_num+1, 1, bias = True)
        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, candidate_newindex, user_clicked_newindex,
                candidate_new_feature, user_clicked_new_feature):
        candidate_new_feature = candidate_new_feature.to(device)
        user_clicked_new_feature = user_clicked_new_feature.to(device)
        candidate_newindex = candidate_newindex.to(device)
        user_clicked_newindex = user_clicked_newindex.to(device)

        score = None
        for i in range(self.sample_num):
            candidate_new_feature_one = candidate_new_feature[:, i, :]
            candidate_new_feature_one = candidate_new_feature_one.unsqueeze(1)
            feature = torch.cat([candidate_new_feature_one, user_clicked_new_feature] ,dim=1)
            feature = F.dropout(feature, p=self.dropout_prob, training=self.training)
            candidate_newindex_one = candidate_newindex[:, i].unsqueeze(1)
            newindex = torch.cat([candidate_newindex_one, user_clicked_newindex] ,dim=1)
            score_one = self.linear(newindex) + self.fm(feature)
            if i == 0:
                score = score_one
            else:
                score = torch.cat([score, score_one], dim = 1)
        return score

    def loss(self, candidate_newindex, user_clicked_newindex,
             candidate_new_word_embedding, user_clicked_word_embedding, label):
        label = label.to(device)
        sorce = self.forward(candidate_newindex, user_clicked_newindex, candidate_new_word_embedding, user_clicked_word_embedding)
        loss = torch.nn.functional.cross_entropy(sorce, torch.argmax(label, dim =1))
        try:
            auc = roc_auc_score(label.cpu().numpy(), F.softmax(sorce.cpu(), dim=1).detach().numpy())
        except ValueError:
            auc = 0.5
        return loss, auc
