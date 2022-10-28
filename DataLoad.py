import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import csv
path = os.path.dirname(os.getcwd())
print(path)

def data_generator(path):
    # 嵌入数据
    # 新闻流行度嵌入
    new_popularity = np.load(path + '/data3/new_popularity.npy')
    # 预训练特征嵌入
    new_title_bert = np.load(path + '/data3/pretrain-data/new_title_embedding.npy')
    # 单词嵌入
    new_feature= np.load(path + '/data3/new_feature.npy')
    # 单词嵌入
    new_word_embedding = np.load(path + '/data3/new_title_word_embedding.npy')
    new_word_index = np.load(path + '/data3/new_title_word_index.npy')
    # 标题实际长度
    new_title_length = np.load(path + '/data3/new_title_length.npy')
    # 实体嵌入
    new_entity_index = np.load(path + '/data3/new_entity_index.npy')
    new_entity_pos = np.load(path + '/data3/new_entity_pos_index.npy')
    new_entity_freq = np.load(path + '/data3/new_entity_freq_index.npy')
    new_entity_cate = np.load(path + '/data3/new_entity_cate_index.npy')
    new_entity_embedding = np.load(path + '/data3/new_entity_embedding.npy')
    # 邻居实体嵌入
    neigh_entity_index = np.load(path + '/data3/kg/neigh_entity_index.npy')
    neigh_entity_embedding = np.load(path + '/data3/kg/neigh_entity_embedding.npy')
    # 邻居实体关系嵌入
    neigh_relation_index = np.load(path + '/data3/kg/neigh_relation_index.npy')
    neigh_relation_embedding = np.load(path + '/data3/kg/neigh_relation_embedding.npy')
    # 主题和副主题
    new_category_index = np.load(path + '/data3/new_category_index.npy')
    new_subcategory_index = np.load(path + '/data3/new_subcategory_index.npy')
    # 用户点击新闻
    user_clicked_newindex = np.load(path + '/data3/user_clicked_newindex.npy')
    # 用户实际点击新闻新闻
    user_clicked_new_length = np.load(path + '/data3/user_clicked_new_length.npy')
    ## 用户异构图
    heterogeneous_user_graph_A =  np.load(path +'/data3/heterogeneous_graph/heterogeneous_A.npy')
    heterogeneous_user_graph_newindex = np.load(path + '/data3/heterogeneous_graph/heterogeneous_newindex.npy')
    heterogeneous_user_graph_entityindex = np.load(path + '/data3/heterogeneous_graph/heterogeneous_entityindex.npy')
    heterogeneous_user_graph_categoryindex = np.load(path + '/data3/heterogeneous_graph/heterogeneous_categoryindex.npy')
    # 用户交互图相关
    user_one_hop_neighbor = np.load(path +'/data3/user_one_hop_neighbor.npy')
    user_two_hop_neighbor = np.load(path + '/data3/user_two_hop_neighbor.npy')
    new_one_hop_neighbor = np.load(path + '/data3/new_one_hop_neighbor.npy')
    new_two_hop_neighbor = np.load(path + '/data3/new_two_hop_neighbor.npy')
    # 训练集
    candidate_newindex_train = np.load(path + '/data3/candidate_newindex.npy')
    user_index_train = np.load(path + '/data3/user_index.npy')
    label_train = np.load(path + '/data3/label.npy')
    # 测试集
    candidate_newindex_test = np.load(path + '/data3/test/test_candidate_newindex.npy')
    user_index_test = np.load(path + '/data3/test/test_user_index.npy')
    label_test = np.load(path + '/data3/test/test_label.npy')
    Bound_test = np.load(path + '/data3/test/test_bound.npy')
    # 选择bound
    candidate_newindex_select = []
    user_index_select = []
    label_select = []
    bound_select = []
    test_size = int(len(Bound_test) * 0.3)
    train_size = int(len(Bound_test) * 0.7)
    candidate_newindex_train = candidate_newindex_train[:train_size]
    user_index_train = user_index_train[:train_size]
    label_train = label_train[:train_size]
    bound_test = Bound_test[train_size + 1:]

    # bound_test = random.sample(list(Bound_test), int(len(Bound_test) * 0.4))
    #bound_test = random.sample(list(Bound_test), 5)
    index = 0
    for i in range(len(bound_test)):
        start = bound_test[i][0]
        end = bound_test[i][1]
        temp1 = candidate_newindex_test[start:end]
        temp2 = user_index_test[start:end]
        temp3 = label_test[start:end]
        start_new = index
        end_new = index + len(temp1)
        index = index + len(temp1)
        bound_select.append([start_new,end_new])
        candidate_newindex_select.extend(temp1)
        user_index_select.extend(temp2)
        label_select.extend(temp3)
    candidate_newindex_test = np.array(candidate_newindex_select)
    user_index_test = np.array(user_index_select)
    label_test = np.array(label_select)
    bound_test = np.array(bound_select)
    print('训练集印象数{}'.format(len(label_train)))
    print('测试集印象数{}'.format(len(bound_test)))
    print('训练集点击数{}'.format(len(candidate_newindex_train)))
    print('测试集点击数{}'.format(len(candidate_newindex_test)))
    return new_popularity, new_entity_pos, new_entity_freq, new_entity_cate,\
           new_title_bert, new_feature, new_word_embedding, new_word_index, new_title_length, user_clicked_newindex, user_clicked_new_length, \
           new_entity_index, new_entity_embedding, \
           neigh_entity_index, neigh_entity_embedding, neigh_relation_index, neigh_relation_embedding, \
           new_category_index, new_subcategory_index, \
           candidate_newindex_train, user_index_train, label_train, \
           candidate_newindex_test, user_index_test, label_test, bound_test, \
           heterogeneous_user_graph_A, heterogeneous_user_graph_newindex, heterogeneous_user_graph_entityindex, heterogeneous_user_graph_categoryindex, \
           user_one_hop_neighbor, user_two_hop_neighbor, new_one_hop_neighbor, new_two_hop_neighbor

class Train_Dataset(Dataset):
    def __init__(self, user_one_hop_neighbor, user_two_hop_neighbor, new_one_hop_neighbor, new_two_hop_neighbor,
                 new_popularity, new_entity_pos, new_entity_freq, new_entity_cate,
                 heterogeneous_user_graph_A, heterogeneous_user_graph_newindex,
                 heterogeneous_user_graph_entityindex, heterogeneous_user_graph_categoryindex,
                 new_title_bert, new_featrue, new_word_embedding, new_word_index, new_title_length,
                 user_clicked_newindex, user_clicked_new_length,
                 new_entity_index, new_entity_embedding,
                 neigh_entity_index, neigh_entity_embedding, neigh_relation_index, neigh_relation_embedding,
                 new_category_index,new_subcategory_index,
                 candidate_newindex, user, label, mode):
        self.user_one_hop_neighbor = user_one_hop_neighbor
        self.user_two_hop_neighbor = user_two_hop_neighbor
        self.new_one_hop_neighbor = new_one_hop_neighbor
        self.new_two_hop_neighbor = new_two_hop_neighbor
        self.new_popularity = new_popularity
        self.new_entity_pos = new_entity_pos
        self.new_entity_freq = new_entity_freq
        self.new_entity_cate = new_entity_cate
        self.heterogeneous_user_graph_A = heterogeneous_user_graph_A
        self.heterogeneous_user_graph_newindex = heterogeneous_user_graph_newindex
        self.heterogeneous_user_graph_entityindex = heterogeneous_user_graph_entityindex
        self.heterogeneous_user_graph_categoryindex = heterogeneous_user_graph_categoryindex
        self.new_title_bert = new_title_bert
        self.new_featrue = new_featrue
        self.new_word_embedding = new_word_embedding
        self.new_word_index = new_word_index
        self.new_title_length = new_title_length
        self.user_clicked_newindex = user_clicked_newindex
        self.user_clicked_new_length = user_clicked_new_length
        if mode == 'KIM':
            self.new_entity_index = new_entity_index[ :, 0:3]
        else:
            self.new_entity_index = new_entity_index
        self.new_entity_embedding = new_entity_embedding
        self.neigh_entity_index = neigh_entity_index
        self.neigh_entity_embedding = neigh_entity_embedding
        self.neigh_relation_index = neigh_relation_index
        self.neigh_relation_embedding = neigh_relation_embedding
        self.new_category_index = new_category_index
        self.new_subcategory_index = new_subcategory_index
        self.candidate_newindex = candidate_newindex
        self.user = user
        self.label = label
        self.mode = mode

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        candidate_newindex = self.candidate_newindex[index]
        # with open("newindex.csv", "a") as csvfile:
        #     writer = csv.writer(csvfile)
        #     # 写入多行用writerows
        #     writer.writerow(candidate_newindex)
        user_index = self.user[index]
        label_index = self.label[index]
        #################################用户交互图相关#############################################
        ## 获取候选新闻一跳和二条邻居
        new_one_hop_neighbor = self.new_one_hop_neighbor[candidate_newindex]
        new_two_hop_neighbor = self.new_two_hop_neighbor[candidate_newindex]
        ## 获取用户一跳和二条邻居
        user_one_hop_neighbor = self.user_one_hop_neighbor[user_index]
        user_two_hop_neighbor = self.user_two_hop_neighbor[user_index]
        ## 获取用户一跳新闻标题
        user_one_hop_new_word_index = self.new_word_index[user_one_hop_neighbor]
        user_one_hop_new_word_embedding = self.new_word_embedding[user_one_hop_new_word_index]
        ## 获取新闻二跳新闻标题
        new_two_hop_neighbor_new_word_index = self.new_word_index[new_two_hop_neighbor]
        new_two_hop_neighbor_new_word_embedding = self.new_word_embedding[new_two_hop_neighbor_new_word_index]
        new_two_hop_neighbor_new_entity_cate = self.new_entity_cate[new_two_hop_neighbor]
        new_two_hop_neighbor_new_entity_index = self.new_entity_index[new_two_hop_neighbor]
        new_two_hop_neighbor_new_entity_embedding = self.new_entity_embedding[new_two_hop_neighbor_new_entity_index]
        #################################用户交互图相关#############################################
        ## 获取候选新闻流行度
        candidate_new_popularity = self.new_popularity[candidate_newindex]
        ## 获取候选新闻bert预训练模型特征
        candidate_new_title_bert = self.new_title_bert[candidate_newindex]
        ## 获取候选新闻TF-IDF特征
        candidate_new_featrue = self.new_featrue[candidate_newindex]
        ## 获取候选新闻标题长度
        candidate_new_title_length = self.new_title_length[candidate_newindex]
        ## 获取候选新闻单词嵌入
        candidate_new_word_index = self.new_word_index[candidate_newindex]
        candidate_new_word_embedding = self.new_word_embedding[candidate_new_word_index]
        ## 获取候选新闻实体嵌入
        candidate_new_entity_index = self.new_entity_index[candidate_newindex]
        candidate_new_entity_embedding = self.new_entity_embedding[candidate_new_entity_index]
        ## 获取候选新闻实体位置、频率、类别
        candidate_new_entity_pos = self.new_entity_pos[candidate_newindex]
        candidate_new_entity_freq = self.new_entity_freq[candidate_newindex]
        candidate_new_entity_cate = self.new_entity_cate[candidate_newindex]
        ## 获取候选新闻实体的邻居实体嵌入
        candidate_new_neigh_entity_index = self.neigh_entity_index[candidate_new_entity_index]
        candidate_new_neigh_entity_embedding = self.neigh_entity_embedding[candidate_new_neigh_entity_index]
        ## 获取候选新闻实体的邻居实体关系嵌入
        candidate_new_neigh_relation_index = self.neigh_relation_index[candidate_new_entity_index]
        candidate_new_neigh_relation_embedding = self.neigh_relation_embedding[candidate_new_neigh_relation_index]
        ## 获取候选新闻的主题index
        candidate_new_category_index = self.new_category_index[candidate_newindex]
        ## 获取候选新闻的副主题index
        candidate_new_subcategory_index = self.new_subcategory_index[candidate_newindex]
        ##########################################################################################################
        if self.mode == 'heter_graph':
            ## 获取用户异构图index
            user_clicked_newindex = self.heterogeneous_user_graph_newindex[user_index]
        else:
            ## 获取用户点击新闻
            user_clicked_newindex = self.user_clicked_newindex[user_index]
        ## 获取用户点击新闻流行度
        user_clicked_new_popularity = self.new_popularity[user_clicked_newindex]
        ## 获取用户点击新闻长度
        user_clicked_new_length = self.user_clicked_new_length[user_index]
        ## 获取用户点击新闻bert预训练模型特征
        user_clicked_new_title_bert = self.new_title_bert[user_clicked_newindex]
        ## 获取用户点击新闻TF-IDF特征
        user_clicked_new_featrue = self.new_featrue[user_clicked_newindex]
        ## 获取用户点击新闻单词嵌入
        user_clicked_new_word_index = self.new_word_index[user_clicked_newindex]
        user_clicked_new_word_embedding = self.new_word_embedding[user_clicked_new_word_index]
        ## 获取用户点击新闻标题长度
        user_clicked_new_title_length = self.new_title_length[user_clicked_newindex]
        ## 获取用户点击新闻实体嵌入
        user_clicked_new_entity_index = self.new_entity_index[user_clicked_newindex]
        user_clicked_new_entity_embedding = self.new_entity_embedding[user_clicked_new_entity_index]
        ## 获取用户点击新闻实体位置、频率、类别
        user_clicked_new_entity_pos = self.new_entity_pos[user_clicked_newindex]
        user_clicked_new_entity_freq = self.new_entity_freq[user_clicked_newindex]
        user_clicked_new_entity_cate = self.new_entity_cate[user_clicked_newindex]
        ## 获取用户点击新闻实体的邻居实体嵌入
        user_clicked_new_neigh_entity_index = self.neigh_entity_index[user_clicked_new_entity_index]
        user_clicked_new_neigh_entity_embedding = self.neigh_entity_embedding[user_clicked_new_neigh_entity_index]
        ## 获取用户点击新闻实体的邻居实体关系嵌入
        user_clicked_new_neigh_relation_index = self.neigh_relation_index[user_clicked_new_entity_index]
        user_clicked_new_neigh_relation_embedding = self.neigh_relation_embedding[user_clicked_new_neigh_relation_index]
        ## 获取点击新闻的主题index
        user_clicked_new_category_index = self.new_category_index[user_clicked_newindex]
        ## 获取点击新闻的副主题index
        user_clicked_new_subcategory_index = self.new_subcategory_index[user_clicked_newindex]
        # 获取用户异构图
        heterogeneous_user_graph_A = self.heterogeneous_user_graph_A[user_index]
        heterogeneous_user_graph_entityindex = self.heterogeneous_user_graph_entityindex[user_index]
        heterogeneous_user_graph_entity_embedding = self.new_entity_embedding[heterogeneous_user_graph_entityindex]
        heterogeneous_user_graph_categoryindex = self.heterogeneous_user_graph_categoryindex[user_index]
        if self.mode == 'GNewsRec':
            return torch.Tensor(candidate_new_word_embedding), \
                   torch.Tensor(user_clicked_new_word_embedding), \
                   torch.Tensor(candidate_new_entity_embedding), \
                   torch.Tensor(user_clicked_new_entity_embedding), \
                   torch.Tensor(candidate_new_entity_cate), \
                   torch.Tensor(user_clicked_new_entity_cate), \
                   torch.Tensor(new_one_hop_neighbor), \
                   torch.Tensor(new_two_hop_neighbor_new_word_embedding), \
                   torch.Tensor(new_two_hop_neighbor_new_entity_embedding), \
                   torch.Tensor(new_two_hop_neighbor_new_entity_cate), \
                   torch.Tensor(candidate_new_category_index), \
                   torch.Tensor(label_index)
        if self.mode == 'PENR':
            return torch.Tensor(candidate_new_word_embedding), \
                   torch.Tensor(user_clicked_new_word_embedding), \
                   torch.Tensor(candidate_new_popularity), \
                   torch.Tensor(user_clicked_new_popularity), \
                   torch.Tensor(label_index)
        if self.mode == 'GNUD':
            return torch.Tensor([user_index]), \
                   torch.Tensor(user_clicked_new_word_embedding), \
                   torch.Tensor(user_clicked_new_entity_embedding), \
                   torch.Tensor(user_clicked_new_entity_cate), \
                   torch.Tensor(candidate_new_word_embedding), \
                   torch.Tensor(candidate_new_entity_embedding), \
                   torch.Tensor(candidate_new_entity_cate), \
                   torch.Tensor(new_one_hop_neighbor), \
                   torch.Tensor(label_index)
        if self.mode == 'DAN':
            return torch.Tensor(candidate_new_word_embedding), \
                   torch.Tensor(user_clicked_new_word_embedding), \
                   torch.Tensor(candidate_new_entity_embedding), \
                   torch.Tensor(user_clicked_new_entity_embedding), \
                   torch.Tensor(candidate_new_entity_cate), \
                   torch.Tensor(user_clicked_new_entity_cate), \
                   torch.Tensor(label_index)
        if self.mode == 'GERL':
            return torch.Tensor([user_index]), torch.Tensor(user_one_hop_new_word_embedding), torch.Tensor(user_two_hop_neighbor),\
                   torch.Tensor(candidate_new_word_embedding), torch.Tensor(new_two_hop_neighbor), torch.Tensor(new_two_hop_neighbor_new_word_embedding),\
                   torch.Tensor(label_index)
        if self.mode == 'KRED':
            return torch.Tensor(candidate_new_word_embedding), \
                   torch.Tensor(user_clicked_new_word_embedding), \
                   torch.Tensor(candidate_new_entity_embedding), \
                   torch.Tensor(user_clicked_new_entity_embedding), \
                   torch.Tensor(candidate_new_neigh_entity_embedding), \
                   torch.Tensor(user_clicked_new_neigh_entity_embedding), \
                   torch.Tensor(candidate_new_neigh_relation_embedding), \
                   torch.Tensor(user_clicked_new_neigh_relation_embedding), \
                   torch.Tensor(candidate_new_entity_pos), \
                   torch.Tensor(user_clicked_new_entity_pos), \
                   torch.Tensor(candidate_new_entity_freq), \
                   torch.Tensor(user_clicked_new_entity_freq),\
                   torch.Tensor(candidate_new_entity_cate), \
                   torch.Tensor(user_clicked_new_entity_cate),\
                   torch.Tensor(label_index)
        if self.mode == 'KIM':
            return  torch.Tensor(candidate_new_word_embedding), \
                    torch.Tensor(user_clicked_new_word_embedding), \
                    torch.Tensor(candidate_new_entity_embedding), \
                    torch.Tensor(user_clicked_new_entity_embedding), \
                    torch.Tensor(candidate_new_neigh_entity_embedding), \
                    torch.Tensor(user_clicked_new_neigh_entity_embedding), \
                    torch.Tensor(label_index)
        if self.mode == 'heter_graph':
            return  torch.Tensor(candidate_new_word_embedding), \
                    torch.Tensor(user_clicked_new_word_embedding), \
                    torch.Tensor(candidate_new_entity_embedding), \
                    torch.Tensor(user_clicked_new_entity_embedding), \
                    torch.Tensor(candidate_new_neigh_entity_embedding), \
                    torch.Tensor(user_clicked_new_neigh_entity_embedding), \
                    torch.Tensor(candidate_new_neigh_relation_embedding), \
                    torch.Tensor(user_clicked_new_neigh_relation_embedding), \
                    torch.Tensor(candidate_new_category_index), \
                    torch.Tensor(user_clicked_new_category_index), \
                    torch.Tensor(candidate_new_subcategory_index), \
                    torch.Tensor(user_clicked_new_subcategory_index), \
                    torch.Tensor(heterogeneous_user_graph_A), \
                    torch.Tensor(heterogeneous_user_graph_entity_embedding), \
                    torch.Tensor(heterogeneous_user_graph_categoryindex), \
                    torch.Tensor(label_index)
        if self.mode == 'TANR':
            return torch.Tensor(candidate_new_word_embedding), \
                    torch.Tensor(user_clicked_new_word_embedding), \
                    torch.Tensor(candidate_new_category_index), \
                    torch.Tensor(user_clicked_new_category_index), \
                    torch.Tensor(label_index)
        if self.mode == 'heter_graph_bert':
            return torch.Tensor(candidate_new_title_bert), \
                   torch.Tensor(user_clicked_new_title_bert), \
                   torch.Tensor(candidate_new_entity_embedding), \
                   torch.Tensor(user_clicked_new_entity_embedding), \
                   torch.Tensor(candidate_new_neigh_entity_embedding), \
                   torch.Tensor(user_clicked_new_neigh_entity_embedding), \
                   torch.Tensor(candidate_new_neigh_relation_embedding), \
                   torch.Tensor(user_clicked_new_neigh_relation_embedding), \
                   torch.Tensor(candidate_new_category_index), \
                   torch.Tensor(user_clicked_new_category_index), \
                   torch.Tensor(candidate_new_subcategory_index), \
                   torch.Tensor(user_clicked_new_subcategory_index), \
                   torch.Tensor(heterogeneous_user_graph_A), \
                   torch.Tensor(heterogeneous_user_graph_entity_embedding), \
                   torch.Tensor(heterogeneous_user_graph_categoryindex), \
                   torch.Tensor(label_index)
        if self.mode == 'FM' or self.mode == 'DeepFM' or self.mode == 'WideDeep':
            return torch.Tensor(candidate_newindex), \
                   torch.Tensor(user_clicked_newindex), \
                   torch.Tensor(candidate_new_featrue),\
                   torch.Tensor(user_clicked_new_featrue), \
                   torch.Tensor(label_index)
        if self.mode == 'NRMS' or self.mode == 'CNN':
            return torch.Tensor(candidate_new_word_embedding), \
                   torch.Tensor(user_clicked_new_word_embedding), \
                   torch.Tensor(label_index)
        if self.mode == 'NRMS_mask':
            return torch.Tensor(candidate_new_word_embedding), \
                   torch.Tensor(user_clicked_new_word_embedding),\
                   torch.Tensor(candidate_new_title_length),\
                   torch.Tensor(user_clicked_new_title_length),\
                   torch.Tensor([user_clicked_new_length]), \
                   torch.Tensor(label_index)
        if self.mode == 'NRMS&GCN_mask':
            return torch.Tensor(candidate_new_word_embedding), \
                   torch.Tensor(user_clicked_new_word_embedding), \
                   torch.Tensor(candidate_new_entity_embedding), \
                   torch.Tensor(user_clicked_new_entity_embedding), \
                   torch.Tensor(candidate_new_title_length), \
                   torch.Tensor(user_clicked_new_title_length), \
                   torch.Tensor([user_clicked_new_length]), \
                   torch.Tensor(label_index)
        if self.mode == 'NRMS&GCN&KGAT_mask':
            return torch.Tensor(candidate_new_word_embedding), \
                   torch.Tensor(user_clicked_new_word_embedding), \
                   torch.Tensor(candidate_new_entity_embedding), \
                   torch.Tensor(user_clicked_new_entity_embedding), \
                   torch.Tensor(candidate_new_neigh_entity_embedding), \
                   torch.Tensor(user_clicked_new_neigh_entity_embedding), \
                   torch.Tensor(candidate_new_neigh_relation_embedding), \
                   torch.Tensor(user_clicked_new_neigh_relation_embedding),\
                   torch.Tensor(candidate_new_title_length), \
                   torch.Tensor(user_clicked_new_title_length), \
                   torch.Tensor([user_clicked_new_length]), \
                   torch.Tensor(label_index)
        if self.mode == 'GCN':
            return torch.Tensor(candidate_new_entity_embedding), \
                   torch.Tensor(user_clicked_new_entity_embedding), \
                   torch.Tensor(label_index)
        if self.mode == 'GRU':
            return torch.Tensor(candidate_new_word_embedding), \
                   torch.Tensor(user_clicked_new_word_embedding), \
                   torch.IntTensor([user_clicked_new_length]), \
                   torch.Tensor(label_index)
        if self.mode == 'GCN&KGAT':
            return torch.Tensor(candidate_new_entity_embedding), \
                   torch.Tensor(user_clicked_new_entity_embedding), \
                   torch.Tensor(candidate_new_neigh_entity_embedding), \
                   torch.Tensor(user_clicked_new_neigh_entity_embedding), \
                   torch.Tensor(candidate_new_neigh_relation_embedding), \
                   torch.Tensor(user_clicked_new_neigh_relation_embedding), \
                   torch.Tensor(label_index)
        if self.mode == 'NRMS&GCN' or self.mode == 'DKN':
            return torch.Tensor(candidate_new_word_embedding), \
                   torch.Tensor(user_clicked_new_word_embedding), \
                   torch.Tensor(candidate_new_entity_embedding), \
                   torch.Tensor(user_clicked_new_entity_embedding), \
                   torch.Tensor(label_index)
        if self.mode == 'NPA':
            return torch.Tensor(candidate_new_word_embedding), \
                   torch.Tensor(user_clicked_new_word_embedding), \
                   torch.IntTensor([user_index]), \
                   torch.Tensor(label_index)
        if self.mode == 'LSTUR':
            return torch.Tensor(candidate_new_word_embedding), \
                   torch.Tensor(user_clicked_new_word_embedding), \
                   torch.Tensor(candidate_new_category_index), \
                   torch.Tensor(user_clicked_new_category_index), \
                   torch.Tensor(candidate_new_subcategory_index), \
                   torch.Tensor(user_clicked_new_subcategory_index), \
                   torch.IntTensor([user_clicked_new_length]), \
                   torch.IntTensor([user_index]), \
                   torch.Tensor(label_index)
        if self.mode == 'NRMS&MV':
            return torch.Tensor(candidate_new_word_embedding), \
                   torch.Tensor(user_clicked_new_word_embedding), \
                   torch.Tensor(candidate_new_category_index), \
                   torch.Tensor(user_clicked_new_category_index), \
                   torch.Tensor(candidate_new_subcategory_index), \
                   torch.Tensor(user_clicked_new_subcategory_index), \
                   torch.Tensor(label_index)
        if self.mode == 'NAML':
            return torch.Tensor(candidate_new_word_embedding), \
                   torch.Tensor(user_clicked_new_word_embedding), \
                   torch.Tensor(candidate_new_category_index), \
                   torch.Tensor(user_clicked_new_category_index), \
                   torch.Tensor(candidate_new_subcategory_index), \
                   torch.Tensor(user_clicked_new_subcategory_index), \
                   torch.Tensor(label_index)

        if self.mode == 'FIM':
            return torch.Tensor(candidate_new_word_embedding), \
                   torch.Tensor(user_clicked_new_word_embedding), \
                   torch.Tensor(candidate_new_category_index), \
                   torch.Tensor(user_clicked_new_category_index), \
                   torch.Tensor(candidate_new_subcategory_index), \
                   torch.Tensor(user_clicked_new_subcategory_index), \
                   torch.Tensor(label_index)

        if self.mode == 'NRMS&GCN&KGAT':
            return torch.Tensor(candidate_new_word_embedding), \
                   torch.Tensor(user_clicked_new_word_embedding), \
                   torch.Tensor(candidate_new_entity_embedding), \
                   torch.Tensor(user_clicked_new_entity_embedding), \
                   torch.Tensor(candidate_new_neigh_entity_embedding), \
                   torch.Tensor(user_clicked_new_neigh_entity_embedding), \
                   torch.Tensor(candidate_new_neigh_relation_embedding), \
                   torch.Tensor(user_clicked_new_neigh_relation_embedding), \
                   torch.Tensor(label_index)
        if self.mode == 'NRMS&GCN&KGAT&MV':
            return torch.Tensor(candidate_new_word_embedding), \
                   torch.Tensor(user_clicked_new_word_embedding), \
                   torch.Tensor(candidate_new_entity_embedding), \
                   torch.Tensor(user_clicked_new_entity_embedding), \
                   torch.Tensor(candidate_new_neigh_entity_embedding), \
                   torch.Tensor(user_clicked_new_neigh_entity_embedding), \
                   torch.Tensor(candidate_new_neigh_relation_embedding), \
                   torch.Tensor(user_clicked_new_neigh_relation_embedding), \
                   torch.Tensor(candidate_new_category_index), \
                   torch.Tensor(user_clicked_new_category_index), \
                   torch.Tensor(candidate_new_subcategory_index), \
                   torch.Tensor(user_clicked_new_subcategory_index), \
                   torch.Tensor(label_index)
        if self.mode == 'NRMS&GCN&KGAT&MV_bert':
            return torch.Tensor(candidate_new_title_bert), \
                   torch.Tensor(user_clicked_new_title_bert), \
                   torch.Tensor(candidate_new_entity_embedding), \
                   torch.Tensor(user_clicked_new_entity_embedding), \
                   torch.Tensor(candidate_new_neigh_entity_embedding), \
                   torch.Tensor(user_clicked_new_neigh_entity_embedding), \
                   torch.Tensor(candidate_new_neigh_relation_embedding), \
                   torch.Tensor(user_clicked_new_neigh_relation_embedding), \
                   torch.Tensor(candidate_new_category_index), \
                   torch.Tensor(user_clicked_new_category_index), \
                   torch.Tensor(candidate_new_subcategory_index), \
                   torch.Tensor(user_clicked_new_subcategory_index), \
                   torch.Tensor(label_index)
        if self.mode == 'exp7':
            return torch.Tensor(candidate_new_word_embedding), \
                   torch.Tensor(user_clicked_new_word_embedding), \
                   torch.Tensor(candidate_new_entity_embedding), \
                   torch.Tensor(user_clicked_new_entity_embedding), \
                   torch.Tensor(candidate_new_neigh_entity_embedding), \
                   torch.Tensor(user_clicked_new_neigh_entity_embedding), \
                   torch.Tensor(candidate_new_neigh_relation_embedding), \
                   torch.Tensor(user_clicked_new_neigh_relation_embedding), \
                   torch.Tensor(candidate_new_category_index), \
                   torch.Tensor(user_clicked_new_category_index), \
                   torch.Tensor(candidate_new_subcategory_index), \
                   torch.Tensor(user_clicked_new_subcategory_index), \
                   torch.IntTensor([user_index]), \
                   torch.Tensor(label_index)
        if self.mode == 'NRMS&GCN&KGAT&MV_mask':
            return torch.Tensor(candidate_new_word_embedding), \
                   torch.Tensor(user_clicked_new_word_embedding), \
                   torch.Tensor(candidate_new_entity_embedding), \
                   torch.Tensor(user_clicked_new_entity_embedding), \
                   torch.Tensor(candidate_new_neigh_entity_embedding), \
                   torch.Tensor(user_clicked_new_neigh_entity_embedding), \
                   torch.Tensor(candidate_new_neigh_relation_embedding), \
                   torch.Tensor(user_clicked_new_neigh_relation_embedding), \
                   torch.Tensor(candidate_new_category_index), \
                   torch.Tensor(user_clicked_new_category_index), \
                   torch.Tensor(candidate_new_subcategory_index), \
                   torch.Tensor(user_clicked_new_subcategory_index), \
                   torch.Tensor(candidate_new_title_length), \
                   torch.Tensor(user_clicked_new_title_length), \
                   torch.Tensor([user_clicked_new_length]), \
                   torch.Tensor(label_index)
        if self.mode == 'exp1':
            return torch.Tensor(candidate_new_word_embedding), \
                   torch.Tensor(user_clicked_new_word_embedding), \
                   torch.Tensor(candidate_new_entity_embedding), \
                   torch.Tensor(user_clicked_new_entity_embedding), \
                   torch.Tensor(candidate_new_neigh_entity_embedding), \
                   torch.Tensor(user_clicked_new_neigh_entity_embedding), \
                   torch.Tensor(candidate_new_neigh_relation_embedding), \
                   torch.Tensor(user_clicked_new_neigh_relation_embedding), \
                   torch.Tensor(candidate_new_category_index), \
                   torch.Tensor(user_clicked_new_category_index), \
                   torch.Tensor(candidate_new_subcategory_index), \
                   torch.Tensor(user_clicked_new_subcategory_index), \
                   torch.Tensor(candidate_new_title_length), \
                   torch.Tensor(user_clicked_new_title_length), \
                   torch.Tensor([user_clicked_new_length]), \
                   torch.IntTensor([user_index]), \
                   torch.Tensor(label_index)
        if self.mode == 'exp2':
            return torch.Tensor(candidate_new_word_embedding), \
                   torch.Tensor(user_clicked_new_word_embedding), \
                   torch.Tensor(candidate_new_entity_embedding), \
                   torch.Tensor(user_clicked_new_entity_embedding), \
                   torch.Tensor(candidate_new_neigh_entity_embedding), \
                   torch.Tensor(user_clicked_new_neigh_entity_embedding), \
                   torch.Tensor(candidate_new_neigh_relation_embedding), \
                   torch.Tensor(user_clicked_new_neigh_relation_embedding), \
                   torch.Tensor(candidate_new_category_index), \
                   torch.Tensor(user_clicked_new_category_index), \
                   torch.Tensor(candidate_new_subcategory_index), \
                   torch.Tensor(user_clicked_new_subcategory_index), \
                   torch.Tensor(candidate_new_title_length), \
                   torch.Tensor(user_clicked_new_title_length), \
                   torch.Tensor([user_clicked_new_length]), \
                   torch.IntTensor([user_index]), \
                   torch.Tensor(label_index)
        if self.mode == 'exp3' or self.mode == 'exp4' or self.mode == 'exp5' or self.mode == 'exp6':
            return torch.Tensor(candidate_new_word_embedding), \
                   torch.Tensor(user_clicked_new_word_embedding), \
                   torch.Tensor(candidate_new_entity_embedding), \
                   torch.Tensor(user_clicked_new_entity_embedding), \
                   torch.Tensor(candidate_new_neigh_entity_embedding), \
                   torch.Tensor(user_clicked_new_neigh_entity_embedding), \
                   torch.Tensor(candidate_new_neigh_relation_embedding), \
                   torch.Tensor(user_clicked_new_neigh_relation_embedding), \
                   torch.Tensor(candidate_new_category_index), \
                   torch.Tensor(user_clicked_new_category_index), \
                   torch.Tensor(candidate_new_title_length), \
                   torch.Tensor(user_clicked_new_title_length), \
                   torch.Tensor([user_clicked_new_length]), \
                   torch.IntTensor([user_index]), \
                   torch.Tensor(label_index)

class Test_Dataset(Dataset):
    def __init__(self, user_one_hop_neighbor, user_two_hop_neighbor, new_one_hop_neighbor, new_two_hop_neighbor,
                 new_popularity, new_entity_pos, new_entity_freq, new_entity_cate,
                 heterogeneous_user_graph_A, heterogeneous_user_graph_newindex,
                 heterogeneous_user_graph_entityindex, heterogeneous_user_graph_categoryindex,
                 new_title_bert, new_featrue, new_word_embedding, new_word_index, new_title_length, user_clicked_newindex, user_clicked_new_length,
                 new_entity_index, new_entity_embedding,
                 neigh_entity_index, neigh_entity_embedding, neigh_relation_index, neigh_relation_embedding,
                 new_category_index, new_subcategory_index,
                 candidate_newindex, user_index, mode):
        self.user_one_hop_neighbor = user_one_hop_neighbor
        self.user_two_hop_neighbor = user_two_hop_neighbor
        self.new_one_hop_neighbor = new_one_hop_neighbor
        self.new_two_hop_neighbor = new_two_hop_neighbor
        self.new_popularity = new_popularity
        self.new_entity_pos = new_entity_pos
        self.new_entity_freq = new_entity_freq
        self.new_entity_cate = new_entity_cate
        self.heterogeneous_user_graph_A = heterogeneous_user_graph_A
        self.heterogeneous_user_graph_newindex = heterogeneous_user_graph_newindex
        self.heterogeneous_user_graph_entityindex = heterogeneous_user_graph_entityindex
        self.heterogeneous_user_graph_categoryindex = heterogeneous_user_graph_categoryindex
        self.new_title_bert = new_title_bert
        self.new_featrue = new_featrue
        self.new_word_embedding = new_word_embedding
        self.new_word_index = new_word_index
        self.new_title_length = new_title_length
        self.user_clicked_new_length = user_clicked_new_length
        if mode == 'KIM':
            self.new_entity_index = new_entity_index[:, 0:5]
        else:
            self.new_entity_index = new_entity_index
        self.new_entity_embedding = new_entity_embedding
        self.neigh_entity_index = neigh_entity_index
        self.neigh_entity_embedding = neigh_entity_embedding
        self.neigh_relation_index = neigh_relation_index
        self.neigh_relation_embedding = neigh_relation_embedding
        self.new_category_index = new_category_index
        self.new_subcategory_index = new_subcategory_index
        self.user_clicked_newindex = user_clicked_newindex
        self.candidate_newindex = candidate_newindex
        self.user_index = user_index
        self.mode = mode

    def __len__(self):
        return len(self.candidate_newindex)

    def __getitem__(self, index):
        candidate_newindex = self.candidate_newindex[index]
        # with open("newindex.csv", "a") as csvfile:
        #     writer = csv.writer(csvfile)
        #     # 写入多行用writerows
        #     writer.writerow(candidate_newindex)
        # print("候选新闻index{}".format(candidate_newindex))
        user_index = self.user_index[index]
        #################################用户交互图相关#############################################
        ## 获取候选新闻一跳和二条邻居
        new_one_hop_neighbor = self.new_one_hop_neighbor[candidate_newindex]
        new_two_hop_neighbor = self.new_two_hop_neighbor[candidate_newindex]
        ## 获取用户一跳和二条邻居
        user_one_hop_neighbor = self.user_one_hop_neighbor[user_index]
        user_two_hop_neighbor = self.user_two_hop_neighbor[user_index]
        ## 获取用户一跳新闻标题
        user_one_hop_new_word_index = self.new_word_index[user_one_hop_neighbor]
        user_one_hop_new_word_embedding = self.new_word_embedding[user_one_hop_new_word_index]
        ## 获取新闻二跳新闻标题
        new_two_hop_neighbor_new_word_index = self.new_word_index[new_two_hop_neighbor]
        new_two_hop_neighbor_new_word_embedding = self.new_word_embedding[new_two_hop_neighbor_new_word_index]
        new_two_hop_neighbor_new_entity_cate = self.new_entity_cate[new_two_hop_neighbor]
        new_two_hop_neighbor_new_entity_index = self.new_entity_index[new_two_hop_neighbor]
        new_two_hop_neighbor_new_entity_embedding = self.new_entity_embedding[new_two_hop_neighbor_new_entity_index]
        #######################################################################################
        ## 获取候选新闻流行度
        candidate_new_popularity = self.new_popularity[candidate_newindex]
        ## 获取候选新闻bert预训练模型特征
        candidate_new_title_bert = self.new_title_bert[candidate_newindex]
        ## 获取候选新闻TF-IDF特征
        candidate_new_featrue = self.new_featrue[candidate_newindex]
        ## 获取候选新闻标题长度
        candidate_new_title_length = self.new_title_length[candidate_newindex]
        ## 获取候选新闻单词嵌入
        candidate_new_word_index = self.new_word_index[candidate_newindex]
        candidate_new_word_embedding = self.new_word_embedding[candidate_new_word_index]
        ## 获取候选新闻实体嵌入
        candidate_new_entity_index = self.new_entity_index[candidate_newindex]
        candidate_new_entity_embedding = self.new_entity_embedding[candidate_new_entity_index]
        ## 获取候选新闻实体位置、频率、类别
        candidate_new_entity_pos = self.new_entity_pos[candidate_newindex]
        candidate_new_entity_freq = self.new_entity_freq[candidate_newindex]
        candidate_new_entity_cate = self.new_entity_cate[candidate_newindex]
        ## 获取候选新闻实体的邻居实体嵌入
        candidate_new_neigh_entity_index = self.neigh_entity_index[candidate_new_entity_index]
        candidate_new_neigh_entity_embedding = self.neigh_entity_embedding[candidate_new_neigh_entity_index]
        ## 获取候选新闻实体的邻居实体关系嵌入
        candidate_new_neigh_relation_index = self.neigh_relation_index[candidate_new_entity_index]
        candidate_new_neigh_relation_embedding = self.neigh_relation_embedding[candidate_new_neigh_relation_index]
        ## 获取候选新闻的主题index
        candidate_new_category_index = self.new_category_index[candidate_newindex]
        ## 获取候选新闻的副主题index
        candidate_new_subcategory_index = self.new_subcategory_index[candidate_newindex]
        ##########################################################################################################
        ## 获取用户点击新闻
        if self.mode == 'heter_graph':
            ## 获取用户异构图index
            user_clicked_newindex = self.heterogeneous_user_graph_newindex[user_index]
        else:
            ## 获取用户点击新闻
            user_clicked_newindex = self.user_clicked_newindex[user_index]
        ## 获取候选新闻流行度
        user_clicked_new_popularity = self.new_popularity[user_clicked_newindex]
        ## 获取用户点击新闻长度
        user_clicked_new_length = self.user_clicked_new_length[user_index]
        ## 获取用户点击新闻bert预训练模型特征
        user_clicked_new_title_bert = self.new_title_bert[user_clicked_newindex]
        ## 获取用户点击新闻TF-IDF特征
        user_clicked_new_featrue = self.new_featrue[user_clicked_newindex]
        ## 获取用户点击新闻单词嵌入
        user_clicked_new_word_index = self.new_word_index[user_clicked_newindex]
        user_clicked_new_word_embedding = self.new_word_embedding[user_clicked_new_word_index]
        ## 获取用户点击新闻标题长度
        user_clicked_new_title_length = self.new_title_length[user_clicked_newindex]
        ## 获取用户点击新闻实体嵌入
        user_clicked_new_entity_index = self.new_entity_index[user_clicked_newindex]
        user_clicked_new_entity_embedding = self.new_entity_embedding[user_clicked_new_entity_index]
        ## 获取用户点击新闻实体位置、频率、类别
        user_clicked_new_entity_pos = self.new_entity_pos[user_clicked_newindex]
        user_clicked_new_entity_freq = self.new_entity_freq[user_clicked_newindex]
        user_clicked_new_entity_cate = self.new_entity_cate[user_clicked_newindex]
        ## 获取用户点击新闻实体的邻居实体嵌入
        user_clicked_new_neigh_entity_index = self.neigh_entity_index[user_clicked_new_entity_index]
        user_clicked_new_neigh_entity_embedding = self.neigh_entity_embedding[user_clicked_new_neigh_entity_index]
        ## 获取用户点击新闻实体的邻居实体关系嵌入
        user_clicked_new_neigh_relation_index = self.neigh_relation_index[user_clicked_new_entity_index]
        user_clicked_new_neigh_relation_embedding = self.neigh_relation_embedding[user_clicked_new_neigh_relation_index]
        ## 获取点击新闻的主题index
        user_clicked_new_category_index = self.new_category_index[user_clicked_newindex]
        ## 获取点击新闻的副主题index
        user_clicked_new_subcategory_index = self.new_subcategory_index[user_clicked_newindex]
        # 获取用户异构图
        heterogeneous_user_graph_A = self.heterogeneous_user_graph_A[user_index]
        heterogeneous_user_graph_entityindex = self.heterogeneous_user_graph_entityindex[user_index]
        heterogeneous_user_graph_entity_embedding = self.new_entity_embedding[heterogeneous_user_graph_entityindex]
        heterogeneous_user_graph_categoryindex = self.heterogeneous_user_graph_categoryindex[user_index]
        if self.mode == 'GNewsRec':
            return torch.Tensor(candidate_new_word_embedding), \
                   torch.Tensor(user_clicked_new_word_embedding), \
                   torch.Tensor(candidate_new_entity_embedding), \
                   torch.Tensor(user_clicked_new_entity_embedding), \
                   torch.Tensor(candidate_new_entity_cate), \
                   torch.Tensor(user_clicked_new_entity_cate), \
                   torch.Tensor(new_one_hop_neighbor), \
                   torch.Tensor(new_two_hop_neighbor_new_word_embedding), \
                   torch.Tensor(new_two_hop_neighbor_new_entity_embedding), \
                   torch.Tensor(new_two_hop_neighbor_new_entity_cate), \
                   torch.Tensor(candidate_new_category_index)

        if self.mode == 'PENR':
            return torch.Tensor(candidate_new_word_embedding), \
                   torch.Tensor(user_clicked_new_word_embedding), \
                   torch.Tensor(candidate_new_popularity), \
                   torch.Tensor(user_clicked_new_popularity)
        if self.mode == 'GNUD':
            return torch.Tensor([user_index]), \
                   torch.Tensor(user_clicked_new_word_embedding), \
                   torch.Tensor(user_clicked_new_entity_embedding), \
                   torch.Tensor(user_clicked_new_entity_cate), \
                   torch.Tensor(candidate_new_word_embedding), \
                   torch.Tensor(candidate_new_entity_embedding), \
                   torch.Tensor(candidate_new_entity_cate), \
                   torch.Tensor(new_one_hop_neighbor)
        if self.mode == 'DAN':
            return torch.Tensor(candidate_new_word_embedding), \
                   torch.Tensor(user_clicked_new_word_embedding), \
                   torch.Tensor(candidate_new_entity_embedding), \
                   torch.Tensor(user_clicked_new_entity_embedding), \
                   torch.Tensor(candidate_new_entity_cate), \
                   torch.Tensor(user_clicked_new_entity_cate)
        if self.mode == 'GERL':
            return torch.Tensor([user_index]), torch.Tensor(user_one_hop_new_word_embedding), torch.Tensor(user_two_hop_neighbor),\
                   torch.Tensor(candidate_new_word_embedding), torch.Tensor(new_two_hop_neighbor), torch.Tensor(new_two_hop_neighbor_new_word_embedding)
        if self.mode == 'KRED':
            return torch.Tensor(candidate_new_word_embedding), \
                   torch.Tensor(user_clicked_new_word_embedding), \
                   torch.Tensor(candidate_new_entity_embedding), \
                   torch.Tensor(user_clicked_new_entity_embedding), \
                   torch.Tensor(candidate_new_neigh_entity_embedding), \
                   torch.Tensor(user_clicked_new_neigh_entity_embedding), \
                   torch.Tensor(candidate_new_neigh_relation_embedding), \
                   torch.Tensor(user_clicked_new_neigh_relation_embedding), \
                   torch.Tensor(candidate_new_entity_pos), \
                   torch.Tensor(user_clicked_new_entity_pos), \
                   torch.Tensor(candidate_new_entity_freq), \
                   torch.Tensor(user_clicked_new_entity_freq),\
                   torch.Tensor(candidate_new_entity_cate), \
                   torch.Tensor(user_clicked_new_entity_cate)
        if self.mode == 'KIM':
            return  torch.Tensor(candidate_new_word_embedding), \
                    torch.Tensor(user_clicked_new_word_embedding), \
                    torch.Tensor(candidate_new_entity_embedding), \
                    torch.Tensor(user_clicked_new_entity_embedding), \
                    torch.Tensor(candidate_new_neigh_entity_embedding), \
                    torch.Tensor(user_clicked_new_neigh_entity_embedding)
        if self.mode == 'TANR':
            return  torch.Tensor(candidate_new_word_embedding), \
                    torch.Tensor(user_clicked_new_word_embedding), \
                    torch.Tensor(candidate_new_category_index), \
                    torch.Tensor(user_clicked_new_category_index)
        if self.mode == 'heter_graph':
            return  torch.Tensor(candidate_new_word_embedding), \
                    torch.Tensor(user_clicked_new_word_embedding), \
                    torch.Tensor(candidate_new_entity_embedding), \
                    torch.Tensor(user_clicked_new_entity_embedding), \
                    torch.Tensor(candidate_new_neigh_entity_embedding), \
                    torch.Tensor(user_clicked_new_neigh_entity_embedding), \
                    torch.Tensor(candidate_new_neigh_relation_embedding), \
                    torch.Tensor(user_clicked_new_neigh_relation_embedding), \
                    torch.Tensor(candidate_new_category_index), \
                    torch.Tensor(user_clicked_new_category_index), \
                    torch.Tensor(candidate_new_subcategory_index), \
                    torch.Tensor(user_clicked_new_subcategory_index), \
                    torch.Tensor(heterogeneous_user_graph_A), \
                    torch.Tensor(heterogeneous_user_graph_entity_embedding), \
                    torch.Tensor(heterogeneous_user_graph_categoryindex)
        if self.mode == 'heter_graph_bert':
            return torch.Tensor(candidate_new_title_bert), \
                   torch.Tensor(user_clicked_new_title_bert), \
                   torch.Tensor(candidate_new_entity_embedding), \
                   torch.Tensor(user_clicked_new_entity_embedding), \
                   torch.Tensor(candidate_new_neigh_entity_embedding), \
                   torch.Tensor(user_clicked_new_neigh_entity_embedding), \
                   torch.Tensor(candidate_new_neigh_relation_embedding), \
                   torch.Tensor(user_clicked_new_neigh_relation_embedding), \
                   torch.Tensor(candidate_new_category_index), \
                   torch.Tensor(user_clicked_new_category_index), \
                   torch.Tensor(candidate_new_subcategory_index), \
                   torch.Tensor(user_clicked_new_subcategory_index), \
                   torch.Tensor(heterogeneous_user_graph_A), \
                   torch.Tensor(heterogeneous_user_graph_entity_embedding), \
                   torch.Tensor(heterogeneous_user_graph_categoryindex)
        if self.mode == 'FM' or self.mode == 'DeepFM' or self.mode == 'WideDeep':
            return torch.Tensor(candidate_newindex), \
                   torch.Tensor(user_clicked_newindex), \
                   torch.Tensor(candidate_new_featrue),\
                   torch.Tensor(user_clicked_new_featrue)
        if self.mode == 'NRMS' or self.mode == 'CNN':
            return torch.Tensor(candidate_new_word_embedding), \
                   torch.Tensor(user_clicked_new_word_embedding)
        if self.mode == 'NRMS_mask':
            return torch.Tensor(candidate_new_word_embedding), \
                   torch.Tensor(user_clicked_new_word_embedding),\
                   torch.Tensor(candidate_new_title_length),\
                   torch.Tensor(user_clicked_new_title_length),\
                   torch.Tensor([user_clicked_new_length])
        if self.mode == 'NRMS&GCN_mask':
            return torch.Tensor(candidate_new_word_embedding), \
                   torch.Tensor(user_clicked_new_word_embedding), \
                   torch.Tensor(candidate_new_entity_embedding), \
                   torch.Tensor(user_clicked_new_entity_embedding), \
                   torch.Tensor(candidate_new_title_length), \
                   torch.Tensor(user_clicked_new_title_length), \
                   torch.Tensor([user_clicked_new_length])
        if self.mode == 'NRMS&GCN&KGAT_mask':
            return torch.Tensor(candidate_new_word_embedding), \
                   torch.Tensor(user_clicked_new_word_embedding), \
                   torch.Tensor(candidate_new_entity_embedding), \
                   torch.Tensor(user_clicked_new_entity_embedding), \
                   torch.Tensor(candidate_new_neigh_entity_embedding), \
                   torch.Tensor(user_clicked_new_neigh_entity_embedding), \
                   torch.Tensor(candidate_new_neigh_relation_embedding), \
                   torch.Tensor(user_clicked_new_neigh_relation_embedding),\
                   torch.Tensor(candidate_new_title_length), \
                   torch.Tensor(user_clicked_new_title_length), \
                   torch.Tensor([user_clicked_new_length])
        if self.mode == 'GCN':
            return torch.Tensor(candidate_new_entity_embedding), \
                   torch.Tensor(user_clicked_new_entity_embedding)
        if self.mode == 'GRU':
            return torch.Tensor(candidate_new_word_embedding), \
                   torch.Tensor(user_clicked_new_word_embedding), \
                   torch.Tensor([user_clicked_new_length])
        if self.mode == 'GCN&KGAT':
            return torch.Tensor(candidate_new_entity_embedding), \
                   torch.Tensor(user_clicked_new_entity_embedding), \
                   torch.Tensor(candidate_new_neigh_entity_embedding), \
                   torch.Tensor(user_clicked_new_neigh_entity_embedding), \
                   torch.Tensor(candidate_new_neigh_relation_embedding), \
                   torch.Tensor(user_clicked_new_neigh_relation_embedding)
        if self.mode == 'NRMS&GCN' or self.mode == 'DKN':
            return torch.Tensor(candidate_new_word_embedding), \
                   torch.Tensor(user_clicked_new_word_embedding), \
                   torch.Tensor(candidate_new_entity_embedding), \
                   torch.Tensor(user_clicked_new_entity_embedding)
        if self.mode == 'LSTUR':
            return torch.Tensor(candidate_new_word_embedding), \
                   torch.Tensor(user_clicked_new_word_embedding), \
                   torch.Tensor(candidate_new_category_index), \
                   torch.Tensor(user_clicked_new_category_index), \
                   torch.Tensor(candidate_new_subcategory_index), \
                   torch.Tensor(user_clicked_new_subcategory_index), \
                   torch.Tensor([user_clicked_new_length]), \
                   torch.IntTensor([user_index])
        if self.mode == 'NRMS&MV':
            return torch.Tensor(candidate_new_word_embedding), \
                   torch.Tensor(user_clicked_new_word_embedding), \
                   torch.Tensor(candidate_new_category_index), \
                   torch.Tensor(user_clicked_new_category_index), \
                   torch.Tensor(candidate_new_subcategory_index), \
                   torch.Tensor(user_clicked_new_subcategory_index)
        if self.mode == 'NPA':
            return torch.Tensor(candidate_new_word_embedding), \
                   torch.Tensor(user_clicked_new_word_embedding), \
                   torch.IntTensor([user_index])
        if self.mode == 'NAML':
            return torch.Tensor(candidate_new_word_embedding), \
                   torch.Tensor(user_clicked_new_word_embedding), \
                   torch.Tensor(candidate_new_category_index), \
                   torch.Tensor(user_clicked_new_category_index), \
                   torch.Tensor(candidate_new_subcategory_index), \
                   torch.Tensor(user_clicked_new_subcategory_index)

        if self.mode == 'FIM':
            return torch.Tensor(candidate_new_word_embedding), \
                   torch.Tensor(user_clicked_new_word_embedding), \
                   torch.Tensor(candidate_new_category_index), \
                   torch.Tensor(user_clicked_new_category_index), \
                   torch.Tensor(candidate_new_subcategory_index), \
                   torch.Tensor(user_clicked_new_subcategory_index)


        if self.mode == 'NRMS&GCN&KGAT':
            return torch.Tensor(candidate_new_word_embedding), \
                   torch.Tensor(user_clicked_new_word_embedding), \
                   torch.Tensor(candidate_new_entity_embedding), \
                   torch.Tensor(user_clicked_new_entity_embedding), \
                   torch.Tensor(candidate_new_neigh_entity_embedding), \
                   torch.Tensor(user_clicked_new_neigh_entity_embedding), \
                   torch.Tensor(candidate_new_neigh_relation_embedding), \
                   torch.Tensor(user_clicked_new_neigh_relation_embedding)
        if self.mode == 'NRMS&GCN&KGAT&MV':
            return torch.Tensor(candidate_new_word_embedding), \
                   torch.Tensor(user_clicked_new_word_embedding), \
                   torch.Tensor(candidate_new_entity_embedding), \
                   torch.Tensor(user_clicked_new_entity_embedding), \
                   torch.Tensor(candidate_new_neigh_entity_embedding), \
                   torch.Tensor(user_clicked_new_neigh_entity_embedding), \
                   torch.Tensor(candidate_new_neigh_relation_embedding), \
                   torch.Tensor(user_clicked_new_neigh_relation_embedding), \
                   torch.Tensor(candidate_new_category_index), \
                   torch.Tensor(user_clicked_new_category_index), \
                   torch.Tensor(candidate_new_subcategory_index), \
                   torch.Tensor(user_clicked_new_subcategory_index)
        if self.mode == 'NRMS&GCN&KGAT&MV_bert':
            return torch.Tensor(candidate_new_title_bert), \
                   torch.Tensor(user_clicked_new_title_bert), \
                   torch.Tensor(candidate_new_entity_embedding), \
                   torch.Tensor(user_clicked_new_entity_embedding), \
                   torch.Tensor(candidate_new_neigh_entity_embedding), \
                   torch.Tensor(user_clicked_new_neigh_entity_embedding), \
                   torch.Tensor(candidate_new_neigh_relation_embedding), \
                   torch.Tensor(user_clicked_new_neigh_relation_embedding), \
                   torch.Tensor(candidate_new_category_index), \
                   torch.Tensor(user_clicked_new_category_index), \
                   torch.Tensor(candidate_new_subcategory_index), \
                   torch.Tensor(user_clicked_new_subcategory_index)
        if self.mode == 'exp7':
            return torch.Tensor(candidate_new_word_embedding), \
                   torch.Tensor(user_clicked_new_word_embedding), \
                   torch.Tensor(candidate_new_entity_embedding), \
                   torch.Tensor(user_clicked_new_entity_embedding), \
                   torch.Tensor(candidate_new_neigh_entity_embedding), \
                   torch.Tensor(user_clicked_new_neigh_entity_embedding), \
                   torch.Tensor(candidate_new_neigh_relation_embedding), \
                   torch.Tensor(user_clicked_new_neigh_relation_embedding), \
                   torch.Tensor(candidate_new_category_index), \
                   torch.Tensor(user_clicked_new_category_index), \
                   torch.Tensor(candidate_new_subcategory_index), \
                   torch.Tensor(user_clicked_new_subcategory_index), \
                   torch.IntTensor([user_index])

        if self.mode == 'NRMS&GCN&KGAT&MV_mask':
            return torch.Tensor(candidate_new_word_embedding), \
                   torch.Tensor(user_clicked_new_word_embedding), \
                   torch.Tensor(candidate_new_entity_embedding), \
                   torch.Tensor(user_clicked_new_entity_embedding), \
                   torch.Tensor(candidate_new_neigh_entity_embedding), \
                   torch.Tensor(user_clicked_new_neigh_entity_embedding), \
                   torch.Tensor(candidate_new_neigh_relation_embedding), \
                   torch.Tensor(user_clicked_new_neigh_relation_embedding), \
                   torch.Tensor(candidate_new_category_index), \
                   torch.Tensor(user_clicked_new_category_index), \
                   torch.Tensor(candidate_new_subcategory_index), \
                   torch.Tensor(user_clicked_new_subcategory_index), \
                   torch.Tensor(candidate_new_title_length), \
                   torch.Tensor(user_clicked_new_title_length), \
                   torch.Tensor([user_clicked_new_length])
        if self.mode == 'exp1':
            return torch.Tensor(candidate_new_word_embedding), \
                   torch.Tensor(user_clicked_new_word_embedding), \
                   torch.Tensor(candidate_new_entity_embedding), \
                   torch.Tensor(user_clicked_new_entity_embedding), \
                   torch.Tensor(candidate_new_neigh_entity_embedding), \
                   torch.Tensor(user_clicked_new_neigh_entity_embedding), \
                   torch.Tensor(candidate_new_neigh_relation_embedding), \
                   torch.Tensor(user_clicked_new_neigh_relation_embedding), \
                   torch.Tensor(candidate_new_category_index), \
                   torch.Tensor(user_clicked_new_category_index), \
                   torch.Tensor(candidate_new_subcategory_index), \
                   torch.Tensor(user_clicked_new_subcategory_index), \
                   torch.Tensor(candidate_new_title_length), \
                   torch.Tensor(user_clicked_new_title_length), \
                   torch.Tensor([user_clicked_new_length]), \
                   torch.IntTensor([user_index])
        if self.mode == 'exp2':
            return torch.Tensor(candidate_new_word_embedding), \
                   torch.Tensor(user_clicked_new_word_embedding), \
                   torch.Tensor(candidate_new_entity_embedding), \
                   torch.Tensor(user_clicked_new_entity_embedding), \
                   torch.Tensor(candidate_new_neigh_entity_embedding), \
                   torch.Tensor(user_clicked_new_neigh_entity_embedding), \
                   torch.Tensor(candidate_new_neigh_relation_embedding), \
                   torch.Tensor(user_clicked_new_neigh_relation_embedding), \
                   torch.Tensor(candidate_new_category_index), \
                   torch.Tensor(user_clicked_new_category_index), \
                   torch.Tensor(candidate_new_subcategory_index), \
                   torch.Tensor(user_clicked_new_subcategory_index), \
                   torch.Tensor(candidate_new_title_length), \
                   torch.Tensor(user_clicked_new_title_length), \
                   torch.Tensor([user_clicked_new_length]), \
                   torch.IntTensor([user_index])
        if self.mode == 'exp3' or self.mode == 'exp4' or self.mode == 'exp5' or self.mode == 'exp6':
            return torch.Tensor(candidate_new_word_embedding), \
                   torch.Tensor(user_clicked_new_word_embedding), \
                   torch.Tensor(candidate_new_entity_embedding), \
                   torch.Tensor(user_clicked_new_entity_embedding), \
                   torch.Tensor(candidate_new_neigh_entity_embedding), \
                   torch.Tensor(user_clicked_new_neigh_entity_embedding), \
                   torch.Tensor(candidate_new_neigh_relation_embedding), \
                   torch.Tensor(user_clicked_new_neigh_relation_embedding), \
                   torch.Tensor(candidate_new_category_index), \
                   torch.Tensor(user_clicked_new_category_index), \
                   torch.Tensor(candidate_new_title_length), \
                   torch.Tensor(user_clicked_new_title_length), \
                   torch.Tensor([user_clicked_new_length]), \
                   torch.IntTensor([user_index])

if __name__ == '__main__':
    new_popularity, new_entity_pos, new_entity_freq, new_entity_cate, \
    new_title_bert, new_featrue, new_word_embedding, new_word_index, new_title_length, user_clicked_newindex, user_clicked_new_length, \
    new_entity_index, new_entity_embedding, \
    neigh_entity_index, neigh_entity_embedding, neigh_relation_index, neigh_relation_embedding, \
    new_category_index, new_subcategory_index, \
    candidate_newindex_train, user_index_train, label_train, \
    candidate_newindex_test, user_index_test, label_test, bound_test,\
    heterogeneous_user_graph_A, heterogeneous_user_graph_newindex, \
    heterogeneous_user_graph_entityindex, heterogeneous_user_graph_categoryindex, \
    user_one_hop_neighbor, user_two_hop_neighbor, new_one_hop_neighbor, new_two_hop_neighbor = data_generator(path)

    # 训练集数据
    train_dataset = Train_Dataset(user_one_hop_neighbor, user_two_hop_neighbor, new_one_hop_neighbor, new_two_hop_neighbor,
                                  new_popularity, new_entity_pos, new_entity_freq, new_entity_cate,
                                  heterogeneous_user_graph_A, heterogeneous_user_graph_newindex,
                                  heterogeneous_user_graph_entityindex, heterogeneous_user_graph_categoryindex,
                                  new_title_bert, new_featrue, new_word_embedding, new_word_index, new_title_length, user_clicked_newindex, user_clicked_new_length,
                                  new_entity_index, new_entity_embedding,
                                  neigh_entity_index, neigh_entity_embedding, neigh_relation_index, neigh_relation_embedding,
                                  new_category_index, new_subcategory_index,
                                  candidate_newindex_train, user_index_train, label_train, mode='LSTUR')
    train_dataloader = DataLoader(train_dataset, batch_size=50, shuffle=True)

    for data in train_dataloader:
        candidate_new_word_embedding, user_clicked_new_word_embedding, \
        candidate_new_category_index, user_clicked_new_category_index, \
        candidate_new_subcategory_index, user_clicked_new_subcategory_index, \
        user_clicked_new_length, user_index, label = data
    # 测试集数据
    test_dataset = Test_Dataset(user_one_hop_neighbor, user_two_hop_neighbor, new_one_hop_neighbor, new_two_hop_neighbor,
                                new_popularity, new_entity_pos, new_entity_freq, new_entity_cate,
                                heterogeneous_user_graph_A, heterogeneous_user_graph_newindex,
                                heterogeneous_user_graph_entityindex, heterogeneous_user_graph_categoryindex,
                                new_title_bert, new_featrue, new_word_embedding, new_word_index, new_title_length, user_clicked_newindex, user_clicked_new_length,
                                new_entity_index, new_entity_embedding,
                                neigh_entity_index, neigh_entity_embedding, neigh_relation_index, neigh_relation_embedding,
                                new_category_index, new_subcategory_index,
                                candidate_newindex_test, user_index_test, mode='LSTUR')
    test_dataloader = DataLoader(test_dataset, batch_size=50, shuffle=True)
    for data in test_dataloader:
        candidate_new_word_embedding, user_clicked_new_word_embedding, \
        candidate_new_category_index, user_clicked_new_category_index, \
        candidate_new_subcategory_index, user_clicked_new_subcategory_index, user_clicked_new_length =  data


