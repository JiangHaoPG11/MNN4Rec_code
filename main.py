import torch
import random
import argparse
import os
from model_FM import FM
from model_DeepFM import DeepFM
from model_WideDeep import WideDeep
from model_CNN import CNN
from model_GCN import GCN
from model_GCN_KGAT import GCN_KGAT
from model_GRU import GRU
from model_NAML import NAML
from model_DKN import DKN
from model_LSTUR import LSTUR
from model_NPA import NPA
from model_NRMS import NRMS
from model_NRMS_mask import NRMS_mask
from model_NRMS_GCN import NRMS_GCN
from model_NRMS_GCN_mask import NRMS_GCN_mask
from model_NRMS_GCN_KGAT import NRMS_GCN_KGAT
from model_NRMS_GCN_KGAT_mask import NRMS_GCN_KGAT_mask
from model_NRMS_GCN_KGAT_MV_cp import NRMS_GCN_KGAT_MV
from model_NRMS_GCN_KGAT_MV_bert import NRMS_GCN_KGAT_MV_bert
from model_NRMS_GCN_KGAT_MV_mask import NRMS_GCN_KGAT_MV_mask
from model_heter_graph_simple import heter_graph
from model_heter_graph_bert_simple import heter_graph_bert
from model_KIM import KIM
from model_KRED import KRED
from model_TANR import TANR
from model_GERL import GERL
from model_DAN import DAN
from model_GNUD import GNUD
from model_PENR import PENR
from model_GNewsRec import GNewsRec
from model_FIM import FIM
from model_exp1 import exp1
from model_exp2 import exp2
from model_exp3 import exp3
from model_exp4 import exp4
from model_exp5 import exp5
from model_exp6 import exp6
from model_exp7 import exp7
from DataLoad import *
from Train import *
from Test import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default= 5)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--optimizer_type', type=str, default='adam')
    parser.add_argument('--loss_type', type=str, default='cross_entropy')
    parser.add_argument('--mode', type=str,default='NRMS&GCN&KGAT&MV')
    parser.add_argument('--query_vector_dim', type=int, default=200, help='询问向量维数')
    parser.add_argument('--user_embedding_size', type=int, default=50, help='用户嵌入向量维数')
    # FM模型相关参数
    parser.add_argument('--mlp_dims', type=list, default=[64, 32])
    parser.add_argument('--fm_dropout', type=float, default=0.2)
    # NRMS 模型相关参数
    parser.add_argument('--batch_size', type=int, default= 50 )
    parser.add_argument('--user_size', type=int, default = 100, help='总用户个数')
    parser.add_argument('--title_size', type=int, default = 4139, help='标题新闻总数')
    parser.add_argument('--user_clicked_new_num', type=int, default = 50, help='单个用户点击的新闻个数')
    parser.add_argument('--total_word_size', type=int, default = 11969, help='词袋中总的单词数量')
    parser.add_argument('--title_word_size', type=int, default =23, help='每个title中的单词数量')
    # 多头注意力相关参数
    parser.add_argument('--word_embedding_dim', type=int, default=300, help='单词嵌入维数')
    parser.add_argument('--attention_heads', type=int, default=20, help='多头注意力的头数')
    parser.add_argument('--num_units', type=int, default=20, help='多头注意力输出维数')
    parser.add_argument('--newencoder_epoch', type=int, default=1, help='编码器迭代次数')
    parser.add_argument('--attention_dim', type=int, default=20, help='注意力层的维数')
    # KGAT模型相关参数
    parser.add_argument('--total_entity_size', type=int, default=3795, help='总实体特征个数')
    parser.add_argument('--new_entity_size', type=int, default=23, help='单个新闻最大实体个数')
    parser.add_argument('--entity_embedding_dim', type=int, default=100, help='实体嵌入维数')
    parser.add_argument('--sample_num', type=int, default=5, help='采样点数')
    parser.add_argument('--neigh_num', type=int, default=5, help='邻居节点个数')
    # Multiview模型相关参数
    parser.add_argument('--category_size', type=int, default=15, help='主题总数')
    parser.add_argument('--subcategory_size', type=int, default=170, help='自主题总数')
    parser.add_argument('--category_dim', type=int, default=100, help='主题总数')
    parser.add_argument('--subcategory_dim', type=int, default=100, help='自主题总数')
    # DKN参数
    parser.add_argument('--kcnn_num_filters', type=int, default=50, help='卷积核个数')
    parser.add_argument('--kcnn_window_sizes', type=list, default=[2,3,4], help='窗口大小')
    parser.add_argument('--use_context', type=bool, default=None, help='自主题总数')
    # NAML参数
    parser.add_argument('--cnn_num_filters', type=int, default=400, help='卷积核个数')
    parser.add_argument('--cnn_window_sizes', type=int, default=3, help='窗口大小')
    parser.add_argument('--drop_prob', type=bool, default=0.2, help='丢弃概率')
    # LSTUR参数
    parser.add_argument('--long_short_term_method', type=str, default='ini', help='ini or con')
    parser.add_argument('--lstur_num_filters', type=int, default=300, help='卷积核个数')
    parser.add_argument('--lstur_window_sizes', type=int, default=3, help='窗口大小')
    parser.add_argument('--masking_probability', type=int, default=0.3, help='遮掩概率')
    # KRED参数
    parser.add_argument('--entity_pos_size', type=int, default=3, help='实体位置')
    parser.add_argument('--entity_freq_size', type=int, default=22, help='实体频率')
    parser.add_argument('--entity_cate_size', type=int, default=22, help='实体类别')
    # GERL参数
    parser.add_argument('--max_one_hop', type=int, default=50, help='交互最大一跳数')
    parser.add_argument('--max_two_hop', type=int, default=3, help='交互最大两跳数')
    # KIM参数
    parser.add_argument('--feature_dim', type=int, default=300, help='新闻特征维度')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    path = os.path.dirname(os.getcwd())
    print(os.path.dirname(os.getcwd()))
    print('模型名称：{}'.format(args.mode))
    new_popularity, new_entity_pos, new_entity_freq, new_entity_cate, \
    new_title_bert, new_new_feature, new_word_embedding, new_word_index, new_title_length, user_clicked_newindex, user_clicked_new_length, \
    new_entity_index, new_entity_embedding, \
    neigh_entity_index, neigh_entity_embedding, neigh_relation_index, neigh_relation_embedding, \
    new_category_index, new_subcategory_index, \
    candidate_newindex_train, user_index_train, label_train, \
    candidate_newindex_test, user_index_test, label_test, bound_test, \
    heterogeneous_user_graph_A, heterogeneous_user_graph_newindex, \
    heterogeneous_user_graph_entityindex, heterogeneous_user_graph_categoryindex, \
    user_one_hop_neighbor, user_two_hop_neighbor, new_one_hop_neighbor, new_two_hop_neighbor = data_generator(path)
    model = None
    #########################
    if args.mode == 'FM':
        model = FM(args)
    if args.mode == 'DeepFM':
        model = DeepFM(args)
    if args.mode == 'WideDeep':
        model = WideDeep(args)
    if args.mode == 'CNN':
        model = CNN(args)
    if args.mode == 'GCN':
        model = GCN(args)
    if args.mode == 'GRU':
        model = GRU(args)
    if args.mode == 'GCN&KGAT':
        model = GCN_KGAT(args)
    if args.mode == 'DKN':
        model = DKN(args)
    if args.mode == 'NAML':
        model = NAML(args)
    if args.mode == 'LSTUR':
        model = LSTUR(args)
    if args.mode == 'NPA':
        model = NPA(args)
    if args.mode == 'NRMS':
        model = NRMS(args)
    if args.mode == 'NRMS_mask':
        model = NRMS_mask(args)
    if args.mode == 'NRMS&GCN_mask':
        model = NRMS_GCN_mask(args)
    if args.mode == 'NRMS&GCN':
        model = NRMS_GCN(args)
    if args.mode == 'NRMS&GCN&KGAT':
        model = NRMS_GCN_KGAT(args)
    if args.mode == 'NRMS&GCN&KGAT_mask':
        model = NRMS_GCN_KGAT_mask(args)
    if args.mode == 'NRMS&GCN&KGAT&MV':
        model = NRMS_GCN_KGAT_MV(args)
    if args.mode == 'NRMS&GCN&KGAT&MV_bert':
        model = NRMS_GCN_KGAT_MV_bert(args)
    if args.mode == 'NRMS&GCN&KGAT&MV_mask':
        model = NRMS_GCN_KGAT_MV_mask(args)
    if args.mode == 'exp1':
        model = exp1(args)
    if args.mode == 'exp2':
        model = exp2(args)
    if args.mode == 'exp3':
        model = exp3(args)
    if args.mode == 'exp4':
        model = exp4(args)
    if args.mode == 'exp5':
        model = exp5(args)
    if args.mode == 'exp6':
        model = exp6(args)
    if args.mode == 'exp7':
        model = exp7(args)
    if args.mode == 'heter_graph':
        model = heter_graph(args)
    if args.mode == 'heter_graph_bert':
        model = heter_graph_bert(args)
    if args.mode == 'TANR':
        model = TANR(args)
    if args.mode == 'KIM':
        model = KIM(args)
    if args.mode == 'KRED':
        model = KRED(args)
    if args.mode == 'GERL':
        model = GERL(args)
    if args.mode == 'DAN':
        model = DAN(args)
    if args.mode == 'GNUD':
        model = GNUD(args)
    if args.mode == 'PENR':
        model = PENR(args)
    if args.mode == 'GNewsRec':
        model = GNewsRec(args)
    if args.mode == 'FIM':
        model = FIM(args)
    #########################
    print("torch.cuda.is_available() = ", torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)  # 移动模型到cuda
    # for name, parameters in model.named_parameters():
    #     print(name, ':', parameters.size())
    train_dataset = Train_Dataset(user_one_hop_neighbor, user_two_hop_neighbor, new_one_hop_neighbor, new_two_hop_neighbor,
                                  new_popularity, new_entity_pos, new_entity_freq, new_entity_cate,
                                  heterogeneous_user_graph_A, heterogeneous_user_graph_newindex,
                                  heterogeneous_user_graph_entityindex, heterogeneous_user_graph_categoryindex,
                                  new_title_bert, new_new_feature, new_word_embedding, new_word_index, new_title_length, user_clicked_newindex, user_clicked_new_length,
                                  new_entity_index, new_entity_embedding,
                                  neigh_entity_index, neigh_entity_embedding, neigh_relation_index, neigh_relation_embedding,
                                  new_category_index, new_subcategory_index,
                                  candidate_newindex_train, user_index_train, label_train, mode=args.mode)
    train_dataloader = DataLoader(train_dataset, batch_size=50, shuffle=True)
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': args.learning_rate}])
    for epoch in range(args.epoch):
        loss = train(epoch, len(train_dataset), train_dataloader, model, optimizer, args.batch_size, args.mode)
        if torch.isnan(loss):
            with open('result.txt','a') as save_file:
                save_file.write('epoch：{}, loss: {}'.format(epoch, loss))
            break

    ## 测试集
    pred_label_list = []
    pred_label = None
    test_dataset = Test_Dataset(user_one_hop_neighbor, user_two_hop_neighbor, new_one_hop_neighbor, new_two_hop_neighbor,
                                new_popularity, new_entity_pos, new_entity_freq, new_entity_cate,
                                heterogeneous_user_graph_A, heterogeneous_user_graph_newindex,
                                heterogeneous_user_graph_entityindex, heterogeneous_user_graph_categoryindex,
                                new_title_bert, new_new_feature, new_word_embedding, new_word_index, new_title_length, user_clicked_newindex,user_clicked_new_length,
                                new_entity_index, new_entity_embedding,
                                neigh_entity_index, neigh_entity_embedding,
                                neigh_relation_index, neigh_relation_embedding,
                                new_category_index, new_subcategory_index,
                                candidate_newindex_test, user_index_test, args.mode)
    test_dataloader = DataLoader(test_dataset, batch_size=50, shuffle=False)
    test_AUC, test_MRR, test_nDCG5, test_nDCG10 = test(len(test_dataset), test_dataloader, label_test, bound_test, model, args.batch_size, args.mode)

