from tqdm import tqdm
from torch.autograd import no_grad
from measure import *

def test( length, test_dataloader, label_test, bound_test, model, batch_size, mode):
    model.eval()
    pred_label_list = []
    pbar = tqdm(total=length)
    with no_grad():
        for data in test_dataloader:
            if mode == 'FM' or mode == 'DeepFM' or mode == 'WideDeep':
                candidate_newindex, user_clicked_newindex, candidate_new_featrue, user_clicked_new_featrue = data
                if candidate_new_featrue.shape[0] < batch_size:
                    break
                pred_label = model.forward(candidate_newindex, user_clicked_newindex, candidate_new_featrue, user_clicked_new_featrue)
            if mode == 'NRMS' or mode == 'CNN':
                candidate_new_word_embedding, user_clicked_new_word_embedding = data
                if user_clicked_new_word_embedding.shape[0] < batch_size:
                    break
                pred_label = model.forward(candidate_new_word_embedding, user_clicked_new_word_embedding)
            if mode == 'NRMS_mask':
                candidate_new_word_embedding, user_clicked_new_word_embedding,\
                candidate_new_title_length, user_clicked_new_title_length, user_clicked_new_length= data
                if user_clicked_new_word_embedding.shape[0] < batch_size:
                    break
                pred_label = model.forward(candidate_new_word_embedding, user_clicked_new_word_embedding,
                                           candidate_new_title_length, user_clicked_new_title_length, user_clicked_new_length)
            if mode == 'NRMS&GCN_mask':
                candidate_new_word_embedding, user_clicked_new_word_embedding, \
                candidate_new_entity_embedding, user_clicked_new_entity_embedding , \
                candidate_new_title_length, user_clicked_new_title_length, user_clicked_new_length = data
                if user_clicked_new_word_embedding.shape[0] < batch_size:
                    break
                pred_label = model.forward(candidate_new_word_embedding, user_clicked_new_word_embedding,
                                           candidate_new_entity_embedding, user_clicked_new_entity_embedding,
                                           candidate_new_title_length, user_clicked_new_title_length, user_clicked_new_length)
            if mode == 'NRMS&GCN&KGAT_mask':
                candidate_new_word_embedding, user_clicked_new_word_embedding, \
                candidate_new_entity_embedding, user_clicked_new_entity_embedding , \
                candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding, \
                candidate_new_neigh_relation_embedding, user_clicked_new_neigh_relation_embedding, \
                candidate_new_title_length, user_clicked_new_title_length, user_clicked_new_length = data
                if user_clicked_new_word_embedding.shape[0] < batch_size:
                    break
                pred_label = model.forward(candidate_new_word_embedding, user_clicked_new_word_embedding,
                                           candidate_new_entity_embedding, user_clicked_new_entity_embedding,
                                           candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding,
                                           candidate_new_neigh_relation_embedding,user_clicked_new_neigh_relation_embedding,
                                           candidate_new_title_length, user_clicked_new_title_length, user_clicked_new_length)
            if mode == 'GCN':
                candidate_new_entity_embedding, user_clicked_new_entity_embedding = data
                if candidate_new_entity_embedding.shape[0] < batch_size:
                    break
                pred_label = model.forward(candidate_new_entity_embedding, user_clicked_new_entity_embedding)
            if mode == 'GRU':
                candidate_new_word_embedding, user_clicked_new_word_embedding, user_clicked_new_length = data
                if user_clicked_new_word_embedding.shape[0] < batch_size:
                    break
                pred_label = model.forward(candidate_new_word_embedding, user_clicked_new_word_embedding, user_clicked_new_length)
            if mode == 'GCN&KGAT':
                candidate_new_entity_embedding, user_clicked_new_entity_embedding, \
                candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding, \
                candidate_new_neigh_relation_embedding, user_clicked_new_neigh_relation_embedding = data
                if candidate_new_entity_embedding.shape[0] < batch_size:
                    break
                pred_label = model.forward(candidate_new_entity_embedding, user_clicked_new_entity_embedding,
                                           candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding,
                                           candidate_new_neigh_relation_embedding, user_clicked_new_neigh_relation_embedding)
            if mode == 'NRMS&GCN' or mode == 'DKN':
                candidate_new_word_embedding, user_clicked_new_word_embedding, \
                candidate_new_entity_embedding, user_clicked_new_entity_embedding = data
                if user_clicked_new_word_embedding.shape[0] < batch_size:
                    break
                pred_label = model.forward(candidate_new_word_embedding, user_clicked_new_word_embedding,
                                           candidate_new_entity_embedding, user_clicked_new_entity_embedding)
            if mode == 'NAML':
                candidate_new_word_embedding, user_clicked_new_word_embedding, \
                candidate_new_category_index, user_clicked_new_category_index, \
                candidate_new_subcategory_index, user_clicked_new_subcategory_index = data
                if user_clicked_new_word_embedding.shape[0] < batch_size:
                    break
                pred_label = model.forward(candidate_new_word_embedding, user_clicked_new_word_embedding,
                                           candidate_new_category_index, user_clicked_new_category_index,
                                           candidate_new_subcategory_index, user_clicked_new_subcategory_index)

            if mode == 'FIM':
                candidate_new_word_embedding, user_clicked_new_word_embedding, \
                candidate_new_category_index, user_clicked_new_category_index, \
                candidate_new_subcategory_index, user_clicked_new_subcategory_index = data
                if user_clicked_new_word_embedding.shape[0] < batch_size:
                    break
                pred_label = model.forward(candidate_new_word_embedding, user_clicked_new_word_embedding,
                                           candidate_new_category_index, user_clicked_new_category_index,
                                           candidate_new_subcategory_index, user_clicked_new_subcategory_index)

            if mode == 'LSTUR':
                candidate_new_word_embedding, user_clicked_new_word_embedding, \
                candidate_new_category_index, user_clicked_new_category_index, \
                candidate_new_subcategory_index, user_clicked_new_subcategory_index, \
                user_clicked_new_length, user_index = data
                if user_clicked_new_word_embedding.shape[0] < batch_size:
                    break
                pred_label = model.forward(candidate_new_word_embedding, user_clicked_new_word_embedding,
                                           candidate_new_category_index, user_clicked_new_category_index,
                                           candidate_new_subcategory_index, user_clicked_new_subcategory_index,
                                           user_clicked_new_length, user_index)
            if mode == 'NRMS&MV':
                candidate_new_word_embedding, user_clicked_new_word_embedding, \
                candidate_new_category_index, user_clicked_new_category_index, \
                candidate_new_subcategory_index, user_clicked_new_subcategory_index = data
                if user_clicked_new_word_embedding.shape[0] < batch_size:
                    break
                pred_label = model.forward(candidate_new_word_embedding, user_clicked_new_word_embedding,
                                           candidate_new_category_index, user_clicked_new_category_index,
                                           candidate_new_subcategory_index, user_clicked_new_subcategory_index)
            if mode == 'NPA':
                candidate_new_word_embedding, user_clicked_new_word_embedding, user_index = data
                if user_clicked_new_word_embedding.shape[0] < batch_size:
                    break
                pred_label = model.forward(candidate_new_word_embedding, user_clicked_new_word_embedding, user_index)
            if mode == 'NRMS&GCN&KGAT':
                candidate_new_word_embedding, user_clicked_new_word_embedding, \
                candidate_new_entity_embedding, user_clicked_new_entity_embedding, \
                candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding, \
                candidate_new_neigh_relation_embedding, user_clicked_new_neigh_relation_embedding = data
                if user_clicked_new_word_embedding.shape[0] < batch_size:
                    break
                pred_label = model.forward(candidate_new_word_embedding, user_clicked_new_word_embedding,
                                           candidate_new_entity_embedding, user_clicked_new_entity_embedding,
                                           candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding,
                                           candidate_new_neigh_relation_embedding, user_clicked_new_neigh_relation_embedding)
            if mode == 'NRMS&GCN&KGAT&MV':
                candidate_new_word_embedding, user_clicked_new_word_embedding, \
                candidate_new_entity_embedding, user_clicked_new_entity_embedding, \
                candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding, \
                candidate_new_neigh_relation_embedding, user_clicked_new_neigh_relation_embedding, \
                candidate_new_category_index, user_clicked_new_category_index, \
                candidate_new_subcategory_index, user_clicked_new_subcategory_index = data
                if user_clicked_new_word_embedding.shape[0] < batch_size:
                    break
                pred_label = model.forward(candidate_new_word_embedding, user_clicked_new_word_embedding,
                                           candidate_new_entity_embedding, user_clicked_new_entity_embedding,
                                           candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding,
                                           candidate_new_neigh_relation_embedding, user_clicked_new_neigh_relation_embedding,
                                           candidate_new_category_index, user_clicked_new_category_index,
                                           candidate_new_subcategory_index, user_clicked_new_subcategory_index)
            if mode == 'heter_graph':
                candidate_new_word_embedding, user_clicked_new_word_embedding, \
                candidate_new_entity_embedding, user_clicked_new_entity_embedding, \
                candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding, \
                candidate_new_neigh_relation_embedding, user_clicked_new_neigh_relation_embedding, \
                candidate_new_category_index, user_clicked_new_category_index, \
                heterogeneous_user_graph_A, heterogeneous_user_graph_entity_embedding, \
                heterogeneous_user_graph_categoryindex= data

                if user_clicked_new_word_embedding.shape[0] < batch_size:
                    break
                pred_label = model.forward(candidate_new_word_embedding, user_clicked_new_word_embedding,
                                           candidate_new_entity_embedding, user_clicked_new_entity_embedding,
                                           candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding,
                                           candidate_new_neigh_relation_embedding, user_clicked_new_neigh_relation_embedding,
                                           candidate_new_category_index, user_clicked_new_category_index,
                                           candidate_new_subcategory_index, user_clicked_new_subcategory_index,
                                           heterogeneous_user_graph_A, heterogeneous_user_graph_entity_embedding,
                                           heterogeneous_user_graph_categoryindex)
            if mode == 'TANR':
                candidate_new_word_embedding, user_clicked_new_word_embedding, \
                candidate_new_category_index, user_clicked_new_category_index = data

                if user_clicked_new_word_embedding.shape[0] < batch_size:
                    break
                pred_label,_ = model.forward(candidate_new_word_embedding, user_clicked_new_word_embedding,
                                           candidate_new_category_index, user_clicked_new_category_index, )
            if mode == 'heter_graph_bert':
                candidate_new_title_bert, user_clicked_new_title_bert, \
                candidate_new_entity_embedding, user_clicked_new_entity_embedding, \
                candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding, \
                candidate_new_neigh_relation_embedding, user_clicked_new_neigh_relation_embedding, \
                candidate_new_category_index, user_clicked_new_category_index, \
                candidate_new_subcategory_index, user_clicked_new_subcategory_index, \
                heterogeneous_user_graph_A, heterogeneous_user_graph_entity_embedding, \
                heterogeneous_user_graph_categoryindex = data

                if candidate_new_title_bert.shape[0] < batch_size:
                    break
                pred_label = model.forward(candidate_new_title_bert, user_clicked_new_title_bert,
                                           candidate_new_entity_embedding, user_clicked_new_entity_embedding,
                                           candidate_new_neigh_entity_embedding,
                                           user_clicked_new_neigh_entity_embedding,
                                           candidate_new_neigh_relation_embedding,
                                           user_clicked_new_neigh_relation_embedding,
                                           candidate_new_category_index, user_clicked_new_category_index,
                                           candidate_new_subcategory_index, user_clicked_new_subcategory_index,
                                           heterogeneous_user_graph_A, heterogeneous_user_graph_entity_embedding,
                                           heterogeneous_user_graph_categoryindex
                                           )
            if mode == 'NRMS&GCN&KGAT&MV_bert':
                candidate_new_title_bert, user_clicked_new_title_bert, \
                candidate_new_entity_embedding, user_clicked_new_entity_embedding, \
                candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding, \
                candidate_new_neigh_relation_embedding, user_clicked_new_neigh_relation_embedding, \
                candidate_new_category_index, user_clicked_new_category_index, \
                candidate_new_subcategory_index, user_clicked_new_subcategory_index = data

                if candidate_new_title_bert.shape[0] < batch_size:
                    break
                pred_label = model.forward(candidate_new_title_bert, user_clicked_new_title_bert,
                                           candidate_new_entity_embedding, user_clicked_new_entity_embedding,
                                           candidate_new_neigh_entity_embedding,
                                           user_clicked_new_neigh_entity_embedding,
                                           candidate_new_neigh_relation_embedding,
                                           user_clicked_new_neigh_relation_embedding,
                                           candidate_new_category_index, user_clicked_new_category_index,
                                           candidate_new_subcategory_index, user_clicked_new_subcategory_index)
            if mode == 'exp7':
                candidate_new_word_embedding, user_clicked_new_word_embedding, \
                candidate_new_entity_embedding, user_clicked_new_entity_embedding, \
                candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding, \
                candidate_new_neigh_relation_embedding, user_clicked_new_neigh_relation_embedding, \
                candidate_new_category_index, user_clicked_new_category_index, \
                candidate_new_subcategory_index, user_clicked_new_subcategory_index, user_index = data

                if user_clicked_new_word_embedding.shape[0] < batch_size:
                    break
                pred_label = model.forward(candidate_new_word_embedding, user_clicked_new_word_embedding,
                                           candidate_new_entity_embedding, user_clicked_new_entity_embedding,
                                           candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding,
                                           candidate_new_neigh_relation_embedding, user_clicked_new_neigh_relation_embedding,
                                           candidate_new_category_index, user_clicked_new_category_index,
                                           candidate_new_subcategory_index, user_clicked_new_subcategory_index, user_index)

            if mode == 'exp1':
                candidate_new_word_embedding, user_clicked_new_word_embedding, \
                candidate_new_entity_embedding, user_clicked_new_entity_embedding, \
                candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding, \
                candidate_new_neigh_relation_embedding, user_clicked_new_neigh_relation_embedding, \
                candidate_new_category_index, user_clicked_new_category_index, \
                candidate_new_subcategory_index, user_clicked_new_subcategory_index ,\
                candidate_new_title_length, user_clicked_new_title_length, user_clicked_new_length, user_index = data

                if user_clicked_new_word_embedding.shape[0] < batch_size:
                    break
                pred_label = model.forward(candidate_new_word_embedding, user_clicked_new_word_embedding,
                                           candidate_new_entity_embedding, user_clicked_new_entity_embedding,
                                           candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding,
                                           candidate_new_neigh_relation_embedding, user_clicked_new_neigh_relation_embedding,
                                           candidate_new_category_index, user_clicked_new_category_index,
                                           candidate_new_subcategory_index, user_clicked_new_subcategory_index,
                                           candidate_new_title_length, user_clicked_new_title_length, user_clicked_new_length, user_index)
            if mode == 'exp2':
                candidate_new_word_embedding, user_clicked_new_word_embedding, \
                candidate_new_entity_embedding, user_clicked_new_entity_embedding, \
                candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding, \
                candidate_new_neigh_relation_embedding, user_clicked_new_neigh_relation_embedding, \
                candidate_new_category_index, user_clicked_new_category_index, \
                candidate_new_subcategory_index, user_clicked_new_subcategory_index ,\
                candidate_new_title_length, user_clicked_new_title_length, user_clicked_new_length, user_index = data
                if user_clicked_new_word_embedding.shape[0] < batch_size:
                    break
                pred_label,new_loss = model.forward(candidate_new_word_embedding, user_clicked_new_word_embedding,
                                           candidate_new_entity_embedding, user_clicked_new_entity_embedding,
                                           candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding,
                                           candidate_new_neigh_relation_embedding, user_clicked_new_neigh_relation_embedding,
                                           candidate_new_category_index, user_clicked_new_category_index,
                                           candidate_new_subcategory_index, user_clicked_new_subcategory_index,
                                           candidate_new_title_length, user_clicked_new_title_length, user_clicked_new_length, user_index)
            if mode == 'exp3' or mode == 'exp4' or mode == 'exp5' or mode == 'exp6':
                candidate_new_word_embedding, user_clicked_new_word_embedding, \
                candidate_new_entity_embedding, user_clicked_new_entity_embedding, \
                candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding, \
                candidate_new_neigh_relation_embedding, user_clicked_new_neigh_relation_embedding, \
                candidate_new_category_index, user_clicked_new_category_index, \
                candidate_new_title_length, user_clicked_new_title_length, user_clicked_new_length, user_index = data
                if user_clicked_new_word_embedding.shape[0] < batch_size:
                    break
                pred_label = model.forward(candidate_new_word_embedding, user_clicked_new_word_embedding,
                                           candidate_new_entity_embedding, user_clicked_new_entity_embedding,
                                           candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding,
                                           candidate_new_neigh_relation_embedding, user_clicked_new_neigh_relation_embedding,
                                           candidate_new_category_index, user_clicked_new_category_index,
                                           candidate_new_title_length, user_clicked_new_title_length, user_clicked_new_length, user_index)
            if mode == 'NRMS&GCN&KGAT&MV_mask':
                candidate_new_word_embedding, user_clicked_new_word_embedding, \
                candidate_new_entity_embedding, user_clicked_new_entity_embedding, \
                candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding, \
                candidate_new_neigh_relation_embedding, user_clicked_new_neigh_relation_embedding, \
                candidate_new_category_index, user_clicked_new_category_index, \
                candidate_new_subcategory_index, user_clicked_new_subcategory_index ,\
                candidate_new_title_length, user_clicked_new_title_length, user_clicked_new_length = data
                if user_clicked_new_word_embedding.shape[0] < batch_size:
                    break
                pred_label = model.forward(candidate_new_word_embedding, user_clicked_new_word_embedding,
                                           candidate_new_entity_embedding, user_clicked_new_entity_embedding,
                                           candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding,
                                           candidate_new_neigh_relation_embedding, user_clicked_new_neigh_relation_embedding,
                                           candidate_new_category_index, user_clicked_new_category_index,
                                           candidate_new_subcategory_index, user_clicked_new_subcategory_index,
                                           candidate_new_title_length, user_clicked_new_title_length, user_clicked_new_length)

            if mode == 'KIM':
                candidate_new_word_embedding, user_clicked_new_word_embedding, \
                candidate_new_entity_embedding, user_clicked_new_entity_embedding, \
                candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding = data
                if user_clicked_new_word_embedding.shape[0] < batch_size:
                    break
                pred_label = model.forward(candidate_new_word_embedding, user_clicked_new_word_embedding,
                                           candidate_new_entity_embedding, user_clicked_new_entity_embedding,
                                           candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding)
            if mode == 'KRED':
                candidate_new_word_embedding, user_clicked_new_word_embedding, \
                candidate_new_entity_embedding, user_clicked_new_entity_embedding, \
                candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding, \
                candidate_new_neigh_relation_embedding, user_clicked_new_neigh_relation_embedding, \
                candidate_new_entity_pos, user_clicked_new_entity_pos, \
                candidate_new_entity_freq, user_clicked_new_entity_freq, \
                candidate_new_entity_cate, user_clicked_new_entity_cate = data
                if user_clicked_new_word_embedding.shape[0] < batch_size:
                    break
                pred_label = model.forward(candidate_new_word_embedding, user_clicked_new_word_embedding,
                                           candidate_new_entity_embedding, user_clicked_new_entity_embedding,
                                           candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding,
                                           candidate_new_neigh_relation_embedding, user_clicked_new_neigh_relation_embedding,
                                           candidate_new_entity_pos, user_clicked_new_entity_pos,
                                           candidate_new_entity_freq, user_clicked_new_entity_freq,
                                           candidate_new_entity_cate, user_clicked_new_entity_cate)
            if mode == 'GERL':
                user_index, user_one_hop_new_word_embedding, \
                user_two_hop_neighbor, candidate_new_word_embedding, \
                new_two_hop_neighbor, new_two_hop_neighbor_new_word_embedding = data
                if user_index.shape[0] < batch_size:
                    break
                pred_label = model.forward(user_index, user_one_hop_new_word_embedding,
                                           user_two_hop_neighbor, candidate_new_word_embedding,
                                           new_two_hop_neighbor, new_two_hop_neighbor_new_word_embedding)
            if mode == 'DAN':
                candidate_new_word_embedding, user_clicked_new_word_embedding, \
                candidate_new_entity_embedding, user_clicked_new_entity_embedding, \
                candidate_new_entity_cate, user_clicked_new_entity_cate = data
                if user_clicked_new_word_embedding.shape[0] < batch_size:
                    break
                pred_label = model.forward(candidate_new_word_embedding, user_clicked_new_word_embedding,
                                           candidate_new_entity_embedding, user_clicked_new_entity_embedding,
                                           candidate_new_entity_cate, user_clicked_new_entity_cate)
            if mode == 'GNUD':
                user_index, user_clicked_new_word_embedding, user_clicked_new_entity_embedding, user_clicked_new_entity_cate, \
                candidate_new_word_embedding, candidate_new_entity_embedding, candidate_clicked_new_entity_cate, new_one_hop_neighbor = data
                if user_clicked_new_word_embedding.shape[0] < batch_size:
                    break
                pred_label = model.forward(user_index, user_clicked_new_word_embedding, user_clicked_new_entity_embedding, user_clicked_new_entity_cate,
                                           candidate_new_word_embedding, candidate_new_entity_embedding, candidate_clicked_new_entity_cate, new_one_hop_neighbor)
            if mode == 'PENR':
                candidate_new_word_embedding, user_clicked_new_word_embedding, \
                candidate_new_popularity, user_clicked_new_popularity = data
                if user_clicked_new_word_embedding.shape[0] < batch_size:
                    break
                pred_label = model.forward(candidate_new_word_embedding, user_clicked_new_word_embedding,
                                           candidate_new_popularity, user_clicked_new_popularity)
            if mode == 'GNewsRec':
                candidate_new_word_embedding, user_clicked_new_word_embedding, \
                candidate_new_entity_embedding, user_clicked_new_entity_embedding, \
                candidate_new_entity_cate, user_clicked_new_entity_cate, \
                new_one_hop_neighbor, new_two_hop_neighbor_new_word_embedding,\
                new_two_hop_neighbor_new_entity_embedding, new_two_hop_neighbor_new_entity_cate, \
                candidate_new_category_index = data
                if user_clicked_new_word_embedding.shape[0] < batch_size:
                    break
                pred_label = model.forward(candidate_new_word_embedding, user_clicked_new_word_embedding,
                                           candidate_new_entity_embedding, user_clicked_new_entity_embedding,
                                           candidate_new_entity_cate, user_clicked_new_entity_cate,
                                           new_one_hop_neighbor, new_two_hop_neighbor_new_word_embedding,
                                           new_two_hop_neighbor_new_entity_embedding, new_two_hop_neighbor_new_entity_cate,
                                           candidate_new_category_index)
            pbar.update(batch_size)
            pred_label_list.extend(pred_label.cpu().numpy())
        pred_label_list = np.vstack(pred_label_list)
        pbar.close()
        test_AUC, test_MRR, test_nDCG5, test_nDCG10 = evaluate(pred_label_list, label_test, bound_test)
        print("test_AUC = %.4lf, test_MRR = %.4lf, test_nDCG5 = %.4lf, test_nDCG10 = %.4lf" %
              (test_AUC, test_MRR, test_nDCG5, test_nDCG10))

        return test_AUC, test_MRR, test_nDCG5, test_nDCG10
