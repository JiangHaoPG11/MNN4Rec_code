from tqdm import tqdm

def train( epoch, length, dataloader, model, optimizer, batch_size, mode, new_model = None , new_optimizer = None):
    model.train()
    loss = None
    sum_loss = 0.0
    sum_auc = 0.0
    step = 0.0
    pbar = tqdm(total=length)
    num_pbar = 0
    for data in dataloader:
        if mode == 'FM' or mode == 'DeepFM' or mode == 'WideDeep':
            candidate_newindex, user_clicked_newindex, candidate_new_featrue, user_clicked_new_featrue, label = data
            if label.shape[0] < batch_size:
                break
            optimizer.zero_grad()
            loss, auc = model.loss(candidate_newindex, user_clicked_newindex, candidate_new_featrue, user_clicked_new_featrue, label)
        if mode == 'NRMS' or mode == 'CNN':
            candidate_new_word_embedding, user_clicked_new_word_embedding, label = data
            if label.shape[0] < batch_size:
                break
            optimizer.zero_grad()
            loss, auc = model.loss(candidate_new_word_embedding, user_clicked_new_word_embedding, label)
        if mode == 'NRMS_mask':
            candidate_new_word_embedding, user_clicked_new_word_embedding,\
            candidate_new_title_length, user_clicked_new_title_length, user_clicked_new_length, label = data
            if label.shape[0] < batch_size:
                break
            optimizer.zero_grad()
            loss, auc = model.loss(candidate_new_word_embedding, user_clicked_new_word_embedding,
                                   candidate_new_title_length, user_clicked_new_title_length, user_clicked_new_length, label)
        if mode == 'NRMS&GCN_mask':
            candidate_new_word_embedding, user_clicked_new_word_embedding, \
            candidate_new_entity_embedding, user_clicked_new_entity_embedding, \
            candidate_new_title_length, user_clicked_new_title_length, user_clicked_new_length, label = data
            if label.shape[0] < batch_size:
                break
            optimizer.zero_grad()
            loss, auc = model.loss(candidate_new_word_embedding, user_clicked_new_word_embedding,
                                   candidate_new_entity_embedding, user_clicked_new_entity_embedding,
                                   candidate_new_title_length, user_clicked_new_title_length, user_clicked_new_length, label)
        if mode == 'NRMS&GCN&KGAT_mask':
            candidate_new_word_embedding, user_clicked_new_word_embedding, \
            candidate_new_entity_embedding, user_clicked_new_entity_embedding, \
            candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding, \
            candidate_new_neigh_relation_embedding, user_clicked_new_neigh_relation_embedding,\
            candidate_new_title_length, user_clicked_new_title_length, user_clicked_new_length, label = data
            if label.shape[0] < batch_size:
                break
            optimizer.zero_grad()
            loss, auc = model.loss(candidate_new_word_embedding, user_clicked_new_word_embedding,
                                   candidate_new_entity_embedding, user_clicked_new_entity_embedding,
                                   candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding,
                                   candidate_new_neigh_relation_embedding, user_clicked_new_neigh_relation_embedding,
                                   candidate_new_title_length, user_clicked_new_title_length, user_clicked_new_length, label)

        if mode == 'GCN':
            candidate_new_entity_embedding, user_clicked_new_entity_embedding, label = data
            if label.shape[0] < batch_size:
                break
            optimizer.zero_grad()
            loss, auc = model.loss(candidate_new_entity_embedding, user_clicked_new_entity_embedding, label)
        if mode == 'GRU':
            candidate_new_word_embedding, user_clicked_new_word_embedding, user_clicked_new_length,  label = data
            if label.shape[0] < batch_size:
                break
            optimizer.zero_grad()
            loss,auc = model.loss(candidate_new_word_embedding, user_clicked_new_word_embedding, user_clicked_new_length,label)
        if mode == 'GCN&KGAT':
            candidate_new_entity_embedding, user_clicked_new_entity_embedding, \
            candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding, \
            candidate_new_neigh_relation_embedding, user_clicked_new_neigh_relation_embedding, label = data
            if label.shape[0] < batch_size:
                break
            optimizer.zero_grad()
            loss,auc = model.loss(candidate_new_entity_embedding, user_clicked_new_entity_embedding,
                                  candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding,
                                  candidate_new_neigh_relation_embedding, user_clicked_new_neigh_relation_embedding, label)
        if mode == 'NAML':
            candidate_new_word_embedding, user_clicked_new_word_embedding, \
            candidate_new_category_index, user_clicked_new_category_index, \
            candidate_new_subcategory_index, user_clicked_new_subcategory_index, label = data
            if label.shape[0] < batch_size:
                break
            optimizer.zero_grad()
            loss,auc = model.loss(candidate_new_word_embedding, user_clicked_new_word_embedding,
                                  candidate_new_category_index, user_clicked_new_category_index,
                                  candidate_new_subcategory_index, user_clicked_new_subcategory_index, label)

        if mode == 'FIM':
            candidate_new_word_embedding, user_clicked_new_word_embedding, \
            candidate_new_category_index, user_clicked_new_category_index, \
            candidate_new_subcategory_index, user_clicked_new_subcategory_index, label = data
            if label.shape[0] < batch_size:
                break
            optimizer.zero_grad()
            loss,auc = model.loss(candidate_new_word_embedding, user_clicked_new_word_embedding,
                                  candidate_new_category_index, user_clicked_new_category_index,
                                  candidate_new_subcategory_index, user_clicked_new_subcategory_index, label)


        if mode == 'LSTUR':
            candidate_new_word_embedding, user_clicked_new_word_embedding, \
            candidate_new_category_index, user_clicked_new_category_index, \
            candidate_new_subcategory_index, user_clicked_new_subcategory_index, \
            user_clicked_new_length, user_index, label = data
            if label.shape[0] < batch_size:
                break
            optimizer.zero_grad()
            loss,auc = model.loss(candidate_new_word_embedding, user_clicked_new_word_embedding,
                                  candidate_new_category_index, user_clicked_new_category_index,
                                  candidate_new_subcategory_index, user_clicked_new_subcategory_index,
                                  user_clicked_new_length, user_index, label)
        if mode == 'NRMS&MV':
            candidate_new_word_embedding, user_clicked_new_word_embedding, \
            candidate_new_category_index, user_clicked_new_category_index, \
            candidate_new_subcategory_index, user_clicked_new_subcategory_index, label = data
            if label.shape[0] < batch_size:
                break
            optimizer.zero_grad()
            loss,auc = model.loss(candidate_new_word_embedding, user_clicked_new_word_embedding,
                                  candidate_new_category_index, user_clicked_new_category_index,
                                  candidate_new_subcategory_index, user_clicked_new_subcategory_index, label)
        if mode == 'NPA':
            candidate_new_word_embedding, user_clicked_new_word_embedding, user_index, label = data
            if label.shape[0] < batch_size:
                break
            optimizer.zero_grad()
            loss, auc = model.loss(candidate_new_word_embedding, user_clicked_new_word_embedding, user_index, label)

        if mode == 'NRMS&GCN' or mode == 'DKN':
            candidate_new_word_embedding, user_clicked_new_word_embedding, \
            candidate_new_entity_embedding, user_clicked_new_entity_embedding, label = data
            if label.shape[0] < batch_size:
                break
            optimizer.zero_grad()
            loss, auc = model.loss(candidate_new_word_embedding, user_clicked_new_word_embedding,
                                   candidate_new_entity_embedding, user_clicked_new_entity_embedding, label)
        if mode == 'NRMS&GCN&KGAT':
            candidate_new_word_embedding, user_clicked_new_word_embedding, \
            candidate_new_entity_embedding, user_clicked_new_entity_embedding, \
            candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding, \
            candidate_new_neigh_relation_embedding, user_clicked_new_neigh_relation_embedding, label = data
            if label.shape[0] < batch_size:
                break
            optimizer.zero_grad()
            loss,auc = model.loss(candidate_new_word_embedding, user_clicked_new_word_embedding,
                                  candidate_new_entity_embedding, user_clicked_new_entity_embedding,
                                  candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding,
                                  candidate_new_neigh_relation_embedding, user_clicked_new_neigh_relation_embedding, label)

        if mode == 'NRMS&GCN&KGAT&MV':
            candidate_new_word_embedding, user_clicked_new_word_embedding, \
            candidate_new_entity_embedding, user_clicked_new_entity_embedding, \
            candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding, \
            candidate_new_neigh_relation_embedding, user_clicked_new_neigh_relation_embedding, \
            candidate_new_category_index, user_clicked_new_category_index, \
            candidate_new_subcategory_index, user_clicked_new_subcategory_index, label = data
            if label.shape[0] < batch_size:
                break
            optimizer.zero_grad()
            loss, auc = model.loss(candidate_new_word_embedding, user_clicked_new_word_embedding,
                                   candidate_new_entity_embedding, user_clicked_new_entity_embedding,
                                   candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding,
                                   candidate_new_neigh_relation_embedding, user_clicked_new_neigh_relation_embedding,
                                   candidate_new_category_index, user_clicked_new_category_index,
                                   candidate_new_subcategory_index, user_clicked_new_subcategory_index, label)
        if mode == 'heter_graph':
            candidate_new_word_embedding, user_clicked_new_word_embedding, \
            candidate_new_entity_embedding, user_clicked_new_entity_embedding, \
            candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding, \
            candidate_new_neigh_relation_embedding, user_clicked_new_neigh_relation_embedding, \
            candidate_new_category_index, user_clicked_new_category_index, \
            candidate_new_subcategory_index, user_clicked_new_subcategory_index, \
            heterogeneous_user_graph_A, heterogeneous_user_graph_entity_embedding,\
            heterogeneous_user_graph_categoryindex, label = data
            if label.shape[0] < batch_size:
                break
            optimizer.zero_grad()
            loss, auc = model.loss(candidate_new_word_embedding, user_clicked_new_word_embedding,
                                   candidate_new_entity_embedding, user_clicked_new_entity_embedding,
                                   candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding,
                                   candidate_new_neigh_relation_embedding, user_clicked_new_neigh_relation_embedding,
                                   candidate_new_category_index, user_clicked_new_category_index,
                                   candidate_new_subcategory_index, user_clicked_new_subcategory_index,
                                   heterogeneous_user_graph_A, heterogeneous_user_graph_entity_embedding,
                                   heterogeneous_user_graph_categoryindex, label)
        if mode == 'heter_graph_bert':
            candidate_new_title_bert, user_clicked_new_title_bert, \
            candidate_new_entity_embedding, user_clicked_new_entity_embedding, \
            candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding, \
            candidate_new_neigh_relation_embedding, user_clicked_new_neigh_relation_embedding, \
            candidate_new_category_index, user_clicked_new_category_index, \
            candidate_new_subcategory_index, user_clicked_new_subcategory_index,\
            heterogeneous_user_graph_A, heterogeneous_user_graph_entity_embedding,\
            heterogeneous_user_graph_categoryindex, label = data
            if label.shape[0] < batch_size:
                break
            optimizer.zero_grad()
            loss, auc = model.loss(candidate_new_title_bert, user_clicked_new_title_bert,
                                   candidate_new_entity_embedding, user_clicked_new_entity_embedding,
                                   candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding,
                                   candidate_new_neigh_relation_embedding, user_clicked_new_neigh_relation_embedding,
                                   candidate_new_category_index, user_clicked_new_category_index,
                                   candidate_new_subcategory_index, user_clicked_new_subcategory_index,
                                   heterogeneous_user_graph_A, heterogeneous_user_graph_entity_embedding,
                                   heterogeneous_user_graph_categoryindex, label)
        if mode == 'NRMS&GCN&KGAT&MV_bert':
            candidate_new_title_bert, user_clicked_new_title_bert, \
            candidate_new_entity_embedding, user_clicked_new_entity_embedding, \
            candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding, \
            candidate_new_neigh_relation_embedding, user_clicked_new_neigh_relation_embedding, \
            candidate_new_category_index, user_clicked_new_category_index, \
            candidate_new_subcategory_index, user_clicked_new_subcategory_index, label = data
            if label.shape[0] < batch_size:
                break
            optimizer.zero_grad()
            loss, auc = model.loss(candidate_new_title_bert, user_clicked_new_title_bert,
                                   candidate_new_entity_embedding, user_clicked_new_entity_embedding,
                                   candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding,
                                   candidate_new_neigh_relation_embedding, user_clicked_new_neigh_relation_embedding,
                                   candidate_new_category_index, user_clicked_new_category_index,
                                   candidate_new_subcategory_index, user_clicked_new_subcategory_index, label)
        if mode == 'TANR':
            candidate_new_word_embedding, user_clicked_new_word_embedding, \
            candidate_new_category_index, user_clicked_new_category_index, label = data
            if label.shape[0] < batch_size:
                break
            optimizer.zero_grad()
            loss, auc = model.loss(candidate_new_word_embedding, user_clicked_new_word_embedding,
                                   candidate_new_category_index, user_clicked_new_category_index, label)
        if mode == 'exp7':
            candidate_new_word_embedding, user_clicked_new_word_embedding, \
            candidate_new_entity_embedding, user_clicked_new_entity_embedding, \
            candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding, \
            candidate_new_neigh_relation_embedding, user_clicked_new_neigh_relation_embedding, \
            candidate_new_category_index, user_clicked_new_category_index, \
            candidate_new_subcategory_index, user_clicked_new_subcategory_index, user_index, label = data
            if label.shape[0] < batch_size:
                break
            optimizer.zero_grad()
            loss, auc = model.loss(candidate_new_word_embedding, user_clicked_new_word_embedding,
                                   candidate_new_entity_embedding, user_clicked_new_entity_embedding,
                                   candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding,
                                   candidate_new_neigh_relation_embedding, user_clicked_new_neigh_relation_embedding,
                                   candidate_new_category_index, user_clicked_new_category_index,
                                   candidate_new_subcategory_index, user_clicked_new_subcategory_index, user_index, label)

        if mode == 'NRMS&GCN&KGAT&MV_mask':
            candidate_new_word_embedding, user_clicked_new_word_embedding, \
            candidate_new_entity_embedding, user_clicked_new_entity_embedding, \
            candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding, \
            candidate_new_neigh_relation_embedding, user_clicked_new_neigh_relation_embedding, \
            candidate_new_category_index, user_clicked_new_category_index, \
            candidate_new_subcategory_index, user_clicked_new_subcategory_index, \
            candidate_new_title_length, user_clicked_new_title_length, user_clicked_new_length, label = data

            if label.shape[0] < batch_size:
                break
            optimizer.zero_grad()
            loss, auc = model.loss(candidate_new_word_embedding, user_clicked_new_word_embedding,
                                   candidate_new_entity_embedding, user_clicked_new_entity_embedding,
                                   candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding,
                                   candidate_new_neigh_relation_embedding, user_clicked_new_neigh_relation_embedding,
                                   candidate_new_category_index, user_clicked_new_category_index,
                                   candidate_new_subcategory_index, user_clicked_new_subcategory_index,
                                   candidate_new_title_length, user_clicked_new_title_length, user_clicked_new_length, label)
        if mode == 'exp1':
            candidate_new_word_embedding, user_clicked_new_word_embedding, \
            candidate_new_entity_embedding, user_clicked_new_entity_embedding, \
            candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding, \
            candidate_new_neigh_relation_embedding, user_clicked_new_neigh_relation_embedding, \
            candidate_new_category_index, user_clicked_new_category_index, \
            candidate_new_subcategory_index, user_clicked_new_subcategory_index, \
            candidate_new_title_length, user_clicked_new_title_length, user_clicked_new_length, user_index, label = data

            if label.shape[0] < batch_size:
                break
            optimizer.zero_grad()
            loss, auc = model.loss(candidate_new_word_embedding, user_clicked_new_word_embedding,
                                   candidate_new_entity_embedding, user_clicked_new_entity_embedding,
                                   candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding,
                                   candidate_new_neigh_relation_embedding, user_clicked_new_neigh_relation_embedding,
                                   candidate_new_category_index, user_clicked_new_category_index,
                                   candidate_new_subcategory_index, user_clicked_new_subcategory_index,
                                   candidate_new_title_length, user_clicked_new_title_length, user_clicked_new_length, user_index, label)
        if mode == 'exp2':
            candidate_new_word_embedding, user_clicked_new_word_embedding, \
            candidate_new_entity_embedding, user_clicked_new_entity_embedding, \
            candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding, \
            candidate_new_neigh_relation_embedding, user_clicked_new_neigh_relation_embedding, \
            candidate_new_category_index, user_clicked_new_category_index, \
            candidate_new_subcategory_index, user_clicked_new_subcategory_index, \
            candidate_new_title_length, user_clicked_new_title_length, user_clicked_new_length, user_index, label = data
            if label.shape[0] < batch_size:
                break
            optimizer.zero_grad()
            loss, auc = model.loss(candidate_new_word_embedding, user_clicked_new_word_embedding,
                                   candidate_new_entity_embedding, user_clicked_new_entity_embedding,
                                   candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding,
                                   candidate_new_neigh_relation_embedding, user_clicked_new_neigh_relation_embedding,
                                   candidate_new_category_index, user_clicked_new_category_index,
                                   candidate_new_subcategory_index, user_clicked_new_subcategory_index,
                                   candidate_new_title_length, user_clicked_new_title_length, user_clicked_new_length, user_index, label)
        if mode == 'exp3' or mode == 'exp4' or mode == 'exp5' or mode == 'exp6':
            candidate_new_word_embedding, user_clicked_new_word_embedding, \
            candidate_new_entity_embedding, user_clicked_new_entity_embedding, \
            candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding, \
            candidate_new_neigh_relation_embedding, user_clicked_new_neigh_relation_embedding, \
            candidate_new_category_index, user_clicked_new_category_index, \
            candidate_new_title_length, user_clicked_new_title_length, user_clicked_new_length, user_index, label = data
            if label.shape[0] < batch_size:
                break
            optimizer.zero_grad()
            loss, auc = model.loss(candidate_new_word_embedding, user_clicked_new_word_embedding,
                                   candidate_new_entity_embedding, user_clicked_new_entity_embedding,
                                   candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding,
                                   candidate_new_neigh_relation_embedding, user_clicked_new_neigh_relation_embedding,
                                   candidate_new_category_index, user_clicked_new_category_index,
                                   candidate_new_title_length, user_clicked_new_title_length, user_clicked_new_length, user_index, label)
        if mode == 'KIM':
            candidate_new_word_embedding, user_clicked_new_word_embedding, \
            candidate_new_entity_embedding, user_clicked_new_entity_embedding, \
            candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding, label = data
            if label.shape[0] < batch_size:
                break
            optimizer.zero_grad()
            loss, auc = model.loss(candidate_new_word_embedding, user_clicked_new_word_embedding,
                                   candidate_new_entity_embedding, user_clicked_new_entity_embedding,
                                   candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding, label)

        if mode == 'KRED':
            candidate_new_word_embedding, user_clicked_new_word_embedding, \
            candidate_new_entity_embedding, user_clicked_new_entity_embedding, \
            candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding, \
            candidate_new_neigh_relation_embedding, user_clicked_new_neigh_relation_embedding, \
            candidate_new_entity_pos, user_clicked_new_entity_pos, \
            candidate_new_entity_freq, user_clicked_new_entity_freq, \
            candidate_new_entity_cate, user_clicked_new_entity_cate, label = data
            if label.shape[0] < batch_size:
                break
            optimizer.zero_grad()
            loss, auc = model.loss(candidate_new_word_embedding, user_clicked_new_word_embedding,
                                   candidate_new_entity_embedding, user_clicked_new_entity_embedding,
                                   candidate_new_neigh_entity_embedding, user_clicked_new_neigh_entity_embedding,
                                   candidate_new_neigh_relation_embedding, user_clicked_new_neigh_relation_embedding,
                                   candidate_new_entity_pos, user_clicked_new_entity_pos,
                                   candidate_new_entity_freq, user_clicked_new_entity_freq,
                                   candidate_new_entity_cate, user_clicked_new_entity_cate, label)
        if mode == 'GERL':
            user_index, user_one_hop_new_word_embedding, \
            user_two_hop_neighbor, candidate_new_word_embedding, \
            new_two_hop_neighbor, new_two_hop_neighbor_new_word_embedding, label = data
            if label.shape[0] < batch_size:
                break
            optimizer.zero_grad()
            loss, auc = model.loss(user_index, user_one_hop_new_word_embedding,
                                   user_two_hop_neighbor, candidate_new_word_embedding,
                                   new_two_hop_neighbor, new_two_hop_neighbor_new_word_embedding, label)
        if mode == 'DAN':
            candidate_new_word_embedding, user_clicked_new_word_embedding, \
            candidate_new_entity_embedding, user_clicked_new_entity_embedding, \
            candidate_new_entity_cate, user_clicked_new_entity_cate, label = data
            if label.shape[0] < batch_size:
                break
            optimizer.zero_grad()
            loss, auc = model.loss(candidate_new_word_embedding, user_clicked_new_word_embedding,
                                   candidate_new_entity_embedding, user_clicked_new_entity_embedding,
                                   candidate_new_entity_cate, user_clicked_new_entity_cate, label)

        if mode == 'GNUD':
            user_index, user_clicked_new_word_embedding, user_clicked_new_entity_embedding, user_clicked_new_entity_cate,  \
            candidate_new_word_embedding, candidate_new_entity_embedding, candidate_new_entity_cate, new_one_hop_neighbor,\
            label = data
            if label.shape[0] < batch_size:
                break
            optimizer.zero_grad()
            loss, auc = model.loss(user_index, user_clicked_new_word_embedding, user_clicked_new_entity_embedding, user_clicked_new_entity_cate,
                                   candidate_new_word_embedding, candidate_new_entity_embedding, candidate_new_entity_cate, new_one_hop_neighbor,
                                    label)
        if mode == 'PENR':
            candidate_new_word_embedding, user_clicked_new_word_embedding, \
            candidate_new_popularity, user_clicked_new_popularity, \
            label = data
            if label.shape[0] < batch_size:
                break
            optimizer.zero_grad()
            loss, auc = model.loss(candidate_new_word_embedding, user_clicked_new_word_embedding,
                                   candidate_new_popularity, user_clicked_new_popularity, label)
        if mode == 'GNewsRec':
            candidate_new_word_embedding, user_clicked_new_word_embedding, \
            candidate_new_entity_embedding, user_clicked_new_entity_embedding, \
            candidate_new_entity_cate, user_clicked_new_entity_cate, \
            new_one_hop_neighbor, new_two_hop_neighbor_new_word_embedding,\
            new_two_hop_neighbor_new_entity_embedding, new_two_hop_neighbor_new_entity_cate, \
            candidate_new_category_index, label = data
            if label.shape[0] < batch_size:
                break
            optimizer.zero_grad()
            loss, auc = model.loss(candidate_new_word_embedding, user_clicked_new_word_embedding,
                                   candidate_new_entity_embedding, user_clicked_new_entity_embedding,
                                   candidate_new_entity_cate, user_clicked_new_entity_cate, new_one_hop_neighbor,
                                   new_two_hop_neighbor_new_word_embedding, new_two_hop_neighbor_new_entity_embedding,
                                   new_two_hop_neighbor_new_entity_cate, candidate_new_category_index, label)
        loss.backward()
        optimizer.step()
        pbar.update(batch_size)
        num_pbar += batch_size
        sum_loss += loss.cpu().item()
        sum_auc += auc
        step += 1.0
    pbar.close()
    print('epoch:{}----------------- loss value:{} auc value:{}--------------'.format(epoch+1, sum_loss/step, sum_auc/step))
    return loss
