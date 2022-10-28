import random
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

entity_index_df = pd.read_csv('./df/entity_index_df.csv')
entity_embedding_df = pd.read_table('MIND_small/MINDsmall_train/entity_embedding.vec', header=None)
relation_embedding_df = pd.read_table('MIND_small/MINDsmall_train/relation_embedding.vec', header=None)

### 获取KG三元组
def get_KG_construct():
    '''
    构建知识图谱
    :return: dict(kg)：kg的键是头部实体，kg的值是尾部实体和关系
    '''
    print('constructing adjacency matrix ...')
    graph_file_fp = open('../wikidata-graph/wikidata-graph.tsv', 'r', encoding='utf-8')
    graph = []
    index = 0
    for line in graph_file_fp:
        linesplit = line.split('\n')[0].split('\t')
        if len(linesplit) > 1:
            graph.append([linesplit[0], linesplit[1],  linesplit[2]])
        index += 1
        # if index > 50:
        #     break
    print(len(graph))
    kg = {}
    index = 0
    for triple in graph:
        index += 1
        head = triple[0]
        relation = triple[1]
        tail = triple[2]
        if head not in kg:
            kg[head] = []
        kg[head].append((tail, relation))
        if tail not in kg:
            kg[tail] = []
        kg[tail].append((head, relation))
    return kg

def get_onehop_neighbor(kg, entity_index_df, entity_neighbor_num = 5):
    '''
    得到一跳邻居实体
    :param kg: 知识图谱字典
    :param entity_index_df: 实体id和index对应df
    :param entity_neighbor_num: 搜索的邻居实体个数 = 5
    :return: List(neigh_entity_adj): 对应新闻实体的邻居实体id列表
             List(neigh_relation_adj)：对应新闻实体的邻居实体id列表
             entity_index_df ：添加邻居实体之后的df
    '''
    entity_id_list = entity_index_df['id'].values.tolist()
    print(entity_id_list[0])
    neigh_entity_adj = []
    neigh_relation_adj = []

    for entity_id in entity_id_list:
        neigh_entity_adj_one = []
        neigh_relation_adj_one = []
        if  entity_id in kg.keys():
            for index in range(entity_neighbor_num):
                i = random.randint(0, len(kg[entity_id]) - 1)
                neigh_entity_adj_one.append(kg[entity_id][i][0])
                neigh_relation_adj_one.append(kg[entity_id][i][1])
        else:
            for index in range(entity_neighbor_num):
                neigh_entity_adj_one.append('padding')
                neigh_relation_adj_one.append('padding')
        neigh_entity_adj.append(neigh_entity_adj_one)
        neigh_relation_adj.append(neigh_relation_adj_one)
    entity_index_df['one-hop neighbor entity'] = neigh_entity_adj
    entity_index_df['one-hop neighbor relation'] = neigh_relation_adj
    print(1)
    return neigh_entity_adj, neigh_relation_adj, entity_index_df

def get_onehop_neighbor_entity_index(entity_index_df, neigh_entity_adj):
    '''
    映射邻居实体id为index，并构建邻居实体index字典
    :param entity_index_df: 实体id 和 index对应df
    :param neigh_entity_adj: 邻居实体列表
    :return: List(neigh_entity_index_adj): 邻居实体index列表
             DataFrame(neigh_entity_index_df): 邻居实体id和邻居实体index列表
             DataFrame(entity_index_df): 实体id和实体index对应的df
    '''
    neigh_entity_dict = {}
    neigh_entity_dict['padding'] = 0
    index = 1
    for entity_list in neigh_entity_adj:
        for entity in entity_list:
            if (entity not in neigh_entity_dict.keys()) == True:
                neigh_entity_dict[entity] = index
                index += 1

    print(len(neigh_entity_adj))
    neigh_entity_index_adj = []
    for entity_list in neigh_entity_adj:
        entity_index_adj_one = []
        for entity in entity_list:
            if (entity not in neigh_entity_dict.keys()) == False:
                entity_index_adj_one.append(neigh_entity_dict[entity])
            else:
                entity_index_adj_one.append(neigh_entity_dict['padding'])
        neigh_entity_index_adj.append(entity_index_adj_one)
    print(len(neigh_entity_index_adj))
    entity_index_df['one-hop neighbor entity index'] = neigh_entity_index_adj

    np.save('./kg/neigh_entity_index.npy', np.array(neigh_entity_index_adj))

    neigh_entity_index_df = pd.DataFrame(pd.Series(neigh_entity_dict), columns=[ 'entity_index'])
    neigh_entity_index_df = neigh_entity_index_df.reset_index().rename(columns={'index': 'id'})
    neigh_entity_index_df.to_csv('./df/neigh_entity_index_df.csv')
    print('finally')
    return neigh_entity_index_adj, neigh_entity_index_df, entity_index_df

def get_onehop_neighbor_relation_index(neigh_relation_adj, entity_index_df):
    '''
    映射关系id为index
    :param neigh_relation_adj: 邻居实体关系id列表
    :param entity_index_df: 新闻实体id和index对应的df
    :return: List(neigh_relation_index_adj):邻居实体关系index
             DataFrame(neigh_relation_index_df): 邻居实体关系id对应实体关系index
    '''
    neigh_relation_dict = {}
    neigh_relation_dict['padding'] = 0

    index = 1
    for neigh_relation_list in neigh_relation_adj:
        for neigh_relation in neigh_relation_list:
            if (neigh_relation not in neigh_relation_dict.keys()) == True:
                neigh_relation_dict[neigh_relation] = index
                index += 1
    neigh_relation_index_adj = []

    for neigh_relation_list in neigh_relation_adj:
        neigh_relation_index_adj_one = []
        for neigh_relation in neigh_relation_list:
            if (neigh_relation not in neigh_relation_dict.keys()) == False:
                neigh_relation_index_adj_one.append(neigh_relation_dict[neigh_relation])
            else:
                neigh_relation_index_adj_one.append(neigh_relation_dict['padding'])
        neigh_relation_index_adj.append(neigh_relation_index_adj_one)

    entity_index_df['one-hop neighbor relation index'] = neigh_relation_index_adj
    entity_index_df.to_csv('./df/entity_index_df.csv')
    np.save('./kg/neigh_relation_index.npy',np.array(neigh_relation_index_adj))

    neigh_relation_index_df = pd.DataFrame(pd.Series(neigh_relation_dict), columns=['relation_index'])
    neigh_relation_index_df = neigh_relation_index_df.reset_index().rename(columns={'index': 'id'})
    neigh_relation_index_df.to_csv('./df/neigh_relation_index_df.csv')
    print('finally')
    return  neigh_relation_index_adj, neigh_relation_index_df

def get_onehop_neighbor_entity_embedding( neigh_entity_index_df, entity_embedding_df):
    '''
    获取邻居实体嵌入
    :param neigh_entity_index_df: 实体id和index对应的df
    :param entity_embedding_df: 实体嵌入列表
    :return:
        neigh_entity_index_df: 加入实体id和实体嵌入
    '''
    entity_embedding_df['vector'] = entity_embedding_df.iloc[:, 1:101].values.tolist()
    entity_embedding_df = entity_embedding_df[[0, 'vector']].rename(columns={0: "entity"})
    print(entity_embedding_df)
    neigh_entity_id_list = neigh_entity_index_df['id'].values.tolist()
    neigh_entity_embedding_list = []
    empty_id = 100 * [0]

    df_add = pd.DataFrame()
    df_add['entity'] = ['padding']
    df_add['vector'] = [np.array(empty_id)]
    entity_embedding_df = entity_embedding_df.append(df_add)

    for neigh_entity_id in neigh_entity_id_list:
        neigh_entity_embedding = entity_embedding_df[entity_embedding_df['entity'] == neigh_entity_id]['vector'].values
        if len(neigh_entity_embedding) == 0:

            df_add = pd.DataFrame()
            df_add['entity'] = [neigh_entity_id]
            df_add['vector'] = [np.random.normal(-0.1, 0.1, 100)]
            entity_embedding_df = entity_embedding_df.append(df_add)
            neigh_entity_embedding = entity_embedding_df[entity_embedding_df['entity'] == neigh_entity_id]['vector'].values

            neigh_entity_embedding_list.append(neigh_entity_embedding[0])
        else:
            neigh_entity_embedding_list.append(neigh_entity_embedding[0])
        # print(neigh_entity_embedding_list[-1])
    neigh_entity_index_df['neigh_entity_embedding'] = neigh_entity_embedding_list
    neigh_entity_index_df = neigh_entity_index_df.fillna(method='ffill')
    neigh_embedding_list = neigh_entity_index_df['neigh_entity_embedding'].values.tolist()

    neigh_entity_embedding_index = np.array(neigh_embedding_list)
    np.save('./kg/neigh_entity_embedding.npy', neigh_entity_embedding_index)

    return neigh_entity_index_df

def get_onehop_neighbor_relation_embedding(neigh_relation_index_df, relation_embedding_df):
    '''
    获取邻居实体关系嵌入
    :param neigh_relation_index_df:
    :param relation_embedding_df:
    :return: DataFrame(neigh_relation_index_df):
    '''
    relation_embedding_df['vector'] = relation_embedding_df.iloc[:, 1:101].values.tolist()
    relation_embedding_df = relation_embedding_df[[0, 'vector']].rename(columns={0: "relation"})
    neigh_relation_id_list = neigh_relation_index_df['id'].values.tolist()
    neigh_relation_embedding_list = []
    empty_id = 100 * [0]

    df_add = pd.DataFrame()
    df_add['relation'] = ['padding']
    df_add['vector'] = [np.array(empty_id)]
    relation_embedding_df = relation_embedding_df.append(df_add)


    for neigh_relation_id in neigh_relation_id_list:
        neigh_relation_embedding = relation_embedding_df[relation_embedding_df['relation'] == neigh_relation_id]['vector'].values
        if len(neigh_relation_embedding) == 0:
            df_add = pd.DataFrame()
            df_add['relation'] = [neigh_relation_id]
            df_add['vector'] = [np.random.normal(-0.1, 0.1, 100)]
            relation_embedding_df = relation_embedding_df.append(df_add)
            neigh_relation_embedding = relation_embedding_df[relation_embedding_df['relation'] == neigh_relation_id]['vector'].values

            neigh_relation_embedding_list.append(neigh_relation_embedding[0])
        else:
            neigh_relation_embedding_list.append(neigh_relation_embedding[0])
    neigh_relation_index_df['neigh_relation_embedding'] = neigh_relation_embedding_list
    neigh_relation_index_df = neigh_relation_index_df.fillna(method='ffill')
    neigh_embedding_list = neigh_relation_index_df['neigh_relation_embedding'].values.tolist()
    neigh_relation_embedding_index = np.array(neigh_embedding_list)
    np.save('./kg/neigh_relation_embedding.npy', neigh_relation_embedding_index)
    print('finally')
    return neigh_relation_index_df

if __name__ == "__main__":
    kg = get_KG_construct()
    neigh_entity_adj, neigh_relation_adj, entity_index_df = get_onehop_neighbor(kg, entity_index_df, entity_neighbor_num=5)
    neigh_entity_index_adj, neigh_entity_index_df, entity_index_df = get_onehop_neighbor_entity_index(entity_index_df, neigh_entity_adj)
    neigh_relation_index_adj, neigh_relation_index_df = get_onehop_neighbor_relation_index(neigh_relation_adj, entity_index_df)
    neigh_entity_index_df = get_onehop_neighbor_entity_embedding(neigh_entity_index_df, entity_embedding_df)
    neigh_relation_index_df = get_onehop_neighbor_relation_embedding(neigh_relation_index_df, relation_embedding_df)
