import json
from collections import OrderedDict
import random

import numpy as np
import pandas as pd
from ast import literal_eval
import numpy as np
from gensim import models
import gensim.models
from gensim.models import Word2Vec
import torch


#词嵌入部分
# item_id=[]
# feature_list=[]
# for line in open('D:/Download/track2_title.txt/track2_title.txt','r'):
#     temp_item=json.loads(line)
#     item_temp_id=temp_item['item_id']
#     feature=temp_item['title_features']
#     sentence=[]
#     for key, value in feature.items():
#         for i in range(1,value+1):
#             key=int(key)
#             sentence.append(key)
#     item_id.append(item_temp_id)
#     feature_list.append(sentence)
# np.save('croup.npy',np.array(feature_list))



from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile

#训练doc2vec模型
# croup=np.load('croup.npy',allow_pickle=True)
# documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(croup)]
# model = Doc2Vec(documents, vector_size=128, window=3, min_count=1, workers=4)
# model.save('./doc2vec.vec')


# # # 载入已训练的doc2vec模型
# model = Doc2Vec.load('./doc2vec/my_doc2vec_model')
# # 使用模型一次性获取所有'tag_name'列表
# tag_list = model.docvecs.index_to_key
# # 使用模型一次性获取所有与tag_list顺序一致的文档向量矩阵
# vectors = model.docvecs.vectors

#构建pd文件
# item_id=[]
# for line in open('D:/Download/track2_title.txt/track2_title.txt','r'):
#     temp_item=json.loads(line)
#     item_temp_id=temp_item['item_id']
#     item_id.append(item_temp_id)
# vectors=vectors.tolist()
# title_feature_dict = {
#         'item_id':item_id,
#        'feature': vectors}
# title_feature_df = pd.DataFrame(title_feature_dict)
# title_feature_df=title_feature_df.sort_values(by=['item_id'],ascending=[1])
# title_feature_df=title_feature_df[title_feature_df['item_id']<=137000]
# title_feature_df.to_csv('title_df_feature.csv')
# item_id=title_feature_df['item_id'].values.tolist()
# title_feature=title_feature_df['feature'].values.tolist()
# np.save('item_id_title.npy',item_id)
# np.save('feature_list_title.npy',title_feature)
# for i in range(137001):
#     if i not in item_id:
#         temp=[random.random() for j in range(128)]
#         title_feature.insert(i,temp)
# title_feature=np.array(title_feature)
# np.save('feature_title.npy',title_feature)

# import torch
#
# title_feature=np.load('feature_title.npy')
# torch.save(title_feature,'feat_t.pt')
import torch
t_feat=torch.load('feat_t.pt')
t_feat = torch.tensor(t_feat, dtype=torch.float).cuda()
print(t_feat)


