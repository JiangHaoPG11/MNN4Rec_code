import numpy as np
from nltk.text import TextCollection

title_word = np.load('../data3/new_title_word_index.npy')
corpus = TextCollection(title_word)

tf_idf_list = []
index = 0
for title_word_one in title_word:
    index += 1
    tf_idf_list_one = []
    for i in range(len(title_word_one)):
        tf_idf = corpus.tf_idf(title_word_one[i], corpus)
        tf_idf_list_one.append(tf_idf)
    tf_idf_list.append(tf_idf_list_one)
np.save('../data3/new_feature.npy',np.array(tf_idf_list))
