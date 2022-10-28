from transformers import BertTokenizer, TFBertModel
import numpy as np
import tensorflow as tf

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertModel.from_pretrained("bert-base-uncased")
title_text = np.load('pretrain-data/new_title.npy')
title_text = title_text.tolist()

encoded_input = tokenizer(title_text, padding = True, truncation= True, return_tensors='tf')
output = model(encoded_input)
new_title_embedding = output[1].numpy()
print(new_title_embedding)
np.save('pretrain-data/new_title_embedding.npy', new_title_embedding)

