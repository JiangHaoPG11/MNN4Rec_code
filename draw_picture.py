import seaborn as sns
import numpy as np
from numpy.random import randn
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('weight-2.csv')
data.columns = ["word","entity","topic","subtopic"]
word_weight = data['word'].values.tolist()
entity_weight = data['entity'].values.tolist()
topic_weight = data['topic'].values.tolist()
subtopic_weight = data['subtopic'].values.tolist()

c1, c2, c3, c4 = sns.color_palette('Set1', 4)

sns.kdeplot(word_weight, shade=True, color = c1, linewidth = 2,label = 'word_weight')
sns.kdeplot(entity_weight, shade=True, color=c2, linewidth = 2, label = 'entity_weight')
sns.kdeplot(topic_weight, shade=True, color=c3, linewidth = 2,label = 'topic_weight')
sns.kdeplot(subtopic_weight, shade=True, color=c4,linewidth = 2,label = 'subtopic_weight')

plt.legend(loc="best", fontsize = 15)

plt.ylabel('Density',fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig("weight_dis1.png")
plt.show()
plt.close()


label = ('word', 'entity', 'topic', 'subtopic')
mean_num = [np.mean(np.array(word_weight)), np.mean(np.array(entity_weight)), np.mean(np.array(topic_weight)), np.mean(np.array(subtopic_weight))]
plt.ylabel('Weight',fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.bar(label, mean_num)
plt.savefig("weight_bar.png")
plt.show()



