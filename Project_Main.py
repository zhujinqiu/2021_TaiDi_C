import sys

from litNlp.predict import SA_Model_Predict
from matplotlib import pyplot as plt
import numpy as np
from sa_analysis import topic_sa_analysis
import jieba.posseg as pseg
from sa_model_train import model_train
import multiprocessing
from setting import *
from tqdm import tqdm
import os
import pandas as pd

# 识别标点符号进行句子切分
def doc2sentence(resource_text):
    # jieba 预热
    print(pseg.cut('预热'))
    with open(sentence_cut_path, 'w', encoding='utf-8') as sentence_cut:
        for sentence in tqdm(resource_text):
            if len(sentence.strip()) > 1:
                for word, flag in pseg.cut(sentence):
                    if flag != 'x':
                        sentence_cut.write(word)
                    else:
                        sentence_cut.write('\n')

# 多线程主题句查询  Linux系统
def find_topic_sentence():
    if not os.path.exists(topic_path):
        os.mkdir(topic_path)
    task_split = []
    for topic_n, key_words_list in topic_words_list.items():
        task_split.append([topic_path, topic_n, key_words_list])
    # task multiprocessing
    thread_number = len(topic_words_list.keys())
    pool = multiprocessing.Pool(processes=thread_number)
    pool.map(find_key_txt, task_split)
def find_key_txt(data_list):
    sentence_cut = open(sentence_cut_path, 'r', encoding='utf-8')
    with open('{}/{}.txt'.format(data_list[0], data_list[1]), 'w', encoding='utf-8') as key_txt:
        for sentence in sentence_cut.readlines():
            for i in data_list[2]:
                if i in sentence:
                   key_txt.write(sentence)
    # 关闭文件
    sentence_cut.close()
    print('{} 已经查找完成'.format(data_list[1]))

def get_labels(sentiments):
  for i in range(sentiments.shape[0]):
    if sentiments.iloc[i,1] >= 0.75:
      sentiments['labels'][i] = 4
    elif sentiments.iloc[i,1] >= 0.5:
      sentiments['labels'][i] = 3
    elif sentiments.iloc[i,1] >= 0.25:
      sentiments['labels'][i] = 2
    else:
      sentiments['labels'][i] = 1
  return sentiments
if __name__ == '__main__':
    # 情感分析模型训练
    #model_train()
    sentiments_score = []
    sa_model = SA_Model_Predict(tokenize_path, sa_model_path_m, max_len=100)
    for i in range(1,51):
      if i < 10 :
        with open(r'path/text/A/A0'+str(i)+'.txt', 'r', encoding='utf-8') as resource_text:#酒店换成H即可
          resource_text = resource_text.readlines()
        # 整句切分
          doc2sentence(resource_text)
        # 主题句查询
          find_topic_sentence()
        # 主题情感极性可视化
        #a = topic_sa_analysis()
        #print(resource_text)
      else:
        with open(r'path/text/A/A'+str(i)+'.txt', 'r', encoding='utf-8') as resource_text:#酒店换成H即可
          resource_text= resource_text.readlines()
        # 整句切分
          doc2sentence(resource_text)
        # 主题句查询
          find_topic_sentence()
        # 主题情感极性可视化
        #a = topic_sa_analysis()
        #print(resource_text)

      key_txt = open('path/topic_text/服务.txt', 'r', encoding='utf-8').readlines()  #这里服务、性价比、位置、卫生、设施依次训练
      sentiments_score_predict = sa_model.predict(key_txt)
      sentiments = pd.DataFrame(sentiments_score_predict)
      sentiments['labels'] =1
      sentiments_label = get_labels(sentiments)
      a = sentiments_label['labels'].value_counts(normalize = True)
      sentiments_score.append(list(a.sort_index()))  #输出训练特征


#回归过程
A_score = pd.read_excel('path/景区评分.xlsx')
sentiments_score = pd.DataFrame(sentiments_score).fillna(0)
train = pd.concat([sentiments_score,A_score],axis=1)
X = train.iloc[:,:4]
y = train.iloc[:,6]#服务是6列，当训练其他方面模型则改列数

from sklearn.model_selection import train_test_split
# 避免过拟合，采用交叉验证，验证集占训练集10%，固定随机种子（random_state)
train_X,test_X, train_y, test_y = train_test_split(X,y,test_size = 0.20)
from sklearn import ensemble
random_forest_regressor = ensemble.RandomForestRegressor(n_estimators=20)  # n_estimators=15随机森林回归,并使用20个决策树
random_forest_regressor.fit(train_X, train_y)  # 拟合模型
score = random_forest_regressor.score(test_X, test_y)
result = random_forest_regressor.predict(test_X)
plt.figure()
plt.plot(np.arange(len(result)), test_y, "go-", label="True value")
plt.plot(np.arange(len(result)), result, "ro-", label="Predict value")
plt.title(f"RandomForest---score:{score}")
plt.legend(loc="best")
plt.show()

#模型评价
from sklearn.metrics import mean_squared_error
result = random_forest_regressor.predict(train_X)
MSE2 = mean_squared_error(train_y, result)
#酒店服务测试集MSE 0.0016 训练集MSE 0.0085
##酒店服务测试集MSE 0.0014 训练集MSE 0.0029
##酒店性价比测试集MSE 0.005 训练集MSE 0.0021
#酒店设施 ：训练集MSE : 0.007437499999999983测试机MSE : 0.02518999999999999
#酒店卫生  训练集MSE : 0.0027994500000001047 测试机MSE : 0.01492470000000016

#景区卫生 训练集MSE : 0.017310000000000082 测试机MSE : 0.037840000000000054
#景区性价比 训练集MSE : 0.01524821428571431测试机MSE : 0.0308437499999994
#景区设施 训练集MSE : 0.027388095238095298 测试机MSE : 0.060387500000000094
#景区位置  训练集MSE : 0.027388095238095298 测试机MSE : 0.060387500000000094
#景区服务 训练集MSE : 0.016531874999999994 测试机MSE : 0.0404400000000001
result_test = random_forest_regressor.predict(test_X)
MSE2_test = mean_squared_error(test_y, result_test)
print('训练集MSE :',MSE2)
print('测试机MSE :',MSE2_test)
