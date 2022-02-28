import pandas as pd
import jieba
import jieba.analyse
import numpy as np
import pandas as pd

# 读取停用词数据
stopwords = pd.read_csv('highfreinvalid.txt', encoding='utf8', names=['stopword'], index_col=False)
def strip(text):
    try:
        return text.strip()
    except AttributeError:
        return text

word_cut = []
data_A = pd.read_excel('附件1/景区评论.xlsx',names=["景区名称", "评论日期", "评论内容"],
                      converters = {'景区名称' : strip,
                                    '评论日期' : strip,
                                    '评论内容' : strip
                                   })
data_H = pd.read_excel('附件1/酒店评论.xlsx',names=["酒店名称", "评论日期", "评论内容",'入住房型'],
                      converters = {'酒店名称' : strip,
                                    '评论日期' : strip,
                                    '评论内容' : strip,
                                    '入住房型' : strip,

                                   })


def cut_words(data):
    data['cut'] = data['评论内容'].apply(lambda x: list(jieba.cut(x, HMM=True)))
    return data

def remove_stopwords(stopwords,data):


    # 转化词列表
    stop_list = stopwords['stopword'].tolist()

    # 去除停用词
    data['cut'] = data['评论内容'].apply(lambda x: [i for i in jieba.cut(x) if len(i) > 1 and i not in stop_list])
    return data

def hotwords_count(data):
    # 将所有的分词合并
    words = []

    for content in data['cut']:
        words.extend(content)
    # 创建分词数据框
    corpus = pd.DataFrame(words, columns=['word'])
    corpus['词频'] = 1

    # 分组统计
    g = corpus.groupby(['word']).agg({'词频': 'count'}).sort_values('词频', ascending=False)

    hot_words = g.head(20)
    return hot_words


stopwords = pd.read_csv('highfreinvalid.txt', encoding='utf8', names=['stopword'], index_col=False)
for i in range(1,51):
    if i <= 9 :
        data = data_H[data_H['酒店名称'] == 'H0'+ str(i) ]
        data = cut_words(data)
        data = remove_stopwords(stopwords, data)
        hot_words = hotwords_count(data)
        hot_words.to_csv('印象词云表/酒店热门词/H0'+str(i)+'.csv', encoding='utf_8_sig')
    else:
        data = data_H[data_H['酒店名称'] == 'H' + str(i) ]
        data =  cut_words(data)
        data = remove_stopwords(stopwords,data)
        hot_words = hotwords_count(data)
        hot_words.to_csv('印象词云表/酒店热门词/H'+str(i)+'.csv', encoding='utf_8_sig')
