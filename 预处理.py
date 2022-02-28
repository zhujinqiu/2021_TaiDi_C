#去除空格
import pandas as pd
def strip(text):
    try:
        return text.strip()
    except AttributeError:
        return text
#path原始文件路径
data_A = pd.read_excel('附件1/景区评论.xlsx',names=["景区名称", "评论日期", "评论内容"], #景区
                      converters = {'景区名称' : strip,
                                    '评论日期' : strip,
                                    '评论内容' : strip
                                   })
#data_A['评论内容'] = data_A['评论内容'].str.replace(' ', '')
data_H = pd.read_excel('附件1/酒店评论.xlsx',names=["酒店名称", "评论日期", "评论内容",'入住房型'],
                      converters = {'酒店名称' : strip,
                                    '评论日期' : strip,
                                    '评论内容' : strip,
                                    '入住房型' : strip,

                                   })
#data_H['评论内容'] = data_H['评论内容'].str.replace(' ', '')


# 写入文本到TEXT文件


for i in range(1, 51):
    if i < 10:
        data = data_A[data_A['景区名称'] == 'A0' + str(i)]

        data['评论内容'].to_csv('text/A/A0' + str(i) + '.txt', encoding='utf_8_sig', index=0,
                            header=0)
    else:
        data = data_A[data_A['景区名称'] == 'A' + str(i)]

        data['评论内容'].to_csv('text/A/A' + str(i) + '.txt', encoding='utf_8_sig', index=0,
                            header=0)
for i in range(1, 51):
    if i < 10:
        data = data_H[data_H['酒店名称'] == 'H0' + str(i)]
        data['评论内容'].to_csv('text/H/H0' + str(i) + '.txt', encoding='utf_8_sig', index=0,
                            header=0)
    else:
        data = data_H[data_H['酒店名称'] == 'H' + str(i)]
        data['评论内容'].to_csv('text/H/H' + str(i) + '.txt', encoding='utf_8_sig', index=0,
                            header=0)
