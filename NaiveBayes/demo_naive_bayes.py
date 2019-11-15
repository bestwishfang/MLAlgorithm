import jieba
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer


def get_stop_words():
    """
    获取停止词， jieba统计次数时，不统计停止词
    :return: stop_words 停止词
    """
    with open('./stopwords.txt', mode='r', encoding='utf-8') as fp:
        stop_words = [s.strip() for s in fp.readlines()]
    return stop_words


def build_data():
    """
    构建训练集，测试集
    :return: train test 类型 nd.array
    """
    data = pd.read_csv('./data.csv', encoding='ansi')
    print(data.columns)
    content = []
    for t in data['内容 ']:
        # jieba 精确模式 分词
        seg = jieba.cut(t, cut_all=False)
        content.append(' '.join(seg))
    data['内容 '] = content
    # 将文本内容转化为数值类型
    conv = CountVectorizer(stop_words=get_stop_words())
    # x 词数统计
    x = conv.fit_transform(data['内容 '])
    res = x.toarray()
    data.loc[data['评价'] == '差评', :] = 0
    data.loc[data['评价'] == '好评', :] = 1
    data['评价'] = data['评价'].astype(np.int64)
    new_data = np.concatenate((res, data['评价'].values.reshape(-1, 1)), axis=1)
    train, test = new_data[:9, :], new_data[9:, :]
    return train, test


def main():
    train, test = build_data()
    # 朴素贝叶斯算法 进行分类 预测结果
    nb = MultinomialNB(alpha=1.0)  # alpha为拉普拉斯平滑系数
    nb.fit(train[:, :-1], train[:, -1])
    nb_predict = nb.predict(test[:, :-1])
    score = nb.score(test[:, :-1], test[:, -1])
    print('test target value: \n', test[:, -1])
    print('Naive Bayes predict: \n', nb_predict)
    print('Naive Bayes score:', score)


if __name__ == '__main__':
    main()
