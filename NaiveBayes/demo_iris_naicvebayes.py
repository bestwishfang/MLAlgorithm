import pandas as pd
from sklearn.naive_bayes import MultinomialNB


def build_data(file_path):
    """
    简单处理数据，并构建训练集，测试集
    :param file_path: 文件路径
    :return: train， test np.array
    """
    data = pd.read_table(file_path, sep=',', header=None, encoding='ansi')
    # 查看分类 类别性数据
    print(data.groupby(by=4).mean())
    # 将分类结果转为数值型
    data.loc[data[4] == 'Iris-setosa', 4] = 1
    data.loc[data[4] == 'Iris-versicolor', 4] = 2
    data.loc[data[4] == 'Iris-virginica', 4] = 3
    # 构建训练集，测试集
    train_index = [True if i % 5 != 0 else False for i in range(data.shape[0])]
    test_index = [True if i % 5 == 0 else False for i in range(data.shape[0])]
    train, test = data.values[train_index, :], data.values[test_index, :]
    return train, test


def main():
    train, test = build_data('./iris_dataset.txt')
    print(train)
    print(train.shape)
    print(test)
    print(test.shape)
    # Naive Bayes 分析
    nb = MultinomialNB(alpha=1.0)
    nb.fit(train[:, :-1], train[:, -1])
    nb_predict = nb.predict(test[:, :-1])
    score = nb.score(test[:, :-1], test[:, -1])
    print('test target value: \n', test[:, -1])
    print('Naive Bayes predict: \n', nb_predict)
    print('Naive Bayes score:', score)  # 0.9666666666666667


if __name__ == '__main__':
    main()
