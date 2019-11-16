import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gevent


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


def distance(v1, v2):
    """
    计算欧式距离
    :param v1: array
    :param v2: array
    :return: dist
    """
    dist = np.sqrt(np.sum(np.power((v1 - v2), 2)))
    return dist


def knn_owns(train, test, k):
    true_num = 0
    for i in range(test.shape[0]):
        arr_dist = np.zeros(shape=(train.shape[0], 2))
        for j in range(train.shape[0]):
            dist = distance(test[i, :-1], train[j, :-1])
            arr_dist[j, :] = dist, train[j, -1]
        df = pd.DataFrame(data=arr_dist, columns=['dist', 'target'])
        mode = df.sort_values(by='dist')['target'].head(k).mode()[0]
        if mode == test[i, -1]:
            true_num += 1
    score = true_num / test.shape[0]
    score_list.append((k, score))
    return score


def show_res():
    k_list = []
    score_li = []
    for k, s in score_list:
        k_list.append(k)
        score_li.append(s)
    fig = plt.figure()
    # 修改RC参数，来让其支持中文
    plt.rcParams['font.sans-serif'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot(k_list, score_li, color='r', linestyle='-.', linewidth=1.2, marker="o", markersize=7,
             markerfacecolor='r', markeredgecolor='r')
    plt.title('iris种类knn算法预测准确率走势图')
    plt.xlabel('k值')
    plt.ylabel('准确率')
    plt.xticks(k_list)
    for i, j in zip(k_list, score_li):
        plt.text(i, j, "%.3f" % j, horizontalalignment='center')
    # plt.savefig('iris种类knn算法预测准确率走势图.png')
    plt.show()


def main():
    train, test = build_data('./iris_dataset.txt')
    print(train)
    print(train.shape)
    print(test)
    print(test.shape)
    # knn 分析
    k_list = list(range(1,6))
    # 使用协程加速
    gevent_list = []
    for k in k_list:
        g = gevent.spawn(knn_owns, train, test, k)
        gevent_list.append(g)
    gevent.joinall(gevent_list)

    # 结果展示
    flag = True
    while flag:
        if len(score_list) == len(k_list):
            print(score_list)
            show_res()
            flag = False


if __name__ == '__main__':
    score_list = []
    main()
