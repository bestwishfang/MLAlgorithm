import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def build_data():
    train = np.load('./data/train.npy')
    test = np.load('./data/test.npy')
    return train, test


def distance(v1, v2):
    """
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
            dist = distance(test[i, :1024], train[j, :1024])
            arr_dist[j, :] = dist, train[j, -1]
        df = pd.DataFrame(data=arr_dist, columns=['dist', 'target'])
        mode = df.sort_values(by='dist')['target'].head(k).mode()[0]
        if mode == test[i, -1]:
            true_num += 1
    score = true_num / test.shape[0]
    return score


def show_res(k_list, score_list):
    fig = plt.figure()
    # 修改RC参数，来让其支持中文
    plt.rcParams['font.sans-serif'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot(k_list, score_list, color='r', linestyle='-.', linewidth=1.2, marker="o", markersize=7,
             markerfacecolor='r', markeredgecolor='r')
    plt.title('手写字knn算法预测准确率走势图')
    plt.xlabel('k值')
    plt.ylabel('准确率')
    plt.xticks(k_list)
    for i, j in zip(k_list, score_list):
        plt.text(i, j, "%.3f" % j, horizontalalignment='center')
    # plt.savefig('手写字knn算法预测准确率走势图.png')
    plt.show()


def main():
    # 1、加载数据
    train, test = build_data()
    print(train)
    print(train.shape)
    print(test)
    print(test.shape)
    # 2、算法预测
    # 自己实现knn_owns算法
    k_list = list(range(5, 15))
    score_list = []
    for k in k_list:
        score = knn_owns(train, test, k)
        score_list.append(score)

    # 3、结果展示
    print(score_list)
    show_res(k_list, score_list)


if __name__ == '__main__':
    main()
