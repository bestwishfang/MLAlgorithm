import pandas as pd
import numpy as np


def build_data(file_path):
    df = pd.read_excel(file_path, sheetname=0)
    return df


def distance(v1, v2):
    """
    :param v1: array
    :param v2: array
    :return: dist
    """
    dist = np.sqrt(np.sum(np.power((v1-v2), 2)))
    return dist


def knn_owns(train, test):
    # print(test.values)
    # print(type(test.values))
    for i in train.index:
        dist = distance(train.loc[i, '搞笑镜头':'打斗镜头'].values, test.values[1:])
        # print('='*100)
        # print(dist)
        train.loc[i, 'dist'] = dist
    print(train)


def main():
    # 1、加载数据
    file_path = './电影分类数据.xlsx'
    data = build_data(file_path)
    print(data)
    # 2、确定训练集 、 测试集
    train = data.iloc[:, 1: -4]
    test = data.columns[-4:]
    print(train)
    print(test)
    print(train.dtypes)
    # 3、
    # k = int(input("请输入knn算法的k值：").strip())
    # knn_owns(train, test, k)
    knn_owns(train, test)


if __name__ == '__main__':
    main()