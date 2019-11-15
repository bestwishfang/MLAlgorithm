import numpy as np
import os


def build_data(file_path):
    file_name_list = os.listdir(file_path)
    data_arr = np.zeros(shape=(len(file_name_list), 1025))
    for file_index, file_name in enumerate(file_name_list):
        file_content = np.loadtxt(file_path + '/' + file_name, dtype=np.str)
        file_str = ''.join(file_content)
        data_arr[file_index, : 1024] = [int(s) for s in file_str]
        data_arr[file_index, -1] = int(file_name.split('_', 1)[0])
    return data_arr


def save_file(file_path, data):
    if not os.path.exists('./data'):
        os.makedirs('./data')
    np.save('./data/' + file_path, data)
    print("{}保存成功！".format('./data/' + file_path + '.npy'))


def main():
    train = build_data('./trainingDigits')
    test = build_data('./testDigits')

    print(train)
    print(train.shape)
    print(test)
    print(test.shape)
    save_file('train', train)
    save_file('test', test)


if __name__ == '__main__':
    main()
