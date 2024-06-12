import os
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import torch


# データを読み込む
def data_load(input_dir, file_name):
    data_path = os.path.join(input_dir, file_name)
    data = pd.read_csv(data_path)
    return data


# データ内ですべて0のミニバッチを削除する
# feature,label:3次元リスト
# feature_filterd,label_filterd:3次元リスト
def Del0(data, label, n, delindex):
    data_filtered = []
    label_filtered = []
    for i in range(len(data)):
        if all(data[i][j][delindex] == 0 for j in range(n)):
            # if feature[i][n-1][delindex] != 0:
            None
        else:
            f = data[i]
            data_filtered.append(f)
            label_filtered.append(label[i])
    return data_filtered, label_filtered


# データの前処理
# 入力data:2次元DataFrame型
# label,data:2次元DataFrame型
def make_data(data, input_size):
    data_list = []
    for list in data:
        data_list.append([a_list[2 : 2 + input_size] for a_list in list])
    return data_list


# データのバッチ化
# 入力,data,label:DataFrame型 2次元
# data,label:3次元リスト
# module 0,1:NN 2,3,4:RNN,CNN 5:middle
def split_data(data, label, n):
    data = [
        data.iloc[i : i + n].values.tolist()
        for i in range(len(data) - n + 2)
        if i + n < len(data)
    ]

    label = [
        label.values[i + n] for i in range(len(label) - n + 2) if i + n < len(label)
    ]

    return data, label


def make_dataloader(data_x, data_y, batch_size):
    data_x = torch.tensor(data_x, dtype=torch.float)  # .to(device)
    data_y = torch.tensor(data_y, dtype=torch.long)  # .to(device)
    out_data = TensorDataset(data_x, data_y)
    data_loader = DataLoader(out_data, batch_size=batch_size, shuffle=True)

    return data_loader


def pr_data(input_dir, n, batch_size, module, input_size):
    csv_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]
    data_x = []
    data_y = []
    for csv_file in csv_files:
        # データ読み込み
        data_path = os.path.join(input_dir, csv_file)
        data = pd.read_csv(data_path)

        label = data["label"].replace(-1, 0)
        data = data.drop("label", axis=1)

        # データをバッチに分割
        data, label = split_data(data, label, n)

        # 0を消す処理
        data, label = Del0(data, label, n, 7)

        # データの前処理
        data = make_data(data, input_size)

        # バッチデータを配列に加える
        data_x = data_x + data
        data_y = data_y + label

    data_loader = make_dataloader(data_x, data_y, batch_size)

    return data_loader
