import json
import os
import pdb
import pickle

import numpy as np
import pandas as pd
import math
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm
from tool import read_txt, write_dict


# from autobase import dataloader
# stock_dl = dataloader.Dataloader()
# stock_dl.init("/data/cache/stock")


class StockJson():
    def __init__(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.dates = [int(date) for date in self.data.keys()]
        self.codes = [code for code in self.data[str(self.dates[0])].keys()]  # 2012 stocks
        self.feature_names = ["feat_a1", "feat_a2", "feat_a3", "feat_a4", "feat_a5", "feat_a6",
                              "feat_b1", "feat_b2", "feat_b3", "feat_b4", "feat_b5", "feat_b6",
                              "feat_c1", "feat_c2", "feat_c3", "feat_c4", "feat_c5", "feat_c6",
                              "feat_c7", "feat_c8", "feat_c9", "feat_c10", "feat_c11"]  # 23 features
        self.label_name = "label_a"  # 1 label

    def get_data(self, date, code, feature_name):
        return self.data[str(date)][code][feature_name]

    def get_di(self, value):
        return value

    def get_di_ii(self, value):
        return value


class DataNorm():
    def __init__(self, json_path=fr"data/data_norm.json"):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.dictionary = json.load(f)

    def Norm(self, x, feature):
        x = (x - self.dictionary["min"][feature]) / (self.dictionary["max"][feature] - self.dictionary["min"][feature])
        return x


def get_train_data(stock_codes, stock_dates, config):
    """
    形成训练数据
    """
    # params
    features = config.feature_names
    features.append(config.label_name)
    timestep = config.timestep
    input_size = config.input_size
    cache_train = config.cache_train

    dataX = []
    dataY = []

    if not os.path.exists(cache_train):
        os.makedirs(cache_train)

    cache_nameX = fr"{cache_train}/codes{len(stock_codes)}dates{stock_dates[0]}_{stock_dates[-1]}_timestep{config.timestep}_dataX.npy"
    cache_nameY = fr"{cache_train}/codes{len(stock_codes)}dates{stock_dates[0]}_{stock_dates[-1]}_timestep{config.timestep}_dataY.npy"
    if os.path.exists(cache_nameX) and os.path.exists(cache_nameX) and config.use_cache:
        print(fr"load data cache: {cache_nameX}")
        print(fr"load data cache: {cache_nameY}")
        dataX = np.load(cache_nameX)
        dataY = np.load(cache_nameY)
    else:
        stock_dl = StockJson("data/data.json")
        data_norm = DataNorm()

        # dictionary = {}
        # dictionary["max"] = {}
        # dictionary["min"] = {}
        # for feature in features:
        #     dictionary["max"][feature] = -100
        #     dictionary["min"][feature] = 100
        # for stock_code in tqdm(stock_codes):
        #     for stock_date in stock_dates:
        #         for feature in features:
        #             factor = stock_dl.get_data(stock_date, stock_code, feature)
        #             if not math.isnan(factor):
        #                 dictionary["max"][feature] = factor if factor > dictionary["max"][feature] else dictionary["max"][feature]
        #                 dictionary["min"][feature] = factor if factor < dictionary["min"][feature] else dictionary["min"][feature]
        # write_dict(fr"data/data_norm.json", dictionary)
        # pdb.set_trace()

        for stock_code in tqdm(stock_codes):
            res = []
            for stock_date in stock_dates:
                # di = stock_dl.get_di(stock_date)
                # ii = stock_dl.get_di_ii(di, stock_code)
                # label = stock_dl.get_data(features[-1])[di][ii]
                label = stock_dl.get_data(stock_date, stock_code, features[-1])
                if math.isnan(label):  # 缺失值处理，判断是否含有label，若没有，则跳过本条数据
                    continue
                if len(res) > 2 and res[-1][-1] == 0 and label == 0:  # 如果连续两天为0，即为退市，不需要后面时间的数据了
                    del res[-1]
                    break

                cur = []  # 用于存储feature
                for feature in features:
                    # factor = stock_dl.get_data(feature)[di][ii]
                    factor = stock_dl.get_data(stock_date, stock_code, feature)
                    cur.append(0 if math.isnan(factor) else data_norm.Norm(factor, feature))  # 缺失值处理，将nan全部变为0
                res.append(cur)

            if len(res) <= timestep:  # 如果该股票数据长度不够，就跳过该股票
                continue

            # 开始划分数据集
            dataframe = pd.DataFrame(res)
            dataframeX = dataframe.iloc[:, :-1]
            dataframeY = dataframe.iloc[:, -1]
            # dataframeX = (dataframeX - dataframeX.min()) / (dataframeX.max() - dataframeX.min())
            # dataframeY = (dataframeY - dataframeY.min()) / (dataframeY.max() - dataframeY.min())
            for index in range(len(dataframe) - timestep):
                dataX.append(dataframeX.iloc[index: index + timestep, :])
                dataY.append(dataframeY.iloc[index + timestep])

        dataX = np.array(dataX)
        dataY = np.array(dataY)

        np.save(cache_nameX, dataX)
        np.save(cache_nameY, dataY)

    train_size = int(np.round(0.8 * dataX.shape[0]))

    x_train = dataX[: train_size, :].reshape(-1, timestep, input_size)
    y_train = dataY[: train_size].reshape(-1, 1)

    x_test = dataX[train_size:, :].reshape(-1, timestep, input_size)
    y_test = dataY[train_size:].reshape(-1, 1)

    x_train_tensor = torch.from_numpy(x_train).to(torch.float32)
    y_train_tensor = torch.from_numpy(y_train).to(torch.float32)
    x_test_tensor = torch.from_numpy(x_test).to(torch.float32)
    y_test_tensor = torch.from_numpy(y_test).to(torch.float32)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    return [train_dataset, test_dataset]


def get_test_data(stock_codes, dates_test, config):
    # params
    STOCK_DATES = [int(date) for date in read_txt("data/stock_dl_dates.txt", ",")[0] if
                   20150101 <= int(date) <= 20230717]
    features = config.feature_names
    features.append(config.label_name)
    timestep = config.timestep
    input_size = config.input_size

    data_x = {}

    if not os.path.exists(config.cache_test):
        os.makedirs(config.cache_test)

    cache_name = fr"{config.cache_test}/codes{len(stock_codes)}dates{dates_test[0]}_{dates_test[-1]}_timestep{config.timestep}.npy"
    if os.path.exists(cache_name) and config.use_cache:
        print(fr"load data cache: {cache_name}")
        with open(cache_name, 'rb') as f:
            data_x = pickle.load(f)
    else:
        stock_dl = StockJson("data/data.json")
        data_norm = DataNorm()
        for i, stock_date in enumerate(tqdm(dates_test)):
            data_x[stock_date] = {}
            for stock_code in stock_codes:
                label = stock_dl.get_data(stock_date, stock_code, "label_a")
                if math.isnan(label):
                    continue

                cur_data_x = []
                cur_date_index = STOCK_DATES.index(stock_date) - 1
                zero_counter = 0
                while len(cur_data_x) < timestep and cur_date_index > 0:
                    # todo: 获取新的数据特征
                    cur_stock_date = STOCK_DATES[cur_date_index]
                    # di = stock_dl.get_di(cur_stock_date)
                    # ii = stock_dl.get_di_ii(di, stock_code)
                    # label = stock_dl.get_data(features[-1])[di][ii]
                    label = stock_dl.get_data(cur_stock_date, stock_code, "label_a")
                    if math.isnan(label):
                        break
                    if label == 0:  # 缺失值处理，判断是否含有label，若没有，则跳过本条数据
                        zero_counter += 1
                        if zero_counter == 3:
                            break
                        else:
                            continue
                    else:
                        zero_counter = 0

                    cur = []
                    cur_stock_date = STOCK_DATES[cur_date_index]
                    # di = stock_dl.get_di(cur_stock_date)
                    # ii = stock_dl.get_di_ii(di, stock_code)
                    for feature in features[0:-1]:
                        # factor = stock_dl.get_data(feature)[di][ii]
                        factor = stock_dl.get_data(cur_stock_date, stock_code, feature)
                        cur.append(0 if math.isnan(factor) else data_norm.Norm(factor, feature))  # 缺失值处理，将nan全部变为0
                    cur_data_x.insert(0, cur)

                    cur_date_index -= 1

                if len(cur_data_x) != timestep:
                    continue

                # label = get_feature(stock_date, stock_code, "label_a")
                label = stock_dl.get_data(stock_date, stock_code, "label_a")
                x = np.array(cur_data_x).reshape(-1, timestep, input_size)
                y = np.array(label).reshape(-1, 1)
                x_tensor = torch.from_numpy(x).to(torch.float32)
                y_tensor = torch.from_numpy(y).to(torch.float32)
                data_x[stock_date][stock_code] = {'x': x_tensor, 'y': y_tensor}

        with open(cache_name, 'wb') as f:
            pickle.dump(data_x, f)

    return data_x