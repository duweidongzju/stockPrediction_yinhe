import math
from autobase import dataloader
from tqdm import tqdm
from config import Config
from tool import write_txt, write_dict, read_dict

stock_dl = dataloader.Dataloader()
stock_dl.init("/data/cache/stock")
STOCK_DATES = [date for date in stock_dl.dates if 20150101 <= date <= 20230717]
config = Config()
print(len(STOCK_DATES))

write_txt("files/stock_dl_dates.txt", [stock_dl.dates], separator=",")


def get_feature(stock_date, stock_code, feature_name):
    di = stock_dl.get_di(stock_date)
    ii = stock_dl.get_di_ii(di, stock_code)
    feature = stock_dl.get_data(feature_name)[di][ii]

    return feature


def func1():
    """
    计算每个股票的有效 label 数量
    :return:
    """
    stock_codes = [code for code in stock_dl.codes if code.split(".")[-1] in ['SH', 'SZ']]

    data = {}
    for stock_code in stock_codes:  # init
        data[stock_code] = 0

    for stock_date in tqdm(STOCK_DATES):
        di = stock_dl.get_di(stock_date)
        for stock_code in stock_codes:
            ii = stock_dl.get_di_ii(di, stock_code)
            label = stock_dl.get_data("label_a")[di][ii]
            if not math.isnan(label):
                data[stock_code] += 1

    write_dict(fr"../files/stock_codes_counter.json", data)


def func2():
    """
    计算有效 label数量的股票集合
    :return:
    """
    data = read_dict(fr"../files/stock_codes_counter.json")
    res = {}
    for i in range(2077):  # init
        res[str(i)] = []
    for key in data.keys():
        res[str(data[key])].append(key)
    write_dict(fr"../files/stock_codes_counter2.json", res)


def func3():
    """
    去掉存在连续三个 label为 0的股票，（退市股票）

    :return:
    """
    stock_codes = read_dict(fr"../files/stock_codes_counter2.json")["2076"]
    new_stock_codes = []

    for stock_code in tqdm(stock_codes):
        is_all = True
        label_last, label_last_last = 1, 1
        for stock_date in STOCK_DATES:
            di = stock_dl.get_di(stock_date)
            ii = stock_dl.get_di_ii(di, stock_code)
            label = stock_dl.get_data("label_a")[di][ii]
            if label == 0 and label_last == 0 and label_last_last == 0:
                is_all = False
                break

        if is_all:
            new_stock_codes.append(stock_code)

    print(new_stock_codes)
    write_txt(fr"files/stock_codes.csv", [new_stock_codes], separator=",")

#
# stock_codes = read_txt(fr"files/stock_codes.csv", separator=",")[0]
# # # pdb.set_trace()
#
# for stock_code in stock_codes[0:1]:
#     res = []
#     for stock_date in tqdm(STOCK_DATES):
#         di = stock_dl.get_di(stock_date)
#         ii = stock_dl.get_di_ii(di, stock_code)
#         cur = []  # 用于存储feature
#         for feature in config.feature_name:
#             factor = stock_dl.get_data(feature)[di][ii]
#             cur.append(factor)
#             # cur.append(0 if math.isnan(factor) else factor)  # 缺失值处理，将nan全部变为0
#         res.append(cur)
#     print(res)
#     name, place = stock_code.split('.')
#     write_txt(fr"data/{name+place}.csv", res, separator=",")


