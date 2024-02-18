from autobase import dataloader
from tqdm import tqdm
from tool import read_txt, write_dict

stock_dl = dataloader.Dataloader()
stock_dl.init("/data/cache/stock")

print(len(stock_dl.dates))
STOCK_DATES = [date for date in stock_dl.dates if 20150101 <= date <= 20230718]
print(len(STOCK_DATES))



stock_dates = stock_dl.dates
stock_codes = stock_dl.codes
stock_codes = read_txt(fr"files/stock_codes.csv", separator=",")[0]
fearure_names = ["feat_a1", "feat_a2", "feat_a3", "feat_a4", "feat_a5", "feat_a6",
                             "feat_b1", "feat_b2", "feat_b3", "feat_b4", "feat_b5", "feat_b6",
                             "feat_c1", "feat_c2", "feat_c3", "feat_c4", "feat_c5", "feat_c6",
                             "feat_c7", "feat_c8", "feat_c9", "feat_c10", "feat_c11",
                             "label_a"]
data = {}
for stock_date in tqdm(STOCK_DATES):
    data[str(stock_date)] = {}
    di = stock_dl.get_di(stock_date)
    for stock_code in stock_codes:
        data[str(stock_date)][stock_code] = {}
        ii = stock_dl.get_di_ii(di, stock_code)
        for feature_name in fearure_names:
            factor = stock_dl.get_data(feature_name)[di][ii]
            data[str(stock_date)][stock_code][feature_name] = factor.tolist()
write_dict(fr"files/data.json", data)

# data = read_dict(fr"files/test_data.json")
# print(math.isnan(data["20230718"]["601989.SH"]["label_a"]))