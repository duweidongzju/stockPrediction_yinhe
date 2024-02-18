import os
import yaml
import pdb
import torch

from tool import read_txt


class Config():
    def __init__(self, model_name="gru", loss_function="MSELoss"):
        self._set_config(fr"configuration/data.yaml")
        self.stock_codes = read_txt(fr"data/stock_codes.csv", separator=",")[0]
        self.stock_dates_train = [int(date) for date in read_txt("data/stock_dl_dates.txt", ",")[0] if self.train_start_date <= int(date) <= self.train_end_date]
        self.stock_dates_test = [int(date) for date in read_txt("data/stock_dl_dates.txt", ",")[0] if self.test_start_date <= int(date) <= self.test_end_date]
        self.model_name = model_name
        self._set_model_config(fr"configuration/model.yaml")

        self.loss_function = loss_function

        self.checkpoint = False
        self.checkpoint_model_path = f"./output/checkpoint_{self.model_name}_{self.loss_function}_{len(self.stock_codes)}stocks_{self.stock_dates_train[0]}_{self.stock_dates_train[-1]}.pth"

        self.batch_size = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.learning_rate = 0.001
        self.epochs = 20
        self.best_loss = 100
        self.save_model_path = f"./output/best_{self.model_name}_{self.loss_function}_{len(self.stock_codes)}stocks_{self.stock_dates_train[0]}_{self.stock_dates_train[-1]}.pth"
        if not os.path.exists("output"):
            os.mkdir("output")
        # test set
        self.train_data_min_days = 300
        self.train_data_max_days = 600
        self.train_update_days = 60

        self.test_result_root = fr"test_result"
        if not os.path.exists(self.test_result_root):
            os.mkdir(self.test_result_root)

    def _set_config(self, yaml_path):
        with open(yaml_path, 'r', encoding='utf-8') as f:
            dictinoary = yaml.load(f.read(), Loader=yaml.FullLoader)
        for key in dictinoary.keys():
            setattr(self, key, dictinoary[key])

    def _set_model_config(self, yaml_path):
        with open(yaml_path, 'r', encoding='utf-8') as f:
            dictionary = yaml.load(f.read(), Loader=yaml.FullLoader)[self.model_name]
        for key in dictionary.keys():
            setattr(self, key, dictionary[key])


if __name__ == "__main__":
    config = Config("gru")
    pdb.set_trace()
    a = 1
