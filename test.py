import pdb

import torch
from tqdm import tqdm
from config import Config
from estimator import NDCG, MyEstimator
from get_data import get_test_data
from model import GRU, GRU_TransAm, TransAm
from tool import write_dict


def run_test(stock_codes, dates_test, config):
    # Load data
    print("=====>> prepare data")
    data_test = get_test_data(stock_codes, dates_test, config)

    # Load MODEL
    print("=====>> load model")
    if config.model_type == "gru":
        model = GRU(config.feature_size, config.hidden_size, config.num_layers, config.output_size)
    elif config.model_type == "transformer":
        model = TransAm(config.feature_size, config.num_layers, config.dropout)
    elif config.model_type == "gru+transformer":
        model = GRU_TransAm(config.feature_size, config.num_layers_tran, config.dropout, config.hidden_size_gru,
                            config.num_layers_gru, config.output_size)
    state_dict = torch.load(config.save_model_path)
    model.load_state_dict(state_dict["model"])
    model.to(config.device)

    # Set estimator
    # estimator = NDCG()
    estimator = MyEstimator()

    # Test
    print("=====>>>  test")
    dates_test_bar = tqdm(dates_test)
    for day, date_test in enumerate(dates_test_bar):
        if data_test[date_test] is None:  # 当天无数据，跳过
            continue
        for code in data_test[date_test].keys():
            if data_test[date_test][code] is None:  # 当天股票无数据，跳过
                continue

            model.eval()
            with torch.no_grad():
                x = data_test[date_test][code]["x"]
                x = x.to(config.device)
                output = model(x)

                data_test[date_test][code]["output"] = output.cpu()
        # pdb.set_trace()
        # write_dict(fr"test_result/output_{config.model_name}_{config.loss_function}_{len(stock_codes)}stocks_{dates_test[0]}_{dates_test[-1]}.pkl", data_test)

        estimator.calculate(data_test[date_test], date_test)

        dates_test_bar.desc = "test day[{}/{}] score_NDCG:{:.3f}".format(day + 1, len(dates_test), estimator.result[day][1])
    # write_dict(fr"test_result/output_{config.model_name}_{config.loss_function}_{len(stock_codes)}stocks_{dates_test[0]}_{dates_test[-1]}.pkl", data_test)
    print(estimator.result)
    print(estimator.mean_score())
    estimator.save_result(fr"test_result/{estimator.name}_{config.model_name}_{config.loss_function}_{len(stock_codes)}stocks_{dates_test[0]}_{dates_test[-1]}.csv")


if __name__ == "__main__":
    """
    model_names = ["gru", "gru1", "TransAm", "GRU_TransAm"]
    loss_functions = ["MSELoss", "MyLoss1"]
    config = Config(model_name="gru", loss_function="MSELoss")
    run_test(config.stock_codes, config.stock_dates_test[0:60], config)
    """
    # config = Config(model_name="GRU_TransAm", loss_function="MyLoss1")
    # run_test(config.stock_codes, config.stock_dates_test[0:60], config)
    for model_name in ["gru", "gru1", "TransAm", "GRU_TransAm"][0:3]:
        for loss_function in ["MSELoss", "MyLoss1"]:
            print(fr"model_name: {model_name}")
            print(fr"loss_function: {loss_function}")
            config = Config(model_name=model_name, loss_function=loss_function)
            run_test(config.stock_codes, config.stock_dates_test[0:60], config)

