import os
import pdb

import torch
from torch import nn
from tqdm import tqdm
from config import Config
from get_data import get_train_data
from mylossfunction import MyLoss1, Myloss2, NDCGLoss
from model import GRU, TransAm, GRU_TransAm


def run_train(config):
    # Load data
    print("=====>> prepare data")
    train_dataset, test_dataset = get_train_data(config.stock_codes, config.stock_dates_train, config)
    train_loader = torch.utils.data.DataLoader(train_dataset, config.batch_size, False)
    test_loader = torch.utils.data.DataLoader(test_dataset, config.batch_size, False)

    # Load MODEL
    print(fr"=====>> load model: {config.model_name}")
    if config.model_type == "gru":
        model = GRU(config.feature_size, config.hidden_size, config.num_layers, config.output_size)
    elif config.model_type == "transformer":
        model = TransAm(config.feature_size, config.num_layers, config.dropout)
    elif config.model_type == "gru+transformer":
        model = GRU_TransAm(config.feature_size, config.num_layers_tran, config.dropout, config.hidden_size_gru,
                            config.num_layers_gru, config.output_size)

    config.checkpoint_model_path = config.save_model_path
    if config.checkpoint and os.path.exists(config.checkpoint_model_path):  # 是否加载断点
        print(fr"load checkpoint: {config.checkpoint_model_path}")
        state_dict = torch.load(config.checkpoint_model_path)
        model.load_state_dict(state_dict["model"])

    model.to(config.device)

    if os.path.exists(config.save_model_path):  # 是否已有 best_model
        print(fr"load best_model val_loss: {config.save_model_path}")
        state_dict = torch.load(config.save_model_path)
        config.best_loss = state_dict["val_loss"]

    # Set Loss function
    print(fr"=====>> Set Loss function: {config.loss_function}")
    if config.loss_function == "MSELoss":
        loss_function = nn.MSELoss()
    elif config.loss_function == "NDCGLoss":
        loss_function = NDCGLoss()
    elif config.loss_function == "MyLoss1":
        loss_function = MyLoss1()
    elif config.loss_function == "Myloss2":
        loss_function = Myloss2()
    loss_function.to(config.device)

    # Set optimizer
    print(fr"=====>> Set optimizer: AdamW")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # start train
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0
        train_bar = tqdm(train_loader)  # 形成进度条
        for x_train, y_train in train_bar:
            x_train = x_train.to(config.device)
            y_train = y_train.to(config.device)
            optimizer.zero_grad()
            y_train_pred = model(x_train)
            loss = loss_function(y_train_pred, y_train.reshape(-1, 1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, config.epochs, loss)
        print("epoch: train loss: {:.3f}".format(train_loss * config.batch_size / len(train_loader)))

        # eval
        model.eval()
        val_loss_total = 0
        with torch.no_grad():
            test_bar = tqdm(test_loader)
            for x_test, y_test in test_bar:
                x_test = x_test.to(config.device)
                y_test = y_test.to(config.device)
                y_test_pred = model(x_test)
                loss = loss_function(y_test_pred, y_test.reshape(-1, 1))
                val_loss_total += loss.item()
                test_bar.desc = "test  loss:{:.3f}".format(loss.item())
            val_loss = val_loss_total * config.batch_size / len(test_loader)
            print("val loss: {:.3f}".format(val_loss))

        if val_loss < config.best_loss:
            config.best_loss = val_loss
            torch.save({"model": model.state_dict(), "val_loss": config.best_loss}, config.save_model_path)
            print(fr"=====>>  save model: {config.save_model_path}")

    print('Finished Training')


if __name__ == "__main__":
    """
    model_names = ["gru", "gru1", "TransAm", "GRU_TransAm"]
    loss_functions = ["MSELoss", "MyLoss1"]
    
    config = Config(model_name="GRU_TransAm", loss_function="MyLoss1")
    run_train(config)
    """
    # stock_codes = ["000001.SZ", "000002.SZ"]
    # stock_codes = [code for code in stock_dl.codes if code.split(".")[-1] in ['SH', 'SZ']]

    config = Config(model_name="GRU_TransAm", loss_function="MyLoss1")
    run_train(config)

    models = ["gru", "gru1", "TransAm", "GRU_TransAm"]
    loss_functions = ["MSELoss", "NDCGLoss", "MyLoss1", "MyLoss2"]
    # for loss_function in loss_functions:
    #     config = Config(model_name="gru", loss_function=loss_function)
    #     run_train(config)


