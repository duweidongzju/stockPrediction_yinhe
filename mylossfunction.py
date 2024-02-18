import torch
from torch import nn
from torch.autograd import Variable
import pdb
import math


class NDCGLoss(nn.Module):
    def __init__(self):
        super(NDCGLoss, self).__init__()

    def forward(self, output, label):
        label_max = 13.982028007507324
        label_min = -11.172813415527344

        dictionary = {}
        for code, label in enumerate(label):
            dictionary[str(code)] = {}
            dictionary[str(code)]["label"] = label.item() * (label_max - label_min) + label_min
            dictionary[str(code)]["output"] = output[code].item()
        output_set = sorted(dictionary.items(), key=lambda dic: dic[1]["output"], reverse=True)
        output_codes = [item[0] for item in output_set]
        label_set = sorted(dictionary.items(), key=lambda dic: dic[1]["label"], reverse=True)
        label_codes = [item[0] for item in label_set]

        # add G and eta_output, calculate DCG
        DCG = 0
        for i, code in enumerate(output_codes):
            dictionary[code]["gain"] = pow(2, dictionary[code]["label"]) - 1
            dictionary[code]["eta_output"] = 1 / math.log(i + 2, 2)
            DCG += dictionary[code]["gain"] * dictionary[code]["eta_output"]

        # add eta_label, calculate Zk
        Zk = 0
        for i, code in enumerate(label_codes):
            dictionary[code]["eta_label"] = 1 / math.log(i + 2, 2)
            Zk += dictionary[code]["gain"] * dictionary[code]["eta_label"]

        NDCG = torch.tensor(DCG / Zk)
        NDCG = NDCG.to(torch.device("cuda"))
        loss = Variable(NDCG, requires_grad=True)

        return loss


class MyLoss1(nn.Module):
    def __init__(self):
        super(MyLoss1, self).__init__()

    def forward(self, output, label):
        label_max = 13.982028007507324
        label_min = -11.172813415527344

        origin_label = label * (label_max - label_min) + label_min
        sign = [-1 if i < 0 else 1 for i in origin_label]
        amp = torch.tensor([-i if i < 0 else i for i in origin_label])

        weight1 = torch.log(amp * 3 + 2) + 0.5  # 幅值作为权重,整体往上平移
        alpha = torch.tensor([0.5 if i < 0 else 1 for i in sign])  # 关注涨的股票
        weight1 = torch.mul(weight1, alpha)
        weight1 = weight1.to(torch.device("cuda"))

        weight2 = 0.5 - torch.pow(origin_label / 15, 2)

        weight = weight1 + weight2

        loss = torch.pow(label - output, 2)
        loss = torch.mul(loss, weight)
        loss = torch.mean(loss)
        loss = Variable(loss, requires_grad=True)

        return loss


class Myloss2(nn.Module):
    def __init__(self):
        super(Myloss2, self).__init__()

    def forward(self, output, label):
        loss = torch.pow(label - output, 2)
        loss = torch.abs(loss)
        loss = torch.mul(loss, label)
        loss = torch.mean(loss)
        loss = Variable(loss, requires_grad=True)

        return loss