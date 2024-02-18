import math
import pdb

import torch
from torch import nn


class GRU(nn.Module):
    def __init__(self, feature_size, hidden_size, num_layers, output_size):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size  # 隐层大小
        self.num_layers = num_layers  # gru层数
        # feature_size为特征维度，就是每个时间点对应的特征数量，这里为1
        self.gru = nn.GRU(feature_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        batch_size = x.shape[0]  # 获取批次大小

        # 初始化隐层状态
        if hidden is None:
            h_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float().cuda()
        else:
            h_0 = hidden
        x.cuda()

        output, h_0 = self.gru(x, h_0)  # GRU运算
        batch_size, timestep, hidden_size = output.shape  # 获取GRU输出的维度信息
        output = output.reshape(-1, hidden_size)  # 将output变成 batch_size * timestep, hidden_dim
        output = self.fc(output)  # 全连接层, 形状为batch_size * timestep, 1
        output = output.reshape(timestep, batch_size, -1)  # 转换维度，用于输出

        # 我们只需要返回最后一个时间片的数据即可
        return output[-1]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)[:, 0:11]
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransAm(nn.Module):
    def __init__(self, feature_size=23, num_layers=1, dropout=0.1):
        super(TransAm, self).__init__()
        self.model_type = 'transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=1, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        batch_size, timestep, hidden_size = src.shape  # 获取GRU输出的维度信息
        output = output.reshape(-1, hidden_size)
        output = self.decoder(output)
        output = output.reshape(timestep, batch_size, -1)  # 转换维度，用于输出

        return output[-1]

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class GRU_TransAm(nn.Module):
    def __init__(self, feature_size=23, num_layers_tran=1, dropout=0.1, hidden_size_gru=512, num_layers_gru=3, output_size=1):
        super(GRU_TransAm, self).__init__()
        self.model_type = 'gru+transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=1, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers_tran)
        self.decoder = nn.Linear(feature_size, 1)
        self.init_weights()

        # gru
        self.hidden_size_gru = hidden_size_gru  # 隐层大小
        self.num_layers_gru = num_layers_gru  # gru层数
        # feature_size为特征维度，就是每个时间点对应的特征数量，这里为1
        self.gru = nn.GRU(feature_size, hidden_size_gru, num_layers_gru, batch_first=True)
        self.fc = nn.Linear(hidden_size_gru, output_size)


    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, hidden=None):
        x = src

        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask
        src = self.pos_encoder(src)

        batch_size = x.shape[0]  # 获取批次大小
        # 初始化隐层状态
        if hidden is None:
            h_0 = x.data.new(self.num_layers_gru, batch_size, self.hidden_size_gru).fill_(0).float().cuda()
        else:
            h_0 = hidden
        x.cuda()

        output_tran = self.transformer_encoder(src, self.src_mask)
        output_gru, h_0 = self.gru(x, h_0)  # GRU运算

        batch_size, timestep, hidden_size_tran = src.shape  # 获取GRU输出的维度信息
        output_tran = output_tran.reshape(-1, hidden_size_tran)
        batch_size, timestep, hidden_size_gru = output_gru.shape  # 获取GRU输出的维度信息
        output_gru = output_gru.reshape(-1, hidden_size_gru)  # 将output变成 batch_size * timestep, hidden_dim

        output_tran = self.decoder(output_tran)

        output_tran = output_tran.reshape(timestep, batch_size, -1)  # 转换维度，用于输出

        output_gru = self.fc(output_gru)  # 全连接层, 形状为batch_size * timestep, 1
        output_gru = output_gru.reshape(timestep, batch_size, -1)  # 转换维度，用于输出

        output = (output_tran[-1] + output_gru[-1]) * 0.5

        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
