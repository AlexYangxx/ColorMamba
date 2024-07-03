import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchsummary import summary
from thop import profile
import time

from torch.nn import Softmax


def INF(B, H, W, device):
    return -torch.diag(torch.tensor(float("inf")).to(device).repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim):
        super().__init__()
        self.proj = nn.Conv1d(in_channels, embed_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.proj(x)
        x = x.transpose(1, 2)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.att_drop = nn.Dropout(0.1)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(0.1)

    def forward(self, x, y):
        B, N, C = x.shape

        q, k, v = x, y, y
        q = q.reshape(B, N, self.num_heads, C // self.num_heads)
        k = k.reshape(B, N, self.num_heads, C // self.num_heads)
        v = v.reshape(B, N, self.num_heads, C // self.num_heads)
        attn = (q @ k.transpose(-2, -1)) / (C // self.num_heads) ** 0.5
        attn = attn.softmax(dim=-1)
        attn = self.att_drop(attn)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm11 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), embed_dim)

    def forward(self, x, y):
        n_x = self.norm1(x)
        n_y = self.norm11(y)
        x = x + self.attn(n_x, n_y)
        x = x + self.mlp(self.norm2(x))
        return x


def conv_diff(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU()
    )


class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""

    def __init__(self, in_dim, in_dim_1):
        super(CrissCrossAttention, self).__init__()
        self.query_conv = conv_diff(in_channels=in_dim, out_channels=in_dim // 8)
        self.key_conv = conv_diff(in_channels=in_dim, out_channels=in_dim // 8)
        self.value_conv = conv_diff(in_channels=in_dim_1, out_channels=in_dim)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.conv_skip = conv_diff(in_dim + in_dim_1, in_dim)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, y):
        # print(x.shape, y.shape)
        x_skip = self.conv_skip(torch.cat([x, y], dim=1))
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2,
                                                                                                                 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2,
                                                                                                                 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        proj_value = self.value_conv(y)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H) +
                    self.INF(m_batchsize, height, width, x.device)).view(m_batchsize, width,
                                                                         height,
                                                                         height).permute(0, 2, 1, 3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        return self.gamma * (out_H + out_W) + x_skip
