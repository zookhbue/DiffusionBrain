import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math



# define GFE block
class GFE(nn.Module):
    def __init__(self, in_channels):
        super(GFE, self).__init__()

        #　DW Conv
        self.depthwise_conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, groups=in_channels, bias=False)
        self.depthwise_conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False)
        self.depthwise_conv3 = nn.Conv3d(in_channels, in_channels, kernel_size=5, stride=1, padding=2, groups=in_channels, bias=False)
        self.bn = nn.BatchNorm3d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        # MaxPool
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2)

    def forward(self, input):
        x = self.depthwise_conv1(input)
        x = self.depthwise_conv2(input)
        x = self.depthwise_conv3(input)
        y = self.bn(input + x)
        z = self.maxpool(self.relu(y))
        o = self.relu(z)

        return o


# define SFENet
class SFENet(nn.Module):
    def __init__(self, in_channels):
        super(SFENet, self).__init__()

        self.layer1 = GFE(in_channels)
        self.layer2 = GFE(in_channels)
        self.layer3 = GFE(in_channels)
        self.layer4 = GFE(in_channels)

    def forward(self, input):
        x = self.layer1(input)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        F_s = torch.flatten(x, start_dim=2)

        return F_s


# define FFENet
class FFENet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FFENet, self).__init__()

        #　FC Layer
        self.fc = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, input):
        F_f = self.fc(input)

        return F_f


# Cross-Attention
class CrossAttention(nn.Module):
    def __init__(self, d_q, d_kv, d_out):
        super(CrossAttention, self).__init__()
        self.q_layer = nn.Linear(d_q, d_out)
        self.k_layer = nn.Linear(d_kv, d_out)
        self.v_layer = nn.Linear(d_kv, d_out)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, queries, keys, values):
        # Linear projections
        q = self.q_layer(queries)
        k = self.k_layer(keys)
        v = self.v_layer(values)

        # attention scores
        scores = torch.bmm(q, k.transpose(1, 2)) / (k.size(-1) ** 0.5)
        a = self.softmax(scores)
        o = torch.bmm(a, v)
        
        return o


# define CFFM
class CFFM(nn.Module):
    def __init__(self, d_s, d_f, d):
        super(CFFM, self).__init__()
        self.sca = CrossAttention(d_f, d_s, d)
        self.fca = CrossAttention(d_s, d_f, d)
        self.fc = nn.Linear(2*d, d,  bias=False)

    def forward(self, F_s, F_f):
        x = self.sca(F_f, F_s, F_s)
        y = self.fca(F_s, F_f, F_f)
        z = torch.cat((x, y), dim=-1)
        F_c = self.fc(z)

        return F_c


# define GPFM
class GPFM(nn.Module):
    def __init__(self):
        super(GPFM, self).__init__()
        self.sfenet = SFENet(90)
        self.ffenet = FFENet(187, 90)
        self.cffm = CFFM(80, 90, 90)

    def forward(self, X_s, X_f):
        F_s = self.sfenet(X_s)
        F_f = self.ffenet(X_f)
        F_c =self.cffm(F_s, F_f)

        return F_c



# define Multi-Head Self-Attention
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, out_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        
        self.out_dim = out_dim
        self.num_heads = num_heads
        
        self.Q_layer = nn.Linear(out_dim, out_dim*num_heads)
        self.K_layer = nn.Linear(out_dim, out_dim*num_heads)
        self.V_layer = nn.Linear(out_dim, out_dim*num_heads)
        self.O_layer = nn.Linear(out_dim*num_heads, out_dim)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        bs = x.size(0)
        Q = self.Q_layer(x)
        K = self.K_layer(x)
        V = self.V_layer(x)
        
        # Split into multiple heads
        q = Q.view(bs, -1, self.num_heads, self.out_dim).transpose(1, 2)
        k = K.view(bs, -1, self.num_heads, self.out_dim).transpose(1, 2)
        v = V.view(bs, -1, self.num_heads, self.out_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.out_dim ** 0.5)
        a = self.softmax(scores)
        context = torch.matmul(a, v)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(bs, -1, self.out_dim*self.num_heads)
        
        # Final linear layer
        O = self.O_layer(context)
        
        return O


# define GCN
class GCN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCN, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.relu = nn.ReLU(inplace=True)
        nn.init.eye_(self.weight)

    def forward(self, A, F):
        x = torch.matmul(F, self.weight)
        o = self.relu(torch.matmul(A, x))

        return o


# get time embedding
def get_timestep_embedding(t, emb_dim, max_timestep):
    assert len(t.shape) == 1

    half_dim = emb_dim // 2
    emb = math.log(max_timestep) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=t.device)
    emb = t.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if emb_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))

    return emb


# calculate Pearson correlation coefficient
def Corr(feature):
    mu = feature.mean(dim=-1, keepdim=True)  
    std = feature.std(dim=-1, keepdim=True)   

    norm = (feature - mu) / (std + 1e-8) 
    corr = torch.bmm(norm, norm.transpose(1, 2)) / feature.shape[-1]
    
    return corr


# difine DGT
class DGT(nn.Module):
    def __init__(self, in_channels, d, out_dim, num_heads):
        super(DGT, self).__init__()
        self.in_channels = in_channels
        self.multi_head_self_attention = MultiHeadSelfAttention(out_dim, num_heads)
        self.gcn = GCN(d, out_dim)


    def forward(self, A_t, F_c, t):

        t_emb = get_timestep_embedding(t, self.in_channels, 1000)
        c = F_c + t_emb[:, :, None]
        h = self.multi_head_self_attention(c)
        feature = self.gcn(A_t, h)
        A_tau = Corr(feature)

        return A_tau


# difine DDGT
class DDGT(nn.Module):
    def __init__(self, in_channels, d, out_dim, num_heads):
        super(DDGT, self).__init__()
        self.sdgt = DGT(in_channels, d, out_dim, num_heads)
        self.fdgt = DGT(in_channels, d, out_dim, num_heads)

    def forward(self, A_s_t, A_f_t, F_c, t):
        A_s_tau = self.sdgt(A_s_t, F_c, t)
        A_f_tau = self.fdgt(A_f_t, F_c, t)

        return A_s_tau, A_f_tau



# difine GAM
class GAM(nn.Module):
    def __init__(self, in_channels, d, C):
        super(GAM, self).__init__()
        self.gcn = GCN(in_channels, in_channels)
        self.fc = nn.Linear(d, C)

    def forward(self, A_s, A_f, F_c):
        mask_s = torch.triu(torch.ones(90, 90))
        mask_f = torch.tril(torch.ones(90, 90) - torch.ones(90, 90))
        A_init = abs(A_s) * mask_s + abs(A_f) * mask_f
        A_init2 = torch.matmul(A_init,A_init)
        F = self.gcn(A_init2, F_c)
        A = abs(Corr(F))
        F_g = torch.mean(F, dim=1)
        y = self.fc(F_g)

        return A, F, y
