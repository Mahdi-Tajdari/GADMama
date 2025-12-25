# mamba_gcn_modules.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
from einops import repeat, einsum
from mamba_ssm import Mamba

# --- توابع کمکی ---
class GCN_mamba_liner(torch.nn.Module):
    def __init__(self, in_features, out_features, with_bias=False):
        super(GCN_mamba_liner, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if with_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        output = input @ self.weight
        if self.bias is not None:
            return output + self.bias
        else:
            return output

# --- انکودر اصلی (شامل حالت جدید Ablation) ---
class GCN_mamba_Net_Encoder(torch.nn.Module):
    def __init__(self, n_features, args, mode='local'):
        super(GCN_mamba_Net_Encoder, self).__init__()
        self.args = args
        self.mode = mode
        self.dropout_rate = args.mamba_dropout
        
        print(f"\n--- Encoder Initialized in '{self.mode.upper()}' Mode ---\n")

        # 1. لایه تبدیل ویژگی اولیه (مشترک)
        self.embedding = GCN_mamba_liner(n_features, args.d_model, with_bias=args.bias)
        self.bn_in = torch.nn.BatchNorm1d(args.d_model)

        # 2. ماژول‌های اختصاصی
        if self.mode == 'global':
            self.mamba_global = Mamba(
                d_model=args.d_model, d_state=16, d_conv=4, expand=2
            )
            self.bn_global = torch.nn.BatchNorm1d(args.d_model)
        
        elif self.mode == 'local':
            self.gcn_weight = Parameter(torch.FloatTensor(args.d_model, args.d_model))
            self.mamba_local = Mamba(
                d_model=args.d_model, d_state=16, d_conv=4, expand=2
            )
            self.bn_local = torch.nn.BatchNorm1d(args.d_model)
            self.reset_gcn_parameters()
            
        elif self.mode == 'gcn_only':
            # <<< ABLATION MODE >>>
            # فقط وزن GCN را دارد، خبری از مامبا نیست
            self.gcn_weight = Parameter(torch.FloatTensor(args.d_model, args.d_model))
            self.bn_local = torch.nn.BatchNorm1d(args.d_model) # نرمال‌سازی ساده
            self.reset_gcn_parameters()

    def reset_gcn_parameters(self):
        if self.mode in ['local', 'gcn_only']:
            stdv = 1. / math.sqrt(self.gcn_weight.size(1))
            self.gcn_weight.data.uniform_(-stdv, stdv)

    def forward(self, x, adj, labels=None, epoch=-1):
        # --- تبدیل ویژگی‌ها ---
        x_emb = self.embedding(x)
        x_emb = self.bn_in(x_emb)
        x_emb = F.relu(x_emb)
        x_emb = F.dropout(x_emb, p=self.dropout_rate, training=self.training)
        
        output = x_emb 

        # --- لاگ‌گیری ---
        do_log = (self.training and epoch != -1 and epoch % 20 == 0)

        if self.mode == 'linear':
            output = x_emb

        elif self.mode == 'global':
            x_seq = x_emb.unsqueeze(0) 
            mamba_out = self.mamba_global(x_seq).squeeze(0)
            output = self.bn_global(x_emb + mamba_out)

        elif self.mode == 'local':
            # 1. GCN
            support = torch.mm(x_emb, self.gcn_weight)
            x_gcn = torch.spmm(adj, support) 
            # 2. Mamba
            x_seq = x_gcn.unsqueeze(0)
            mamba_out = self.mamba_local(x_seq).squeeze(0)
            # Residual
            output = self.bn_local(x_gcn + mamba_out)

            if do_log:
                norm_mamba = mamba_out.norm(p=2, dim=1).mean().item()
                print(f"[Local] Mamba Active. Output Norm: {norm_mamba:.2f}")

        elif self.mode == 'gcn_only':
            # <<< ABLATION LOGIC >>>
            # 1. GCN (دقیقا مشابه حالت Local)
            support = torch.mm(x_emb, self.gcn_weight)
            x_gcn = torch.spmm(adj, support)
            
            # 2. NO MAMBA
            # مستقیماً خروجی GCN را نرمال کرده و می‌دهیم بیرون
            output = self.bn_local(x_gcn) 
            
            if do_log:
                norm_gcn = x_gcn.norm(p=2, dim=1).mean().item()
                print(f"--- [Ablation: GCN Only] Epoch {epoch} ---")
                print(f"    GCN Output Norm: {norm_gcn:.2f}")
                print(f"    (Note: Mamba is removed from pipeline)")

        return output
