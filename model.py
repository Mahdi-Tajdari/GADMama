# model.py

import torch.nn as nn
import torch.nn.functional as F
import torch
from mamba_gcn_modules import GCN_mamba_Net_Encoder 

class MambaGCNEncoder(nn.Module):
    """ A wrapper for the novelty model to be used as an encoder. """
    def __init__(self, feat_size, mamba_args, encoder_mode):
        super().__init__()
        self.encoder = GCN_mamba_Net_Encoder(n_features=feat_size, args=mamba_args, mode=encoder_mode)
        
    def forward(self, x, adj, labels=None, epoch=-1):
        return self.encoder(x, adj, labels, epoch)

# <<< NEW: SUPER SIMPLE DECODER >>>
class Simple_Attribute_Decoder(nn.Module):
    """
    این دیکودر هیچ قدرت گراف یا لایه پنهانی ندارد.
    فقط یک تبدیل خطی ساده است: Z * W + b -> X_hat
    """
    def __init__(self, nfeat, nhid, dropout):
        super(Simple_Attribute_Decoder, self).__init__()
       
        self.linear = nn.Linear(nhid, nfeat)
        self.dropout = dropout

    def forward(self, x, adj):

        x = F.dropout(x, self.dropout, training=self.training)
        return self.linear(x)

# <<< NEW: PARAMETER-FREE STRUCTURE DECODER >>>
class Simple_Structure_Decoder(nn.Module):
    """
    ساده‌ترین دیکودر ساختار ممکن: Z * Z.T
    هیچ پارامتر قابل یادگیری ندارد. تمام فشار روی انکودر است تا Z را درست بسازد.
    """
    def __init__(self, dropout):
        super(Simple_Structure_Decoder, self).__init__()
        self.dropout = dropout

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
       
        score = x @ x.T
        return torch.sigmoid(score)

class Dominant(nn.Module):
    def __init__(self, feat_size, dropout, mamba_args, encoder_mode):
        super(Dominant, self).__init__()
        hidden_size = mamba_args.d_model
        
        
        self.shared_encoder = MambaGCNEncoder(feat_size, mamba_args, encoder_mode)
        
        
        self.attr_decoder = Simple_Attribute_Decoder(feat_size, hidden_size, dropout)
        
        
        self.struct_decoder = Simple_Structure_Decoder(dropout)
    
    def forward(self, x, adj, labels=None, epoch=-1):
        
        encoded = self.shared_encoder(x, adj, labels, epoch)
        
      
        x_hat = self.attr_decoder(encoded, adj)
        
      
        struct_reconstructed = self.struct_decoder(encoded)
        
        return struct_reconstructed, x_hat



class Real_Local_Encoder(nn.Module):
    def __init__(self, n_features, args):
        super(Real_Local_Encoder, self).__init__()
        self.args = args
        self.dropout = args.mamba_dropout
        
        # لایه ۱: GCN واقعی (اجبار به دیدن همسایه‌ها)
        # این لایه وزن W دارد و فیچرها را با همسایه‌ها ترکیب می‌کند
        self.gcn_layer = GraphConvolution(n_features, args.d_model)
        
        # لایه ۲: نرمال‌سازی و فعالیت
        self.bn = nn.BatchNorm1d(args.d_model)
        
        # لایه ۳: مامبا (برای یادگیری الگوهای پیچیده از فیچرهای ترکیب شده)
        self.mamba = Mamba(
                d_model=args.d_model, 
                d_state=16, 
                d_conv=4, 
                expand=2
            )
            
    def forward(self, x, adj):
        # قدم ۱: ترکیب اطلاعات همسایه‌ها (اینجاست که Linear شکست می‌خورد)
        x = self.gcn_layer(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.bn(x)
        
        # قدم ۲: پردازش ترتیبی/پیچیده با مامبا
        # چون خروجی GCN برای همه نودهاست، ما به عنوان یک Sequence (1, N, D) به مامبا می‌دهیم
        x_seq = x.unsqueeze(0) 
        x_mamba = self.mamba(x_seq)
        x_out = x_mamba.squeeze(0)
        
        # قدم ۳: Residual Connection (جمع با خروجی GCN)
        # این باعث می‌شود مامبا فقط "بهبود" را یاد بگیرد
        return x_out + x