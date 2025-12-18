import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import random
import numpy as np
import os
from sklearn.metrics import roc_auc_score
from torch_sparse import SparseTensor

# فرض بر این است که فایل‌های شما در این مسیرها هستند
from dataset_loader import DataLoader
from models import GCN_mamba_Net

# --- تابع کمکی برای تکرارپذیری ---
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['OMP_NUM_THREADS'] = '1'

# --- مدل اصلی برای تشخیص ناهنجاری ---
class AnomalyGNN(nn.Module):
    def __init__(self, encoder, in_features, hidden_features):
        super(AnomalyGNN, self).__init__()
        self.encoder = encoder
        self.decoder = nn.Linear(hidden_features, in_features)

    def forward(self, x, adj_t, return_z=False):
        """
        ### تغییر ۱: این متد حالا می‌تواند فضای نهفته (z) را هم برگرداند ###
        """
        # انکودر شما یک تاپل (embeddings, log_softmax_output) برمی‌گرداند
        embeddings, _ = self.encoder(x, adj_t)
        reconstructed_x = self.decoder(embeddings)
        
        if return_z:
            return reconstructed_x, embeddings
        
        return reconstructed_x

# --- توابع تمرین و تست با لاگ‌گیری دقیق ---

def train(model, optimizer, data):
    """
    ### تغییر ۲: این تابع حالا لاگ‌های بسیار دقیق‌تری را برمی‌گرداند ###
    """
    model.train()
    optimizer.zero_grad()
    
    # ما z (فضای نهفته) را هم برای تحلیل دریافت می‌کنیم
    reconstructed_x, z = model(data.x, data.adj_t, return_z=True)
    loss = F.mse_loss(reconstructed_x, data.x)
    
    loss.backward()

    # --- شروع لاگ‌گیری دقیق ---
    log_stats = {}

    # ۱. آمار فضای نهفته (z)
    # این به ما می‌گوید آیا انکودر یک خروجی بی‌معنی تولید می‌کند یا نه
    log_stats['z_mean'] = z.mean().item()
    log_stats['z_std'] = z.std().item()

    # ۲. محاسبه گرادیان انکودر و دیکودر به صورت جداگانه
    encoder_grad_norm = 0.0
    decoder_grad_norm = 0.0
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2).item()
            if 'encoder' in name:
                encoder_grad_norm += param_norm ** 2
            elif 'decoder' in name:
                decoder_grad_norm += param_norm ** 2
    
    log_stats['encoder_grad'] = np.sqrt(encoder_grad_norm)
    log_stats['decoder_grad'] = np.sqrt(decoder_grad_norm)
    
    # ۳. اضافه کردن Loss به لاگ
    log_stats['loss'] = loss.item()
    # --- پایان لاگ‌گیری ---
    
    optimizer.step()
    
    return log_stats

def test(model, data):
    """
    این تابع بدون تغییر باقی می‌ماند.
    """
    model.eval()
    with torch.no_grad():
        reconstructed_x = model(data.x, data.adj_t)
        anomaly_scores = torch.sum((data.x - reconstructed_x) ** 2, dim=1).cpu().numpy()
        true_labels = data.y.cpu().numpy()
        auc_score = roc_auc_score(true_labels, anomaly_scores)
        
        scores_normal = anomaly_scores[true_labels == 0]
        scores_anomaly = anomaly_scores[true_labels == 1]
        mean_normal, std_normal = np.mean(scores_normal), np.std(scores_normal)
        mean_anomaly, std_anomaly = np.mean(scores_anomaly), np.std(scores_anomaly)
        
    return auc_score, (mean_normal, std_normal), (mean_anomaly, std_anomaly)

# --- بخش اصلی اجرا ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unsupervised Anomaly Detection with MambaGNN (Detailed Logging)')
    # ... آرگومان‌های شما بدون تغییر باقی می‌مانند ...
    parser.add_argument('--dataset', type=str, required=True, choices=['cora', 'citeseer', 'pubmed', 'bitotc', 'bitcoinotc', 'bitalpha'], help='Dataset to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0., help='Weight decay.')
    parser.add_argument('--device', type=int, default=0, help='GPU device ID.')
    parser.add_argument('--d_model', type=int, default=128, help='Hidden dimension size (embedding size).')
    parser.add_argument('--d_state', type=int, default=16, help='Mamba state dimension.')
    parser.add_argument('--alpha', type=float, default=0.8)
    parser.add_argument('--graph_weight', type=float, default=0.9)
    parser.add_argument('--mamba_dropout', type=float, default=0.5)
    parser.add_argument('--layer_num', type=int, default=3)
    parser.add_argument('--d_inner', type=int, default=128)
    parser.add_argument('--dt_rank', type=int, default=16)
    parser.add_argument('--bias', action='store_true')
    parser.add_argument('--d_conv', type=int, default=4)
    parser.add_argument('--expand', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    
    args = parser.parse_args()

    print("--- Configuration ---")
    for k, v in vars(args).items():
        print(f"{k:<20}: {v}")
    print("---------------------")

    fix_seed(args.seed)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    dataset = DataLoader(args.dataset)
    data = dataset[0].to(device)
    adj_t = SparseTensor(row=data.edge_index[0], col=data.edge_index[1],
                         value=data.edge_attr,
                         sparse_sizes=(data.num_nodes, data.num_nodes)).to(device)
    data.adj_t = adj_t

    encoder = GCN_mamba_Net(dataset, args).to(device)
    model = AnomalyGNN(encoder=encoder, in_features=data.num_features, hidden_features=args.d_model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_auc = 0
    best_epoch = 0

    print("\n--- Starting Training ---")
    ### تغییر ۳: فرمت لاگ جدید ###
    print("Log Format: Epoch | Loss | AUC | Z_Mean | Z_Std | Grad(Enc) | Grad(Dec) | Scores(N) | Scores(A)")
    
    with tqdm(range(1, args.epochs + 1)) as pbar:
        for epoch in pbar:
            # تابع train حالا یک دیکشنری از لاگ‌ها را برمی‌گرداند
            train_logs = train(model, optimizer, data)
            pbar.set_description(f"Epoch {epoch:03d} | Loss: {train_logs['loss']:.4f}")

            # ارزیابی دوره‌ای
            if epoch % 10 == 0 or epoch == 1 or epoch == args.epochs:
                auc, (mean_n, std_n), (mean_a, std_a) = test(model, data)
                
                ### تغییر ۴: پیام لاگ جدید و کامل‌تر ###
                log_message = (
                    f"Epoch {epoch:03d} | Loss: {train_logs['loss']:.4f} | AUC: {auc:.4f} | "
                    f"Z_Mean: {train_logs['z_mean']:.3f} | Z_Std: {train_logs['z_std']:.3f} | "
                    f"Grad(Enc): {train_logs['encoder_grad']:.4f} | "
                    f"Grad(Dec): {train_logs['decoder_grad']:.4f} | "
                    f"Scores(N): {mean_n:.3f}±{std_n:.3f} | "
                    f"Scores(A): {mean_a:.3f}±{std_a:.3f}"
                )
                print(log_message)

                if auc > best_auc:
                    best_auc = auc
                    best_epoch = epoch
                    torch.save(model.state_dict(), f'best_model_{args.dataset}.pkl')

    print("\n--- Training Finished ---")
    print(f"Best AUC Score: {best_auc:.4f} at epoch {best_epoch}")
    
    print("\n--- Loading Best Model for Final Detailed Evaluation ---")
    model.load_state_dict(torch.load(f'best_model_{args.dataset}.pkl'))
    final_auc, (mean_n, std_n), (mean_a, std_a) = test(model, data)
    
    print(f"Final AUC: {final_auc:.4f}")
    print(f"Final Normal Node Scores: {mean_n:.4f} (std: {std_n:.4f})")
    print(f"Final Anomaly Node Scores: {mean_a:.4f} (std: {std_a:.4f})")

