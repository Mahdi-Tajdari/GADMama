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

# Assuming your data loader and models are in these files
from dataset_loader import DataLoader
from models import GCN_mamba_Net

# --- Helper Function for Reproducibility ---
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

# --- Model Definition for Anomaly Detection ---
class AnomalyGNN(nn.Module):
    def __init__(self, encoder, in_features, hidden_features):
        super(AnomalyGNN, self).__init__()
        self.encoder = encoder
        self.decoder = nn.Linear(hidden_features, in_features)

    def forward(self, x, adj_t):
        embeddings, _ = self.encoder(x, adj_t)
        reconstructed_x = self.decoder(embeddings)
        return reconstructed_x

# --- Training and Evaluation Functions with Detailed Logging ---
def train(model, optimizer, data):
    """
    Executes one training epoch and returns loss and average gradient norm.
    """
    model.train()
    optimizer.zero_grad()
    
    reconstructed_x = model(data.x, data.adj_t)
    loss = F.mse_loss(reconstructed_x, data.x)
    
    loss.backward()
    
    # --- Calculate Average Gradient Norm ---
    total_norm = 0
    num_params = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            num_params += 1
    avg_grad_norm = (total_norm ** 0.5) / num_params if num_params > 0 else 0
    # --- End Gradient Calculation ---
    
    optimizer.step()
    return loss.item(), avg_grad_norm

def test(model, data):
    """
    Evaluates the model and returns detailed stats:
    AUC, (mean, std) of normal scores, (mean, std) of anomaly scores.
    """
    model.eval()
    with torch.no_grad():
        reconstructed_x = model(data.x, data.adj_t)
        anomaly_scores = torch.sum((data.x - reconstructed_x) ** 2, dim=1).cpu().numpy()
        true_labels = data.y.cpu().numpy()
        
        auc_score = roc_auc_score(true_labels, anomaly_scores)
        
        # --- Calculate score distributions ---
        scores_normal = anomaly_scores[true_labels == 0]
        scores_anomaly = anomaly_scores[true_labels == 1]
        
        mean_normal, std_normal = np.mean(scores_normal), np.std(scores_normal)
        mean_anomaly, std_anomaly = np.mean(scores_anomaly), np.std(scores_anomaly)
        # --- End distribution calculation ---
        
    return auc_score, (mean_normal, std_normal), (mean_anomaly, std_anomaly)

# --- Main Execution Block ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unsupervised Anomaly Detection with MambaGNN (Detailed Logging)')
    # General arguments
    parser.add_argument('--dataset', type=str, required=True, choices=['cora', 'citeseer', 'pubmed', 'bitotc', 'bitcoinotc', 'bitalpha'], help='Dataset to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0., help='Weight decay.')
    parser.add_argument('--device', type=int, default=0, help='GPU device ID.')
    
    # Mamba GNN Encoder arguments
    parser.add_argument('--d_model', type=int, default=128, help='Hidden dimension size (embedding size).')
    parser.add_argument('--d_state', type=int, default=16, help='Mamba state dimension.')
    # Other Mamba args...
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
    print("Log Format: Epoch | Loss | AUC | Grad Norm | Scores(Normal) | Scores(Anomaly)")
    
    with tqdm(range(1, args.epochs + 1)) as pbar:
        for epoch in pbar:
            loss, grad_norm = train(model, optimizer, data)
            pbar.set_description(f"Epoch {epoch:03d} | Loss: {loss:.4f}")
            
            # Evaluate periodically
            if epoch % 10 == 0 or epoch == 1 or epoch == args.epochs:
                auc, (mean_n, std_n), (mean_a, std_a) = test(model, data)
                
                log_message = (
                    f"Epoch {epoch:03d} | Loss: {loss:.4f} | AUC: {auc:.4f} | "
                    f"Grad Norm: {grad_norm:.4f} | "
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
    # The warning here is safe to ignore as you are loading a model you just saved.
    model.load_state_dict(torch.load(f'best_model_{args.dataset}.pkl', weights_only=True))
    final_auc, (mean_n, std_n), (mean_a, std_a) = test(model, data)
    
    print(f"Final AUC: {final_auc:.4f}")
    print(f"Final Normal Node Scores: {mean_n:.4f} (std: {std_n:.4f})")
    print(f"Final Anomaly Node Scores: {mean_a:.4f} (std: {std_a:.4f})")
