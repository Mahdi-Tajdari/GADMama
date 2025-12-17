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
    """
    A Graph Auto-Encoder model for anomaly detection.
    It uses the GCN_mamba_Net as an encoder and a simple linear layer as a decoder.
    """
    def __init__(self, encoder, in_features, hidden_features):
        super(AnomalyGNN, self).__init__()
        self.encoder = encoder
        self.decoder = nn.Linear(hidden_features, in_features)

    def forward(self, x, adj_t):
        # The GCN_mamba_Net encoder needs to return the final node embeddings
        # before the classification head. Let's assume the first element it
        # returns is this embedding tensor.
        embeddings, _ = self.encoder(x, adj_t)
        
        # Decode the embeddings to reconstruct the original features
        reconstructed_x = self.decoder(embeddings)
        return reconstructed_x

# --- Training and Evaluation Functions ---
def train(model, optimizer, data):
    """
    Executes one training epoch for the unsupervised GAE model.
    """
    model.train()
    optimizer.zero_grad()
    
    reconstructed_x = model(data.x, data.adj_t)
    loss = F.mse_loss(reconstructed_x, data.x)
    
    loss.backward()
    optimizer.step()
    return loss.item()

def test(model, data):
    """
    Evaluates the model by calculating the AUC-ROC score.
    """
    model.eval()
    with torch.no_grad():
        reconstructed_x = model(data.x, data.adj_t)
        anomaly_scores = torch.sum((data.x - reconstructed_x) ** 2, dim=1).cpu().numpy()
        true_labels = data.y.cpu().numpy()
        auc_score = roc_auc_score(true_labels, anomaly_scores)
    return auc_score

# --- Main Execution Block ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unsupervised Anomaly Detection with MambaGNN')
    # --- General arguments ---
    parser.add_argument('--dataset', type=str, required=True, choices=['cora', 'citeseer', 'pubmed', 'bitotc', 'bitcoinotc', 'bitalpha'], help='Dataset to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    # THIS IS THE CORRECTED LINE:
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0., help='Weight decay for the optimizer.')
    parser.add_argument('--device', type=int, default=0, help='GPU device ID.')
    
    # --- Mamba GNN Encoder arguments (from your original file) ---
    parser.add_argument('--alpha', type=float, default=0.8)
    parser.add_argument('--graph_weight', type=float, default=0.9)
    parser.add_argument('--d_model', type=int, default=64, help='Hidden dimension size (embedding size).')
    parser.add_argument('--mamba_dropout', type=float, default=0.5, help='Dropout for mamba layers.')
    parser.add_argument('--layer_num', type=int, default=3, help='Number of layers.')
    parser.add_argument('--d_inner', type=int, default=64, help='')
    parser.add_argument('--dt_rank', type=int, default=16, help='')
    parser.add_argument('--d_state', type=int, default=16, help='')
    parser.add_argument('--bias', action='store_true', help='Use bias if set')
    parser.add_argument('--d_conv', type=int, default=4, help='')
    parser.add_argument('--expand', type=int, default=2, help='')
    # You had a 'dropout' argument in the original file, which is different from 'mamba_dropout'.
    # If GCN_mamba_Net uses it, we should add it back.
    parser.add_argument('--dropout', type=float, default=0.2, help='General dropout rate.')


    args = parser.parse_args()
    print("--- Configuration ---")
    print(args)
    print("---------------------")

    fix_seed(args.seed)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    dataset = DataLoader(args.dataset)
    data = dataset[0].to(device)

    adj_t = SparseTensor(row=data.edge_index[0], col=data.edge_index[1],
                         value=data.edge_attr,
                         sparse_sizes=(data.num_nodes, data.num_nodes)).to(device)
    data.adj_t = adj_t

    # IMPORTANT: You need to make a small change in your `models.py` file.
    # In the `forward` method of `GCN_mamba_Net`, please ensure it returns the `output`
    # tensor which represents the node embeddings before they are passed to the final classifier.
    # Change: return all_layers_output, F.log_softmax(y, dim=-1)
    # To:     return output, F.log_softmax(y, dim=-1) # Where 'output' is the variable before `self.lin2(output)`
    encoder = GCN_mamba_Net(dataset, args).to(device)
    
    model = AnomalyGNN(
        encoder=encoder,
        in_features=data.num_features,
        hidden_features=args.d_model
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # --- Training Loop ---
    best_auc = 0
    best_epoch = 0
    
    print("Starting training...")
    with tqdm(range(1, args.epochs + 1)) as pbar:
        for epoch in pbar:
            loss = train(model, optimizer, data)
            
            if epoch % 10 == 0 or epoch == args.epochs:
                auc_score = test(model, data)
                pbar.set_description(f"Epoch {epoch:03d} | Loss: {loss:.4f} | AUC: {auc_score:.4f}")

                if auc_score > best_auc:
                    best_auc = auc_score
                    best_epoch = epoch
                    torch.save(model.state_dict(), f'best_model_{args.dataset}.pkl')

    print("\n--- Training Finished ---")
    print(f"Best AUC Score: {best_auc:.4f} at epoch {best_epoch}")

    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load(f'best_model_{args.dataset}.pkl'))
    final_auc = test(model, data)
    print(f"Final AUC Score on loaded best model: {final_auc:.4f}")
