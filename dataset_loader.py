import torch
import os.path as osp
import numpy as np
import scipy.sparse as sp
import scipy.io
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import from_scipy_sparse_matrix
import torch_geometric.transforms as T

# --- Keep other original imports if needed ---
from torch_geometric.datasets import Planetoid, Reddit # etc.

# In your dataset_loader.py file

def load_anomaly_mat_dataset(name, root='data/'):
    """
    Loads and preprocesses graph anomaly detection datasets from .mat files.
    
    IMPORTANT: This version REMOVES the feature normalization, allowing the model's
    initial nn.Linear layer to act as a proper embedding layer on the raw features,
    which prevents over-smoothing.
    """
    filepath = osp.join(root, f'{name}.mat')
    if not osp.exists(filepath):
        raise FileNotFoundError(f"Dataset .mat file not found at: {filepath}")

    print(f"Loading '{name}' from {filepath}. SKIPPING feature pre-processing to use model's embedding layer.")
    mat_data = scipy.io.loadmat(filepath)

    # 1. Load data
    adj = mat_data.get('Network', mat_data.get('A'))
    features = mat_data.get('Attributes', mat_data.get('X'))
    labels = mat_data.get('Label', mat_data.get('gnd'))

    # Convert to sparse formats
    adj = sp.csr_matrix(adj)
    features = sp.lil_matrix(features)

    # ------------------- THE CRUCIAL CHANGE IS HERE -------------------
    #
    # 2. Convert raw features directly to a dense tensor.
    # We NO LONGER perform row-normalization here. The model's first linear
    # layer (`lin1`) will handle this transformation much more effectively.
    #
    if sp.issparse(features):
        features_tensor = torch.FloatTensor(features.toarray())
    else:
        features_tensor = torch.FloatTensor(features)
    #
    # ------------------------------------------------------------------

    # 3. Preprocess Adjacency Matrix (Symmetric Normalization - this is still correct)
    adj_normalized = adj + sp.eye(adj.shape[0])
    row_sum = np.array(adj_normalized.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj_normalized = adj_normalized.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    edge_index, edge_weight = from_scipy_sparse_matrix(adj_normalized)

    # 4. Convert labels to tensor
    labels_tensor = torch.LongTensor(labels).squeeze()

    # 5. Create PyG Data object
    pyg_data = Data(x=features_tensor, edge_index=edge_index, edge_attr=edge_weight, y=labels_tensor)
    pyg_data.num_nodes = features_tensor.shape[0]

    return pyg_data

def DataLoader(name):
    """
    Main data loading function.
    It now uses the AD-GCL specific loader for your anomaly datasets.
    """
    name = name.lower()
    
    # These datasets will now be processed using the AD-GCL methodology
    anomaly_datasets = ['cora', 'citeseer', 'pubmed', 'bitotc', 'bitcoinotc', 'bitalpha']

    if name in anomaly_datasets:
        data = load_anomaly_mat_dataset(name, root='data/')

        # Wrapper to maintain compatibility with your training script's dataset[0] access
        class AnomalyDatasetWrapper(InMemoryDataset):
            def __init__(self, data_obj):
                super(AnomalyDatasetWrapper, self).__init__()
                self.data, self.slices = self.collate([data_obj])
            
            @property
            def num_features(self):
                return self.data.num_features

            @property
            def num_classes(self): # For anomaly detection, it's always binary
                return 2

        return AnomalyDatasetWrapper(data)
    
    # --- Your original classification loaders can remain here for other datasets ---
    else:
        # This part is for any other datasets you might use that are NOT for anomaly detection
        print(f"Loading classification dataset '{name}'...")
        # Example for other datasets:
        # if name == 'computers':
        #     dataset = Amazon('./data', name, T.NormalizeFeatures())
        # else:
        raise ValueError(f"Dataset '{name}' is not configured in the anomaly or classification loaders.")

    return dataset
