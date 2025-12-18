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
import torch
import os.path as osp
import numpy as np
import scipy.sparse as sp
import scipy.io
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix

def load_anomaly_mat_dataset(name, root='data/'):
    """
    دیتاست‌های تشخیص ناهنجاری را از فایل‌های .mat بارگذاری و پیش‌پردازش می‌کند.
    
    نکته بسیار مهم: این نسخه شامل نرمال‌سازی سطری (row-normalization) برای ویژگی‌هاست
    که یک مرحله حیاتی برای جلوگیری از فروپاشی مدل (model collapse) و صفر شدن گرادیان‌هاست.
    """
    filepath = osp.join(root, f'{name}.mat')
    if not osp.exists(filepath):
        raise FileNotFoundError(f"فایل .mat دیتاست در این مسیر پیدا نشد: {filepath}")

    print(f"در حال بارگذاری '{name}' از {filepath} همراه با نرمال‌سازی ویژگی‌ها.")
    mat_data = scipy.io.loadmat(filepath)

    # ۱. بارگذاری داده‌های خام از فایل
    adj = mat_data.get('Network', mat_data.get('A'))
    features = mat_data.get('Attributes', mat_data.get('X'))
    labels = mat_data.get('Label', mat_data.get('gnd'))

    # تبدیل به فرمت‌های اسپارس برای کارایی بهتر
    adj = sp.csr_matrix(adj)
    features = sp.lil_matrix(features)

    # ------------------- تغییر اصلی و حیاتی اینجاست -------------------
    #
    # ۲. پیش‌پردازش و نرمال‌سازی ویژگی‌ها (Row-Normalization)
    # این کار باعث می‌شود که ویژگی‌های هر نود مقیاس مشابهی داشته باشند و مدل
    # به جای پیدا کردن راه‌حل ساده (خروجی صفر)، مجبور به یادگیری ساختار داده شود.
    #
    print("در حال اعمال نرمال‌سازی سطری روی ویژگی‌ها. این مرحله بسیار مهم است.")
    if sp.issparse(features):
        # محاسبه معکوس مجموع هر سطر
        row_sum = np.array(features.sum(axis=1), dtype=np.float32).flatten()
        r_inv = np.power(row_sum, -1)
        r_inv[np.isinf(r_inv)] = 0. # مقادیر بی‌نهایت (ناشی از تقسیم بر صفر) را صفر می‌کنیم
        r_mat_inv = sp.diags(r_inv) # ساخت ماتریس قطری از مقادیر معکوس
        
        # ضرب ماتریس قطری در ماتریس ویژگی‌ها برای نرمال‌سازی
        features = r_mat_inv.dot(features)
        features_tensor = torch.FloatTensor(features.toarray())
    else: # اگر ویژگی‌ها اسپارس نباشند (که در این دیتاست‌ها هستند)
        row_sum = features.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1 # جلوگیری از تقسیم بر صفر
        features = features / row_sum
        features_tensor = torch.FloatTensor(features)
    #
    # ------------------------------------------------------------------

    # ۳. پیش‌پردازش ماتریس مجاورت (Symmetric Normalization - بدون تغییر)
    adj_normalized = adj + sp.eye(adj.shape[0])
    row_sum = np.array(adj_normalized.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj_normalized = adj_normalized.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    
    # تبدیل به فرمت مورد نیاز PyG
    edge_index, edge_weight = from_scipy_sparse_matrix(adj_normalized)

    # ۴. تبدیل لیبل‌ها به تنسور
    labels_tensor = torch.LongTensor(labels).squeeze()

    # ۵. ساخت آبجکت نهایی PyTorch Geometric Data
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
