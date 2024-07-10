import scipy.sparse as sp
import torch
import numpy as np

def sparse_np_to_torch(A):
    Acoo = A.tocoo()
    values = Acoo.data
    indices = np.vstack((Acoo.row, Acoo.col))
    shape = Acoo.shape
    return torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(shape)).coalesce()

def convert_dict(np_dict, device="cpu"):
    torch_dict = {}
    for k, value in np_dict.items():
        if sp.issparse(value):
            torch_dict[k] = sparse_np_to_torch(value).to(device)
            if torch_dict[k].dtype == torch.int32:
                torch_dict[k] = torch_dict[k].long().to(device)
        elif isinstance(value, np.ndarray):
            torch_dict[k] = torch.from_numpy(value).to(device)
            if torch_dict[k].dtype == torch.int32:
                torch_dict[k] = torch_dict[k].squeeze().long().to(device)
        else:
            torch_dict[k] = value
    return torch_dict

def batchify_dict(torch_dict):
    for k, value in torch_dict.items():
        if isinstance(value, torch.Tensor):
            if torch_dict[k].dtype != torch.int64:
                torch_dict[k] = torch_dict[k].unsqueeze(0)
