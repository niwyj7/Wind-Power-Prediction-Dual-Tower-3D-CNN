

import torch
from torch.utils.data import Dataset

class DualGridDataset3D(Dataset):
    """
    Encapsulates both south and north grid features simultaneously 
    to ensure strict index alignment across batches.
    """
    def __init__(self, X_south, X_north, y=None):
        assert len(X_south) == len(X_north), "South and North feature arrays must have the same length!"
        self.X_south = torch.FloatTensor(X_south)
        self.X_north = torch.FloatTensor(X_north)
        self.y = torch.FloatTensor(y) if y is not None and len(y) > 0 else None
        
    def __len__(self):
        return len(self.X_south)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X_south[idx], self.X_north[idx], self.y[idx]
        else:
            return self.X_south[idx], self.X_north[idx]
