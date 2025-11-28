import numpy as np
import torch.nn.functional as F
from torch.utils import data
import torch

class Dataset(data.Dataset):
    """Characterizes a dataset for PyTorch"""
    def __init__(self, data, k, num_classes, T):
        """Initialization"""
        self.k = k
        self.num_classes = num_classes
        self.T = T

        x = self.prepare_x(data)
        y = self.get_label(data)
        x, y = self.data_classification(x, y, self.T)
        y = y[:,self.k]
        self.length = len(x)

        x = torch.from_numpy(x)
        self.x = torch.unsqueeze(x, 1)
        self.y = torch.from_numpy(y)

    def __len__(self):
        """Denotes the total number of samples"""
        return self.length

    def __getitem__(self, index):
        """Generates samples of data"""
        return self.x[index], self.y[index]

    def prepare_x(self,data):
        df1 = data[:, :40]
        return np.array(df1)

    def get_label(self,data):
        lob = data[:, -3:]
        return lob

    def data_classification(self,X, Y, T):
        [N, D] = X.shape
        df = np.array(X)

        dY = np.array(Y)

        dataY = dY[T - 1:N]

        dataX = np.zeros((N - T + 1, T, D))
        for i in range(T, N + 1):
            dataX[i - T] = df[i - T:i, :]

        return dataX, dataY

    def torch_data(self,x, y):
        x = torch.from_numpy(x)
        x = torch.unsqueeze(x, 1)
        y = torch.from_numpy(y)
        y = F.one_hot(y, num_classes=3)
        return x, y

    def merge(self, other):
        """
        Merge this Dataset with another Dataset of the same structure.
        Returns a NEW merged Dataset.
        """
        # Basic shape + structural checks
        assert self.T == other.T, "T mismatch"
        assert self.num_classes == other.num_classes, "num_classes mismatch"
        assert self.x.shape[2:] == other.x.shape[2:], "Feature dimension mismatch"

        # Concatenate samples
        new_x = torch.cat([self.x, other.x], dim=0)
        new_y = torch.cat([self.y, other.y], dim=0)

        # Create an empty placeholder instance
        merged = object.__new__(Dataset)

        # Copy attributes
        merged.k = self.k
        merged.num_classes = self.num_classes
        merged.T = self.T
        merged.length = new_x.shape[0]

        # Assign concatenated tensors
        merged.x = new_x
        merged.y = new_y

        return merged
