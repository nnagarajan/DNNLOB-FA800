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