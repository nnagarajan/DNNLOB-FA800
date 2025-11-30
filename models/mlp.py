import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim=(40, 100), output_dim=3, hidden_layer_dim=128, p_dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        flat_dims = self.input_dim[0] * self.input_dim[1]
        self.linear1 = nn.Linear(flat_dims, hidden_layer_dim)
        self.leakyReLU = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=p_dropout)
        self.linear2 = nn.Linear(hidden_layer_dim, self.output_dim)

    def forward(self, x):
        # [batch_size x 40 x observation_length]
        x = x.view(x.size(0), -1).float()
        out = self.linear1(x)
        #out = self.leakyReLU(out)
        #out = self.dropout(out)
        out = self.linear2(out)
        out = F.softmax(out, dim=1)
        return out


