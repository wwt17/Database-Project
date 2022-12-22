import torch.nn as nn


class MLP(nn.Module):
    """A 2-layer MLP.
    """
    def __init__(self, input_dim, output_dim, hidden_dim, act='relu', dropout=0.0):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        if act == 'relu':
            self.activation = nn.ReLU()
        elif act == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise NotImplementedError
        self.dropout = nn.Dropout(p=dropout)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x
