import torch
import torch.nn as nn
import torch.nn.functional as F

class PLM_nn(nn.Module):
    #class PocketTypeEstimator(nn.Module):
    def __init__(self, input_dim=2560,layer_width=100,dropout=0.3):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=input_dim, out_features=layer_width)
        self.dropout1 = nn.Dropout(dropout)

        self.layer_2 = nn.Linear(in_features=layer_width, out_features=layer_width)
        self.dropout2 = nn.Dropout(dropout)

        self.layer_3 = nn.Linear(in_features=layer_width, out_features=1)

        self.relu = nn.ReLU()

    def forward(self, x):
      # Intersperse the ReLU activation function between layers
       return self.layer_3(self.dropout1(self.relu(self.layer_2(self.dropout1(self.relu(self.layer_1(x)))))))
