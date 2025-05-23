import torch
import torch.nn as nn

class DeepSTARR(nn.Module):
  """Based on https://github.com/bernardo-de-almeida/DeepSTARR"""
  def __init__(
    self,
    seq_len=271,
    kernel_sizes=[7, 3, 5, 3],
    n_channels=[4, 256, 60, 60, 120],
    conv_padding='same',
    conv_dropout=0.3,
    fc_dim=128,
    fc_dropout=0.2,
  ):
    super().__init__()

    self.conv_layers = nn.ModuleList([
        nn.Sequential(
            nn.Conv1d(
              in_channels=n_channels[i-1],
              out_channels=n_channels[i],
              kernel_size=kernel_sizes[i-1],
              padding='same',
            ),
            nn.BatchNorm1d(n_channels[i]),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(conv_dropout)
        )
        for i in range(1, len(n_channels))
    ])

    conv_out_dim = n_channels[-1] * (seq_len // (2 ** len(kernel_sizes)))
    self.reg_head = nn.Sequential(
            nn.Linear(conv_out_dim, fc_dim),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(fc_dim, 1),
        )

  def forward(self, x):
    """
    x: Tensor of shape (batch_size, seq_len, 4)
    Returns:
      Tuple of Tensors: (regression output, classification output)
      Each Tensor has shape (batch_size, 1)
    """
    x = x.permute(0, 2, 1)
    for conv_layer in self.conv_layers:
      x = conv_layer(x)
    x = x.view(x.size(0), -1)
    return self.reg_head(x)
