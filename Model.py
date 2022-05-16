from torch import nn
import torch


class CPNet(nn.Module):
    """mini cnn structure
    input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1.09e-03)
        self.loss_fn = torch.nn.SmoothL1Loss()
        self.layers.to(device='cuda')

    def forward(self, x):
        return self.layers(x)
