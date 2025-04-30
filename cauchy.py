import torch
import torch.nn as nn

class CauchyActivation(nn.Module):
    def __init__(self):
        super(CauchyActivation, self).__init__()
        self.lambda1 = nn.Parameter(torch.tensor(1.0))
        self.lambda2 = nn.Parameter(torch.tensor(1.0))
        self.d = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        x2_d2 = x ** 2 + self.d ** 2
        return self.lambda1 * x / x2_d2 + self.lambda2 / x2_d2


