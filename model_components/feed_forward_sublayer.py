import torch
from torch import nn
from torch.nn import functional


class FeedForwardSublayer(nn.Module):
    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()

        self.linear_1 = nn.Linear(d_model, d_hidden)
        self.linear_2 = nn.Linear(d_hidden, d_model)

    def forward(self, x: torch.Tensor):
        linear_1_output = self.linear_1(x)
        linear_1_output_relu = functional.gelu(linear_1_output)    # In the BERT paper they use GELU instead of ReLU
        linear_2_output = self.linear_2(linear_1_output_relu)
        return linear_2_output
