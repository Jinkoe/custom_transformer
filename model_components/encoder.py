import torch
from torch import nn
from model_components.encoder_layer import EncoderLayer


class Encoder(nn.Module):
    def __init__(self, n_encoder_layers, d_model: int, d_ff_hidden: int, h: int):
        super().__init__()

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model=d_model, d_ff_hidden=d_ff_hidden, h=h)
                                                   for _ in range(n_encoder_layers)])

    def forward(self, x: torch.Tensor):
        encoder_layer_input = x
        encoder_layer_output = None
        for encoder_layer in self.encoder_layers:
            encoder_layer_output = encoder_layer(encoder_layer_input)
            encoder_layer_input = encoder_layer_output

        return encoder_layer_output