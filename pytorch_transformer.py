import torch
from torch import nn


class PytorchTransformerEncoderModel(nn.Module):
    def __init__(self, device: str, vocab_size: int, n_encoder_layers, d_model: int, d_ff_hidden: int, h: int, max_input_size: int):
        super().__init__()
        self.d_model = d_model

        self.tok_embedder = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.pos_embedder = nn.Embedding(num_embeddings=max_input_size, embedding_dim=d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model,
                                                   nhead=h,
                                                   dim_feedforward=d_ff_hidden,
                                                   activation=nn.functional.gelu)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=n_encoder_layers)

        self.linear_final = nn.Linear(d_model, vocab_size)
        self.softmax_final = nn.LogSoftmax(dim=-1)    # Using LogSoftmax to get negative values for NLLLoss

        # self.apply(self._init_weights)      # weights initialization ?

        self.device = device
        self.to(device)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.1)
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.normal_(mean=0.0, std=0.1)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.1)

    def forward(self, x: torch.Tensor):
        tok_embeddings = self.tok_embedder(x)
        pos_embeddings = self.pos_embedder(torch.IntTensor(range(x.shape[1])).to(self.device))
        input_embeddings = tok_embeddings + pos_embeddings

        encoder_output = self.encoder(input_embeddings)

        final_linear_output = self.linear_final(encoder_output)
        final_softmax_output = self.softmax_final(final_linear_output)

        return final_softmax_output, encoder_output
