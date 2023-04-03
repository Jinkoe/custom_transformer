import torch
from torch import nn
from torch.nn import functional
import math


class CustomTransformerEncoderModel(nn.Module):
    def __init__(self, device: str, vocab_size: int, n_encoder_layers, d_model: int, d_ff_hidden: int, h: int, max_input_size: int):
        super().__init__()
        self.d_model = d_model

        self.tok_embedder = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.pos_embedder = nn.Embedding(num_embeddings=max_input_size, embedding_dim=d_model)

        self.encoder = Encoder(
            n_encoder_layers=n_encoder_layers,
            d_model=d_model,
            d_ff_hidden=d_ff_hidden,
            h=h
        )

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


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, d_ff_hidden: int, h: int):
        super().__init__()

        self.multi_head_attention = MultiHeadAttentionCustom(d_k=d_model, d_v=d_model, d_model=d_model, h=h)
        self.attention_layer_norm = nn.LayerNorm(d_model)
        self.feed_forward = FeedForwardSublayer(d_model=d_model, d_hidden=d_ff_hidden)
        self.feed_forward_layer_norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x: torch.Tensor):
        attention_output = self.multi_head_attention(x, x, x)
        attention_output = self.dropout(attention_output)
        attention_add_and_norm_output = self.attention_layer_norm(attention_output + x)
        feed_forward_output = self.feed_forward(attention_add_and_norm_output)
        feed_forward_output = self.dropout(feed_forward_output)
        feed_forward_add_and_norm_output = self.feed_forward_layer_norm(feed_forward_output + attention_add_and_norm_output)

        return feed_forward_add_and_norm_output


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


class MultiHeadAttentionCustom(nn.Module):
    def __init__(self, d_k: int, d_v: int, d_model: int, h: int):
        super().__init__()

        self.h = h
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model

        assert self.d_model % self.h == 0
        self.d_head = int(d_model / h)

        self.linear_q = nn.Linear(d_k, self.d_model)
        self.linear_k = nn.Linear(d_k, self.d_model)
        self.linear_v = nn.Linear(d_v, self.d_model)
        self.linear_final = nn.Linear(self.d_model, self.d_model)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        batch_size = q.shape[0]

        # Linearly projecting q, k and v
        linear_q_output = self.linear_q(q)
        linear_k_output = self.linear_k(k)
        linear_v_output = self.linear_v(v)

        # Computing the attention
        attention_output = self.scaled_dot_product_attention(q=linear_q_output,
                                                             k=linear_k_output,
                                                             v=linear_v_output)

        # Reshaping from (batch_size, h, d_input, d_head) to (batch_size, d_input, h, d_head)
        # and then concatenating heads results to (batch_size, d_input, h * d_head)
        # which is equivalent to (batch_size, d_input, d_model)
        attention_output = attention_output.transpose(1, 2)
        attention_output_concat = attention_output.reshape(batch_size, -1, self.d_model)

        # Linearly projecting the concatenated attentions results
        linear_final_output = self.linear_final(attention_output_concat)

        return linear_final_output

    def scaled_dot_product_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        batch_size = q.shape[0]

        # Reshaping from (batch_size, input_size, h * d_head)
        # to (batch_size, input_size, h, d_head)
        # and then to (batch_size, h, input_size, d_head)
        q = q.view(batch_size, -1, self.h, self.d_head).transpose(1, 2)
        k = k.view(batch_size, -1, self.h, self.d_head).transpose(1, 2)
        v = v.view(batch_size, -1, self.h, self.d_head).transpose(1, 2)

        # Computing the dot products Q*K.T => result is of shape (batch_size, h, input_size, input_size)
        q_kt = torch.matmul(q, k.transpose(-2, -1))

        # Scaling the result
        q_kt_scaled = q_kt / math.sqrt(self.d_k)

        # Softmax
        q_kt_scaled_softmax = torch.softmax(q_kt_scaled, dim=-1)

        # Computing the dot products softmax(...) * V => result is of shape (batch_size, h, d_head)
        q_kt_scaled_softmax_v = torch.matmul(q_kt_scaled_softmax, v)

        return q_kt_scaled_softmax_v
