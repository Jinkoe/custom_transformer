import torch
from torch import nn
from model_components.multi_head_attention import MultiHeadAttentionCustom
from model_components.feed_forward_sublayer import FeedForwardSublayer


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
