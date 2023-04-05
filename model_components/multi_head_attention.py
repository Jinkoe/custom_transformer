import math
import torch
from torch import nn


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