import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, n_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.embedding_dim = embedding_dim
        self.head_dim = embedding_dim // n_heads
        assert self.head_dim * n_heads == embedding_dim, "embedding_dim must be divisible by n_heads"

        self.proj_q = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.proj_k = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.proj_v = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.proj_out = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.norm_factor = 1 / math.sqrt(self.head_dim)

        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_uniform_(self.proj_q.weight)
        nn.init.xavier_uniform_(self.proj_k.weight)
        nn.init.xavier_uniform_(self.proj_v.weight)
        nn.init.xavier_uniform_(self.proj_out.weight)

    def forward(self, q, k=None, v=None, attn_mask=None):
        if k is None:
            k = q
        if v is None:
            v = q

        # q, k, v are [batch_size, num_nodes, embedding_dim]
        batch_size, num_nodes, embedding_dim = q.size()

        # Compute queries, keys, and values
        Q = self.proj_q(q)  # [batch_size, num_nodes, embedding_dim]
        K = self.proj_k(k)  # [batch_size, num_nodes, embedding_dim]
        V = self.proj_v(v)  # [batch_size, num_nodes, embedding_dim]

        # Split into heads
        Q = Q.view(batch_size, num_nodes, self.n_heads, self.head_dim).transpose(1, 2)  # [batch_size, n_heads, num_nodes, head_dim]
        K = K.view(batch_size, num_nodes, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, num_nodes, self.n_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.norm_factor  # [batch_size, n_heads, num_nodes, num_nodes]

        if attn_mask is not None:
            # Ensure attn_mask has shape [batch_size, 1, num_nodes, num_nodes]
            attn_mask = attn_mask.unsqueeze(1)  # [batch_size, 1, num_nodes, num_nodes]
            attn_mask = attn_mask.to(dtype=torch.bool, device=attn_scores.device)
            # Mask positions where attn_mask == 1 (no connection)
            attn_scores = attn_scores.masked_fill(attn_mask, float('-inf'))

        # Compute attention weights
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [batch_size, n_heads, num_nodes, num_nodes]

        # Apply attention weights to V
        context = torch.matmul(attn_weights, V)  # [batch_size, n_heads, num_nodes, head_dim]

        # Concatenate heads: [batch_size, n_heads, num_nodes, head_dim] -> [batch_size, num_nodes, embedding_dim]
        context = context.transpose(1, 2).contiguous().view(batch_size, num_nodes, self.embedding_dim)

        # Final linear projection
        out = self.proj_out(context)  # [batch_size, num_nodes, embedding_dim]

        return out, attn_weights
