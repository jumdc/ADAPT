"""Transformer module implementing masked cross attention between time series and video."""

# pylint:disable=C0303

import copy
import einops
import torch
from torch import nn


def clone(module, number_of_copies):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(number_of_copies)])


def default(val, default_val):
    """default value for the input."""
    return default_val if val is None else val


class Attention(nn.Module):
    """Multihead attention module."""

    def __init__(self, d_model, n_heads, dropout: int = 0.0, *args, **kwargs) -> None:
        """
        Initialize the attention module.

        Parameters
        ----------
        dim : int.
            Dimension of the input.
        num_heads : int.
            Number of heads in the multihead attention.
        qkv_bias : bool, default = False.
            Whether to use bias in the query, key and value projection.
        attn_drop : int, optional
            Dropout rate fro the attention operation. The default is 0.0.
        proj_drop : int, optional
            Dropout rate for the projection operation. The default is 0.0.
        """
        super().__init__()

        self.num_heads = n_heads
        self.d_model = d_model
        self.head_dim = self.d_model // self.num_heads
        self.scale = None or self.head_dim**-0.5
        self.qkv = clone(nn.Linear(self.d_model, self.d_model, bias=False), 3)
        self.attn_drop = nn.Dropout(dropout)
        self.ffn_drop = nn.Dropout(dropout)
        self.concat_head = nn.Linear(self.d_model, self.d_model)
        self.projection = nn.Linear(self.d_model, self.d_model)
        self.ffn = nn.Sequential(
            *[
                nn.Linear(self.d_model, self.d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.d_model * 4, self.d_model),
                nn.Dropout(dropout),
            ]
        )
        self.post_attn_ln = nn.LayerNorm(self.d_model, eps=1e-5)
        self.post_ffn_ln = nn.LayerNorm(self.d_model, eps=1e-5)
        self._initialize_params()

    def forward(self, src, context=None, mask=None, modality=None, softmax_dim=-1):
        """Forward pass of the attention module."""
        query, key, value = (
            self.qkv[0](default(context, default_val=src)),
            self.qkv[1](src),
            self.qkv[2](src),
        )
        q, k, v = map(
            lambda t: einops.rearrange(t, "b n (h d) -> b h n d", h=self.num_heads),
            (query, key, value),
        )
        qk = torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        # mask
        if mask is not None:
            mask_attn = self._create_mask(
                src, mask, context=context, num_heads=self.num_heads, modality=modality
            )
            max_value = -torch.max(qk)
            qk_softmax = (
                torch.softmax(qk + (1 - mask_attn) * max_value, dim=softmax_dim)
                * mask_attn
            )  # change dim
        else:
            qk_softmax = torch.softmax(qk, dim=softmax_dim)
        # attn map
        out = torch.einsum("b h i j, b h j d -> b h i d", qk_softmax, v)
        out = einops.rearrange(out, "b h n d -> b n (h d)", h=self.num_heads)
        out = self.concat_head(out)
        if hasattr(self, "resweight"):
            ## Attn: residual connection
            out = self.resweight * out
            out = src + self.attn_drop(out)
            ## FFN: residual connection
            out = self.ffn_drop(self.resweight * self.ffn(out)) + out
        elif hasattr(self, "ls1"):
            out = src + self.attn_drop(self.ls1(out))
            out = out + self.ffn_drop(self.ls2(self.ffn(out)))
        else:
            ## Attn: Add & norm
            out = self.post_attn_ln(self.attn_drop(out) + src)
            ## FFN: Add & norm
            out = self.post_ffn_ln(self.ffn_drop(self.ffn(out)) + out)
        return out

    @staticmethod
    def _create_mask(x, mask=None, num_heads=1, context=None, modality=None):
        """create the mask for the missing modalities."""
        BATCH = x.shape[0]
        ROW = context.shape[1] if context is not None else x.shape[1]
        COL = x.shape[1]
        masked = torch.ones((BATCH, ROW, COL), device=x.device)
        for i in range(ROW):
            # if the modality is missing : all the rows are masked.
            if modality is not None:
                masked[mask[:, modality], :, :] = 0
            masked[mask[:, i], i, :] = 0  #
            if COL > 1:
                masked[mask[:, i], :, i] = (
                    0  # put one for the column of the missing modality
                )
        masked = einops.repeat(masked, "b n d -> b h n d", h=num_heads)
        return masked

    def _initialize_params(self):
        """Initialize parameters.

        Same initilization scheme than in CLIP.
        """
        std = self.d_model**-0.5
        nn.init.normal_(self.qkv[0].weight, std=std)
        nn.init.normal_(self.qkv[1].weight, std=std)
        nn.init.normal_(self.qkv[2].weight, std=std)
