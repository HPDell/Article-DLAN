"""Reviewed local GeoAggregator implementation, extracted without changes.

Source: P18_GeoAggregator.ipynb, code cells 4--7.
Source code SHA256: ba53a746d1504c9b8ddef4d0f66b7f7eead2887065fa3b243ecfb868b053d901
Retains the P18 ParameterList registration and rotary-width corrections.
This is the repository implementation, not a new official-source reproduction.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _calc_attention_bias(geo_proxy, attn_bias_factor):
    """
    :param geo_proxy: [bs, sl]
    :param attn_bias_factor: [4] (n-head = nh = 4)

    :return: [bs, nh, sl]
    """
    batch_size, seq_len = geo_proxy.shape
    geo_proxy **= 2
    max_, _ = geo_proxy.max(dim=1, keepdim=True)
    min_, _ = geo_proxy.min(dim=1, keepdim=True)
    geo_proxy = (geo_proxy - min_) / (max_ - min_ + 1e-8)
    if isinstance(attn_bias_factor, torch.Tensor):
        geo_proxy = torch.mul(
            geo_proxy.unsqueeze(1).repeat(1, 4, 1),   # [bs, sl] -> [bs, nh (4), sl]
            attn_bias_factor.unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, seq_len)   # -> [bs, nh, sl]
        ) # -> [bs, nh, sl]
    else:
        geo_proxy *= attn_bias_factor

    return geo_proxy


class CartesianPerceiver(nn.Module):
    def __init__(self,
                 d_model,
                 n_attn_layer,
                 n_hidden_token=4,
                 attn_dropout=0.1,
                 attn_bias_factor=None):
        """
        A Perceiver-like architecture (Encoder-Processor-Decoder)
        Jaegle, A., et al. (2021). Perceiver io: A general architecture for structured inputs & outputs.

        :param d_model:
            Embedding dimension throughout the transformer model.
        :param n_attn_layer:
            Number of total attention layers.
        :param n_hidden_token:
            Number of hidden token, or inducing points.
        :param attn_dropout:
            Dropout probability for the attention mechanism.
        :param attn_bias_factor:
            Scaler controlling the amount of attention bias added to the attention
            (see GeoAggregator paper).
        """
        super(CartesianPerceiver, self).__init__()
        # ----------------------------------------------------------------
        self.d_model = d_model
        self.n_attn_layer = n_attn_layer
        self.n_hidden_token = n_hidden_token
        self.attn_dropout = attn_dropout
        # ----------------------------------------------------------------
        if attn_bias_factor is not None:
            self.attn_bias_factor = attn_bias_factor
        else:
            self.attn_bias_factor = nn.Parameter(torch.zeros(4))

        self.attn_layers = nn.ModuleList([
            MaskedCartesianAttention(
                d_model=d_model,
                n_a_head=2,
                n_b_head=2,
                attn_dropout=self.attn_dropout,
                attn_bias_factor=self.attn_bias_factor
            )
            for _ in range(self.n_attn_layer)
        ])
        self.a_ffds = nn.ModuleList([
            nn.Linear(self.d_model // 2, self.d_model // 2)
            for _ in range(self.n_attn_layer)
        ])
        self.b_ffds = nn.ModuleList([
            nn.Linear(self.d_model // 2, self.d_model // 2)
            for _ in range(self.n_attn_layer)
        ])

        if self.n_attn_layer > 1:
            self.latent_queries_a = nn.ParameterList([
                nn.Parameter(torch.randn(self.n_hidden_token, self.d_model // 2))
                for _ in range(self.n_attn_layer - 1)
            ])
            self.latent_queries_b = nn.ParameterList([
                nn.Parameter(torch.randn(self.n_hidden_token, self.d_model // 2))
                for _ in range(self.n_attn_layer - 1)
            ])

        self.softmax = nn.Softmax(dim=3)

        self.a_attn_layer_norms = nn.ModuleList([
            nn.LayerNorm(self.d_model // 2, eps=1e-6)
            for _ in range(self.n_attn_layer)
        ])  # layer_norm the last dimension
        self.b_attn_layer_norms = nn.ModuleList([
            nn.LayerNorm(self.d_model // 2, eps=1e-6)
            for _ in range(self.n_attn_layer)
        ])

        self.a_ffd_layer_norms = nn.ModuleList([
            nn.LayerNorm(self.d_model // 2, eps=1e-6)
            for _ in range(self.n_attn_layer)
        ])  # layer_norm the last dimension
        self.b_ffd_layer_norms = nn.ModuleList([
            nn.LayerNorm(self.d_model // 2, eps=1e-6)
            for _ in range(self.n_attn_layer)
        ])

    def forward(self,
                a_embed,
                b_embed,
                q_a_embed,
                q_b_embed,
                ctx_spa_embed,
                q_spa_embed,
                mask,
                geo_proxy):
        """
        :param ctx_spa_embed: [bs, sl-1, dm//8, dm//8]
        :param q_spa_embed: [bs, 1, dm//8, dm//8]
        :param a_embed: [bs, sl-1, dm//2]
        :param b_embed: [bs, sl-1, dm//2]
        :param q_a_embed: [bs, 1, dm//2]
        :param q_b_embed: [bs, 1, dm//2]
        :param mask: [bs, sl-1], masked: 1, not masked: 0.
        :param geo_proxy: [bs, sl-1]
        """
        batch_size = ctx_spa_embed.shape[0]
        attn_weights = torch.FloatTensor([])
        gaussian_bias = _calc_attention_bias(geo_proxy, self.attn_bias_factor)  # -> [bs, nh, sl]

        embedding = torch.stack((a_embed, b_embed), dim=0)  # -> [2, bs, sl, dm//2]
        query = torch.stack((
            q_a_embed,
            q_b_embed
        ), dim=0)  # -> [2, bs, 1, dm//2]

        # Perceiver
        for idx, (atn,
                  a_ffd, b_ffd,
                  a_atn_ln, b_atn_ln,
                  a_ffd_ln, b_ffd_ln) in enumerate(zip(self.attn_layers[:-1],
                                                       self.a_ffds[:-1],
                                                       self.b_ffds[:-1],
                                                       self.a_attn_layer_norms[:-1],
                                                       self.b_attn_layer_norms[:-1],
                                                       self.a_ffd_layer_norms[:-1],
                                                       self.b_ffd_layer_norms[:-1])):
            attn_output, attn_weight = atn(
                q=torch.stack((
                    self.latent_queries_a[idx].unsqueeze(0).repeat(batch_size, 1, 1),
                    self.latent_queries_b[idx].unsqueeze(0).repeat(batch_size, 1, 1)
                ), dim=0).clone(),
                k=embedding.clone(),
                v=embedding.clone(),
                q_spa_embed=q_spa_embed,
                kv_spa_embed=ctx_spa_embed,
                mask=mask,
                attention_bias=gaussian_bias
            )
            attn_output = torch.stack((
                a_atn_ln(
                    self.latent_queries_a[idx].unsqueeze(0).repeat(batch_size, 1, 1) \
                    + attn_output[0]
                ),
                b_atn_ln(
                    self.latent_queries_b[idx].unsqueeze(0).repeat(batch_size, 1, 1) \
                    + attn_output[1]
                )
            ), dim=0)
            attn_output = torch.stack((
                a_ffd_ln(attn_output[0] + a_ffd(attn_output[0])),
                b_ffd_ln(attn_output[1] + b_ffd(attn_output[1]))
            ), dim=0)
            embedding = attn_output

        # Stand-alone
        if self.n_attn_layer != 1:
            mask = torch.zeros_like(mask, device=mask.device)
            gaussian_bias = torch.zeros_like(gaussian_bias, device=gaussian_bias.device)

        attn_output, attn_weight = self.attn_layers[-1](
            q=query.clone(),
            k=embedding.clone(),
            v=embedding.clone(),
            q_spa_embed=q_spa_embed,
            kv_spa_embed=ctx_spa_embed,
            mask=mask,
            attention_bias=gaussian_bias
        )  # -> [2, bs, ql(1), dm//2]
        # attn_weights = torch.cat((attn_weights, attn_weight))
        # residual connection valid when layer goes deep (YouTube stanford perceiver video)
        attn_output = torch.stack((
            self.a_attn_layer_norms[-1](query[0] + attn_output[0]),
            self.b_attn_layer_norms[-1](query[1] + attn_output[1])
        ), dim=0)  # -> [2, bs, ql(1), dm//2]
        attn_output = torch.stack((
            self.a_ffd_layer_norms[-1](attn_output[0] + self.a_ffds[-1](attn_output[0])),
            self.b_ffd_layer_norms[-1](attn_output[1] + self.b_ffds[-1](attn_output[1]))
        ), dim=0)  # -> [2, bs, ql(1), dm//2]

        return attn_output, attn_weights


class MaskedCartesianAttention(nn.Module):
    def __init__(self,
                 d_model,
                 n_a_head,
                 n_b_head,
                 attn_dropout=0.1,
                 attn_bias_factor=None):
        super(MaskedCartesianAttention, self).__init__()

        self.d_model = d_model
        self.n_a_head = n_a_head
        self.n_b_head = n_b_head
        self.n_head = self.n_a_head * self.n_b_head
        self.attn_bias_factor = attn_bias_factor

        self.w_q_a = nn.Linear(self.d_model // 2, self.d_model // 4, bias=False)
        self.w_q_b = nn.Linear(self.d_model // 2, self.d_model // 4, bias=False)
        self.w_k_a = nn.Linear(self.d_model // 2, self.d_model // 4, bias=False)
        self.w_k_b = nn.Linear(self.d_model // 2, self.d_model // 4, bias=False)
        self.w_v_a = nn.Linear(self.d_model // 2, self.d_model // 4, bias=False)
        self.w_v_b = nn.Linear(self.d_model // 2, self.d_model // 4, bias=False)
        self.w_o_a = nn.Linear(self.d_model // 2, self.d_model // 2, bias=False)
        self.w_o_b = nn.Linear(self.d_model // 2, self.d_model // 2, bias=False)

        self.softmax = nn.Softmax(dim=-1)
        self.drop_out = nn.Dropout(attn_dropout)

    def forward(self,
                q,
                k,
                v,
                q_spa_embed,
                kv_spa_embed,
                mask,
                attention_bias=None):
        """
        :param q: [2, bs, ql, dm//2]
        :param k: [2, bs, sl, dm//2]
        :param v: [2, bs, sl, dm//2]
        :param q_spa_embed: [bs, 1, dk, dk]
        :param kv_spa_embed: [bs, sl, dk, dk]
        :param mask: [bs, sl] element to mask => 1, valid => 0.
        :param attention_bias: [bs, nh, sl]

        Note that: n-head = nh = 4
        """
        _, batch_size, seq_len, _ = k.shape
        _, _, query_len, _ = q.shape

        # Project, Cross concat & Rotate
        # n-head = nh = 4
        # [bs, l, dm//2] -> [bs, l, dm//4]
        q_a = self.w_q_a(q[0])
        q_b = self.w_q_b(q[1])
        k_a = self.w_k_a(k[0])
        k_b = self.w_k_b(k[1])
        v_a = self.w_v_a(v[0])
        v_b = self.w_v_b(v[1])

        q_a = q_a.view(batch_size, query_len, 2, self.d_model // 8)
        q_b = q_b.view(batch_size, query_len, 2, self.d_model // 8)
        k_a = k_a.view(batch_size, seq_len, 2, self.d_model // 8)
        k_b = k_b.view(batch_size, seq_len, 2, self.d_model // 8)
        v_a = v_a.view(batch_size, seq_len, 2, self.d_model // 8)
        v_b = v_b.view(batch_size, seq_len, 2, self.d_model // 8)

        # Cartesian product
        # [bs, l, 2, dm//8] -> [bs, l, 4, dm//4] -> [bs, 4, l, dm//4]
        q = torch.cat((q_a.repeat(1, 1, 2, 1), q_b.repeat_interleave(2, dim=-2)), dim=-1).transpose(1, 2)
        k = torch.cat((k_a.repeat(1, 1, 2, 1), k_b.repeat_interleave(2, dim=-2)), dim=-1).transpose(1, 2)
        v = torch.cat((v_a.repeat(1, 1, 2, 1), v_b.repeat_interleave(2, dim=-2)), dim=-1).transpose(1, 2)

        # Add spatial embedding at FIRST layer (ll = ql = latent_len)
        if seq_len == kv_spa_embed.shape[1]:
            # Adapt spatial embeddings to match query and key dimensions
            # [bs, sl, dk, dk] -> [bs, 1, sl, dk, dk] -> [bs, 4, sl, dk//2, dk//2]
            q_spa_embed_reshaped = q_spa_embed.unsqueeze(1).repeat(1, 4, query_len, 1, 1)
            kv_spa_embed_reshaped = kv_spa_embed.unsqueeze(1).repeat(1, 4, 1, 1, 1)
            
            # Ensure dimensions match for matrix multiplication
            q_dim = q.size(-1)
            spa_dim = q_spa_embed_reshaped.size(-1)
            
            if spa_dim != q_dim:
                # Reshape spatial embeddings or pad as needed
                if spa_dim > q_dim:
                    # Use a subset of the spatial dimensions
                    q_spa_embed_reshaped = q_spa_embed_reshaped[..., :q_dim, :q_dim]
                    kv_spa_embed_reshaped = kv_spa_embed_reshaped[..., :q_dim, :q_dim]
                else:
                    # Pad spatial dimensions (this should not happen with current setup)
                    pad_size = q_dim - spa_dim
                    q_spa_embed_reshaped = F.pad(q_spa_embed_reshaped, (0, pad_size, 0, pad_size))
                    kv_spa_embed_reshaped = F.pad(kv_spa_embed_reshaped, (0, pad_size, 0, pad_size))
            
            # Apply spatial embeddings to queries and keys
            # [bs, 4, ll, dm//4, dm//4] * [bs, 4, ll, dm//4, 1] -> [bs, 4, ll, dm//4]
            q = q_spa_embed_reshaped.matmul(q.unsqueeze(-1)).squeeze(-1)
            # [bs, 4, sl, dm//4, dm//4] * [bs, 4, sl, dm//4, 1]
            k = kv_spa_embed_reshaped.matmul(k.unsqueeze(-1)).squeeze(-1)
        # Add spatial embedding at PROCESSOR layers:
        elif seq_len == query_len:
            # Adapt spatial embeddings to match query and key dimensions
            q_spa_embed_reshaped = q_spa_embed.unsqueeze(1).repeat(1, 4, query_len, 1, 1)
            
            # Ensure dimensions match
            q_dim = q.size(-1)
            spa_dim = q_spa_embed_reshaped.size(-1)
            
            if spa_dim != q_dim:
                if spa_dim > q_dim:
                    q_spa_embed_reshaped = q_spa_embed_reshaped[..., :q_dim, :q_dim]
                else:
                    pad_size = q_dim - spa_dim
                    q_spa_embed_reshaped = F.pad(q_spa_embed_reshaped, (0, pad_size, 0, pad_size))
            
            # Apply spatial embeddings
            q = q_spa_embed_reshaped.matmul(q.unsqueeze(-1)).squeeze(-1)
            k = q_spa_embed_reshaped.matmul(k.unsqueeze(-1)).squeeze(-1)
        # Add spatial embedding at DECODER layer (ql = 1)
        elif query_len == 1:
            # Adapt spatial embeddings to match query and key dimensions
            q_spa_embed_reshaped = q_spa_embed.unsqueeze(1).repeat(1, 4, query_len, 1, 1)
            
            # Ensure dimensions match
            q_dim = q.size(-1)
            spa_dim = q_spa_embed_reshaped.size(-1)
            
            if spa_dim != q_dim:
                if spa_dim > q_dim:
                    q_spa_embed_reshaped = q_spa_embed_reshaped[..., :q_dim, :q_dim]
                else:
                    pad_size = q_dim - spa_dim
                    q_spa_embed_reshaped = F.pad(q_spa_embed_reshaped, (0, pad_size, 0, pad_size))
            
            # Apply spatial embeddings
            q = q_spa_embed_reshaped.matmul(q.unsqueeze(-1)).squeeze(-1)
            k = q_spa_embed_reshaped.repeat(1, 1, seq_len, 1, 1).matmul(k.unsqueeze(-1)).squeeze(-1)

        # [bs, nh, ql, dm//4] * [bs, nh, dm//4, sl] -> [bs, nh, ql, sl]
        attn_score = q.matmul(k.transpose(-1, -2)) / math.sqrt(self.d_model // 4)  # dk

        if attention_bias.shape[-1] == seq_len:
            if attention_bias is None:
                raise NotImplementedError()
            else:
                # [bs, nh, sl] -> [bs, nh, ql, sl]
                attention_bias = attention_bias.unsqueeze(2).repeat(1, 1, query_len, 1)
        else:
            attention_bias = torch.zeros_like(attn_score, device=attn_score.device)

        if mask.shape[1] == seq_len:
            # [bs, nh, ql, sl] + [bs, nh, ql, sl] -> [bs, nh, ql, sl]
            attn_score += mask.unsqueeze(1).unsqueeze(1).repeat(1, self.n_head, query_len, 1) * -1e9

        attn_score = self.softmax(attn_score - attention_bias)
        attn_score = self.drop_out(attn_score)

        # [bs, nh, ql, l] * [bs, nh, l, dm//4] -> [bs, nh, ql, dm//4]
        v = attn_score.matmul(v)
        # [bs, nh, ql, dm//4] -> [bs, nh, ql, 2, dm//8]
        v = v.view(batch_size, self.n_head, query_len, 2, self.d_model // 8)

        v_a, v_b = v[:, :, :, 0], v[:, :, :, 1]
        v_a = v_a.transpose(1, 2).contiguous().view(batch_size, query_len, -1)
        v_b = v_b.transpose(1, 2).contiguous().view(batch_size, query_len, -1)

        v_a = self.w_o_a(v_a)  # -> [bs, ql, dm//2]
        v_b = self.w_o_b(v_b)  # -> [bs, ql, dm//2]
        v = torch.stack((v_a, v_b), dim=0)  # -> [2, bs, ql, dm//2]

        return v, attn_score


class VanillaPerceiver(nn.Module):
    def __init__(self,
                 d_model,
                 n_head,
                 n_attn_layer,
                 n_hidden_token=4,
                 attn_dropout=0.1,
                 attn_bias_factor=1):
        super(VanillaPerceiver, self).__init__()
        # ----------------------------------------------------------------
        self.d_model = d_model
        self.n_head = n_head
        self.n_attn_layer = n_attn_layer
        self.n_hidden_token = n_hidden_token
        self.attn_dropout = attn_dropout
        self.attn_bias_factor = attn_bias_factor
        # ----------------------------------------------------------------
        if attn_bias_factor:
            self.attn_bias_factor = attn_bias_factor
        else:
            self.attn_bias_factor = nn.Parameter(torch.zeros(1))

        self.attn_layers = nn.ModuleList([
            MaskedVanillaAttention(
                d_model=self.d_model,
                n_head=self.n_head,
                attn_dropout=self.attn_dropout,
                attn_bias_factor=self.attn_bias_factor
            )

            for _ in range(self.n_attn_layer)
        ])

        self.ffds = nn.ModuleList([
            nn.Linear(self.d_model, self.d_model)
            for _ in range(self.n_attn_layer)
        ])

        if self.n_attn_layer > 1:
            self.latent_queries = {
                n: nn.Parameter(torch.randn(self.n_hidden_token, self.d_model))
                for n in range(self.n_attn_layer - 1)
            }

        self.softmax = nn.Softmax(dim=3)

        self.attn_layer_norms = nn.ModuleList([
            nn.LayerNorm(self.d_model, eps=1e-6)
            for _ in range(self.n_attn_layer)
        ])
        self.ffd_layer_norms = nn.ModuleList([
            nn.LayerNorm(self.d_model, eps=1e-6)
            for _ in range(self.n_attn_layer)
        ])

    def forward(self,
                ctx_embed,
                q_embed,
                ctx_spa_embed,
                q_spa_embed,
                mask,
                geo_proxy):
        """
        :param ctx_embed: [bs, sl-1, dm]
        :param q_embed: [bs, sl-1, dm]
        :param ctx_spa_embed: [bs, sl-1, dm//4, dm//4]
        :param q_spa_embed: [bs, 1, dm//4, dm//4]
        :param mask: [bs, sl-1], masked:1, masked:0
        :param geo_proxy: [bs, sl-1]
        """
        batch_size = ctx_embed.shape[0]
        attn_weights = torch.FloatTensor([], device=ctx_embed.device)
        gaussian_bias = _calc_attention_bias(geo_proxy, self.attn_bias_factor)  # [bs, sl]
        embedding = ctx_embed

        # [bs, sl, dm], [bs, ql, dm]
        # Perceiver:
        for idx, (atn, ffd, atn_ln, ffd_ln) in enumerate(zip(
                self.attn_layers[:-1], self.ffds[:-1],
                self.attn_layer_norms[:-1], self.ffd_layer_norms[:-1]
        )):
            attn_output, attn_weight = atn(
                q=self.latent_queries[idx].unsqueeze(0).repeat(batch_size, 1, 1).clone(),  # -> [bs, ql, dm]
                k=ctx_embed.clone(),
                v=ctx_embed.clone(),
                q_spa_embed=q_spa_embed,
                kv_spa_embed=ctx_spa_embed,
                mask=mask,
                attention_bias=gaussian_bias
            )  # -> [bs, ql, dm]
            # residual connection:
            attn_output = atn_ln(
                self.latent_queries[idx] + attn_output
            )
            attn_output = ffd_ln(
                attn_output + ffd(attn_output)
            )
            embedding = attn_output

        # Stand-alone attention:
        if self.n_attn_layer != 1:
            mask = torch.zeros_like(mask, device=mask.device)
            gaussian_bias = torch.zeros_like(gaussian_bias, device=gaussian_bias.device)

        attn_output, attn_weight = self.attn_layers[-1](
            q=q_embed.clone(),
            k=embedding.clone(),
            v=embedding.clone(),
            q_spa_embed=q_spa_embed,
            kv_spa_embed=ctx_spa_embed,
            mask=mask,
            attention_bias=gaussian_bias
        )  # -> [bs, ql, dm]

        attn_output = self.attn_layer_norms[-1](
            q_embed + attn_output
        )
        attn_output = self.ffd_layer_norms[-1](
            attn_output + self.ffds[-1](attn_output)
        )  # -> [bs, ql, dm]

        return attn_output, attn_weights


class MaskedVanillaAttention(nn.Module):
    def __init__(self,
                 d_model,
                 n_head=4,
                 attn_dropout=0.1,
                 attn_bias_factor=1):
        super(MaskedVanillaAttention, self).__init__()

        self.d_model = d_model
        self.n_head = n_head
        self.attn_bias_factor = attn_bias_factor

        self.w_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_v = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_o = nn.Linear(self.d_model, self.d_model, bias=False)

        self.softmax = nn.Softmax(dim=-1)
        self.drop_out = nn.Dropout(attn_dropout)

    def forward(self,
                q,
                k,
                v,
                q_spa_embed,
                kv_spa_embed,
                mask=None,
                attention_bias=None):
        """
        :param q: [bs, ql, dm]
        :param k: [bs, sl, dm]
        :param v: [bs, sl, dm]
        :param q_spa_embed: [bs, 1, dm//4, dm//4]
        :param kv_spa_embed: [bs, sl, dm//4, dm//4]
        :param mask: [bs, sl] element to mask => 1, valid => 0.
        :param attention_bias: [bs, nh, ql, sl]
        """
        batch_size, key_len, _ = k.shape
        _, query_len, _ = q.shape

        # Project
        # [bs, l, dm] -> [bs, l, dm]
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)
        # [bs, l, dm] -> [bs, l, nh, dm//nh] -> [bs, nh, l, dm//nh]
        q = q.view(batch_size, query_len, self.n_head, self.d_model // self.n_head).transpose(1, 2)
        k = k.view(batch_size, key_len, self.n_head, self.d_model // self.n_head).transpose(1, 2)
        v = v.view(batch_size, key_len, self.n_head, self.d_model // self.n_head).transpose(1, 2)

        # Add spatial embedding at FIRST layer (ll = ql = latent_len)
        if key_len == kv_spa_embed.shape[1]:
            # [bs, nh, ll, dm//4, dm//4] * [bs, nh, ll, dm//4, 1] -> [bs, nh, ll, dm//4]
            q = q_spa_embed.unsqueeze(1).repeat(1, self.n_head, query_len, 1, 1).matmul(q.unsqueeze(-1)).squeeze(-1)
            # [bs, 4, sl, dm//4, dm//4] * [bs, 4, sl, dm//4, 1] -> [bs, 4, sl, dm//4]
            k = kv_spa_embed.unsqueeze(1).repeat(1, self.n_head, 1, 1, 1).matmul(k.unsqueeze(-1)).squeeze(-1)
        # Add spatial embedding at PROCESSOR layers:
        elif key_len == query_len:
            # [bs, nh, ll, dm//4, dm//4] * [bs, nh, ll, dm//4, 1] -> [bs, nh, ll, dm//4]
            q = q_spa_embed.unsqueeze(1).repeat(1, self.n_head, query_len, 1, 1).matmul(q.unsqueeze(-1)).squeeze(-1)
            # [bs, nh, ll, dm//4, dm//4] * [bs, nh, ll, dm//4, 1] -> [bs, nh, ll, dm//4]
            k = q_spa_embed.unsqueeze(1).repeat(1, self.n_head, query_len, 1, 1).matmul(k.unsqueeze(-1)).squeeze(-1)
        # Add spatial embedding at DECODER layer (ql = 1)
        elif query_len == 1:
            # [bs, nh, ql, dm//4, dm//4] * [bs, nh, ql, dm//4, 1] -> [bs, nh, ql, dm//4]
            q = q_spa_embed.unsqueeze(1).repeat(1, self.n_head, query_len, 1, 1).matmul(q.unsqueeze(-1)).squeeze(-1)
            # [bs, nh, ll, dm//4, dm//4] * [bs, nh, ll, dm//4, 1] -> [bs, nh, ll, dm//4]
            k = q_spa_embed.unsqueeze(1).repeat(1, self.n_head, key_len, 1, 1).matmul(k.unsqueeze(-1)).squeeze(-1)

        # [bs, nh, ql, dm//4] * [bs, nh, dm//4, sl] -> [bs, nh, ql, sl]
        attn_score = q.matmul(k.transpose(-1, -2)) / math.sqrt(self.d_model // self.n_head)  # dk

        if attention_bias.shape[1] == key_len:
            if attention_bias is None:
                raise NotImplementedError()
            else:
                # [bs, sl] -> [bs, nh, ql, sl]
                attention_bias = attention_bias.unsqueeze(1).unsqueeze(1).repeat(1, self.n_head, query_len, 1)
        else:
            attention_bias = torch.zeros_like(attn_score, device=attn_score.device)

        if mask.shape[1] == key_len:
            # [bs, nh, ql, sl] + [bs, nh, ql, sl] -> [bs, nh, ql, sl]
            attn_score += mask.unsqueeze(1).unsqueeze(1).repeat(1, self.n_head, query_len, 1) * -1e9

        attn_score = self.softmax(attn_score - attention_bias)
        attn_score = self.drop_out(attn_score)

        # [bs, nh, ql, l] * [bs, nh, l, dm//nh] -> [bs, nh, ql, dm//nh] -> [bs, ql, nh, dm//nh] -> [bs, ql, dm]
        v = attn_score.matmul(v).transpose(2, 3).contiguous().view(batch_size, query_len, self.d_model)
        v = self.w_o(v)  # -> [bs, ql, dm]

        return v, attn_score


import torch
from torch import nn as nn


class NanBatchNorm1dNaive(nn.Module):
    def __init__(self,
                 n_feature,
                 eps=1e-5,
                 momentum=0.1,
                 affine=False,
                 track_running_stats=True):
        super(NanBatchNorm1dNaive, self).__init__()
        self.num_features = n_feature
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = nn.Parameter(torch.ones(self.num_features))
            self.bias = nn.Parameter(torch.zeros(self.num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(1, 1, self.num_features))
            self.register_buffer('running_std', torch.ones(1, 1, self.num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0.))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_std', None)
            self.register_parameter('num_batches_tracked', None)

    def forward(self, input):
        """
        :param input: [bs, sl, fd]
        """
        if self.training:
            mask = torch.isnan(input)
            nan_mean_per_feat = torch.nanmean(input, dim=(0, 1), keepdim=True)
            count_no_nan_per_feat = (~mask).float().sum(dim=(0, 1), keepdim=True)
            sum_no_mean_per_feat = torch.nansum((input - nan_mean_per_feat) ** 2, dim=(0, 1), keepdim=True)
            nan_std_per_feat = torch.sqrt(sum_no_mean_per_feat / count_no_nan_per_feat)

            if self.track_running_stats:
                with torch.no_grad():
                    self.running_mean = self.momentum * nan_mean_per_feat + \
                                        (1 - self.momentum) * self.running_mean
                    self.running_std = self.momentum * nan_std_per_feat + \
                                       (1 - self.momentum) * self.running_std

        input_norm = (input - self.running_mean) / (self.running_std + self.eps)

        return input_norm


class GADefaultHyperParameters:
    variants = {
        "mini": {"idu_points": 0, "seq_len": 81, "n_attn_layer": 1},
        "small": {"idu_points": 4, "seq_len": 144, "n_attn_layer": 2},
        "large": {"idu_points": 8, "seq_len": 256, "n_attn_layer": 3},
    }


GAParams = GADefaultHyperParameters

from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F



class FCNTokenizer(nn.Module):
    """
    FCN encodes raw features to higher-dimensional (d_model // 2) features.
    Not encoding spa_feats, where we use 2-D rotary PE instead.
    """

    def __init__(self, d_model, x_dims, y_dims):
        self.d_model = d_model
        self.x_dims, self.y_dims = x_dims, y_dims
        super(FCNTokenizer, self).__init__()
        self.x_encoder = nn.Sequential(
            nn.Linear(len(self.x_dims), self.d_model // 2, bias=True),
            nn.Tanhshrink(),
            nn.Linear(self.d_model // 2, self.d_model // 2, bias=True)
        )
        self.y_encoder = nn.Sequential(
            nn.Linear(len(self.y_dims), self.d_model // 2, bias=True),
            nn.Tanhshrink(),
            nn.Linear(self.d_model // 2, self.d_model // 2, bias=True)
        )
        self.learnable_y = nn.Parameter(torch.randn(1, 1, self.d_model // 2))

    def forward(self, x_tensor, y_tensor):
        """
        :param x_tensor: [bs, sl, x_dim]
        :param y_tensor: [bs, sl-1, 1]
        """
        batch_size, seq_len = x_tensor.shape[0], x_tensor.shape[1]
        x_embed = self.x_encoder(x_tensor)  # -> [bs, sl, dm//2]
        y_embed = self.y_encoder(y_tensor)  # -> [bs, sl-1, dm//2]

        y_embed = torch.concat((
            y_embed,
            self.learnable_y.repeat(batch_size, 1, 1)
        ), dim=1)  # -> [bs, sl, dm//2]

        return x_embed, y_embed  # [bs, sl, dm//2], [bs, sl, dm//2]


class FCNRegressor(nn.Module):
    def __init__(self, linear_dims):
        super(FCNRegressor, self).__init__()
        self.decoder = nn.ModuleList([
            nn.Linear(linear_dims[lyr], linear_dims[lyr + 1]) for lyr in range(len(linear_dims) - 1)
        ])

    def forward(self, encoding_tensor):
        """
        :param encoding_tensor: encoder output. [bs, 1, dm]
        :return: regression result [bs, 1, 1]
        """
        in_mat = encoding_tensor
        for lyr in self.decoder[:-1]:
            in_mat = lyr(in_mat)
            in_mat = F.tanhshrink(in_mat)

        in_mat = self.decoder[-1](in_mat)

        return in_mat


class RotaryEmbedding2D(nn.Module):
    """
    Optimized 2D Rotary Positional Embeddings for geospatial coordinates.
    Key optimizations:
    1. Pre-compute and cache frequency bands
    2. Use efficient batch operations
    3. Support both training and inference with minimal overhead
    4. Maintain compatibility with existing GeoAggregator architecture
    """

    def __init__(self, d: int = 4, base: int = 10000, scale: float = 1.0, n_heads: int = 4):
        """
        Args:
            d: Model dimension (must be divisible by 4 for 2D coordinates)
            base: Base for the frequency bands
            scale: Optional scaling factor for positions
            n_heads: Number of attention heads (default: 4)
        """
        super().__init__()
        assert d % 4 == 0, f"Model dimension {d} must be divisible by 4 for 2D coordinates"
        
        self.d = d
        self.base = base
        self.scale = scale
        self.n_heads = n_heads
        
        # Cache frequency bands - shape: [d//4] for each coordinate
        inv_freq = 1.0 / (base ** (torch.arange(0, d//4, 1).float() / (d//4)))
        self.register_buffer("inv_freq", inv_freq)
        
        # # Cache for inference
        # self.cached_shape = None
        # self.cached_sin_cos = None

    def _compute_sin_cos(self, coords: torch.Tensor) -> tuple:
        """
        Compute sin and cos values for given coordinates efficiently.
        Args:
            coords: [batch_size, seq_len, 2] coordinates tensor
        Returns:
            tuple of (sin_x, cos_x, sin_y, cos_y) each with shape [batch_size, seq_len, d//4]
        """
        x, y = coords[..., 0], coords[..., 1]
        batch_size, seq_len = x.shape
        
        # Scale and reshape coordinates
        x = x.view(batch_size, seq_len, 1) * self.scale
        y = y.view(batch_size, seq_len, 1) * self.scale
        
        # Compute frequency products efficiently
        freq_x = x @ self.inv_freq.view(1, -1)  # [batch_size, seq_len, d//4]
        freq_y = y @ self.inv_freq.view(1, -1)
        
        # Compute sin and cos
        sin_x, cos_x = torch.sin(freq_x), torch.cos(freq_x)
        sin_y, cos_y = torch.sin(freq_y), torch.cos(freq_y)
        
        return sin_x, cos_x, sin_y, cos_y

    def embed(self, spa_feat: torch.Tensor) -> torch.Tensor:
        """
        Create rotation matrices for spatial coordinates.
        Args:
            spa_feat: [batch_size, seq_len, 2] coordinates tensor
        Returns:
            Rotation matrix R [batch_size, seq_len, d, d]
        """
        batch_size, seq_len, _ = spa_feat.shape
        
        # # Check if we can use cached computations for inference
        # if not self.training:
        #     current_shape = (spa_feat.shape, spa_feat.device, spa_feat.dtype)
        #     if self.cached_shape == current_shape and self.cached_sin_cos is not None:
        #         sin_emb, cos_emb = self.cached_sin_cos
        #         return self._construct_rotation_matrix(sin_emb, cos_emb, batch_size, seq_len)
        
        # Compute sin and cos values
        sin_x, cos_x, sin_y, cos_y = self._compute_sin_cos(spa_feat)
        
        # Combine x and y components
        sin_emb = torch.cat([sin_x, sin_y], dim=-1)  # [batch_size, seq_len, d//2]
        cos_emb = torch.cat([cos_x, cos_y], dim=-1)
        
        # # Cache for inference
        # if not self.training:
        #     self.cached_shape = (spa_feat.shape, spa_feat.device, spa_feat.dtype)
        #     self.cached_sin_cos = (sin_emb, cos_emb)
        
        # Construct rotation matrix
        return self._construct_rotation_matrix(sin_emb, cos_emb, batch_size, seq_len)

    def _construct_rotation_matrix(self, sin_emb: torch.Tensor, cos_emb: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        """
        Construct rotation matrix from sin and cos embeddings.
        Args:
            sin_emb: [batch_size, seq_len, d//2]
            cos_emb: [batch_size, seq_len, d//2]
            batch_size: Batch size
            seq_len: Sequence length
        Returns:
            Rotation matrix [batch_size, seq_len, d, d]
        """
        # The sin and cos embeddings have shape [batch_size, seq_len, d//2]
        # For a proper 2x2 rotation matrix, we need to create a block diagonal matrix
        
        # First create block-wise matrices
        d_half = sin_emb.shape[-1]  # This is d//2
        
        # For each position (batch element, sequence position), we'll create 
        # a matrix of shape [d//2, d//2] where each 2x2 block is a rotation matrix
        
        # Initialize an empty rotation matrix
        R = torch.zeros(batch_size, seq_len, self.d, self.d, device=sin_emb.device)
        
        # Fill in the rotation matrices
        for i in range(d_half // 2):  # For each 2x2 block
            # Get the sin and cos values for this block
            sin_block = sin_emb[:, :, i].unsqueeze(-1).unsqueeze(-1)  # [b, s, 1, 1]
            cos_block = cos_emb[:, :, i].unsqueeze(-1).unsqueeze(-1)  # [b, s, 1, 1]
            
            # Compute 2x2 rotation matrix for this block
            block = torch.cat([
                torch.cat([cos_block, -sin_block], dim=-1),
                torch.cat([sin_block, cos_block], dim=-1)
            ], dim=-2)  # [b, s, 2, 2]
            
            # Place this block along the diagonal
            row_idx = i * 2
            col_idx = i * 2
            R[:, :, row_idx:row_idx+2, col_idx:col_idx+2] = block
        
        # Similarly for the y-coordinate blocks
        for i in range(d_half // 2):
            # Get the sin and cos values for this block
            sin_block = sin_emb[:, :, i + d_half//2].unsqueeze(-1).unsqueeze(-1)  # [b, s, 1, 1]
            cos_block = cos_emb[:, :, i + d_half//2].unsqueeze(-1).unsqueeze(-1)  # [b, s, 1, 1]
            
            # Compute 2x2 rotation matrix for this block
            block = torch.cat([
                torch.cat([cos_block, -sin_block], dim=-1),
                torch.cat([sin_block, cos_block], dim=-1)
            ], dim=-2)  # [b, s, 2, 2]
            
            # Place this block along the diagonal
            row_idx = (i + d_half//2) * 2
            col_idx = (i + d_half//2) * 2
            R[:, :, row_idx:row_idx+2, col_idx:col_idx+2] = block
            
        return R

    def __repr__(self):
        return f"RotaryEmbedding2D(d={self.d}, base={self.base}, scale={self.scale}, n_heads={self.n_heads})"


class GeoAggregator(nn.Module):
    def __init__(self,
                 x_dims: tuple,
                 spa_dims: tuple,
                 y_dims: tuple,
                 attn_variant: Literal['MCPA', 'vanilla'] = 'MCPA',
                 model_variant: Optional[Literal['mini', 'small', 'large']] = None,
                 d_model: int = 32,
                 n_attn_layer: int = 2,
                 idu_points: int = 4,
                 attn_dropout: float = 0.2,
                 attn_bias_factor: float = None,
                 dc_lin_dims: list = None):
        """
        An encoder-processor-decoder architecture (see GeoAggregator paper).

        :param x_dims:
            Dim indices for x features (co-variates).
        :param spa_dims:
            Dim indices for spatial coordinates.
        :param y_dims:
            Dim indices for y feature dims.
        :param attn_variant:
            Variant of the attention mechanism: Cartesian attention (MCPA) or
            vanilla attention (see GeoAggregator paper).
        :param model_variant:
            Variant of the GeoAggregator model (see GeoAggregator paper).
            When this param is not None, params `n_attn_layer` and `idu_points`
            are ignored.
        :param d_model:
            Embedding dimension throughout the transformer model.
            Note that, as indicated in the paper (section 3.4), the following equation
            holds:
                2 * d_c * H^2 = d_{model}
            In our implementation, we set H = 2, thus the hyperparameter `d_model`
            must be divisible by 8.
        :param n_attn_layer:
            Number of total attention layers. Default = 2.
            - When n_attn_layer = 1, GA aggregates the input points into 1
            point, without further decoding.
            - When n_attn_layer = 2, the first attention layer (the encoder)
            aggregates the input points into `idu_points`-many inducing points;
            the second attention layer (the decoder) aggregates the
            inducing points into one point, before final regression.
            - When n_attn_layer = n (n >= 3), the first attention layer (the encoder)
            aggregates the input points into `idu_points`-many inducing points;
            the following n-2 attention layers (the processor) learn the interaction
            between these inducing points; the final attention layer (the decoder)
            aggregates the inducing points into one point, before final regression.
        :param idu_points:
            Number of inducing points.
        :param attn_dropout:
            Dropout probability for the attention mechanism.
        :param attn_bias_factor:
            If the attention bias factor is not given (None), the model dynamically
            learns the optimized value for the attention bias factor.
        :param dc_lin_dims:
            Decoder ffn dims.
        """
        super(GeoAggregator, self).__init__()
        # ----------------------------------------------------------------
        self.x_dims = x_dims
        self.spa_dims = spa_dims
        self.y_dims = y_dims
        self.attn_variant = attn_variant

        if model_variant is not None:
            idu_points = GAParams.variants[model_variant]['idu_points']
            n_attn_layer = GAParams.variants[model_variant]['n_attn_layer']

        # Check dimensions:
        assert d_model % 8 == 0, ('`d_model` must be divisible by 8. '
                                  'Refer to the docstring for details.')
        # ----------------------------------------------------------------
        # Encoder-Processor-Decoder architecture
        self._n_batch_norm = NanBatchNorm1dNaive(
            n_feature=len(self.x_dims + self.y_dims)
        )
        self._tokenizer = FCNTokenizer(
            d_model=d_model,
            x_dims=self.x_dims,
            y_dims=self.y_dims
        )
        self._rotary_embed = RotaryEmbedding2D(
            d=d_model // 4,
            base=10000,
            scale=1.0,
            n_heads=4
        )
        self._perceiver = CartesianPerceiver(
            d_model=d_model,
            n_attn_layer=n_attn_layer,
            attn_dropout=attn_dropout,
            attn_bias_factor=attn_bias_factor,
            n_hidden_token=idu_points
        )
        if attn_variant == 'MCPA':
            self._perceiver = CartesianPerceiver(
                d_model=d_model,
                n_attn_layer=n_attn_layer,
                attn_dropout=attn_dropout,
                attn_bias_factor=attn_bias_factor,
                n_hidden_token=idu_points
            )
        else:
            self._perceiver = VanillaPerceiver(
                d_model=d_model,
                n_head=4,
                n_attn_layer=n_attn_layer,
                n_hidden_token=idu_points,
                attn_dropout=attn_dropout,
                attn_bias_factor=attn_bias_factor
            )
        # ----------------------------------------------------------------
        # Regressor
        if dc_lin_dims is None:
            dc_lin_dims = [d_model // 2, 1]

        self._dc_lin_dims = dc_lin_dims.copy()
        self._dc_lin_dims.insert(
            0, d_model + len(self.x_dims + self.spa_dims)
        )
        self._regressor = FCNRegressor(linear_dims=self._dc_lin_dims)

    def __enc_proc_dec(self,
                       input_tensor,
                       mask,
                       geo_proximity):
        """
        The Encoder-Processor-Decoder.
        :param input_tensor: [bs, sl, fd].
        :param mask: [bs, sl-1], masked: 1, not masked: 0.
        :param geo_proximity: [bs, sl-1], distances.
        """
        # 2D Rotary Positional Embedding
        # -> [bs, sl, 2]
        spa_embed = input_tensor[:, :, self.spa_dims]
        # [bs, sl, 2] -> [bs, sl, dk, dk]
        spa_embed = self._rotary_embed.embed(spa_embed)

        # Feature Projection
        # -> [bs, sl, x_dims]
        x_embed = input_tensor[:, :, self.x_dims]
        # -> [bs, sl-1, y_dims]
        y_embed = input_tensor[:, :-1, self.y_dims]
        # -> [bs, sl, dm//2], [bs, sl, dm//2]
        x_embed, y_embed = self._tokenizer(x_tensor=x_embed, y_tensor=y_embed)

        # Attention
        if self.attn_variant == 'MCPA':
            # -> [2, bs, ql(1), dm//2]
            return self._perceiver(
                a_embed=x_embed[:, :-1, :],
                b_embed=y_embed[:, :-1, :],
                q_a_embed=x_embed[:, -1:, :],
                q_b_embed=y_embed[:, -1:, :],
                ctx_spa_embed=spa_embed[:, :-1, :, :],
                q_spa_embed=spa_embed[:, -1:, :, :],
                mask=mask,
                geo_proxy=geo_proximity
            )
        else:
            # -> [bs, sl-1, dm]
            ctx_embed = torch.cat(
                (x_embed[:, :-1, :], y_embed[:, :-1, :]),
                dim=-1
            )
            q_embed = torch.cat(
                (x_embed[:, -1:, :], y_embed[:, -1:, :]),
                dim=-1
            )

            # -> [bs, ql(1), dm]
            return self._perceiver(
                ctx_embed=ctx_embed,
                q_embed=q_embed,
                ctx_spa_embed=spa_embed[:, :-1, :, :],
                q_spa_embed=spa_embed[:, -1:, :, :],
                mask=mask,
                geo_proxy=geo_proximity
            )

    def forward(self,
                input_tensor,
                geo_proximity,
                get_attn_score=False):
        """
        :param geo_proximity: [bs, sl]
        :param input_tensor: [bs, sl, x_dims + y_dims + spa_dims]
        :param get_attn_score: return attention scores, True or False.
        """
        # Preprocess
        mask = torch.where(input_tensor.isnan(), 1, 0)[:, :-1, 0]  # -> [bs, sl-1]
        input_tensor[:, :, self.x_dims + self.y_dims] = \
            self._n_batch_norm(input_tensor[:, :, self.x_dims + self.y_dims])

        input_tensor = input_tensor.nan_to_num()
        geo_proximity = geo_proximity.nan_to_num()[:, :-1]  # -> [bs, sl] -> [bs, sl-1]

        target_point_x = input_tensor[:, -1][:, :-1].clone()  # -> [bs, fd] -> [bs, fd-1]
        target_point_x = target_point_x.unsqueeze(1)  # [bs, fd-1] -> [bs, 1, fd-1]

        # Perceiver
        encoding_tensor, attn_weights = self.__enc_proc_dec(
            input_tensor=input_tensor,
            mask=mask,
            geo_proximity=geo_proximity
        )

        # Regression
        _, batch_size, query_len, d_model_half = encoding_tensor.shape
        encoding_tensor = encoding_tensor.permute(1, 2, 0, 3).contiguous().view(batch_size, query_len, -1)
        encoding_tensor = torch.concat((encoding_tensor, target_point_x), dim=-1)

        pred = self._regressor(encoding_tensor=encoding_tensor)  # -> [bs, 1, 1]

        if get_attn_score:
            return pred, attn_weights
        else:
            return pred

