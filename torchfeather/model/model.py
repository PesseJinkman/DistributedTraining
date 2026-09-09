import math
from typing import Optional

import torch
from torch import nn

from torchfeather.model.attention import (
    ScaledDotProductAttentionWrapper,
)
from torchfeather.model.model_args import DeepSeekV3ModelArgs
from torchfeather.model.moe import FeedForward, MoE
from torchfeather.model.rope import apply_rotary_emb, precompute_freqs_cis


class Attention(nn.Module):
    def __init__(self, model_args: DeepSeekV3ModelArgs):
        super().__init__()

        self.dim = model_args.dim 
        self.n_heads = model_args.n_heads
        self.q_lora_rank = model_args.q_lora_rank
        self.kv_lora_rank = model_args.kv_lora_rank
        self.qk_nope_head_dim = model_args.qk_nope_head_dim
        self.qk_rope_head_dim = model_args.qk_rope_head_dim
        self.qk_head_dim = (
            model_args.qk_nope_head_dim + model_args.qk_rope_head_dim
        )
        self.v_head_dim = model_args.v_head_dim

        if self.q_lora_rank == 0: # As stated in DeepSeekV2 paper it helps in reducing acivation memory by doing normal attention
            self.wq = nn.Linear(self.dim, self.n_heads*self.qk_head_dim, bias=False)
        else:
            self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=False)
            self.q_norm = nn.RMSNorm(self.q_lora_rank, eps=model_args.norm_eps)
            self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads*self.qk_head_dim, bias=False)

        self.wkv_a = nn.Linear(self.dim, self.kv_lora_rank+self.qk_rope_head_dim, bias=False) # for block multiplication
        self.kv_norm = nn.RMSNorm(self.kv_lora_rank, eps=model_args.norm_eps)
        self.wkv_b = nn.Linear(
            self.kv_lora_rank, 
            self.n_heads * (self.qk_nope_head_dim+self.v_head_dim),
            bias=False
        )

        self.wo = nn.Linear(self.n_heads*self.v_head_dim, self.dim, bias=False)
        self.softmax_scale = self.qk_head_dim**-0.5

        if model_args.max_seq_len > model_args.original_seq_len:
            mscale = 0.1*model_args.mscale*math.log(model_args.rope_factor)+1.0
            self.softmax_scale = self.softmax_scale*mscale*mscale

        self.inner_attention = ScaledDotProductAttentionWrapper()

    def init_weights(
        self,
        init_std: float,
        buffer_device: torch.device | None = None,
    ):
        linear_list = [
            self.wkv_a,
            self.wkv_b,
        ]
        if self.q_lora_rank > 0:
            linear_list.extend([self.wq_a, self.wq_b])
        else:
            linear_list.append(self.wq)

        for linear in linear_list:
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)

        self.kv_norm.reset_parameters()
        if self.q_lora_rank > 0:
            self.q_norm.reset_parameters()

    def forward(
        self, 
        x: torch.Tensor,
        freqs_cis: torch.Tensor
    ):
        batch_size, seq_len, _ = x.size()

        # Query projection
        if self.q_lora_rank == 0:
            q = self.wq(x) # (batch_size, seq_len, n_heads * qk_head_dim)
        else:
            q = self.wq_a(x) # (batch_size, seq_len, q_lora_rank)
            q = self.wq_b(self.q_norm(q)) # (batch_size, seq_len, n_heads * qk_head_dim)

        q = q.view(batch_size, seq_len, -1, self.qk_head_dim) # (batch_size, seq_len, n_heads, qk_head_dim)

        q_nope, q_pe = torch.split( # (batch_size, seq_len, n_heads, qk_nope_head_dim), (batch_size, seq_len, n_heads, qk_rope_head_dim)
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1 
        )

        q_pe = apply_rotary_emb(q_pe, freqs_cis) # (batch_size, seq_len, n_heads, qk_rope_head_dim)
        q = torch.cat([q_nope, q_pe], dim=-1) # (batch_size, seq_len, n_heads, qk_head_dim)

        kv = self.wkv_a(x) # (batch_size, seq_len, kv_lora_rank+qk_rope_head_dim)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1) # (batch_size, seq_len, kv_lora_rank), (batch_size, seq_len, qk_rope_head_dim) 

        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis) # (batch_size, seq_len, 1, qk_rope_head_dim)

        kv = self.wkv_b(self.kv_norm(kv)) # (batch_size, seq_len, n_heads * (qk_nope_head_dim+v_head_dim))
        kv = kv.view(batch_size, seq_len, -1, self.qk_nope_head_dim+self.v_head_dim) # (batch_size, seq_len, n_heads, (qk_nope_head_dim+v_head_dim))

        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1) # (batch_size, seq_len, n_heads, qk_nope_head_dim), (batch_size, seq_len, n_heads, v_head_dim)
        k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_heads, -1)], dim=-1) # (batch_size, seq_len, n_heads, qk_head_dim)

        q = q.transpose(1, 2) # (batch_size, n_heads, seq_len, qk_head_dim)
        k = k.transpose(1, 2) # (batch_size, n_heads, seq_len, qk_head_dim)
        v = v.transpose(1, 2) # (batch_size, n_heads, seq_len, v_head_dim)

        output = self.inner_attention(q, k, v, scale=self.softmax_scale) # (batch_size, n_heads, seq_len, v_head_dim)

        output = output.transpose(1, 2).contiguous() # (batch_size, seq_len, n_heads, v_head_dim)
        output = output.view(batch_size, seq_len, -1) # (batch_size, seq_len, n_heads*v_head_dim)

        return self.wo(output) # (batch_size, seq_len, dim)

    @torch.no_grad()
    def absorb_mla_weights(self) -> None:
        if self.q_lora_rank != 0:
            raise NotImplementedError()

        n_heads = self.n_heads
        dim = self.dim
        qk_nope_head_dim = self.qk_nope_head_dim
        qk_rope_head_dim = self.qk_rope_head_dim
        v_head_dim = self.v_head_dim
        kv_lora_rank = self.kv_lora_rank

        device = self.wq.weight.device
        dtype = self.wq.weight.dtype

        wq = self.wq.weight.view(
            n_heads,
            qk_nope_head_dim + qk_rope_head_dim,
            dim,
        )
        # qk_nope: [n_heads, qk_nope_head_dim, dim]
        # qk_rope: [n_heads, qk_rope_head_dim, dim]
        wq_nope, wq_rope = torch.split(
            wq,
            [qk_nope_head_dim, qk_rope_head_dim],
            dim=1,
        )

        wkv_b = self.wkv_b.weight.view(
            n_heads,
            qk_nope_head_dim + v_head_dim,
            kv_lora_rank,
        )
        # w_uk: [n_heads, qk_nope_head_dim, kv_lora_rank]
        # w_uv: [n_heads, v_head_dim, kv_lora_rank]
        w_uk, w_uv = torch.split(
            wkv_b,
            [qk_nope_head_dim, v_head_dim],
            dim=1,
        )

        # [n_heads, kv_lora_rank, dim]
        wq_abs_nope = torch.bmm(
            w_uk.float().transpose(1, 2), # [n_heads, qk_nope_head_dim, kv_lora_rank] -> [n_heads, kv_lora_rank, qk_nope_head_dim]
            wq_nope.float(), # [n_heads, qk_nope_head_dim, dim]
        ).to(dtype=dtype)

        # Each new query head is [absorbed nope | original RoPE].
        wq_abs = torch.cat(
            [wq_abs_nope, wq_rope],
            dim=1,
        ).reshape(
            n_heads * (kv_lora_rank + qk_rope_head_dim),
            dim,
        )

        self.wq_abs = nn.Linear(
            dim,
            n_heads * (kv_lora_rank + qk_rope_head_dim),
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.wq_abs.weight.copy_(wq_abs)
        self.wq_abs.requires_grad_(False)

        # [dim, n_heads, v_head_dim] -> [n_heads, dim, v_head_dim]
        w_o = self.wo.weight.view(
            dim,
            n_heads,
            v_head_dim,
        ).permute(1, 0, 2)

        # [n_heads, dim, v_head_dim] @ [n_heads, v_head_dim, kv_lora_rank] -> [n_heads, dim, kv_lora_rank]
        w_o_abs_per_head = torch.bmm(
            w_o.float(),
            w_uv.float(),
        ).to(dtype=dtype)

        # [n_heads, dim, kv_lora_rank] -> [dim, n_heads, kv_lora_rank] -> [dim, n_heads * kv_lora_rank]
        w_o_abs = w_o_abs_per_head.permute(
            1,
            0,
            2,
        ).reshape(
            dim,
            n_heads * kv_lora_rank,
        )

        self.wo_abs = nn.Linear(
            n_heads * kv_lora_rank,
            dim,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.wo_abs.weight.copy_(w_o_abs)
        self.wo_abs.requires_grad_(False)


    def forward_absorbed(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        assert self.wq_abs is not None
        assert self.wo_abs is not None

        batch_size, seq_len, _ = x.shape

        q = self.wq_abs(x)
        q = q.view(
            batch_size,
            seq_len,
            self.n_heads,
            self.kv_lora_rank + self.qk_rope_head_dim,
        )

        # q_nope: [batch_size, seq_len, n_heads, kv_lora_rank]
        # q_rope: [batch_size, seq_len, n_heads, qk_rope_head_dim]
        q_nope, q_rope = torch.split(
            q,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        q_rope = apply_rotary_emb(q_rope, freqs_cis)

        # [batch_size, seq_len, n_heads, kv_lora_rank + qk_rope_head_dim] -> [batch_size, n_heads, seq_len, kv_lora_rank + qk_rope_head_dim]
        q = torch.cat([q_nope, q_rope], dim=-1).transpose(1, 2)

        # latent_raw: [batch_size, seq_len, kv_lora_rank]
        # k_rope: [batch_size, seq_len, qk_rope_head_dim]
        latent_raw, k_rope = torch.split(
            self.wkv_a(x), # [batch_size, seq_len, dim] -> [batch_size, seq_len, kv_lora_rank + qk_rope_head_dim]
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )

        # This is the latent that should be cached.
        # latent: [batch_size, seq_len, kv_lora_rank]
        latent = self.kv_norm(latent_raw)

        # [batch_size, seq_len, 1, qk_rope_head_dim]
        k_rope = apply_rotary_emb(k_rope.unsqueeze(2), freqs_cis)

        # A single shared storage tensor:
        # shared_cache: [batch_size, seq_len, 1, kv_lora_rank + qk_rope_head_dim] -> [batch_size, 1, seq_len, kv_lora_rank + qk_rope_head_dim]
        shared_cache = torch.cat(
            [latent.unsqueeze(2), k_rope],
            dim=-1,
        ).transpose(1, 2)

        # k: [batch_size, 1, seq_len, kv_lora_rank + qk_rope_head_dim]
        k = shared_cache

        # v: [batch_size, 1, seq_len, kv_lora_rank]
        v = shared_cache[..., : self.kv_lora_rank]

        # latent_output: [batch_size, n_heads, seq_len, kv_lora_rank]
        latent_output = self.inner_attention(q, k, v, scale=self.softmax_scale)

        # [batch_size, seq_len, n_heads * kv_lora_rank]
        latent_output = (
            latent_output.transpose(1, 2) # [batch_size, n_heads, seq_len, kv_lora_rank] -> [batch_size, seq_len, n_heads, kv_lora_rank]
            .contiguous()
            .view(
                batch_size,
                seq_len,
                self.n_heads * self.kv_lora_rank,
            )
        )

        # output: [batch_size, seq_len, dim]
        return self.wo_abs(latent_output)

class TransformerBlock(nn.Module):

    def __init__(self, layer_id: int, model_args: DeepSeekV3ModelArgs):
        super().__init__()
        self.attention = Attention(model_args)
        self.attention_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.ffn_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)

        self.moe_enabled = layer_id >= model_args.n_dense_layers
        if self.moe_enabled:
            self.moe = MoE(
                model_args.moe_args,
                dim=model_args.dim,
                hidden_dim=model_args.moe_inter_dim
            )
        else:
            self.feed_forward = FeedForward(model_args.dim, model_args.inter_dim)

        self.weight_init_std = 0.02/(2*(layer_id+1)) ** 0.5
        self.layer_id = layer_id

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor):
        x = x + self.attention(self.attention_norm(x), freqs_cis)
        if self.moe_enabled:
            x = x + self.moe(self.ffn_norm(x))
        else:
            x = x + self.feed_forward(self.ffn_norm(x))
        return x

    def init_weights(
        self,
        init_std: Optional[float] = None,
        buffer_device: Optional[torch.device] = None
    ):
        if buffer_device is None:
            raise ValueError(
                "buffer_device must be provided for TransformerBlock weight initialization"
            )
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        if self.moe_enabled:
            self.moe.init_weights(
                init_std=self.weight_init_std, buffer_device=buffer_device
            )
        else:
            self.feed_forward.init_weights(self.weight_init_std)

class DeepSeekV3Model(nn.Module):

    def __init__(self, model_args: DeepSeekV3ModelArgs):
        super().__init__()
        self.model_args = model_args
        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)
        self.register_buffer(
            "freqs_cis", precompute_freqs_cis(model_args), persistent=False
        )

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(layer_id, model_args)

        self.norm = nn.RMSNorm(model_args.dim)
        self.output = nn.Linear(
            model_args.dim,
            model_args.vocab_size,
            dtype=torch.get_default_dtype(),
            bias=False
        )

    def init_weights(
        self,
        init_std: Optional[float] = None,
        buffer_device: Optional[torch.device] = None,
    ):

        buffer_device = buffer_device or self.freqs_cis.device
        with torch.device(buffer_device):
            self.freq_cis = precompute_freqs_cis(self.model_args)
        if self.tok_embeddings is not None:
            nn.init.normal_(self.tok_embeddings.weight)
        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights(init_std=init_std, buffer_device=buffer_device)
        if self.norm is not None:
            self.norm.reset_parameters()
        final_out_std = self.model_args.dim**-0.5
        cutoff_factor = 3
        if self.output is not None:
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=final_out_std,
                a=-cutoff_factor*final_out_std,
                b=cutoff_factor*final_out_std
            )

    def forward(self, tokens: torch.Tensor):

        h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens

        for layer in self.layers.values():
            h = layer(h, self.freq_cis)

        h = self.norm(h) if self.norm is not None else h
        output = self.output(h) if self.output is not None else h
        return output
