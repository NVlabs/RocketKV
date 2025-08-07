# Copyright (c) Graphcore 2024
# All rights reserved.
# This source code is licensed under the BSD-3 license,
# see the LICENSE file in the root directory of this source tree.

# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# NVIDIA License

# =======================================================================

# 1. Definitions

# “Licensor” means any person or entity that distributes its Work.

# “Work” means (a) the original work of authorship made available under
# this license, which may include software, documentation, or other files,
# and (b) any additions to or derivative works thereof that are made
# available under this license.

# The terms “reproduce,” “reproduction,” “derivative works,” and “distribution”
# have the meaning as provided under U.S. copyright law; provided, however,
# that for the purposes of this license, derivative works shall not include works
# that remain separable from, or merely link (or bind by name) to the
# interfaces of, the Work.

# Works are “made available” under this license by including in or with the Work
# either (a) a copyright notice referencing the applicability of
# this license to the Work, or (b) a copy of this license.

# 2. License Grant

# 2.1 Copyright Grant. Subject to the terms and conditions of this license, each
# Licensor grants to you a perpetual, worldwide, non-exclusive, royalty-free,
# copyright license to use, reproduce, prepare derivative works of, publicly display,
# publicly perform, sublicense and distribute its Work and any resulting derivative
# works in any form.

# 3. Limitations

# 3.1 Redistribution. You may reproduce or distribute the Work only if (a) you do so under
# this license, (b) you include a complete copy of this license with your distribution,
# and (c) you retain without modification any copyright, patent, trademark, or
# attribution notices that are present in the Work.

# 3.2 Derivative Works. You may specify that additional or different terms apply to the use,
# reproduction, and distribution of your derivative works of the Work (“Your Terms”) only
# if (a) Your Terms provide that the use limitation in Section 3.3 applies to your derivative
# works, and (b) you identify the specific derivative works that are subject to Your Terms.
# Notwithstanding Your Terms, this license (including the redistribution requirements in
# Section 3.1) will continue to apply to the Work itself.

# 3.3 Use Limitation. The Work and any derivative works thereof only may be used or
# intended for use non-commercially. Notwithstanding the foregoing, NVIDIA Corporation
# and its affiliates may use the Work and any derivative works commercially.
# As used herein, “non-commercially” means for research or evaluation purposes only.

# 3.4 Patent Claims. If you bring or threaten to bring a patent claim against any Licensor
# (including any claim, cross-claim or counterclaim in a lawsuit) to enforce any patents that
# you allege are infringed by any Work, then your rights under this license from
# such Licensor (including the grant in Section 2.1) will terminate immediately.

# 3.5 Trademarks. This license does not grant any rights to use any Licensor’s or its
# affiliates’ names, logos, or trademarks, except as necessary to reproduce
# the notices described in this license.

# 3.6 Termination. If you violate any term of this license, then your rights under
# this license (including the grant in Section 2.1) will terminate immediately.

# 4. Disclaimer of Warranty.

# THE WORK IS PROVIDED “AS IS” WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING WARRANTIES OR CONDITIONS OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE OR NON-INFRINGEMENT.
# YOU BEAR THE RISK OF UNDERTAKING ANY ACTIVITIES UNDER THIS LICENSE.

# 5. Limitation of Liability.

# EXCEPT AS PROHIBITED BY APPLICABLE LAW, IN NO EVENT AND UNDER NO LEGAL THEORY,
# WHETHER IN TORT (INCLUDING NEGLIGENCE), CONTRACT, OR OTHERWISE SHALL ANY LICENSOR
# BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY DIRECT, INDIRECT, SPECIAL, INCIDENTAL,
# OR CONSEQUENTIAL DAMAGES ARISING OUT OF OR RELATED TO THIS LICENSE, THE USE OR
# INABILITY TO USE THE WORK (INCLUDING BUT NOT LIMITED TO LOSS OF GOODWILL, BUSINESS
# INTERRUPTION, LOST PROFITS OR DATA, COMPUTER FAILURE OR MALFUNCTION, OR ANY
# OTHER DAMAGES OR LOSSES), EVEN IF THE LICENSOR HAS BEEN ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGES.

# =======================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn
import math
from pathlib import Path
import numpy as np

default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

PREFILL = 0
LAST_PREFILL = 1
DECODE = 2


@dataclass(frozen=True)
class RocketArgs:
    token_budget: int = 1024
    window_size: int = 32,
    kernel_size: int = 63,
    # Sorting the output of the top-k takes time, but might result in more contiguous
    # memory accesses. In our experiments we found it was faster not to sort.
    sort_stage_1_top_k: bool = False
    sort_stage_2_top_k: bool = False


def get_params_for_token_budget(
    token_budget: int, sequence_length: int, max_new_tokens: int, head_dim: int
) -> tuple[int, int]:
    """Gets r, k to reduce memory transferred during attention by the given ratio."""
    token_budget = min(sequence_length, token_budget)
    compression_ratio = max(1.0, float(sequence_length)/token_budget)
    alpha = min(0.2+math.log2(compression_ratio)*0.06, 0.8)
    capacity_budget = int(float(sequence_length)/(compression_ratio**alpha))
    prompt_budget = min(sequence_length-max_new_tokens, capacity_budget - max_new_tokens)
    k = min(round(token_budget/2), capacity_budget)
    compression_ratio =  max(1.0, float(capacity_budget)/token_budget)
    chunk_size = min(math.floor(compression_ratio), math.ceil(math.sqrt(compression_ratio)))
    r = min(head_dim, max(1, round(head_dim * chunk_size / compression_ratio)))
    return capacity_budget, prompt_budget, chunk_size, r, k


class RocketAttention(nn.Module):
    def __init__(self, config: RocketArgs, n_head: int, n_local_heads: int) -> None:
        super().__init__()
        self.config = config
        self.n_head = n_head
        self.n_local_heads = n_local_heads
        self.kv_cache = None
        self.capacity_budget : int = None
        self.prompt_budget : int = None
        self.chunk_size: int | None = None
        self.r: int | None = None
        self.k: int | None = None

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Tensor,
        input_pos: Tensor | None,
        layer_idx: int,
        state: int,
    ) -> Tensor:

        if self.kv_cache is not None:
            Kt, K, V = self.kv_cache.update(input_pos, q, k, v, state)
        #K1 = K1.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        #K2 = K2.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        #V = V.repeat_interleave(self.n_head // self.n_local_heads, dim=1)

        if state == PREFILL or state == LAST_PREFILL:
            K = K.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
            V = V.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
            return self._prefill(q, K, V, mask[:,:,:,:K.shape[-2]])
        else:
            return self._generate(q, Kt, K, V, mask[:,:,:,-K.shape[-2]:])

    def _prefill(
        self, 
        q: Tensor, 
        K: Tensor, 
        V: Tensor, 
        mask: Tensor) -> Tensor:
        return F.scaled_dot_product_attention(q, K, V, mask)

    def _generate(
        self, 
        q: Tensor, 
        Kt: Tensor, 
        K: Tensor, 
        V: Tensor, 
        mask: Tensor
    ) -> Tensor:

        assert self.r is not None and self.k is not None
        return rocket_attn(q, Kt, K, V, mask, self.chunk_size, self.r, self.k, self.config)

    def setup_caches(
        self,
        max_batch_size: int,
        max_seq_length: int,
        max_new_tokens: int,
        n_heads: int,
        head_dim: int,
        layer_idx: int,
        dtype=torch.bfloat16,
    ) -> None:

        self.capacity_budget, self.prompt_budget, self.chunk_size, self.r, self.k = get_params_for_token_budget(
            self.config.token_budget, max_seq_length, max_new_tokens, head_dim
        )

        self.kv_cache = RocketKVCache(
            max_batch_size, max_seq_length, self.capacity_budget, self.prompt_budget, self.chunk_size, n_heads, head_dim, self.config, dtype
        )
        self.kv_cache.reset()

    def reset_caches(self):
        self.kv_cache.reset()

class RocketKVCache(nn.Module):
    """KV cache that stores both K and K transpose."""

    def __init__(
        self,
        max_batch_size: int,
        max_seq_length: int,
        capacity_budget: int,
        prompt_budget: int,
        chunk_size: int,
        n_heads: int,
        head_dim: int,
        config: RocketArgs,
        dtype,
    ) -> None:
        super().__init__()
        self.capacity_reduction = max_seq_length - capacity_budget
        self.prompt_budget = prompt_budget
        self.chunk_size = chunk_size
        self.head_dim = head_dim
        self.config = config

        self.orig_cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        cache_shape = (max_batch_size, n_heads, capacity_budget, head_dim)
        cachet_shape = (max_batch_size, n_heads, math.ceil(capacity_budget/chunk_size), head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype))
        if chunk_size >=2 :
            # buffer used to store element-wise min and max values for each chunk
            self.register_buffer(
                "kt_cache",
                torch.cat([torch.full(cachet_shape, float('inf'), dtype=dtype),
                           torch.full(cachet_shape, float('-inf'), dtype=dtype)], dim=-1
                ).transpose(-1, -2).contiguous(),
            )
        else:
            self.register_buffer(
                "kt_cache",
                torch.zeros(cache_shape, dtype=dtype).transpose(-1, -2).contiguous(),
            )
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))
        self.orig_k_cache = torch.zeros(self.orig_cache_shape, dtype=dtype, device=default_device)
        self.orig_v_cache = torch.zeros(self.orig_cache_shape, dtype=dtype, device=default_device)

    def reset(self):
        pass

    def update(
        self, input_pos: Tensor, q: Tensor, k_val: Tensor, v_val: Tensor, state: int
    ) -> tuple[Tensor, Tensor, Tensor]:
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]

        #k_out: Tensor = self.k_cache
        #kt_out: Tensor = self.kt_cache
        #v_out: Tensor = self.v_cache
        #len_q, len_k = q.size(-2), k_val.size(-2)

        if state == PREFILL:
            if self.orig_k_cache is None:
                self.orig_k_cache = torch.zeros(self.orig_cache_shape, dtype=q.dtype, device=q.device)
                self.orig_v_cache = torch.zeros(self.orig_cache_shape, dtype=q.dtype, device=q.device)
            k_out: Tensor = self.orig_k_cache
            v_out: Tensor = self.orig_v_cache
            k_out[:, :, input_pos] = k_val
            v_out[:, :, input_pos] = v_val

            return None, k_out, v_out
            
        elif state == LAST_PREFILL:
            k_tmp: Tensor = self.orig_k_cache
            v_tmp: Tensor = self.orig_v_cache
            k_tmp[:, :, input_pos] = k_val
            v_tmp[:, :, input_pos] = v_val
            k_val = k_tmp[:,:,:input_pos[-1]]
            v_val = v_tmp[:,:,:input_pos[-1]]
            len_q, len_k = q.size(-2), k_val.size(-2)
            assert len_k > self.prompt_budget
            window_size = self.config.window_size[0]
            kernel_size = self.config.kernel_size[0]
            q_observe = q[:, :, -window_size:]
            dist = torch.arange(0, window_size, device=q.device)[:, None] - torch.arange(0, len_k, device=q.device)[None, :] + len_k - window_size
            attention_mask = (dist >= 0)
            k_val2 = k_val.repeat_interleave(q.size(1) // k_val.size(1), dim=1)
            score = torch.matmul(q_observe, k_val2.transpose(-1, -2)) / math.sqrt(q.size(-1))
            score = torch.masked_fill(
                score,
                attention_mask.view(1, 1, window_size, len_k)==False,
                torch.scalar_tensor(float("-inf"), device=score.device, dtype=score.dtype)
            )  
            score = torch.nn.functional.softmax(score, dim=-1)
            # avoid nan in softmax
            score = torch.masked_fill(
                score,
                attention_mask.view(1, 1, window_size, len_k)==False,
                torch.scalar_tensor(0, device=score.device, dtype=score.dtype)
            )
            score = score[:,:,-window_size:,:-window_size].sum(dim=-2)
            score = score.view(k_val.size(0),k_val.size(1),-1,len_k-window_size).sum(dim=2)
            score = torch.nn.functional.max_pool1d(score, kernel_size=kernel_size, padding=kernel_size//2, stride=1)
            indices = score.topk(self.prompt_budget-window_size, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1,-1,-1,q.size(-1))
            k_compress = k_val[:,:,:-window_size].gather(dim=2, index=indices)
            v_compress = v_val[:,:,:-window_size].gather(dim=2, index=indices)
            k_snap = torch.cat([k_compress, k_val[:,:,-window_size:]], dim=2)
            v_snap = torch.cat([v_compress, v_val[:,:,-window_size:]], dim=2)

            k_out: Tensor = self.k_cache
            kt_out: Tensor = self.kt_cache
            v_out: Tensor = self.v_cache

            k_out[:,:,:self.prompt_budget] = k_snap
            # store element-wise min and max values for each chunk
            if self.chunk_size >= 2:
                padding_len = self.chunk_size - ((self.prompt_budget - 1) % self.chunk_size + 1)
                k_snap_min = torch.cat([k_snap, 
                    torch.full((k_snap.size(0),k_snap.size(1),padding_len,k_snap.size(3)), float('inf'), device=k_snap.device, dtype=k_snap.dtype)], dim=-2)
                k_snap_min = k_snap_min.reshape(
                    k_snap_min.size(0),
                    k_snap_min.size(1),
                    k_snap_min.size(2)//self.chunk_size,
                    self.chunk_size,
                    k_snap_min.size(3)
                ).amin(dim=-2).transpose(-1, -2)
                k_snap_max = torch.cat([k_snap, 
                    torch.full((k_snap.size(0),k_snap.size(1),padding_len,k_snap.size(3)), float('-inf'), device=k_snap.device, dtype=k_snap.dtype)], dim=-2)
                k_snap_max = k_snap_max.reshape(
                    k_snap_max.size(0),
                    k_snap_max.size(1),
                    k_snap_max.size(2)//self.chunk_size,
                    self.chunk_size,
                    k_snap_max.size(3)
                ).amax(dim=-2).transpose(-1, -2)
                 
                kt_out[:, :, :, :math.ceil(self.prompt_budget/self.chunk_size)] = torch.cat([k_snap_min, k_snap_max], dim=-2)
            else:
                kt_out[:, :, :, :self.prompt_budget] = k_snap.transpose(-1, -2)
            v_out[:,:,:self.prompt_budget] = v_snap            

            self.orig_k_cache = None
            self.orig_v_cache = None

            return None, k_val, v_val 
        elif state == DECODE:
            k_out: Tensor = self.k_cache
            kt_out: Tensor = self.kt_cache
            v_out: Tensor = self.v_cache
            input_pos_shift = input_pos - self.capacity_reduction
            k_out[:, :, input_pos_shift] = k_val
            if self.chunk_size >= 2:
                kt_out[:, :, :, input_pos_shift//self.chunk_size] = torch.cat(
                                                                 [torch.min(k_val.transpose(-1, -2), 
                                                                  kt_out[:, :, :self.head_dim, input_pos_shift//self.chunk_size]),
                                                                 torch.max(k_val.transpose(-1, -2),
                                                                  kt_out[:, :, self.head_dim:, input_pos_shift//self.chunk_size])],
                                                             dim=-2)
            else:
                kt_out[:, :, :, input_pos_shift] = k_val.transpose(-1, -2)
            v_out[:, :, input_pos_shift] = v_val

            return kt_out.transpose(-1, -2), k_out, v_out



def rocket_attn(
    Q: Tensor,
    K1: Tensor,
    K2: Tensor,
    V: Tensor,
    mask: Tensor,
    chunk_size: int,
    r: int,
    k: int,
    config: RocketArgs,
) -> Tensor:
    # 1. Approximate attention scores using r largest components of Q
    n_head = Q.size(1)
    n_local_heads = V.size(1)
    head_dim = Q.size(-1)
    len_k = K2.size(-2)
    Q = Q.view(Q.shape[0], n_local_heads, -1, Q.shape[2], Q.shape[3]) 
    absQ = torch.abs(Q)
    i1 = torch.topk(absQ.sum(dim=2, keepdim=True), r, dim=-1, sorted=config.sort_stage_1_top_k).indices
    Q_hat =  _gather(Q, -1, i1)
    i1_sign = torch.where(Q_hat.sum(dim=2, keepdim=True) > 0, i1 + head_dim, i1) if chunk_size >= 2 else i1
    K_hat =  _gather(K1.unsqueeze(2), -1, i1_sign)
    QK_hat = Q_hat @ K_hat.transpose(-1, -2)
    QK_hat = QK_hat.repeat_interleave(chunk_size, dim=-1)[:,:,:,:,:len_k]
    masked_QK_hat = torch.where(mask.unsqueeze(2), QK_hat, float("-inf"))
    scale = torch.sqrt(
        Q.shape[-1]
        * torch.abs(Q_hat).sum(dim=-1, keepdim=True)
        / absQ.sum(dim=-1, keepdim=True)
    )
    s_hat = _scaled_softmax(masked_QK_hat, scale, dim=-1)

    # 2. Gather top k positions based on approximate attention scores & run attention
    k = min(k, V.shape[-2])
    i2 = torch.topk(s_hat.sum(dim=2, keepdim=True), k, dim=-1, sorted=config.sort_stage_2_top_k).indices
    iKV = i2[..., 0, :, None]
    QK = Q @ _gather(K2.unsqueeze(2), -2, iKV).transpose(-1, -2)
    masked_QK = torch.where(_gather(mask.unsqueeze(2).expand_as(QK_hat), -1, i2), QK, float("-inf"))
    s = _scaled_softmax(masked_QK, Q.shape[-1] ** 0.5, dim=-1)
    y_ = s @ _gather(V.unsqueeze(2), -2, iKV)
    y_ = y_.view(y_.shape[0], -1, y_.shape[3], y_.shape[4])
    return y_


def _gather(t: Tensor, dim: int, i: Tensor) -> Tensor:
    dim += (dim < 0) * t.ndim
    return t.gather(dim, i.expand(*t.shape[:dim], i.shape[dim], *t.shape[dim + 1 :]))


@torch.compile(disable=not torch.cuda.is_available())
def _scaled_softmax(x: Tensor, divscale: Tensor | float, dim: int) -> Tensor:
    return torch.softmax(x / divscale, dim=dim)
