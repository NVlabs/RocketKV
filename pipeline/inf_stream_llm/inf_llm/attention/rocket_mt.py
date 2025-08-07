# MIT License

# Copyright (c) 2023 Graphcore Ltd. and 2024 jiayi yuan

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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

import torch
import math
from typing import Optional
from .utils import repeat_kv

kv_pos = 0
tmp_key_value = []

def rocket_mt_forward(fattn: bool, topk: int, compression_ratio: float, prompt_budget: int, window_size: int = 32, kernel_size: int = 63, skip_layers: int = 0, *args, **kwargs):
    def forward(self, query : torch.Tensor,
                    key_value : torch.Tensor,
                    position_bias : Optional[torch.Tensor],
                    use_cache: bool,
                    past_key_value,
                    project_q, project_k, project_v, attention_out,
                    dim_head, num_heads, num_heads_kv
    ):

        batch_size = query.size(0)
        len_q = query.size(1)
        len_k = key_value.size(1)

        assert use_cache
        assert prompt_budget >= window_size

        h_q = project_q(query)             # (batch, len_q, num_heads * dim_head)
        h_k = project_k(key_value)         # (batch, len_k, num_heads * dim_head)
        h_v = project_v(key_value)         # (batch, len_k, num_heads * dim_head)


        h_q = h_q.view(batch_size, len_q, num_heads, dim_head).permute(0, 2, 1, 3).contiguous()   # (batch, num_heads, len_q, dim_head)
        h_k = h_k.view(batch_size, len_k, num_heads_kv, dim_head).permute(0, 2, 1, 3).contiguous()   # (batch, num_heads_kv, len_k, dim_head)
        h_v = h_v.view(batch_size, len_k, num_heads_kv, dim_head).permute(0, 2, 1, 3).contiguous()   # (batch, num_heads_kv, len_k, dim_head)

        global kv_pos
        # pass tmp_key_value to the decode phase of the current turn
        global tmp_key_value
        if past_key_value is not None:
            h_q, h_k = position_bias(h_q, h_k, seq_len=len_k + kv_pos)
            h_k2 = torch.cat((past_key_value[0], h_k), dim=-2)
            h_v2 = torch.cat((past_key_value[1], h_v), dim=-2)
            if self.layer_idx == self.config.num_hidden_layers-1:
                kv_pos += len_k
            len_k = h_k2.size(-2)
        else:
            if len(tmp_key_value) < self.layer_idx+1:
                tmp_key_value.append([])
            else:
                tmp_key_value[self.layer_idx] = []
            h_q, h_k = position_bias(h_q, h_k)
            h_k2, h_v2 = h_k, h_v
            if self.layer_idx == self.config.num_hidden_layers-1:
                kv_pos = len_k
        if use_cache:
            current_key_value = h_k2, h_v2

        #detect prefill phase
        prefill_phase = len_q > 1 or past_key_value is None

        if (prefill_phase and fattn) or self.layer_idx < skip_layers:
            h_k3 = repeat_kv(h_k2, num_heads//num_heads_kv)
            h_v3 = repeat_kv(h_v2, num_heads//num_heads_kv)
            #snapkv
            if len_k > prompt_budget and self.layer_idx >= skip_layers:
                obs_window_size = min(window_size, h_q.size(2))
                h_q_observe = h_q[:, :, -obs_window_size:]
                dist = torch.arange(0, obs_window_size, device=h_q.device)[:, None] - torch.arange(0, len_k, device=h_q.device)[None, :] + len_k - obs_window_size
                attention_mask = (dist >= 0)
                score = torch.matmul(h_q_observe, h_k3.transpose(-1, -2)) / math.sqrt(dim_head)
                score = torch.masked_fill(
                    score,
                    attention_mask.view(1, 1, obs_window_size, len_k)==False,
                    torch.scalar_tensor(float("-inf"), device=score.device, dtype=score.dtype)
                )  
                score = torch.nn.functional.softmax(score, dim=-1)
                # avoid nan in softmax
                score = torch.masked_fill(
                    score,
                    attention_mask.view(1, 1, obs_window_size, len_k)==False,
                    torch.scalar_tensor(0, device=score.device, dtype=score.dtype)
                )
                score = score[:,:,-obs_window_size:,:-obs_window_size].sum(dim=-2)
                score = score.view(batch_size,num_heads_kv,-1,len_k-obs_window_size).sum(dim=2)
                score = torch.nn.functional.max_pool1d(score, kernel_size=kernel_size, padding=kernel_size//2, stride=1)
                indices = score.topk(prompt_budget-obs_window_size, dim=-1).indices.sort().values
                indices = indices.unsqueeze(-1).expand(-1,-1,-1,dim_head)
                h_k_cur,h_v_cur = current_key_value
                h_k_compress = h_k_cur[:,:,:-obs_window_size].gather(dim=2, index=indices)
                h_v_compress = h_v_cur[:,:,:-obs_window_size].gather(dim=2, index=indices)
                h_k_snap = torch.cat([h_k_compress, h_k_cur[:,:,-obs_window_size:]], dim=2)
                h_v_snap = torch.cat([h_v_compress, h_v_cur[:,:,-obs_window_size:]], dim=2)
                tmp_key_value[self.layer_idx] = h_k_snap,h_v_snap
            else:
                tmp_key_value[self.layer_idx] = current_key_value
            from flash_attn.flash_attn_interface import flash_attn_func
            h_q = h_q.transpose(1, 2)
            h_k3 = h_k3.transpose(1, 2)
            h_v3 = h_v3.transpose(1, 2)
            o = flash_attn_func(h_q, h_k3, h_v3, causal=True)
                 
        else:
            #hybrid sparse attention
            if len(tmp_key_value[self.layer_idx]) > 0:
                h_k4 = torch.cat((tmp_key_value[self.layer_idx][0], h_k), dim=-2)
                h_v4 = torch.cat((tmp_key_value[self.layer_idx][1], h_v), dim=-2)
                len_k = h_k4.size(-2)
            else:
                h_k4, h_v4 = h_k, h_v
            if use_cache:
                tmp_key_value[self.layer_idx] = h_k4, h_v4
            dist = torch.arange(0, len_q, device=h_q.device)[:, None] - torch.arange(0, len_k, device=h_q.device)[None, :] + len_k - len_q
            attention_mask = (dist >= 0)
            attention_mask = attention_mask.view(1, 1, len_q, len_k)

            def _gather(t: torch.Tensor, dim: int, i: torch.Tensor) -> torch.Tensor:
                dim += (dim < 0) * t.ndim
                return t.gather(dim, i.expand(*t.shape[:dim], i.shape[dim], *t.shape[dim + 1 :]))

            def _scaled_softmax(x: torch.Tensor, divscale: torch.Tensor | float, dim: int) -> torch.Tensor:
                return torch.softmax(x / divscale, dim=dim)
            Q = h_q.view(batch_size, num_heads_kv, -1, len_q, dim_head).contiguous()
            K = h_k4.unsqueeze(2)
            V = h_v4.unsqueeze(2)
            # 1. Approximate attention scores using chunk size max
            sign = (Q.sum(dim=2, keepdim=True) > 0) + (~(Q.sum(dim=2, keepdim=True) > 0)) * -1
            max_key = K * sign
            positive_query = Q * sign
            # compression ratio for quest
            chunk_size = math.ceil(math.sqrt(compression_ratio))
            chunk_size = chunk_size if chunk_size <= compression_ratio else 1

            # expend max_key to be divisible by chunk_size
            padding_length = chunk_size - ((len_k - 1) % chunk_size + 1)
            max_key = torch.cat(
                [
                    max_key,
                    torch.ones(
                        (max_key.shape[0], max_key.shape[1], max_key.shape[2], padding_length, max_key.shape[4]),
                        device=max_key.device, dtype=max_key.dtype
                    )
                    * torch.tensor(torch.finfo(max_key.dtype).min),
                ],
                dim=-2,
            )
   
            # chunk max_key into chunk_size tokens
            chunk_max_key = max_key.reshape(
                max_key.shape[0],
                max_key.shape[1],
                max_key.shape[2],
                max_key.shape[3] // chunk_size,
                chunk_size,
                max_key.shape[4],
            ).amax(dim=-2)

            # 2. Approximate attention scores using r largest components of Q
            # compression ratio for head dim reduction
            r = int(Q.shape[-1]*chunk_size/compression_ratio)
            absQ = torch.abs(Q)
            i1 = torch.topk(absQ.mean(dim=2, keepdim=True), r, dim=-1).indices
            Q_hat, K_hat = _gather(positive_query, -1, i1), _gather(chunk_max_key, -1, i1)
            QK_hat = Q_hat @ K_hat.transpose(-1, -2)
            QK_hat = QK_hat.unsqueeze(-1).repeat(1,1,1,1,1,chunk_size).reshape(
                QK_hat.shape[0],
                QK_hat.shape[1],
                QK_hat.shape[2],
                QK_hat.shape[3],
                -1)[:,:,:,:,:len_k]
            masked_QK_hat = torch.where(attention_mask.unsqueeze(2), QK_hat, float("-inf"))
            scale = torch.sqrt(
                Q.shape[-1]
                * torch.abs(Q_hat).sum(dim=-1, keepdim=True)
                / absQ.sum(dim=-1, keepdim=True)
            )
            s_hat = _scaled_softmax(masked_QK_hat, scale, dim=-1)

            # 3. Gather top k positions based on approximate attention scores & run attention
            k = min(topk, len_k)
            s_hat_i2, i2 = torch.topk(s_hat.mean(dim=2, keepdim=True), k, dim=-1)
            iKV = i2[..., 0, :, None]
            QK = Q @ _gather(K, -2, iKV).transpose(-1, -2)
            masked_QK = torch.where(_gather(attention_mask.unsqueeze(2).expand_as(QK_hat), -1, i2), QK, float("-inf"))
            s = _scaled_softmax(masked_QK, Q.shape[-1] ** 0.5, dim=-1)
            o = s @ _gather(V, -2, iKV)

            o = o.view(batch_size, num_heads, len_q, dim_head).permute(0, 2, 1, 3)

        o = o.reshape(batch_size, len_q, dim_head * num_heads)
        o = attention_out(o)

        if use_cache:
            return o, current_key_value
        else:
            return o

    return forward
