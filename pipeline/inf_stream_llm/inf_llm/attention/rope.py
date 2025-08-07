# MIT License

# Copyright (c) 2024 jiayi yuan

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
from typing import Union, Tuple
import math

class RotaryEmbeddingESM(torch.nn.Module):
    """
    Rotary position embeddings based on those in
    [RoFormer](https://huggingface.co/docs/transformers/model_doc/roformer). Query and keys are transformed by rotation
    matrices which depend on their relative positions.
    """

    def __init__(
        self, 
        dim: int, 
        base: Union[int, float] = 10000,
        distance_scale: Union[int, float] = 1,
        rope_linear_scaling_factor = None,
        config = None
    ):
        super().__init__()
        self.base = base
        self.distance_scale = distance_scale

        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, device="cuda", dtype=torch.int64) / dim)
        )
        # from transformers/modeling_rope_utils.py#L310 
        if config and config.rope_scaling and config.rope_scaling["rope_type"] == "llama3":
            factor = config.rope_scaling["factor"]  # `8` in the original implementation
            low_freq_factor = config.rope_scaling["low_freq_factor"]  # `1` in the original implementation
            high_freq_factor = config.rope_scaling["high_freq_factor"]  # `4` in the original implementation
            old_context_len = config.rope_scaling["original_max_position_embeddings"]  # `8192` in the original implementation

            low_freq_wavelen = old_context_len / low_freq_factor
            high_freq_wavelen = old_context_len / high_freq_factor

            wavelen = 2 * math.pi / inv_freq
            # wavelen < high_freq_wavelen: do nothing
            # wavelen > low_freq_wavelen: divide by factor
            inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
            # otherwise: interpolate between the two, using a smooth factor
            smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
            is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
            inv_freq = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

        if rope_linear_scaling_factor:
            inv_freq /= rope_linear_scaling_factor
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._seq_len_cached = -1
        self._cos_cached = None
        self._sin_cached = None

    def rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    def apply_rotary_pos_emb(self, x, length, right, cos, sin):
        dtype = x.dtype
        if cos.dim() == 2:
            cos = cos[right-length:right, :]
            sin = sin[right-length:right, :]
        elif cos.dim() == 3:
            cos = cos[:, right-length:right, :]
            sin = sin[:, right-length:right, :]
        elif  cos.dim() == 4:
            cos = cos[:, :, right-length:right, :]
            sin = sin[:, :, right-length:right, :]
        elif  cos.dim() == 5:
            cos = cos[:, :, :, right-length:right, :]
            sin = sin[:, :, :, right-length:right, :]
            
        return ((x.float() * cos) + (self.rotate_half(x).float() * sin)).to(dtype)

    def _update_cos_sin_tables(self, x, seq_dim, seq_len = 0):
        if seq_len == 0:
            seq_len = x.size(seq_dim)
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device = x.device).type_as(self.inv_freq)
            freqs = torch.outer(t * self.distance_scale, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim = -1)
            if x.dim() == 2:
                self._cos_cached = emb.cos()
                self._sin_cached = emb.sin()
            elif x.dim() == 3:
                self._cos_cached = emb.cos()[None, :, :]
                self._sin_cached = emb.sin()[None, :, :]
            elif x.dim() == 4:
                self._cos_cached = emb.cos()[None, None, :, :]
                self._sin_cached = emb.sin()[None, None, :, :]
            elif x.dim() == 5:
                self._cos_cached = emb.cos()[None, None, None, :, :]
                self._sin_cached = emb.sin()[None, None, None, :, :]
        return self._cos_cached, self._sin_cached

    def _update_cos_sin_tables_len(self, seq_len, device, dim = None):
        if seq_len > self._seq_len_cached:
            if dim is None:
                assert self._cos_cached is not None
                dim = self._cos_cached.dim()

            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device = device).type_as(self.inv_freq)
            freqs = torch.outer(t * self.distance_scale, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim = -1)
            if dim == 2:
                self._cos_cached = emb.cos()
                self._sin_cached = emb.sin()
            elif dim == 3:
                self._cos_cached = emb.cos()[None, :, :]
                self._sin_cached = emb.sin()[None, :, :]
            elif dim == 4:
                self._cos_cached = emb.cos()[None, None, :, :]
                self._sin_cached = emb.sin()[None, None, :, :]
            elif dim == 5:
                self._cos_cached = emb.cos()[None, None, None, :, :]
                self._sin_cached = emb.sin()[None, None, None, :, :]

        return self._cos_cached, self._sin_cached

    def apply_rotary_pos_emb_one_angle(
        self, x: torch.Tensor, index
    ):
        dtype = x.dtype
        cos, sin = self._update_cos_sin_tables_len(index, x.device)
        if cos.dim() == 2:
            cos = cos[index-1:index, :]
            sin = sin[index-1:index, :]
        elif cos.dim() == 3:
            cos = cos[:, index-1:index, :]
            sin = sin[:, index-1:index, :]
        elif  cos.dim() == 4:
            cos = cos[:, :, index-1:index, :]
            sin = sin[:, :, index-1:index, :]
        elif  cos.dim() == 5:
            cos = cos[:, :, :, index-1:index, :]
            sin = sin[:, :, :, index-1:index, :]
            
        return ((x.float() * cos) + (self.rotate_half(x).float() * sin)).to(dtype)


    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_dim= -2, seq_len= 0) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len == 0:
            self._cos_cached, self._sin_cached = self._update_cos_sin_tables(k, seq_dim=seq_dim)
            return (
                self.apply_rotary_pos_emb(q, q.size(seq_dim), k.size(seq_dim), self._cos_cached, self._sin_cached),
                self.apply_rotary_pos_emb(k, k.size(seq_dim), k.size(seq_dim), self._cos_cached, self._sin_cached),
            )
        else:
            self._cos_cached, self._sin_cached = self._update_cos_sin_tables(k, seq_dim=seq_dim, seq_len=seq_len)
            return (
                self.apply_rotary_pos_emb(q, q.size(seq_dim), seq_len, self._cos_cached, self._sin_cached),
                self.apply_rotary_pos_emb(k, k.size(seq_dim), seq_len, self._cos_cached, self._sin_cached),
            )
