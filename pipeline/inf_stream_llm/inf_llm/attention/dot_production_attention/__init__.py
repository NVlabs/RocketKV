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

from .base import MultiStageDotProductionAttention
from typing import Tuple

def get_multi_stage_dot_production_attention(flash_attn=False) -> Tuple[type, bool]:
    class UseTorch(Exception):
        pass

    try:
        if flash_attn:
            from .triton_impl import TritonMultiStageDotProductionAttention as ret
            fattn = True
        else:
            raise UseTorch

    except Exception as E:
        fattn = False
        if not isinstance(E, UseTorch):
            if get_multi_stage_dot_production_attention.warn:
                from warnings import warn
                warn("Load triton flash attention error. Use torch impl.")
                get_multi_stage_dot_production_attention.warn = False

        from .torch_impl import TorchMultiStageDotProductionAttention as ret


    return ret, fattn


get_multi_stage_dot_production_attention.warn = True
