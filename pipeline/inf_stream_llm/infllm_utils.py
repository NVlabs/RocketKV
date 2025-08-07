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

import os
from tqdm import tqdm
from eval.longbench_utils.constants import LONGBENCH_DATASET
from pipeline.model_utils import build_chat
from inf_llm import patch_hf, GreedySearch, patch_model_center
import torch
import numpy as np
import math

def post_process(pred, chat_template, dataset):
    if chat_template == "qwen":
        pred = pred.split("<|im_end|>")[0]
    elif "llama2" in chat_template.lower():
        pred = (
            pred.split("(Document")[0]
            .split("\n\nQuestion")[0]
            .split("\n\nAnswer")[0]
            .split("[INST]")[0]
            .split("[/INST]")[0]
            .split("(Passage")[0]
            .strip()
        )
    if dataset == "samsum":
        pred = pred.split("\n")[0].strip()

    return pred


def compress(eval_params, pipeline_params, max_seq_len, total_max_new_tokens, model, is_scbench=False):
    group_size = model.config.num_attention_heads//model.config.num_key_value_heads
    assert(pipeline_params['token_budget'] >= 16)
    if pipeline_params['method'] == 'topk':
        pipeline_params['topk'] = pipeline_params['token_budget']
    elif pipeline_params['method'] in ['rocket', 'rocket_mt', 'rocket_r0.5', 'rocket_r0.3', 'rocket_r0.7']:
        pipeline_params['kernel_size'] = 63
        pipeline_params['window_size'] = 128 if is_scbench else 32
        compression_ratio = max(1.0, float(max_seq_len)/pipeline_params['token_budget'])
        if pipeline_params['method'] == 'rocket_r0.5':
            r = 0.5
        elif pipeline_params['method'] == 'rocket_r0.7':
            r = 0.7
        elif pipeline_params['method'] == 'rocket_r0.3':
            r = 0.3
        else:
            r = min(0.2+math.log2(compression_ratio)*0.06, 0.8)
        token_capacity_budget = int(float(max_seq_len)/(compression_ratio**r))
        token_capacity_budget = max(token_capacity_budget , min(2*total_max_new_tokens, max_seq_len))
        pipeline_params['prompt_budget'] = token_capacity_budget - total_max_new_tokens
        pipeline_params['topk'] = int(pipeline_params['token_budget']//2)
        pipeline_params['compression_ratio'] = max(1.0, float(token_capacity_budget)/pipeline_params['token_budget'])
    compressed_model = patch_hf(model, pipeline_params['method'], **pipeline_params)
    return compressed_model

def get_pred(
    model, tokenizer, data,
    eval_params, pipeline_params,
    truncation: str = None, rank: int = None, 
    world_size: int = None, verbose: bool = False
):
    preds = []
    data = list(data)

    if world_size is not None:
        data = data[rank::world_size]

    searcher = GreedySearch(model, tokenizer)
    cur = 0
    total = len(data)
    evaluating_longbench = eval_params['dataset'] in LONGBENCH_DATASET
    evaluating_ruler = eval_params.get('benchmark') == 'synthetic'

    for json_obj in tqdm(data):
        extra_end_token_ids = []
        add_special_tokens = True
        if evaluating_longbench:
            prompt = eval_params['instruction'].format(**json_obj)
            if pipeline_params['chat_template'] == "llama3":
                extra_end_token_ids.append(tokenizer.encode("<|eot_id|>", add_special_tokens=False)[0])

            if pipeline_params['chat_template'] == "qwen":
                extra_end_token_ids.append(tokenizer.encode("<|im_end|>", add_special_tokens=False)[0])

            if eval_params['dataset'] == "samsum":
                    extra_end_token_ids.append(tokenizer.encode("\n", add_special_tokens=False)[-1])
            if eval_params['dataset'] not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                # chat models are better off without build prompts on these tasks
                prompt = build_chat(tokenizer, prompt, pipeline_params['chat_template'])
                if pipeline_params['chat_template'].strip().lower() in ['mistral_instruct']:
                    add_special_tokens = False
        elif evaluating_ruler:
            prompt = json_obj['input']
        else:
            prompt = json_obj

        # Truncation if necessary
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=add_special_tokens).input_ids[0]
        if len(tokenized_prompt) > pipeline_params['model_max_len'] and pipeline_params.get('truncation_mode') == 'middle':
            half = int(pipeline_params['model_max_len']/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=add_special_tokens).input_ids[0]

        max_seq_len = min(len(tokenized_prompt)+eval_params['max_new_tokens'], pipeline_params['model_max_len'])
        searcher = GreedySearch(
                       compress(eval_params, pipeline_params, max_seq_len, eval_params['max_new_tokens'], model), 
                       tokenizer
        )

        output = searcher.generate(
            input_ids = tokenized_prompt,
            max_length=eval_params['max_new_tokens'],
            chunk_size=pipeline_params.get('chunk_size'),
            extra_end_token_ids=extra_end_token_ids
        )
        pred = post_process(output[0], pipeline_params['chat_template'], eval_params['dataset'])
        if eval_params["dataset"] == "magic_city_number_retrieval" or eval_params["dataset"] == 'passkey_retrieval':
            preds.append({"pred": pred})
        elif evaluating_ruler:
            preds.append({
                'index': json_obj['index'], 
                'pred': pred, 
                'input': prompt, 
                'outputs': json_obj['outputs'],
                'others': json_obj.get('others', {}),
                'truncation': json_obj.get('truncation', -1),
                'length': json_obj.get('length', -1)
            })
        else:
            preds.append({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"], "token_length": len(tokenized_prompt) + eval_params['max_new_tokens']})
        searcher.clear()
        cur += 1
        if verbose:
            logger.info(f"----------{cur}/{total}----------")
            logger.info("Length: ", len(tokenized_prompt))
            logger.info("Question:", prompt[-100:])
            logger.info("Pred:", pred)
            logger.info("Answer:", json_obj["answers"])
            logger.info("")

    return preds
