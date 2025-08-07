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

import logging
import os
import json
import pdb

logger = logging.getLogger("main")
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from eval.scbench_utils.compute_scores import compute_scores
from eval.scbench_utils.eval_utils import (
   DATA_NAME_TO_MAX_NEW_TOKENS,
    GreedySearch,
    check_benchmark_availability,
    create_multiturn_prompt,
    create_scdq_prompt,
    dump_jsonl,
    get_compressed_examples,
    get_ground_truth,
    load_data,
)
import torch
import inference as inference
from pipeline.model_utils import build_chat, post_process
from transformers import AutoModelForCausalLM, AutoTokenizer
from eval.longbench_utils.eval_long_bench import load_data

def truncate_input(input: list, max_length: int, manner="middle"):
    if max_length < 0:
        return input
    if len(input) <= max_length:
        return input
    if manner == "middle":
        split = max_length // 2
        return input[0:split] + input[-split:]
    else:
        return None


def truncate_by_tokens(input, tok, max_tokens, manner: str = "middle"):
    tokens = tok.encode(input)
    len_before = len(tokens)
    print(f"# tokens before: {len_before}")
    tokens = truncate_input(tokens, max_length=max_tokens, manner=manner)
    len_after = len(tokens)  # type: ignore
    print(f"# tokens after: {len_after}")
    assert len_after <= len_before
    assert len_after <= max_tokens or max_tokens < 0
    return tokens


def get_pred(
    model,
    eg,
    data_name,
    max_new_tokens,
    max_input_length: int,
    attn_type: str = "vllm",
    tok=None,
    use_chat_template=False,
    scdq_mode=False,
    disable_golden_context=False,
) -> str:
    """
    Truncate down to 128k then make inference.
    """
    if scdq_mode:
        encoded_eg = create_scdq_prompt(
            eg,
            data_name=data_name,
            tok=tok,
            use_chat_template=use_chat_template,
            use_vllm=("vllm" in attn_type),
        )
    else:
        # multi-turn mode
        encoded_eg = create_multiturn_prompt(
            eg,
            data_name=data_name,
            tok=tok,
            use_chat_template=use_chat_template,
            use_vllm=("vllm" in attn_type),
            disable_golden_context=disable_golden_context,
        )
    context = truncate_by_tokens(
        encoded_eg["prompts"][0], model.tokenizer, max_input_length
    )
    encoded_eg["prompts"][0] = context
    if scdq_mode:
        # scdq mode has no action for disable_golden_context
        outputs = model.test_scdq(encoded_eg, max_length=max_new_tokens)
    else:
        # multi-turn mode test
        outputs = model.test(
            encoded_eg,
            max_length=max_new_tokens,
            disable_golden_context=disable_golden_context,
        )

    print("Chunked generation:", json.dumps(outputs, indent=2, ensure_ascii=False))
    return outputs


def eval_scbench(config):
    eval_params = config['eval_params']
    pipeline_params = config['pipeline_params']
    examples = load_data(eval_params)
    data_name = eval_params['dataset']
    model_name = pipeline_params['model_name']
    scdq_mode = pipeline_params['scdq_mode']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}

    model, tokenizer = inference.initialize_model_tokenizer(pipeline_params=config['pipeline_params'])
    model.eval()
    model = GreedySearch(model, tokenizer)

    max_new_tokens = DATA_NAME_TO_MAX_NEW_TOKENS[data_name]
    max_turn_size = len(examples[0]["multi_turns"])
    max_seq_length = pipeline_params['model_max_len']
    real_model_name = model_name.split("/")[-1]
    preds = []
    print(f"==== Evaluation {data_name}====")
    print(f"# examples: {len(examples)}")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"Num of turns: {max_turn_size}")

    output_path = os.path.join(config['management']['output_folder_dir'], 'pred')
    output_path = os.path.join(output_path, pipeline_params['method'])
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    output_path = Path(output_path)
    output_file = (output_path / f'{eval_params["dataset"]}_{pipeline_params["chat_template"]}.jsonl')
    result_file = os.path.join(config['management']['output_folder_dir'], 'results.json')
    done = set()

    for i, eg in tqdm(enumerate(examples)):
        if i < 0 or i in done:
            continue
        if data_name in [
            "scbench_summary_with_needles",
            "scbench_repoqa_and_kv",
        ]:
            max_input_length = max_seq_length - (
                sum(list(max_new_tokens.values())) * max_turn_size // 2
            )
        else:
            max_input_length = max_seq_length - max_new_tokens * max_turn_size
        if scdq_mode:
            max_input_length -= 1000

        pred = get_pred(
            model,
            eg,
            data_name,
            max_new_tokens,
            max_input_length=max_input_length,
            attn_type="vllm",
            tok=tokenizer,
            use_chat_template=True,
            scdq_mode=scdq_mode,
            disable_golden_context=False,
        )
        # a list of ground truth answers for each turn
        gts = get_ground_truth(eg, data_name)
        for turn_idx, (ans, gt, turn) in enumerate(
            zip(pred["answers"], gts, eg["multi_turns"])
        ):
            case = {
                "id": i,
                "turn_idx": turn_idx,
                "prediction": ans,
                "ground_truth": gt,
            }
            if "task" in pred:
                case["task"] = pred["task"][turn_idx]
            if data_name == "scbench_repoqa":
                case["lang"] = eg["lang"]
                case["repo"] = eg["repo"]
                case["func_name"] = turn["name"]
            if data_name == "scbench_repoqa_and_kv":
                case["lang"] = eg["lang"]
                case["repo"] = eg["repo"]
                if turn["task"] == "scbench_repoqa":
                    case["func_name"] = turn["name"]
            if data_name == "scbench_kv_compressible":
                case["task"] = eg["task"]
            preds.append(case)
        dump_jsonl(preds, output_file)
        torch.cuda.empty_cache()
        done.add(i)
    score = compute_scores(
        output_file,
        data_name,
        real_model_name,
        max_seq_length=max_seq_length,
        scdq_mode=scdq_mode,
    )
    results[data_name] = score

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(score, f, indent=4)

    return results, preds
