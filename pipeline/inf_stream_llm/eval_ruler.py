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
logger = logging.getLogger("main")
import os
import subprocess
from pipeline.model_utils import build_chat, post_process
from eval.ruler_utils.data.synthetic.constants import write_manifest
import json
import infllm_utils
import inf_llm
import torch
from tqdm import tqdm

def prepare_data(config):
    eval_params = config['eval_params']
    pipeline_params = config['pipeline_params']
    management = config['management']
    try:
        script = 'eval/ruler_utils/data/prepare.py'
        command = f"""python {script} \
            --save_dir  {management['output_folder_dir']} \
            --benchmark {eval_params['benchmark']} \
            --task {eval_params['dataset']} \
            --tokenizer_path {pipeline_params['tokenizer_name']} \
            --tokenizer_type hf \
            --max_seq_length {eval_params['max_seq_length']} \
            --model_template_type {pipeline_params['chat_template']} \
            --num_samples {eval_params['num_samples']} 
        """
        result = subprocess.run(command, 
                                shell=True, 
                                check=True, 
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE, 
                                text=True)
            
        if result.returncode == 0:
            print("Output:")
            print(result.stdout)
        else:
            print("Error:")
            print(result.stderr)
    except subprocess.CalledProcessError as e:
        print("Error output:", e.stderr)

def get_eval(config):
    eval_params = config['eval_params']
    pipeline_params = config['pipeline_params']
    management = config['management']
    try:
        script = 'eval/ruler_utils/eval/evaluate.py'
        command = f"""python {script} \
            --data_dir  {management['output_folder_dir']} \
            --benchmark {eval_params['benchmark']} \
        """
        result = subprocess.run(command, 
                                shell=True, 
                                check=True, 
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE, 
                                text=True)
            
        if result.returncode == 0:
            print("Output:")
            print(result.stdout)
        else:
            print("Error:")
            print(result.stderr)
    except subprocess.CalledProcessError as e:
        print("Error output:", e.stderr)

def get_pred(model, tokenizer, data, pred_file, device, pipeline_params, eval_params):
    #preds = []
    if os.path.exists(pred_file):
        os.remove(pred_file)
        logger.info(f"clear old pred file in {pred_file}")
    with open(pred_file, 'at', encoding="utf-8", buffering=1) as fout:
        for json_obj in tqdm(data):
            prompt = json_obj['input']
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            pred = inference.batch_generate(input.input_ids, model, tokenizer, eval_params['max_new_tokens'])[0]
            pred = post_process(pred, pipeline_params['chat_template'])
            pred = {
                'index': json_obj['index'], 
                'pred': pred, 
                'input': prompt, 
                'outputs': json_obj['outputs'],
                'others': json_obj.get('others', {}),
                'truncation': json_obj.get('truncation', -1),
                'length': json_obj.get('length', -1)
            }
            fout.write(json.dumps(pred)+ '\n')
    logger.info(f"generate pred file to {pred_file}")

def eval_ruler(config):
    prepare_data(config)
    eval_params = config['eval_params']
    pipeline_params = config['pipeline_params']
    management = config['management']
    with open(f"{management['output_folder_dir']}/{eval_params['dataset']}/validation.jsonl") as json_file:
        data = [json.loads(line) for line in json_file]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info(f'Starting RULER evaluation via {pipeline_params["method"]}.')
    model, tokenizer = inf_llm.initialize_model_tokenizer(pipeline_config=pipeline_params)
    preds = infllm_utils.get_pred(
                                model, tokenizer, data,
                                eval_params, pipeline_params)
    pred_file = os.path.join(config['management']['output_folder_dir'], eval_params['dataset']+'.jsonl')
    if os.path.exists(pred_file):
        os.remove(pred_file)
        logger.info(f"clear old pred file in {pred_file}")
    write_manifest(pred_file, preds)
    logger.info(f"generate pred file to {pred_file}")
    get_eval(config)

