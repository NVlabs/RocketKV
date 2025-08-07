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

import argparse
import sys
import os
import copy
import json
import datetime
from zoneinfo import ZoneInfo
import random

import torch
import numpy as np
import transformers



def lock_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_desc', type=str, help='experiment description, this is purely cosmetic for readability purposes.')
    parser.add_argument('--pipeline_config_dir', type=str, help='file path of pipeline config.')
    parser.add_argument('--eval_config_dir', type=str, help='file path of eval config.')
    parser.add_argument('--output_folder_dir', default='', type=str, help='path of output model')
    parser.add_argument('--job_post_via', default='slurm_sbatch', type=str, help='slurm_sbatch or terminal')    
    parser.add_argument("--method", type=str, help='evaluation method')
    parser.add_argument("--token_budget", type=int, default=4096, help='token_budget')
    parser.add_argument("--dataset", type=str,  help='task for ruler benchmark')
    parser.add_argument("--max_seq_length", type=int, default=4000,  help='max seq length for ruler benchmark')
    parser.add_argument("--scdq_mode", action="store_true")
    # parser.add_argument("--language_model_path", type=str)
    # parser.add_argument("--tokenizer_name", type=str)
    # parser.add_argument("--context_window", type=int, default=3900)
    # parser.add_argument("--max_new_tokens", type=int, default=256)
    # parser.add_argument("--n_gpu_layers", type=int)
    args = parser.parse_args()

    if args.output_folder_dir != '':
        if args.output_folder_dir[-1] != '/':
            args.output_folder_dir  += '/'
    else:
        logger.error(f'Valid {args.output_folder_dir} is required.')

    return args


# Output in terminal and exp.log file under output_folder_dir.
def set_logger(output_folder_dir, args): 
    ct_timezone = ZoneInfo("America/Chicago")
    log_formatter = logging.Formatter("%(asctime)s | %(levelname)s : %(message)s")    
    log_formatter.converter = lambda *args: datetime.datetime.now(ct_timezone).timetuple()
    file_handler = logging.FileHandler(output_folder_dir + 'exp.log', mode = 'w')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)

    return logger


def register_args_and_configs(args):

    # Make outer output dir.
    if not os.path.isdir(args.output_folder_dir):
        os.makedirs(args.output_folder_dir)
        logger.info(f'Output folder dir {args.output_folder_dir} created.')
    else:
        logger.info(f'Output folder dir {args.output_folder_dir} already exist.')


    # Copy input eval config to output dir.
    with open(args.eval_config_dir) as eval_config_f:
        eval_config = json.load(eval_config_f)
        logger.info(f'Input eval config file {args.eval_config_dir} loaded.')
        if eval_config['eval_params'] .get('benchmark') == 'synthetic':
            eval_config['eval_params']['dataset'] = args.dataset
            if 'niah' in args.dataset:
                eval_config['eval_params']['max_new_tokens'] = 128
            elif 'vt' in args.dataset:
                eval_config['eval_params']['max_new_tokens'] = 30
            elif 'cwe' in args.dataset:
                eval_config['eval_params']['max_new_tokens'] = 120
            elif 'fwe' in args.dataset:
                eval_config['eval_params']['max_new_tokens'] = 50
            elif 'qa' in args.dataset:
                eval_config['eval_params']['max_new_tokens'] = 32
            eval_config['eval_params']['max_seq_length'] = args.max_seq_length
 
    # Make subdir under output dir to store input configs.
    input_config_subdir = eval_config['management']['sub_dir']['input_config']
    if not os.path.isdir(args.output_folder_dir + input_config_subdir):
        os.makedirs(args.output_folder_dir + input_config_subdir)
        logger.info(f'Input config subdir {args.output_folder_dir + input_config_subdir} created.')
    else:
        logger.info(f'Input config subdir {args.output_folder_dir + input_config_subdir} already exist.')

    input_eval_config_path = args.output_folder_dir + input_config_subdir + 'input_eval_config.json'
    with open(input_eval_config_path, "w+") as input_eval_config_f:
        json.dump(eval_config, input_eval_config_f, indent = 4)
        logger.info(f'Input eval config file {args.eval_config_dir} saved to {input_eval_config_path}.')

    # Copy input pipeline config to output dir.
    with open(args.pipeline_config_dir) as pipeline_config_f:
        pipeline_config = json.load(pipeline_config_f)
        logger.info(f'Input pipeline config file {args.pipeline_config_dir} loaded.')
        pipeline_config['pipeline_params']['method'] = args.method
        pipeline_config['pipeline_params']['token_budget'] = args.token_budget
        pipeline_config['pipeline_params']['scdq_mode'] = args.scdq_mode
    input_pipeline_config_path = args.output_folder_dir + input_config_subdir + 'input_pipeline_config.json'
    with open(input_pipeline_config_path, "w+") as input_pipeline_config_f:
        json.dump(pipeline_config, input_pipeline_config_f, indent = 4)
        logger.info(f'Input pipeline config file {args.pipeline_config_dir} saved to {input_pipeline_config_path}.')


    # Fuse and complete pipeline config, eval config, and args from argparser into a general config.
    config = dict()
    config['pipeline_params'] = pipeline_config['pipeline_params']
    config['eval_params'] = eval_config['eval_params']
    config['eval_results'] = dict() # processed result

    config['management'] = dict()
    config['management']['exp_desc'] = args.exp_desc
    config['management']['pipeline_config_dir'] = args.pipeline_config_dir
    config['management']['eval_config_dir'] = args.eval_config_dir
    config['management']['output_folder_dir'] = args.output_folder_dir
    config['management']['job_post_via'] = args.job_post_via
    if config['management']['job_post_via'] == 'slurm_sbatch':     # Add slurm info to config['management'] if the job is triggered via slurm sbatch.
        try:
            config['management']['slurm_info'] = register_slurm_sbatch_info()
        except Exception:
            config['management']['job_post_via'] == 'terminal'      # Likely not a slurm job, rollback to terminal post.
    config['management']['sub_dir'] = eval_config['management']['sub_dir']

    return config


def register_slurm_sbatch_info():
    slurm_job_id = os.environ['SLURM_JOB_ID']
    slurm_job_name = os.getenv('SLURM_JOB_NAME')
    slurm_out_file_dir = os.getenv('SLURM_SUBMIT_DIR') + '/slurm-' + os.getenv('SLURM_JOB_ID') + '.out'

    logger.info(f'Slurm job #{slurm_job_id} ({slurm_job_name}) running with slurm.out file at {slurm_out_file_dir}.')

    return {"slurm_job_id": slurm_job_id, "slurm_job_name": slurm_job_name, "slurm_out_file_dir": slurm_out_file_dir}



def register_result(processed_results, raw_results, config):
    
    raw_results_path = config['management']['output_folder_dir'] + config['management']['sub_dir']['raw_results']
    with open(raw_results_path, "w+") as raw_results_f:
        json.dump(raw_results, raw_results_f, indent = 4)
        logger.info(f'raw_results file saved to {raw_results_path}.')


    config['eval_results']['processed_results'] = processed_results
    logger.info('Experiments concluded, below is the raw_results: ')
    logger.info(json.dumps(raw_results, indent=4))

    logger.info('##### And below is the processed_results: #####')
    logger.info(json.dumps(config['eval_results']['processed_results'], indent=4))


def register_exp_time(start_time, end_time, config):
    config['management']['start_time'] = str(start_time)
    config['management']['end_time'] = str(end_time)
    config['management']['exp_duration'] = str(end_time - start_time)


def register_output_config(config):
    output_config_path = config['management']['output_folder_dir'] + config['management']['sub_dir']['output_config']
    with open(output_config_path, "w+") as output_config_f:
        json.dump(config, output_config_f, indent = 4)
        logger.info(f'output_config file saved to {output_config_path}.')
