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
import json
import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, default='output/scbench/')
args = parser.parse_args()

DATA_PATH = args.output_dir
DATASETS = [
        "scbench_kv", "scbench_prefix_suffix", "scbench_vt", "scbench_repoqa", "scbench_qa_eng",
        "scbench_qa_chn", "scbench_choice_eng",  "scbench_many_shot", "scbench_summary",
        "scbench_mf", "scbench_summary_with_needles", "scbench_repoqa_and_kv"
]
TASK_DATASETS = {
    'retr_string': ["scbench_kv", "scbench_prefix_suffix", "scbench_vt"],
    'retr_semantic': ["scbench_repoqa","scbench_qa_eng", "scbench_qa_chn", "scbench_choice_eng"],
    'global': ["scbench_many_shot","scbench_mf","scbench_summary"],
    'multi_task': ["scbench_summary_with_needles","scbench_repoqa_and_kv"]
}    

def get_task_results():
    results = {}
    ind_dataset_result = {}
    task_ave_result = {}

    NA_flag = False
    # Get individual dataset result
    for dataset in DATASETS:
        file_name = os.path.join(DATA_PATH, dataset, 'output_config.json')
        if os.path.isfile(file_name):
            with open(file_name, 'r') as f:
                result = json.load(f)
                result = result['eval_results']['processed_results']
                key = list(result.keys())[0]
                val = [float(element) for element in result[key].split(',')]
                ind_dataset_result[dataset] = np.mean(val)
        else:
            ind_dataset_result[dataset] = 'N/A'
            NA_flag = True
    
    results['individual_dataset_result'] = ind_dataset_result

    # Get task-average dataset result
    for task, datasets in TASK_DATASETS.items():
        task_NA_flag = False
        task_ave_result[task] = 0
        for dataset in datasets:
            if ind_dataset_result[dataset] != 'N/A':
                task_ave_result[task] += ind_dataset_result[dataset]
            else:
                task_NA_flag = True
        if task_NA_flag:
            task_ave_result[task] = 'N/A'
        else:
            task_ave_result[task] =  np.round(task_ave_result[task] / len(datasets), decimals = 2)

    results['task_average_result'] = task_ave_result

    #Get overall average result
    if NA_flag:
        results['average_result'] = 'N/A'
    else:
        average_result = 0
        for dataset in DATASETS:
            average_result += ind_dataset_result[dataset]
        results['average_result'] = np.round(average_result / (len(DATASETS) - 1), decimals = 2)

    # Save result
    output_result_path = os.path.join(DATA_PATH, 'scbench_result_summary.json')
    with open(output_result_path, "w+") as output_file:
        json.dump(results, output_file, indent = 4)
        print(f'Complete writing task summary to {output_result_path}')

get_task_results()
