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


import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, default='output/paulgraham_passkey')
args = parser.parse_args()

def get_dataset_info(model):
    if model in ['longchat-7b-v1.5-32k', 'mistral-7b-instruct-v0.2']:
        return '20480words_10x10x3_7digits'
    return '81920words_10x10x3_7digits'

def generate_plot(output_dir):
    output_dir = os.path.join(output_dir, os.listdir(output_dir)[0])
    file_path = os.path.join(output_dir, 'output_config.json')
    print(f"\nGenerating plot for file: {file_path}")
    with open(file_path, 'r') as f:
        total_file = json.load(f)

    if 'eval_results' not in total_file:
        print("No eval_results in file")
        return
    if 'processed_results' not in total_file['eval_results']:
        print("No processed_results in eval_results")
        return

    file = total_file['eval_results']['processed_results']
    plot_filename = f'result_heatmap.pdf'
    plot_path = os.path.join(output_dir, plot_filename)
    print(f"Plot will be saved as: {plot_path}")

    if not file:
        print("Empty processed_results")
        return

    keys_list = list(file.keys())
    if len(keys_list) >= 2:
        keys_to_delete = keys_list[-2:]
        for key in keys_to_delete:
            del file[key]
    else:
        return

    data = []
    for context_length in file.keys():
        for depth_lvl, results in file[context_length].items():
            if int(context_length) > 1000:
                ctx_length = str(int(context_length) // 1000) + 'K'
            else:
                ctx_length = str(np.round(int(context_length) / 1000, decimals=1)) + 'K'

            if 'exact_match' not in results:
                continue

            data.append({
                "Context Length": ctx_length,
                'Ctx_Length_Value': int(context_length),
                "Document Depth": np.round(float(depth_lvl), decimals=2),
                "Exact Match": results['exact_match']
            })

    if not data:
        return

    df = pd.DataFrame(data)
    df = df.sort_values(by=['Ctx_Length_Value', "Document Depth"])

    pivot_table = pd.pivot_table(
        df,
        values='Exact Match',
        index=['Document Depth', 'Ctx_Length_Value'],
        aggfunc='mean'
    ).reset_index()
    
    pivot_table = pivot_table.pivot(
        index="Document Depth",
        columns="Ctx_Length_Value",
        values="Exact Match"
    )

    if pivot_table.empty:
        return

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(17.5, 8))
    
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])

    NoWords = sorted(list(set([i['Context Length'] for i in data])), key=lambda x: float(x[:-1]))
    NoDepths = sorted(list(set([i['Document Depth'] for i in data])))

    heatmap = sns.heatmap(
        pivot_table,
        fmt="g",
        cbar=False,
        cmap=cmap,
        vmin=0,
        vmax=1,
        ax=ax
    )

    ax.set_xlabel('Word Count', fontsize=30)
    ax.set_ylabel('Depth', fontsize=30)

    ax.set_xticks([i + 0.5 for i in range(len(NoWords))])
    ax.set_xticklabels(NoWords, rotation=45, fontsize=30)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=30)
    ax.set_aspect('equal')

    # Add grid lines
    for i in range(len(NoWords) - 1):
        ax.axvline(i + 1, color='white', linewidth=2)
    for i in range(len(NoDepths) - 1):
        ax.axhline(i + 1, color='white', linewidth=2)

    # Add colorbar with minimal gap
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(heatmap.collections[0], cax=cax)

    plt.savefig(plot_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()


if __name__ == "__main__":
    generate_plot(args.output_dir)
