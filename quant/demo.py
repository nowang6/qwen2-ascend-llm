#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2010-2024. All rights reserved
import os
import sys
import argparse
from dopt.log import Logger
from dopt.dopt_llm.opt_main import parse_args, generate_quant_config, llm_quant_pipline
if __name__ == '__main__':
    args = parse_args()
 
    optimize_info = {
        "model_path" :  "download/Qwen2_0.5B_Instruct",
        "config_yaml":  "quant/config.yaml",
        "dopt_config":  args.dopt_config,
        "output_dir" :  "output",
        "hf_type_save": args.hf_type_save,
        "quant_stage" : args.quant_stage,
        "group_size" :  args.group_size,
        "act_bits":     args.act_bits,
        "w_bits":       args.w_bits,
    }
    ### step2 run llm model Quantification optimization process
    llm_quant_pipline(**optimize_info)