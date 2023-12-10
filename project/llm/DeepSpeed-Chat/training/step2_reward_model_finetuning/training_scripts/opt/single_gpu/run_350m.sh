#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=0
fi
mkdir -p $OUTPUT

deepspeed --num_gpus 1 main.py --model_name_or_path /mnt/f/aigc_data/model/facebook/opt-125m \
   --data_path /mnt/f/aigc_data/data_set/rm-static \
   --num_padding_at_beginning 1 --weight_decay 0.1 --dropout 0.0 --gradient_accumulation_steps 8 --zero_stage $ZERO_STAGE \
   --enable_tensorboard \
   --gradient_checkpointing \
   --tensorboard_path $OUTPUT \
   --deepspeed --output_dir $OUTPUT
