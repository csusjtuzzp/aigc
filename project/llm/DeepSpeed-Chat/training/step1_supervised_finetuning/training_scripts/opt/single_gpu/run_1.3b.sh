#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Note that usually LoRA needs to use larger learning rate
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=0
fi
mkdir -p $OUTPUT

set -x
#    --only_optimize_lora \

deepspeed --num_gpus 1 main.py --model_name_or_path /mnt/f/aigc_data/model/facebook/opt-350m \
   --data_path /mnt/f/aigc_data/data_set/rm-static \
   --data_output_path /mnt/f/aigc_data/data_output_path \
   --gradient_accumulation_steps 8 --lora_dim 128 --zero_stage $ZERO_STAGE \
   --print_loss \
   --gradient_checkpointing \
   --enable_tensorboard \
   --tensorboard_path $OUTPUT \
   --deepspeed --output_dir $OUTPUT

#deepspeed --num_gpus 1 main.py --model_name_or_path /mnt/f/model/facebook/opt-1.3b \
#   --gradient_accumulation_steps 8 --lora_dim 128 --zero_stage $ZERO_STAGE \
#   --enable_tensorboard \
#   --tensorboard_path $OUTPUT \
#   --deepspeed --output_dir $OUTPUT
