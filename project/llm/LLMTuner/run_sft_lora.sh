python tuner/train_lora.py \
  --model_type llama \
  --model_name_or_path /mnt/f/aigc_data/model/TinyLlama-1.1B-intermediate-step-955k-token-2T/ \
  --target_modules "q_proj,k_proj,v_proj,o_proj" \
  --data_path data/dummy.jsonl \
  --output_dir dummy_output \
  --max_length 2048 \
  --use_flash_attn False \
  --use_xformers_attn False \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_grad_norm 1.0 \
  --learning_rate 1e-4 \
  --weight_decay 0. \
  --num_train_epochs 1 \
  --lr_scheduler_type "cosine" \
  --warmup_ratio 0.03 \
  --logging_steps 5 \
  --save_strategy "steps" \
  --save_steps 10 \
  --save_total_limit 1 \
  --bf16 False \
  --tf32 False \
  --report_to "tensorboard" \
  --gradient_checkpointing True \
  --optim "paged_adamw_32bit" \
  --lora_r 64 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --bits 16 \
  --additional_trainable_params "embed,norm"


# python tuner/train_lora.py \
#   --model_type llama \
#   --model_name_or_path /path/to/Llama-2-7b-hf/ \
#   --target_modules "q_proj,k_proj,v_proj,o_proj" \
#   --data_path data/dummy.jsonl \
#   --output_dir dummy_output \
#   --max_length 2048 \
#   --use_flash_attn True \
#   --use_xformers_attn False \
#   --per_device_train_batch_size 2 \
#   --gradient_accumulation_steps 8 \
#   --max_grad_norm 1.0 \
#   --learning_rate 1e-4 \
#   --weight_decay 0. \
#   --num_train_epochs 1 \
#   --lr_scheduler_type "cosine" \
#   --warmup_ratio 0.03 \
#   --logging_steps 5 \
#   --save_strategy "steps" \
#   --save_steps 10 \
#   --save_total_limit 1 \
#   --bf16 True \
#   --tf32 True \
#   --report_to "tensorboard" \
#   --gradient_checkpointing True \
#   --optim "paged_adamw_32bit" \
#   --lora_r 32 \
#   --lora_alpha 16 \
#   --lora_dropout 0.05 \
#   --bits 16 \
#   --additional_trainable_params "embed,norm"
