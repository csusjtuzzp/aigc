# export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="./output"
export MODEL_DIR=/mnt/f/aigc_data/model/stable-diffusion-v1-4/

#  --enable_xformers_memory_efficient_attention \
accelerate launch train_controlnet.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=/mnt/f/aigc_data/data_set/fill50k \
 --use_8bit_adam \
 --resolution=512 \
 --validation_image "./conditioning_image_1.png" "./conditioning_image_2.png" \
 --validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
 --train_batch_size=1 \
 --gradient_accumulation_steps=4 \
 --gradient_checkpointing \
 --set_grads_to_none \
 --mixed_precision fp16