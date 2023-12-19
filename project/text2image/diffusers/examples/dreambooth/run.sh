export MODEL_NAME="/mnt/f/aigc_data/model/stable-diffusion-v1-4/"
export INSTANCE_DIR="dog"
export CLASS_DIR="path-to-class-images"
export OUTPUT_DIR="path-to-save-model"

# accelerate launch --mixed_precision="fp16" train_dreambooth.py \
#   --use_8bit_adam \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --instance_data_dir=$INSTANCE_DIR \
#   --class_data_dir=$CLASS_DIR \
#   --output_dir=$OUTPUT_DIR \
#   --with_prior_preservation --prior_loss_weight=1.0 \
#   --instance_prompt="a photo of sks dog" \
#   --class_prompt="a photo of dog" \
#   --resolution=512 \
#   --train_batch_size=1 \
#   --sample_batch_size=1 \
#   --gradient_accumulation_steps=1 --gradient_checkpointing \
#   --learning_rate=5e-6 \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --num_class_images=200 \
#   --max_train_steps=800 \
#   --push_to_hub

accelerate launch --mixed_precision="fp16" train_dreambooth_lora.py \
  --use_8bit_adam \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks dog" \
  --class_prompt="a photo of dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --sample_batch_size=1 \
  --gradient_accumulation_steps=1 --gradient_checkpointing \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=800 \
  --push_to_hub