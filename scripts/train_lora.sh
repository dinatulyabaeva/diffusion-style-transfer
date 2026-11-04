#!/usr/bin/env bash
# Stage 2 only. This command has NOT been executed in the authoring environment.
set -euo pipefail
cd "$(dirname "$0")/.."
RANK="${1:-4}"
[[ "$RANK" == "4" || "$RANK" == "16" ]] || { echo "Use rank 4 or 16"; exit 2; }
[[ -f data/style/train/metadata.jsonl ]] || { echo "Prepare and review the style data first"; exit 2; }
[[ -f models.lock.json ]] || { echo "Run the baseline first to lock the base model revision"; exit 2; }
if [[ ! -d third_party/diffusers/.git ]]; then
  git clone --depth 1 --branch v0.35.1 https://github.com/huggingface/diffusers.git third_party/diffusers
fi
mkdir -p "outputs/lora_r${RANK}"
git -C third_party/diffusers rev-parse HEAD > "outputs/lora_r${RANK}/training_script_commit.txt"
REVISION=$(python -c 'import json; print(json.load(open("models.lock.json"))["stable-diffusion-v1-5/stable-diffusion-v1-5"])')
python -m pip freeze > "outputs/lora_r${RANK}/environment.txt"
# Effective batch = 1 x 4 accumulation x 1 GPU = 4. No ControlNet/IP-Adapter in training.
accelerate launch --num_processes=1 --num_machines=1 --mixed_precision=fp16 --dynamo_backend=no \
  third_party/diffusers/examples/text_to_image/train_text_to_image_lora.py \
  --pretrained_model_name_or_path=stable-diffusion-v1-5/stable-diffusion-v1-5 \
  --revision="$REVISION" \
  --train_data_dir=data/style/train --caption_column=text \
  --resolution=512 --center_crop \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --gradient_checkpointing --mixed_precision=fp16 \
  --learning_rate=1e-4 --lr_scheduler=constant --lr_warmup_steps=0 \
  --rank="$RANK" --max_train_steps=1000 --max_grad_norm=1.0 \
  --checkpointing_steps=250 --checkpoints_total_limit=4 \
  --dataloader_num_workers=2 --seed=2026 --report_to=tensorboard \
  --output_dir="outputs/lora_r${RANK}" \
  2>&1 | tee "outputs/lora_r${RANK}/train.log"
