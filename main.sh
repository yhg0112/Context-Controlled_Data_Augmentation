#!/bin/bash
# load config

if [[ -z ${SLURM_JOB_ID+x} ]]; then
  OUT_ID=$(date +%Y-%m-%d_%H:%M:%S)
else
  OUT_ID=${SLURM_JOB_ID}
fi

# fine-tune Text Classification model with sub-sampled dataset (GPU: TC Train)
if [ "$1" = "train_classification" ]; then
  MODEL_NAME="roberta-base"
  MODEL_OUT_DIR="./models/classification/roberta-${OUT_ID}"
  python train_classification.py \
    --model_name_or_path "$MODEL_NAME" \
    --subsample_ratio 0.01 \
    --output_dir "$MODEL_OUT_DIR" \
    --evaluation_strategy steps \
    --eval_steps 1000 \
    --learning_rate 2e-5 \
    --fp16 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 64 \
    --num_train_epochs 140 \
    --weight_decay 0.01 \
    --save_total_limit 15 \
    --seed 42
fi


# mask dataset if masked_datset file does not exist (GPU: TC Inference)
if [ "$1" = "mask_dataset" ]; then
  MODEL_NAME="./models/subsample-roberta-base-3029/checkpoint-3000"
  IN_DATA="./data/yelp_subsample=0.01_seed=42_train.jsonl"
  MASK_P=0.2
  python mask_dataset.py \
    --model_name_or_path "$MODEL_NAME" \
    --in_data "$IN_DATA" \
    --out_data "./data/masked/yelp_subsample=0.01_seed=42_mask_p=${MASK_P}_masked_train.jsonl" \
    --mask_p "$MASK_P"
fi


# fine-tune Text Generation model with the masked dataset (GPU: TG Train)
if [ "$1" = "train_generation" ]; then
  MODEL_NAME="google/t5-v1_1-small"
  MODEL_OUT_DIR="./models/generation/t5-${OUT_ID}"
  python train_generation.py \
      --model_name_or_path "$MODEL_NAME" \
      --in_data "./data/masked/yelp_subsample=0.01_seed=42_mask_p=0.7_masked_train.jsonl" \
      --output_dir "$MODEL_OUT_DIR" \
      --evaluation_strategy steps \
      --eval_steps 500 \
      --learning_rate 3e-4 \
      --warmup_steps 50 \
      --per_device_train_batch_size 16 \
      --per_device_eval_batch_size 64 \
      --gradient_accumulation_steps 4 \
      --weight_decay 0.01 \
      --save_total_limit 15 \
      --num_train_epochs 50 \
      --predict_with_generate \
      --bf16 \
      --metric_for_best_model bleu \
      --seed 42
fi


# generate dataset if generated_dataset file does not exist (GPU: TG Inference)
if [ "$1" = "generate_dataset" ]; then
  MODEL_PATH="./models/generation/t5-3036/checkpoint-500"
  IN_DATA="./data/masked/yelp_subsample=0.01_seed=42_mask_p=0.7_masked_train.jsonl"
  OUT_DATA="./data/generated/yelp_subsample=0.01_seed=42_mask_p=0.7_generated_train.jsonl"
  python generate_dataset.py \
    --model_name_or_path "$MODEL_PATH" \
    --in_data "$IN_DATA" \
    --out_data "$OUT_DATA"
fi

# fine-tune TC model again (GPU: TC Train)
