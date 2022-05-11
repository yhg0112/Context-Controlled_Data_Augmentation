#!/bin/bash
# load config

if [[ -z ${SLURM_JOB_ID+x} ]]; then
  OUT_ID=$(date +%Y-%m-%d_%H:%M:%S)
else
  OUT_ID=${SLURM_JOB_ID}
fi

# sub-sample dataset (no gpu)
if [ "$1" = "subsample_dataset" ]; then
  python subsample_yelp_data.py --seed 42 --subsample_ratio 0.02
fi


# fine-tune Text Classification model with sub-sampled dataset (GPU: TC Train)
if [ "$1" = "train_classification" ]; then
  MODEL_NAME="roberta-base"
  MODEL_OUT_DIR="./models/classification/roberta-${OUT_ID}"
  # TRAIN_DATA_PATH="./data/yelp_full_train.jsonl"
  TRAIN_DATA_PATH="./data/yelp_subsample=0.01_seed=42_train.jsonl"
  python train_classification.py \
    --model_name_or_path "$MODEL_NAME" \
    --train_data_paths "$TRAIN_DATA_PATH" \
    --test_data_path ./data/yelp_full_test.jsonl \
    --subsample_eval_set_size 500 \
    --output_dir "$MODEL_OUT_DIR" \
    --learning_rate 2e-5 \
    --fp16 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --metric_for_best_model 'accuracy' \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 64 \
    --eval_every_n_epochs 1 \
    --early_stopping_patience 10 \
    --num_train_epochs 100 \
    --weight_decay 0.01 \
    --save_total_limit 1 \
    --seed 42
fi


# mask dataset if masked_datset file does not exist (GPU: TC Inference)
if [ "$1" = "mask_dataset" ]; then
  # MODEL_NAME="./models/subsample-roberta-base-3029/checkpoint-3000"
  MODEL_NAME="./models/classification/roberta-3536/checkpoint-303"
  IN_DATA="./data/yelp_subsample=0.01_seed=42_train.jsonl"
  MASK_P="$2"
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
  MASK_P=0.3
  TEST_SPLIT_SIZE="$2"
  python train_generation.py \
      --model_name_or_path "$MODEL_NAME" \
      --in_data "./data/masked/yelp_subsample=0.01_seed=42_mask_p=${MASK_P}_masked_train.jsonl" \
      --output_dir "$MODEL_OUT_DIR" \
      --test_split_size "$TEST_SPLIT_SIZE" \
      --bf16 \
      --seed 42 \
      --learning_rate 3e-4 \
      --weight_decay 0.01 \
      --per_device_train_batch_size 32 \
      --gradient_accumulation_steps 2 \
      --per_device_eval_batch_size 64 \
      --evaluation_strategy steps \
      --warmup_steps 50 \
      --num_train_epochs 100 \
      --eval_every_n_epochs 1 \
      --early_stopping_patience 10 \
      --predict_with_generate \
      --load_best_model_at_end \
      --metric_for_best_model 'bleu' \
      --save_total_limit 1
fi


# generate dataset if generated_dataset file does not exist (GPU: TG Inference)
if [ "$1" = "generate_dataset" ]; then
  # MODEL_PATH="./models/generation/t5-3036/checkpoint-500"
  # SETUP=0.3 3588 0.2
  # SETUP=0.3 3589 0.3
  # SETUP=0.3 3590 0.4
  # SETUP=0.3 3591 0.5
  read -r MASK_P MODEL_ID TEST_SPLIT_SIZE <<< "${SETUP}"
  MODEL_PATH="./models/generation/t5-${MODEL_ID}/checkpoint-best"
  IN_DATA="./data/masked/yelp_subsample=0.01_seed=42_mask_p=${MASK_P}_masked_train.jsonl"
  OUT_DATA="./data/generated/yelp_subsample=0.01_seed=42_mask_p=${MASK_P}_generated.jsonl"
  OUT_OPP_DATA="./data/generated/yelp_subsample=0.01_seed=42_mask_p=${MASK_P}_generated_opp.jsonl"
  python generate_dataset.py \
    --model_name_or_path "$MODEL_PATH" \
    --in_data "$IN_DATA" \
    --out_data "$OUT_DATA" \
    --out_opp_data "$OUT_OPP_DATA" \
    --test_split_size "$TEST_SPLIT_SIZE" \
    --seed 42
fi

# fine-tune TC model again (GPU: TC Train)

if [ "$1" = "train_classification_augmented" ]; then
  MODEL_NAME="roberta-base"
  MODEL_OUT_DIR="./models/classification/roberta-aug-${OUT_ID}"
  MASK_P="$2"
  python train_classification.py \
    --model_name_or_path "$MODEL_NAME" \
    --train_data_paths ./data/yelp_subsample=0.01_seed=42_train.jsonl \
                       "./data/generated/yelp_subsample=0.01_seed=42_mask_p=${MASK_P}_generated.jsonl" \
                       "./data/generated/yelp_subsample=0.01_seed=42_mask_p=${MASK_P}_generated_opp.jsonl" \
    --test_data_path ./data/yelp_full_test.jsonl \
    --output_dir "$MODEL_OUT_DIR" \
    --learning_rate 2e-5 \
    --fp16 \
    --evaluation_strategy 'steps' \
    --load_best_model_at_end \
    --metric_for_best_model 'accuracy' \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --eval_every_n_epochs 1 \
    --early_stopping_patience 10 \
    --num_train_epochs 100 \
    --weight_decay 0.01 \
    --save_total_limit 1 \
    --seed 42
fi
