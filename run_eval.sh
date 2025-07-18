#!/bin/bash

# --- Configuration Section ---
export HIP_VISIBLE_DEVICES=1,2
NUM_GPUS=2

MODEL_DIR="/shareddata/dheyo/varunika/benchmarking/llmeval/DYNAMIC_QUANTIZATION/gguf/"
GGUF_FILE="DeepSeek-R1-Distill-Qwen-1.5B-gsm8k_v8_qat.gguf"
TOKENIZER_DIR="/shareddata/dheyo/varunika/benchmarking/llmeval/gsm8k_v8"
OUTPUT_DIR="result_logs/trial5_gsm8k_dheyo_best_v1.json"

# --- Main Evaluation Command ---
echo "Starting evaluation on GPUs : ${HIP_VISIBLE_DEVICES}...."
echo "Results will be saved in the '${OUTPUT_DIR}' directory."


# lm_eval --model hf \
accelerate launch --multi_gpu --num_processes=${NUM_GPUS} -m lm_eval \
    --model hf \
    --model_args pretrained=${MODEL_DIR},gguf_file=${GGUF_FILE},tokenizer=${TOKENIZER_DIR} \
    --tasks gsm8k \
    --device cuda:0 \
    --batch_size 4 \
    --output_path ${OUTPUT_DIR} \
    --log_samples \
    


echo "Evaluation finished. Check the '${OUTPUT_DIR}' directory for results."
