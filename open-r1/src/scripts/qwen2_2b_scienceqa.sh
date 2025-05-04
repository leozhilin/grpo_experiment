# export DATA_PATH=AI4Math/MathVista
export DATA_PATH=derek-thomas/ScienceQA
export CKPT_PATH=Qwen/Qwen2-VL-2B-Instruct
export SAVE_PATH=models/ckpt/qwen2_2b_scienceqa_ckpt

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_2b_GRPO_coco_base65cate_6k.txt"

# Disable P2P and IB communication for RTX 4000 series
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
# export CUDA_VISIBLE_DEVICES=0,1

# Set Hugging Face endpoint to use mirror
export HF_ENDPOINT=https://hf-mirror.com
# export TORCH_HOME=/share/leozhilin/.torch
# export CUDA_LAUNCH_BLOCKING=1

torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    open-r1/src/open_r1/mygrpo.py \
    --output_dir ${SAVE_PATH}  \
    --model_name_or_path ${CKPT_PATH} \
    --dataset_name ${DATA_PATH} \
    --deepspeed open-r1/src/local_scripts/zero3.json \
    --max_prompt_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 2 \
    --run_name Qwen2-VL-2B_GRPO_scienceqa \
    --save_steps 100 \
    --save_only_model true \
    --num_generations 8 \
    --gradient_checkpointing True \