MODEL_PATH=meta-llama/Llama-2-7b-chat-hf
# MODEL_PATH=realzdlegend/Llama-2-7b-chat-hf-8bit

SOURCE_PATH="/data/shared/yuansui/rog"
# DATASET_LIST="datasets/joint_training/align/RoG-cwq/RoG-cwq_train.jsonl datasets/joint_training/align/RoG-webqsp/RoG-webqsp_train.jsonl"
DATASET_LIST="${SOURCE_PATH}/joint_training/align/RoG-webqsp/RoG-webqsp_train_sample5.jsonl ${SOURCE_PATH}/joint_training/align/RoG-cwq/RoG-cwq_train_sample5.jsonl"

SAVE_NAME=rog-mcq
SAVE_PATH=${SOURCE_PATH}/save_models/${SAVE_NAME}
ADD_REL=False

export HF_HOME=${SOURCE_PATH}/.cache/huggingface

# finetune the model
accelerate launch --config_file config/deepspeed_zero3.yml src/joint_training/joint_finetuning.py \
    --data_path_list ${DATASET_LIST}  \
    --model_name_or_path ${MODEL_PATH} \
    --output_dir ${SAVE_PATH} \
    --add_rel_token ${ADD_REL} \
    --bf16 True \
    --use_peft True \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --run_name ${SAVE_NAME} \
    --overwrite_output_dir \
    --report_to "wandb" \


# # finetune the model
# CUDA_VISIBLE_DEVICES=0 python src/joint_training/joint_finetuning.py \
#     --data_path_list ${DATASET_LIST}  \
#     --model_name_or_path ${MODEL_PATH} \
#     --output_dir ${SAVE_PATH} \
#     --add_rel_token ${ADD_REL} \
#     --bf16 True \
#     --use_peft True \
#     --num_train_epochs 3 \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 16 \
#     --evaluation_strategy "no" \
#     --save_strategy "no" \
#     --save_steps 500 \
#     --save_total_limit 1 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --report_to "wandb" \
#     --gradient_checkpointing True \
#     --run_name ${SAVE_NAME} \
#     --overwrite_output_dir