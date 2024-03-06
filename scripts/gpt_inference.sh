SOURCE_PATH="/data/shared/yuansui/rog"
export HF_HOME=${SOURCE_PATH}/.cache/huggingface

# MODEL_NAME="gpt-4-0125-preview"
MODEL_NAME="gpt-3.5-turbo-0125"
DATASET_LIST="RoG-webqsp RoG-cwq"
RETRIEVAL_TYPE="vector_rag"
REASONING_TYPE="llm_reasoning"

for DATA_NAME in $DATASET_LIST; do
   python gpt_mcq_sandbox.py \
      --sample 50 --d ${DATA_NAME} \
      --model_name $MODEL_NAME \
      --retrieval_type $RETRIEVAL_TYPE \
      --reasoning_type $REASONING_TYPE
done