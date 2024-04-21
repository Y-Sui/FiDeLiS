SOURCE_PATH="/data/yuansui/rog"
export HF_HOME=${SOURCE_PATH}/.cache/huggingface

# MODEL_NAME="gpt-4-0125-preview"
MODEL_NAME="gpt-3.5-turbo-0125"
EMBEDDING_MODEL="text-embedding-3-small"
DATASET_LIST="RoG-webqsp RoG-cwq"

# Generate Embeddings
for DATA_NAME in $DATASET_LIST; do
   python run.py \
      --sample 500 \
      --d ${DATA_NAME} \
      --model_name ${MODEL_NAME} \
      --embedding_model ${EMBEDDING_MODEL} \
      --add_hop_information \
      --top_n 30 \
      --top_k 3 \
      --max_length 3
done