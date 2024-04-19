SOURCE_PATH="/data/shared/yuansui/rog"
export HF_HOME=${SOURCE_PATH}/.cache/huggingface

# MODEL_NAME="gpt-4-0125-preview"
# MODEL_NAME="gpt-3.5-turbo-0125"
DATASET_LIST="RoG-webqsp RoG-cwq"
# DATASET_LIST="RoG-webqsp"
RETRIEVAL_TYPE="vector_rag"

for DATA_NAME in $DATASET_LIST; do
   python run.py \
      --sample -1 \
      --d ${DATA_NAME} 
done