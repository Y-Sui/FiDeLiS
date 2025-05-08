SOURCE_PATH="/data/rog"
export HF_HOME=${SOURCE_PATH}/.cache/huggingface

# MODEL_NAME="gpt-4-0125-preview"
MODEL_NAME_LIST="gpt-3.5-turbo-0125 gpt-4-turbo"
EMBEDDING_MODEL="text-embedding-3-small"
DATASET_LIST="RoG-webqsp RoG-cwq"

tmux new-session -d -s mySession

window_id=1
for DATA_NAME in $DATASET_LIST; do
   for MODEL_NAME in $MODEL_NAME_LIST; do
      tmux new-window -t mySession -n "Window$window_id" "python run.py --sample -1 --d ${DATA_NAME} --model_name ${MODEL_NAME} --embedding_model ${EMBEDDING_MODEL} --add_hop_information --top_n 30 --top_k 3 --max_length 3"
      window_id=$((window_id + 1))
   done
done

tmux attach-session -t mySession