SOURCE_PATH="/data/shared/yuansui/rog"
export HF_HOME=${SOURCE_PATH}/.cache/huggingface

for dataset in "RoG-webqsp" "RoG-cwq"; do
   python gpt_mcq_sandbox.py --sample 10 --d ${dataset} --model_name "gpt-3.5-turbo-0125" --whether_filtering true
done