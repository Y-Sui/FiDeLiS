SOURCE_PATH="/data/shared/yuansui/rog"
export HF_HOME=${SOURCE_PATH}/.cache/huggingface

model_name="gpt-4-0125-preview"
model_name="gpt-3.5-turbo-0125"

for dataset in "RoG-webqsp"; do
   python gpt_mcq_sandbox.py --sample 3 --d ${dataset} --model_name "gpt-3.5-turbo-0125" --whether_filtering true
done