for dataset in "RoG-webqsp" "RoG-cwq"; do
   python gpt_mcq_sandbox.py --sample 100 --d ${dataset} --model_name "gpt-3.5-turbo-0125"
done