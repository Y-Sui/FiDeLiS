import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
from utils import *
from transformers import AutoTokenizer
import datasets


N_CPUS = (
    int(os.environ['SLURM_CPUS_PER_TASK']) if 'SLURM_CPUS_PER_TASK' in os.environ else 1
)

# V1
# INSTRUCTION = """
#                     User query: {question} \n
                    
#                     To proceed, you must identify the most relevant reasoning path based on the current reasoning steps: {reasoning_path_str} \n
                    
#                     Please review the following options and select the most appropriate reasoning path for the query, also including the corresponding entity where applicable: \n
                    
#                     {options_str} \n
                    
#                     After evaluating the options, please provide only the index of the selected reasoning path. If the final entity from the current reasoning steps directly answers the query, respond with option 0: EOS, End of Selection.
#                 """
                
                
# V2    
INSTRUCTION = """
                    [INST] <<SYS>>
                    <</SYS>>Your goal is to find a path from a knowledge graph that is useful for answering the following question:  {question} \n
                    *IF AT START*: To proceed, the starting entity is {entity}.\n
                    *IF IN PROGRESS*: The current reasoning path that has been constructed so far is {reasoning_path_str}.\n
                    Now your goal is: examine the reasoning paths to see whether the final entity in the path is the answer to the question; If so, answer EOS.
                    If not, you need to choose the next step in the reasoning: from the following triples starting from the last entity from the reasoning path, select one of them that is likely to lead to useful paths for answering the question. \n
                    {options_str} \n
                    After evaluating the options, please provide only the index of the selected reasoning path. If the final entity from the current reasoning path directly answers the query, respond with 'EOS'.[INST]
                """
SEP = '<SEP>'
BOP = '<PATH>'
EOP = '</PATH>'


def main(args):
    save_dir = "datasets/joint_training/align"
    # prompt_path = "prompts/llama2_chat.txt"
    if args.sample != -1:
        data_template = "datasets/AlignData/{}/{}_mcq_train_sample{}.jsonl"
    else:
        data_template = "datasets/AlignData/{}/{}_mcq_train.jsonl"
    # data_list = ['RoG-webqsp', 'RoG-cwq']
    data_name = args.d
    model_name_or_path = "meta-llama/Llama-2-7b-chat-hf"
    # prompter = InstructFormater(prompt_path)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_fast=False,
    )

    def formatting_prompts_func(example):
        output_label = rule_to_string(
            example["answer"], sep_token=SEP, bop=BOP, eop=EOP
        )
        output_text = (
            INSTRUCTION.format(
                question=example["question"],
                entity=example["starting_entity"],
                reasoning_path_str=example["history"],
                options_str=example["options"],
            )
            + " "
            + output_label
            + tokenizer.eos_token
        )
        return {"text": output_text}

    if args.sample != -1:
        data_path = data_template.format(data_name, data_name, args.sample)
        save_path = os.path.join(
            save_dir, data_name, data_name + f"_train_sample{args.sample}.jsonl"
        )
    else:
        data_path = data_template.format(data_name, data_name)
        save_path = os.path.join(save_dir, data_name, data_name + f"_train.jsonl")
    train_dataset = datasets.load_dataset(
        'json', data_files=data_path, split="train"
    )
    # if not os.path.exists(os.path.dirname(save_path)):
    #     os.makedirs(os.path.dirname(save_path))
    # with open(save_path, "w") as f:
    #     print("Processing {}...".format(data_name))
    #     print("Number of process: {}".format(N_CPUS))
    #     with mp.Pool(N_CPUS) as pool:
    #         for example in tqdm(pool.imap(formatting_prompts_func, train_dataset), total=len(train_dataset)):
    #             f.write(json.dumps(example) + "\n")

    train_dataset = train_dataset.map(
        formatting_prompts_func,
        remove_columns=["question", "answer", "history", "options", "starting_entity"],
        num_proc=N_CPUS,
    )
    train_dataset.to_json(save_path, orient="records", lines=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=int, default=-1)
    parser.add_argument('--d', '-d', type=str, default="RoG-webqsp")
    args = parser.parse_args()
    main(args)
