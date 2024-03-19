# React action space for mcq-style kgqa
import os
import argparse
import os
import json
import ast
import re
import logging
import multiprocessing as mp
import wandb
import numpy as np
import datetime
import random
import aiohttp
import asyncio
from src.qa_prediction.evaluate_results import eval_result
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from functools import partial
from openai import OpenAI
from datasets import load_dataset
from src import utils
from src.utils import prompt_list

# supported by langchain
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import OpenAI

now = datetime.datetime.now()
timestamp = now.strftime(f"%Y_%m_%d_%H_%M")

with open("config.json", "r") as f:
    config = json.load(f)

def prepare_dataset(sample):
    graph = utils.build_graph(sample["graph"])
    paths = utils.get_truth_paths(sample["q_entity"], sample["a_entity"], graph)
    if not paths or all(not path for path in paths): # if there is no path or all paths are empty
        sample["ground_paths"] = [["NA"]] # do not accept null sequence type, use "NA" instead
        sample["hop"] = 0
        return sample
    ground_paths = set()
    for path in paths:
        ground_paths.add(tuple([p[1] for p in path]))  # extract relation path
    sample["ground_paths"] = list(ground_paths) # [list(p) for p in ground_paths], [[], [], ...]
    sample["hop"] = len(list(ground_paths)[0])
    return sample

def process_str(s):
    processed = []

    for item in s:
        if " -> " in item:
            parts = item.split(" -> ")
            for part in parts:
                if not processed or (processed and processed[-1] != part):
                    processed.append(part)
        else:
            processed.append(item)
    return ' -> '.join(processed)


def react_chain():
   tools = [TavilySearchResults(max_results=1)]

async def main(args):
    async with aiohttp.ClientSession() as session:
        input_file = os.path.join(args.data_path, args.d)
        output_dir = os.path.join(args.output_path, args.model_name, timestamp)
        if os.path.exists(output_dir) == False:
            os.makedirs(output_dir)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename=os.path.join(output_dir,'webq.log'),
            filemode='w',
        )

        settings = wandb.Settings(job_name=f"{args.d}-{args.model_name}-{args.sample}")
        wandb.init(
            project="rog-mcq",
            notes="modifying the prompt to be more informative",
            tags=["zero-shot"],
            settings=settings,
            config=args,
        )
        prediction_table = wandb.Table(
            columns=
            ["id", 
             "question", 
             "starting_entity", 
             "prompt", 
             "completion"
            ]
        )
        final_table = wandb.Table(
            columns=[
                "id",
                "question",
                "hop",
                "q_entities", 
                "reasoning_path", 
                "ground_path", 
                "prediction_llm", 
                "prediction_direct", 
                "ground_truth"
            ]
        )
        save_list = []

        if args.sample != -1:
            dataset = load_dataset(input_file, split=args.split)
            dataset = load_dataset(input_file, split=args.split).select(random.sample(range(len(dataset)), args.sample))
            
        else:
            dataset = load_dataset(input_file, split=args.split)

        dataset = dataset.map(
            prepare_dataset,
            num_proc=args.N_CPUS,
        )
        
        dataset = dataset.filter(
            lambda x: x.get("hop") > 0, 
            num_proc=args.N_CPUS
        )

        for data in tqdm(dataset):
            save_list = react_chain()
            
        with open(os.path.join(output_dir, f"{args.d}-{args.model_name}-{args.sample}.jsonl"), "w") as f:
            for item in save_list:
                json_str = json.dumps(item)
                f.write(json_str + "\n")
                
        wandb.log(
            {
                "predictions": prediction_table,
                "reasoning_paths": final_table
            }
        )
        wandb.finish()
        
        # evaluate
        eval_result(os.path.join(output_dir, f"{args.d}-{args.model_name}-{args.sample}.jsonl"), cal_f1=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N_CPUS", type=int, default=mp.cpu_count())
    parser.add_argument("--sample", type=int, default=-1)
    parser.add_argument("--data_path", type=str, default="rmanluo")
    parser.add_argument("--d", "-d", type=str, default="RoG-webqsp")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output_path", type=str, default="results")
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo-0125")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--retrieval_type", type=str, default="vector_rag", choices=["vector_rag", "graph_rag", "graph_vector_rag", "NA"]) #TODO
    parser.add_argument("--shuffle", type=bool, default=True)
    args = parser.parse_args()
    asyncio.run(main(args))