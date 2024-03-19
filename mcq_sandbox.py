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

now = datetime.datetime.now()
timestamp = now.strftime(f"%Y_%m_%d_%H_%M")

with open("config.json", "r") as f:
    config = json.load(f)


class APIQueryError(Exception):
    """Exception raised when API queries fail after all retries."""
    def __init__(self, message="Failed to get a valid response after all retries"):
        self.message = message
        super().__init__(self.message)
        
async def query_api(session, args, prompt):
    api_url = f"https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {config['OPENAI_API_KEY']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": args.model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }
    
    attempt = 0
    while attempt < 5: # retry 3 times if exception occurs
        try: 
            async with session.post(api_url, headers=headers, json=payload) as response:
                response_data = await response.json()
                response_content = response_data['choices'][0]['message']['content']
                logging.info(f"PROMPT: {prompt}")
                logging.info("===" * 50)
                logging.info(f"RECEIVED RESPONSE: {response_content}")
                return {"prompt": prompt, "response": response_content}
        except Exception as e:
            logging.error(f"Error occurred: {e}")
            attempt += 1
            await asyncio.sleep(1)
    
    logging.error(f"Failed to get response for query {prompt} after 3 attempts")
    raise APIQueryError("Failed to get a valid response after all retries.")


async def get_embedding(session, texts, model="text-embedding-3-small"):
    api_url = f"https://api.openai.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {config['OPENAI_API_KEY']}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "input": texts
    }
    
    attempt = 0
    while attempt < 5: # retry 3 times if exception occurs
        try:
            async with session.post(api_url, headers=headers, json=payload) as response:
                response_data = await response.json()
                return [item['embedding'] for item in response_data['data']]
        except Exception as e:
            logging.error(f"Error occurred: {e}")
            attempt += 1
            await asyncio.sleep(1)
            
    logging.error(f"Failed to get embeddings for query {texts} after 3 attempts")
    raise APIQueryError("Failed to get a valid embedding after all retries.")


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
    

async def prepare_options_for_each_step(
    session, 
    q_entity, 
    reasoning_path, 
    query, 
    graph, 
    retrieval_type = "vector_rag"
    ) -> list:
    """
    prepare options for each step of the reasoning path
    """
    if len(reasoning_path) == 0:
        raw_options, neighbors = utils.get_entity_edges([q_entity], graph)
    else:
        if type(reasoning_path) == str:
            next_entity = reasoning_path.split("->")[-1]
        elif type(reasoning_path) == list:
            next_entity = reasoning_path[-1].split("->")[-1]
        raw_options, neighbors = utils.get_entity_edges([next_entity], graph) # get edges of the entities 
    
    async def vector_rag_engine(session, query, options, neighbors, top_k=30) -> list:
        """
        return the top-k similar options' index based on the query simalirity
        """
        try:
            texts = [query] + [opt + "->" + neighbor for opt, neighbor in zip(options, neighbors)]
            embeddings = await get_embedding(session, texts)
            query_embedding = np.array(embeddings[0])
            option_embeddings = np.array(embeddings[1:])
            similarities = cosine_similarity([query_embedding], option_embeddings)
            top_k_indices = np.argsort(similarities[0])[-top_k:][::-1] # index of the top-k similar options
            return top_k_indices
        except Exception as e:
            logging.error(f"Error occurred: {e}")
    
    if retrieval_type == "vector_rag":
        """
        create embedding of query and options; semantic search top-k related options; select the next step reasoning path based on the top-k options
        """
        retrieved_options_index = await vector_rag_engine(session, query, raw_options, neighbors) 
        retrieved_options = [raw_options[i] for i in retrieved_options_index]
        corresponding_neighbors = [neighbors[i] for i in retrieved_options_index]
        
    elif retrieval_type == "graph_rag":
        """
        get n-depth subgraphs of q_entity from KG; select the next step reasoning path based on the related subgraphs
        """
        pass
    
    elif retrieval_type == "graph_vector_rag":
        """
        do retrieval as Vector and Graph RAG; select the next step reasoning path based on both subgraphs and top-k related options 
        """
        pass
    
    processed_path_candidates = [f"{i+1}: {option}->{neighbor}" for i, (option, neighbor) in enumerate(zip(retrieved_options, corresponding_neighbors))]

    return processed_path_candidates
        

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
        
        
async def check_termination_concurrently(session, args, prompt_list, question, q_entity, reasoning_paths):
    prompts = [
        prompt_list.terminals_prune_prompt.format(
            question=question,
            q_entity=q_entity,
            reasoning_path=process_str(reasoning_path)
        ) for reasoning_path in reasoning_paths
    ]
    return await asyncio.gather(*[query_api(session, args, prompt) for prompt in prompts])


async def prepare_options_concurrently(session, q_entity, reasoning_paths, question, graph, args):
    tasks = [
        prepare_options_for_each_step(
            session,
            q_entity,
            reasoning_path,
            question,
            graph,
            args.retrieval_type
        ) for reasoning_path in reasoning_paths
    ]
    
    return await asyncio.gather(*tasks)


async def beam_search(data, session, args):
    id = data['id']
    question = data['question']
    hop = data['hop']
    graph = utils.build_graph(data['graph'])
    answer = data['a_entity']
    pred_list_direct_answer = []
    pred_list_llm_reasoning = []
    reasoning_path_list = []
    ground_reasoning_path_list = data['ground_paths'] # shortest reasoning paths from q_entity to a_entity
    _new_line_char = "\n" # for formatting the prompt
    
    logging.info(f"Processing ID: {id}")
    
    for q_entity in data['q_entity']:
        # start MCQ reasoning
        reasoning_paths = {
            k: [] for k in range(args.top_k) # for each q_entity, we retrieve top-k reasoning paths
        }
        
        # get the top-k reasoning paths for the first hop
        stage_1_path_candidates = await prepare_options_for_each_step(
            session,
            q_entity, 
            [], 
            question, 
            graph, 
            args.retrieval_type
        )
        relation_prune_prompt = prompt_list.relation_prune_prompt.format(
            top_k=args.top_k,
            question=question,
            q_entity=q_entity,
            path_candidates=_new_line_char.join(stage_1_path_candidates)
        )
        try:
            res = await query_api(session, args, relation_prune_prompt)
            stage_1_response = res['response'].strip()
            # prediction_table.add_data(id, question, q_entity, relation_prune_prompt, stage_1_response)
            stage_1_response = ast.literal_eval(stage_1_response) # use ast.literal_eval to convert string to list
            if type(stage_1_response) == list:
                stage_1_index = [int(re.findall(r"\b\d+\b", str(stage_1_res))[0]) - 1 for stage_1_res in stage_1_response] # get the index of the selected reasoning paths
                for k in range(args.top_k):
                    reasoning_paths[k].append(stage_1_path_candidates[stage_1_index[k]].replace(f"{stage_1_index[k]+1}: ", ""))
            logging.info(
                "Stage 1 Reasoning Path:\n {}".format(reasoning_paths.values())
            )
            
        except Exception as e:
            logging.error("Error response: {}".format(e))
            logging.error(
                f"Error occurred at stage 1"
                f"Failed to get response for query {question} due to error {e}"
                f"Error ID: {id}"
            )
            break
        
        round = 0
        while True:
            temp_reasoning_paths = []
            
            # concurrent reasoning path termination check
            termination_checks = await check_termination_concurrently(
                session, 
                args, 
                prompt_list, 
                question, 
                q_entity, 
                [rpth for rpth in reasoning_paths.values()]
            )
            
            to_extend, paths_to_extend = [], []
            for k, check in enumerate(termination_checks):
                next_entity = reasoning_paths[k][-1].split("->")[-1]
                if "No" in check.get('response') and not utils.get_entity_edges([next_entity], graph):
                    paths_to_extend.append(reasoning_paths[k]) # [...]
                    to_extend.append(k)
            
            logging.info(
                "Paths to extend: {}".format(paths_to_extend)
            )
            
            if paths_to_extend:
                # prepare options for each path concurrently
                path_candidates = await prepare_options_concurrently(
                    session, 
                    q_entity, 
                    paths_to_extend, 
                    question, 
                    graph, 
                    args
                )
                
                # construct new reasoning paths (path_to_extend + path_candidates)
                for i in range(len(paths_to_extend)):
                    for j in range(len(path_candidates[i])):
                        path_candidates[i][j] = path_candidates[i][j].replace(f"{j+1}: ", "")
                        temp_reasoning_paths.append(paths_to_extend[i][0] + "->" + path_candidates[i][j])
            else:
                break
            
            # add index for the temp_reasoning_paths
            temp_reasoning_paths = [f"{i+1}: {rpth}" for i, rpth in enumerate(temp_reasoning_paths)]
            
            
            
            # from the new reasoning paths, select the beam_width valid reasoning paths
            beam_search_prompt = prompt_list.beam_search_prompt.format(
                beam_width=args.top_k,
                question=question,
                reasoning_paths=_new_line_char.join(temp_reasoning_paths)
            )
            
            try:
                res = await query_api(session, args, beam_search_prompt)
                beam_search_response = res['response'].strip()
                beam_search_response = ast.literal_eval(beam_search_response) # use ast.literal_eval to convert string to list
                if type(beam_search_response) == list:
                    beam_search_index = [int(re.findall(r"\b\d+\b", str(beam_search_res))[0]) - 1 for beam_search_res in beam_search_response] # [0, 1, 2, 4]
                    
                    # increment the round or any other termination logic
                    round += 1
                    beam_search_res = []
                    for i in beam_search_index:
                        temp_reasoning_paths[i] = temp_reasoning_paths[i].replace(f"{i+1}: ", "")
                        beam_search_res.append(temp_reasoning_paths[i])
                    for i, j in enumerate(to_extend):
                        reasoning_paths[j] = [beam_search_res[i]] # fix a bug here, since reasoning_paths[j] is a list, we need to assign a list to it
                
                logging.info(
                    "Stage {} Reasoning Path:\n {}".format(round, (r + '\n' for _, r in reasoning_paths.items()))
                )
                        
            except Exception as e:
                logging.error("Error response: {}".format(e))
                logging.error(
                    f"Error occurred at stage {round}"
                    f"Failed to get response for query {question} due to error {e}"
                    f"Error ID: {id}"
                )
                break
            
            if round == 4:
                break
        
        # print("question: ", question)    
        # print("reasoning paths: ", reasoning_paths)

        for k in range(args.top_k):
            # answer the question based on the reasoning path
            reasoning_paths[k] = process_str(reasoning_paths[k])
            # reasoning based on the final MCQ reasoning path
            reasoning_prompt = prompt_list.reasoning_prompt.format(
                question=question,
                q_entity=q_entity,
                reasoning_path=reasoning_paths[k]
            )
            # collect both reasoning path and direct answer
            res = await query_api(session, args, reasoning_prompt)
            pred_list_llm_reasoning.append(res['response'].strip())
            if len(reasoning_paths[k]) > 0:
                pred_list_direct_answer.append(reasoning_paths[k].split("->")[-1])
            
        reasoning_path_list.append(reasoning_paths)
    
    # save the results to a jsonl file
    res =  {
            "id": id,
            "question": question,
            "hop": hop,
            "q_entities": data['q_entity'],
            "reasoning_path": reasoning_path_list,
            "ground_path": ground_reasoning_path_list,
            "prediction_llm": "\n".join(set(pred_list_llm_reasoning)), # remove duplicate predictions
            "prediction_direct_answer": "\n".join(set(pred_list_direct_answer)),
            "ground_truth": answer,
        }
    return res


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
            # dataset = load_dataset(input_file, split=args.split).select(random.sample(range(len(dataset)), args.sample))
            dataset = load_dataset(input_file, split=args.split).select(range(args.sample))            
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
            try:
                res = await beam_search(data, session, args) # run the beam search for each sample
                f = open(os.path.join(output_dir, f"{args.d}-{args.model_name}-{args.sample}.jsonl"), "a")
                json_str = json.dumps(res)
                f.write(json_str + "\n")
                
                final_table.add_data(
                    res['id'],
                    res['question'],
                    res['hop'],
                    res['q_entities'],
                    res['reasoning_path'],
                    res['ground_path'],
                    res['prediction_llm'],
                    res['prediction_direct_answer'],
                    res['ground_truth']
                )
            except Exception as e:
                logging.error("Error occurred: {}".format(e))
                f = open(os.path.join(output_dir, "error_sample.jsonl"), "a")
                json_str = json.dumps({"id": data['id'], "error": str(e)})
                f.write(json_str + "\n")
                continue
            
        # evaluate
        eval_result(os.path.join(output_dir, f"{args.d}-{args.model_name}-{args.sample}.jsonl"), cal_f1=True)
        
        wandb.log(
            {
                "reasoning_paths": final_table
            }
        )
        wandb.finish()

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