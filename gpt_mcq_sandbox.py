import sys
import os
import argparse
import os
import json
import ast
import networkx as nx
import re
import logging
import multiprocessing as mp
import wandb
import numpy as np
import datetime
import random
from src.qa_prediction.evaluate_results import eval_result
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from functools import partial
from openai import OpenAI
from datasets import load_dataset
from src import utils

now = datetime.datetime.now()
timestamp = now.strftime(f"%Y_%m_%d_%H_%M")

with open("config.json", "r") as f:
    config = json.load(f)


def get_embedding(texts, model="text-embedding-3-small"):
    client = OpenAI(api_key=config["OPENAI_API_KEY"])
    response = client.embeddings.create(
        model=model, input=texts
    )
    return [item.embedding for item in response.data]


def query_api(args, prompt):
    client = OpenAI(api_key=config["OPENAI_API_KEY"])
    response = (
        client.chat.completions.create(
            model=args.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            # max_tokens=500,
        )
        .choices[0]
        .message.content
    )
    logging.info(f"PROMPT: {prompt}")
    logging.info("===" * 50)
    logging.info(f"RECEIVED RESPONSE: {response}")
    return {"prompt": prompt, "response": response}


def query_api_with_top_k_completions(args, prompt):
    #TODO: implement this function
    """
    generate top-k completions for the given prompt based on the calculated log-likelihood ratio for each token
    """
    client = OpenAI(api_key=config["OPENAI_API_KEY"])
    response = (
        client.chat.completions.create(
            model=args.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            n=args.n_beam,
        )
        .choices[0]
        .message.content
    )


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


def extract_a_entity(s):
    return s.split(" -> ")[-1]


def apply_rules(graph, rules, source_entities):
    results = []
    for entity in source_entities:
        res = utils.bfs_with_rule(graph, entity, rules)
        results.extend(res)
    return results


def prepare_dataset(sample):
    graph = utils.build_graph(sample["graph"])
    paths = utils.get_truth_paths(sample["q_entity"], sample["a_entity"], graph)
    if not paths:
        return sample
    ground_paths = set()
    for path in paths:
        ground_paths.add(tuple([p[1] for p in path]))  # extract relation path
    sample["ground_paths"] = list(ground_paths)
    sample["hop"] = len(list(ground_paths)[0])
    return sample
    

def prepare_options_for_each_step(q_entity, reasoning_path, query, graph, retrieval_type = "vector_rag") -> list:
    """
    prepare options for each step of the reasoning path
    """
    if len(reasoning_path) == 0:
        raw_options, neighbors = utils.get_entity_edges([q_entity], graph)
    else:
        entire_path = apply_rules(graph, reasoning_path, [q_entity])
        next_entities = [p[-1][-1] for p in entire_path] # noted that E_t is an entity set as a head entity and a relation can usually derive multiple tail entities
        raw_options, neighbors = utils.get_entity_edges(next_entities, graph) # get edges of the entities 
    
    def vector_rag_engine(query, options, neighbors, top_k=10) -> list:
        """
        return the top-k similar options' index based on the query simalirity
        """
        texts = [query] + [opt + "->" + neighbor for opt, neighbor in zip(options, neighbors)]
        embeddings = get_embedding(texts)
        query_embedding = np.array(embeddings[0])
        option_embeddings = np.array(embeddings[1:])
        similarities = cosine_similarity([query_embedding], option_embeddings)
        top_k_indices = np.argsort(similarities[0])[-top_k:][::-1] # index of the top-k similar options

        return top_k_indices
    
    if retrieval_type == "vector_rag":
        """
        create embedding of query and options; semantic search top-k related options; select the next step reasoning path based on the top-k options
        """
        retrieved_options_index = vector_rag_engine(query, raw_options, neighbors) 
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
    
    processed_options = [f"{i+1}: {option}" for i, option in enumerate(retrieved_options)]

    return processed_options, corresponding_neighbors

def main(args):
    input_file = os.path.join(args.data_path, args.d)
    output_dir = os.path.join(args.output_path, args.model_name, timestamp)
    # print("Save results to: ", output_dir)
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
    prediction_table = wandb.Table(columns=["id", "question", "starting_entity", "prompt", "completion"])
    final_table = wandb.Table(columns=["id", "question", "hop", "q_entities", "reasoning_path", "ground_path", "prediction_llm", "prediction_direct", "ground_truth"])
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
    
    dataset = dataset.filter(lambda x: "ground_paths" in x.keys(), num_proc=args.N_CPUS) # filter out samples without ground truth reasoning paths / negative samples cannot find the ground truth reasoning paths

    for data in tqdm(dataset):
        id = data['id']
        question = data['question']
        hop = data['hop']
        graph = utils.build_graph(data['graph'])
        answer = data['a_entity']
        pred_list = []
        pred_list_direct_answer = []
        pred_list_llm_reasoning = []
        reasoning_path_list = []
        ground_reasoning_path_list = data['ground_paths'] # shortest reasoning paths from q_entity to a_entity
        _new_line_char = "\n" # for formatting the prompt
        
        logging.info(f"Processing ID: {id}")
        
        for q_entity in data['q_entity']:
            # start MCQ reasoning
            reasoning_path = [[] for _ in range(args.top_k)] # for each q_entity, we retrieve top-k reasoning paths
            stage_1_options, stage_1_neighbors = prepare_options_for_each_step(
                q_entity, 
                [], 
                question, 
                graph, 
                args.retrieval_type
            )
            stage_1_path_candidates = [opt + "->" + nei for opt, nei in zip(stage_1_options, stage_1_neighbors)]
            relation_prune_prompt = (
                f"Your goal is to retrieve {args.top_k} paths from the following candidates that contribute to answering the following question. \n" 
                f"Question: {question} \n"
                f"Starting entity: {q_entity} \n"
                f"Path candidates: \n{_new_line_char.join(stage_1_path_candidates)} \n"
                f"Please only return the index of the {args.top_k} selected reasoning path in a list. (for example [1, 2, 3])"
            )
            try:
                stage_1_response = query_api(args, relation_prune_prompt)['response'].strip() 
                prediction_table.add_data(id, question, q_entity, relation_prune_prompt, stage_1_response)
                stage_1_response = ast.literal_eval(stage_1_response) # use ast.literal_eval to convert string to list
                if type(stage_1_response) == list:
                    stage_1_index = [int(re.findall(r"\b\d+\b", str(stage_1_res))[0]) - 1 for stage_1_res in stage_1_response] # get the index of the selected reasoning paths
                    for k in range(args.top_k):
                        reasoning_path[k].append(stage_1_path_candidates[stage_1_index[k]].replace(f"{stage_1_index[k]+1}: ", ""))
                            
            except Exception as e:
                logging.error("Error response: {}".format(e))
                logging.error(
                    f"Error occurred at stage 1"
                    f"Failed to get response for query due to error {e}: {question}"
                )
                break
            
            # evaluate whether the reasoning path is sufficient to answer the question
            for k in range(args.top_k):
                logging.info(f"Reasoning path {k+1}: {reasoning_path[k]}")
                termination_check_prompt = (
                    f"Your goal is to answer whether it's sufficient for you to answer the question with the following reasoning path and your knowledge \n"
                    f"Question: {question} \n"
                    f"Reasong paths: {q_entity} -> {process_str(reasoning_path[k])} \n"
                    f"If it is sufficient to answer the question, respond with 'Yes'; otherwise, respond with 'No'."
                )
                termination_check = query_api(args, termination_check_prompt)['response'].strip()
                
                if "Yes" in termination_check:
                    flag = False
                elif "No" in termination_check:
                    flag = True
                    
                while flag == True and len(reasoning_path[k]) < 4:
                    stage_n_options, stage_n_neighbors = prepare_options_for_each_step(
                        q_entity, 
                        reasoning_path[k], 
                        question, 
                        graph, 
                        args.retrieval_type
                    )
                    stage_n_path_candidates = [opt + "->" + nei for opt, nei in zip(stage_n_options, stage_n_neighbors)]
                    relation_prune_prompt = (
                        f"Your goal is to find a path from a knowledge graph that is useful for answering the following question. \n" 
                        f"You are asked to consider the reasoning path that has been constructed so far and choose the next step from the following path candidates that is most likely to lead to useful reasoning paths for answering the question. \n"
                        f"Question: {question} \n"
                        f"Reasoning paths: {q_entity} -> {process_str(reasoning_path[k])} \n"
                        f"Path candidates: \n{_new_line_char.join(stage_1_path_candidates)} \n"
                        f"Please only return the index of the selected reasoning path."
                    )   
                    try:
                        stage_n_response = query_api(args, relation_prune_prompt)['response'].strip()
                        prediction_table.add_data(id, question, q_entity, relation_prune_prompt, stage_n_response)
                        index = int(re.findall(r"\b\d+\b", stage_n_response)[0]) - 1 # get the index of the selected reasoning path
                        reasoning_path[k].append(stage_n_path_candidates[index].replace(f"{index+1}: ", ""))
                    except Exception as e:
                        logging.error("Error response: {}".format(e))
                        logging.error(
                            f"Error occurred at stage {k+1}"
                            f"Failed to get response for query due to error {e}: {question}"
                        )
                        break
                    
                    termination_check_prompt = (
                        f"Your goal is to answer whether it's sufficient for you to answer the question with the following reasoning path and your knowledge \n"
                        f"Question: {question} \n"
                        f"Reasong paths: {q_entity} -> {process_str(reasoning_path[k])} \n"
                        f"If it is sufficient to answer the question, respond with 'Yes'; otherwise, respond with 'No'."
                    )
                    
                    termination_check = query_api(args, termination_check_prompt)['response'].strip()
                    if "Yes" in termination_check:
                        flag = False
                    elif "No" in termination_check:
                        flag = True

                # answer the question based on the reasoning path
                reasoning_path[k] = process_str(reasoning_path[k])
                if args.reasoning_type == "llm_reasoning":
                    # reasoning based on the final MCQ reasoning path
                    reasoning_prompt = (
                        f"Your goal is to answer the following question based on the reasoning path and your knowledge. \n"
                        f"Question: {question} \n"
                        f"Reasoning path: {q_entity} -> {reasoning_path[k]} \n"
                        f"Only return the answer to the question."
                    )
                    pred_list.append(query_api(args, reasoning_prompt)['response'].strip())
                elif args.reasoning_type == "direct_answer":
                    # directly answer the question based on the final entity from the reasoning path
                    if len(reasoning_path[k]) > 0:
                        entire_path = apply_rules(graph, reasoning_path[k], [q_entity])
                        for p in entire_path:
                            if len(p) > 0:
                                pred_list.append(p[-1][-1])
                elif args.reasoning_type == "both":
                    # collect both reasoning path and direct answer
                    reasoning_prompt = (
                        f"Your goal is to answer the following question based on the reasoning path and your knowledge. \n"
                        f"Question: {question} \n"
                        f"Reasoning path: {q_entity} -> {reasoning_path[k]} \n"
                        f"Only return the answer to the question."
                    )
                    pred_list_llm_reasoning.append(query_api(args, reasoning_prompt)['response'].strip())
                    if len(reasoning_path[k]) > 0:
                        entire_path = apply_rules(graph, reasoning_path[k], [q_entity])
                        for p in entire_path:
                            if len(p) > 0:
                                pred_list_direct_answer.append(p[-1][-1])
                else:
                    raise ValueError("Invalid reasoning type")
                
            reasoning_path_list.append(reasoning_path)
        
        # save the results to a jsonl file
        if pred_list == []:
            save_list.append(
                {
                    "id": id,
                    "prompt": relation_prune_prompt,
                    "question": question,
                    "hop": hop,
                    "q_entities": data['q_entity'],
                    "reasoning_path": reasoning_path_list,
                    "ground_path": ground_reasoning_path_list,
                    "prediction_llm": "\n".join(set(pred_list_llm_reasoning)), # remove duplicate predictions
                    "prediction_direct_answer": "\n".join(set(pred_list_direct_answer)),
                    "ground_truth": answer,
                }
            )
            final_table.add_data(id, question, hop, data['q_entity'], reasoning_path_list, ground_reasoning_path_list, pred_list_llm_reasoning, pred_list_direct_answer, answer)
        else:
            save_list.append(
                {
                    "id": id,
                    "prompt": relation_prune_prompt,
                    "question": question,
                    "hop": hop,
                    "q_entities": data['q_entity'],
                    "reasoning_path": reasoning_path_list,
                    "ground_path": ground_reasoning_path_list,
                    "prediction": "\n".join(set(pred_list)), # remove duplicate predictions
                    "ground_truth": answer,
                }
            )
            final_table.add_data(id, question, hop, data['q_entity'], reasoning_path_list, pred_list, [],  answer)
        
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
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--retrieval_type", type=str, default="vector_rag", choices=["vector_rag", "graph_rag", "graph_vector_rag", "NA"])
    parser.add_argument("--reasoning_type", type=str, default="direct_answer", choices=["direct_answer", "llm_reasoning", "both"])
    args = parser.parse_args()
    main(args)
