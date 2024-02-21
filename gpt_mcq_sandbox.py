import sys
import os
import argparse
import os
import json
import networkx as nx
import re
import logging
import multiprocessing as mp
import wandb
from tqdm import tqdm
from functools import partial
from openai import OpenAI
from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='webq.log',
    filemode='w',
)


with open("config.json", "r") as f:
    config = json.load(f)

def get_entity_edges_with_neighbors(entity: str, graph: nx.Graph) -> list:
    '''
    given an entity, find all edges and neighbors
    '''
    neighbors = []
    edges = []

    if graph.has_node(entity):
        for neighbor in graph.neighbors(entity):
            neighbors.append(neighbor)
            edges.append(graph[entity][neighbor]['relation'])

    return edges, neighbors


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
    # print("=" * 50)
    # print("RECEIVED RESPONSE: ", response)
    # outputs_file = open(fpath, "w")
    # outputs_file.write(json.dumps({
    #     'prompt': prompt,
    #     'response': response
    # }))
    return {"prompt": prompt, "response": response}


def process_str(s):
    processed = []

    for item in s:
        parts = item.split(" -> ")
        for part in parts:
            if not processed or (processed and processed[-1] != part):
                processed.append(part)

    return ' -> '.join(processed)


def extract_a_entity(s):
    return s.split(" -> ")[-1]

def build_graph(graph: list) -> nx.Graph:
    G = nx.Graph()
    for triplet in graph:
        h, r, t = triplet
        G.add_edge(h, t, relation=r.strip())
    return G


def main(args):
    settings = wandb.Settings(job_name=f"{args.d}-{args.model_name}-{args.sample}")
    wandb.init(
        project="rog-mcq",
        notes="modifying the prompt to be more informative",
        tags=["zero-shot"],
        settings=settings,
        config=args,
    )
    prediction_table = wandb.Table(columns=["id", "question", "prompt", "completion"])
    final_table = wandb.Table(columns=["id", "question", "reasoning_path", "prediction", "ground_truth", "w_o_extra"])
    
    save_list = []
    
    input_file = os.path.join(args.data_path, args.d)
    output_dir = os.path.join(args.output_path, args.model_name)

    # print("Save results to: ", output_dir)
    if os.path.exists(output_dir) == False:
        os.makedirs(output_dir)
        logging.info("Created directory: {}".format(output_dir))

    # Load dataset, select first 10 examples
    if args.sample != -1:
        dataset = load_dataset(input_file, split=args.split).select(range(args.sample))
    else:
        dataset = load_dataset(input_file, split=args.split)

    for data in tqdm(dataset):
        id = data['id']
        question = data['question']
        graph = build_graph(data['graph'])
        answer = data['a_entity']
        starting_entity = data['q_entity'][0]

        # start MCQ reasoning
        reasoning_path = []
        flag = True
        w_o_extra = False # without extra LLM calling
        while flag == True and len(reasoning_path) < 10:
            path_candidates, neighbors = get_entity_edges_with_neighbors(
                starting_entity, graph
            )

            options = []
            options.append(
                "0: EOS -> The final entity of current reasoning steps can directly answers the query. End of Selection."
            )
            i = 1
            for p, n in zip(path_candidates, neighbors):
                options.append(f"{i}: {starting_entity} -> {p} -> {n}")
                i += 1

            options_str = "\n".join(options)

            
            reasoning_path_str = process_str(reasoning_path)
            prompt = f"""
                Your goal is to find a path from a knowledge graph that is useful for answering the following question:  {question} \n
                
                *IF AT START*: To proceed, the starting entity is {starting_entity}. \n
                *IF IN PROGRESS*: The current reasoning path that has been constructed so far is {reasoning_path_str}. \n
                
                Now your goal is: examine the reasoning paths to see whether the final entity in the path is the answer to the question; If so, answer EOS. 
                If not, you need to choose the next step in the reasoning: from the following triples starting from the last entity from the reasoning path, select one of them that is likely to lead to useful paths for answering the question. \n
                
                {options_str} \n
                
                After evaluating the options, please provide only the index of the selected reasoning path. If the final entity from the current reasoning path directly answers the query, respond with 'EOS'.
            """

            try:
                response = query_api(args, prompt)['response'].strip()
                prediction_table.add_data(id, question, prompt, response)
                
            except Exception as e:
                logging.error("Error response: {}".format(e))
                logging.error(
                    f"Failed to get response for query for error {e}: {question}"
                )
                break

            if "EOS" in response:
                logging.info(f"END of SELECTION: {process_str(reasoning_path)}")
                flag = False
                w_o_extra = True
            else:
                index = int(re.findall(r"[-+]?\d*\.\d+|\d+", response)[0]) - 1
                logging.info(f"RESPONSE: {response}; INDEX: {index}")

                path = path_candidates[index]
                neighbor = neighbors[index]
                reasoning_path.append(f"{starting_entity} -> {path} -> {neighbor}")
                starting_entity = neighbor

        if w_o_extra == False:
            # reasoning based on the final MCQ reasoning path
            prompt = f"""
            Your goal is to answer the following question: {question} based on the reasoning path: {process_str(reasoning_path)}. Only return the answer without any additional information.
            """
            
            prediction = query_api(args, prompt)['response'].strip()
        else:
            prediction = extract_a_entity(process_str(reasoning_path))

        
        # save the results to a jsonl file
        save_list.append(
            {
                "question": question,
                "reasoning_path": process_str(reasoning_path),
                "prediction": prediction,
                "ground_truth": answer,
                "w_o_extra": w_o_extra
            }
        )
        final_table.add_data(id, question, process_str(reasoning_path), prediction, answer, w_o_extra)
        
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=-1)
    parser.add_argument("--data_path", type=str, default="rmanluo")
    parser.add_argument("--d", "-d", type=str, default="RoG-webqsp")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output_path", type=str, default="results")
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo-0125")
    args = parser.parse_args()
    main(args)
