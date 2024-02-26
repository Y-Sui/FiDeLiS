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
import numpy as np
from src.qa_prediction.evaluate_results import eval_result
from sklearn.metrics.pairwise import cosine_similarity
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
    # print("=" * 50)
    # print("RECEIVED RESPONSE: ", response)
    # outputs_file = open(fpath, "w")
    # outputs_file.write(json.dumps({
    #     'prompt': prompt,
    #     'response': response
    # }))
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


def prepare_options_for_each_step(entity, query, path_candidates, neighbors, whether_filtering):
    options = [
        "0: EOS -> The final entity of current reasoning steps can directly answers the query. End of Selection."
    ] + [f"{i}: {entity} -> {p} -> {n}" for i, (p, n) in enumerate(zip(path_candidates, neighbors), start=1)]
    
    def semantic_filtering(query, options, top_k=10):
        texts = [query] + options
        embeddings = get_embedding(texts)
        query_embedding = np.array(embeddings[0])
        option_embeddings = np.array(embeddings[1:])
        similarities = cosine_similarity([query_embedding], option_embeddings)
        top_k_indices = np.argsort(similarities[0])[-top_k:][::-1]
        
        # print(f"Top {top_k} options: {top_k_indices}")
        
        return [options[i] for i in top_k_indices]
    
    if whether_filtering:
        filtered_options = semantic_filtering(query, options[1:])
        options = [options[0]] + filtered_options
        
    options_str = "\n".join(options)
    return options_str


def main(args):
    settings = wandb.Settings(job_name=f"{args.d}-{args.model_name}-{args.sample}")
    wandb.init(
        project="rog-mcq",
        notes="modifying the prompt to be more informative",
        tags=["zero-shot"],
        settings=settings,
        config=args,
    )
    prediction_table = wandb.Table(columns=["id", "question", "starting_entity", "prompt", "completion"])
    final_table = wandb.Table(columns=["id", "question", "q_entities", "reasoning_path", "prediction", "ground_truth", "w_o_extra"])
    
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
        # starting_entity = data['q_entity'][0]

        pred_list, reasoning_path_list, w_o_extra_list = [], [], []
        for entity in data['q_entity']:
            # start MCQ reasoning
            reasoning_path = []
            flag = True
            w_o_extra = False # without extra LLM calling
            while flag == True and len(reasoning_path) < 10:
                path_candidates, neighbors = get_entity_edges_with_neighbors(
                    entity, graph
                )
                options_str = prepare_options_for_each_step(entity, question, path_candidates, neighbors, args.whether_filtering)

                reasoning_path_str = process_str(reasoning_path)
                prompt = f"""
                    Your goal is to find a path from a knowledge graph that is useful for answering the following question:  {question} \n
                    
                    *IF AT START*: To proceed, the starting entity is {entity}. \n
                    *IF IN PROGRESS*: The current reasoning path that has been constructed so far is {reasoning_path_str}. \n
                    
                    Now your goal is: examine the reasoning paths to see whether the final entity in the path is the answer to the question; If so, answer EOS. 
                    If not, you need to choose the next step in the reasoning: from the following triples starting from the last entity from the reasoning path, select one of them that is likely to lead to useful paths for answering the question. \n
                    
                    {options_str} \n
                    
                    After evaluating the options, please provide only the index of the selected reasoning path. If the final entity from the current reasoning path directly answers the query, respond with 'EOS'.
                """

                try:
                    response = query_api(args, prompt)['response'].strip()
                    prediction_table.add_data(id, question, entity, prompt, response)
                    if "EOS" in response:
                        logging.info(f"END of SELECTION: {process_str(reasoning_path)}")
                        flag = False
                        w_o_extra = True
                    else:
                        index = int(re.findall(r"\b\d+\b", response)[0]) - 1
                        logging.info(f"RESPONSE: {response}; INDEX: {index}")

                        path = path_candidates[index]
                        neighbor = neighbors[index]
                        reasoning_path.append(f"{entity} -> {path} -> {neighbor}")
                        entity = neighbor
                        
                except Exception as e:
                    logging.error("Error response: {}".format(e))
                    logging.error(
                        f"Failed to get response for query for error {e}: {question}"
                    )
                    break

            reasoning_path_list.append(process_str(reasoning_path))
            w_o_extra_list.append(w_o_extra)
            if w_o_extra == False:
                # reasoning based on the final MCQ reasoning path
                prompt = f"""
                Your goal is to answer the following question: {question} based on the reasoning path: {process_str(reasoning_path)}. Only return the answer without any additional information.
                """
                
                pred_list.append(query_api(args, prompt)['response'].strip())
            else:
                pred_list.append(extract_a_entity(process_str(reasoning_path)))

        
        # save the results to a jsonl file
        save_list.append(
            {
                "question": question,
                "q_entities": data['q_entity'],
                "reasoning_path": reasoning_path_list,
                "prediction": "\n".join(pred_list),
                "ground_truth": answer,
                "w_o_extra": w_o_extra_list
            }
        )
        final_table.add_data(id, question, data['q_entity'], reasoning_path_list, pred_list, answer, w_o_extra_list)
        
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
    parser.add_argument("--sample", type=int, default=-1)
    parser.add_argument("--data_path", type=str, default="rmanluo")
    parser.add_argument("--d", "-d", type=str, default="RoG-webqsp")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output_path", type=str, default="results")
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo-0125")
    parser.add_argument("--n_beam", type=int, default=1)
    parser.add_argument("--whether_filtering", type=bool, default=False)
    args = parser.parse_args()
    main(args)
