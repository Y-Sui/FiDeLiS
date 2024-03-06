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
import datetime
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
    

def build_graph(graph: list) -> nx.Graph:
    # G = nx.Graph()
    G = nx.DiGraph()
    for triplet in graph:
        h, r, t = triplet
        G.add_edge(h, t, relation=r.strip())
    return G


def prepare_options_for_each_step(q_entity, reasoning_path, query, graph, retrieval_type = "vector_rag") -> list:
    """
    prepare options for each step of the reasoning path
    """
    if len(reasoning_path) == 0:
        raw_options = list(set(utils.get_entity_edges([q_entity], graph)))
    else:
        entire_path = apply_rules(graph, reasoning_path, [q_entity])
        next_entities = [p[-1][-1] for p in entire_path] # noted that E_t is an entity set as a head entity and a relation can usually derive multiple tail entities
        raw_options = list(set(utils.get_entity_edges(next_entities, graph))) # get edges of the entities 
    
    def vector_rag_engine(query, options, top_k=10):
        texts = [query] + options
        embeddings = get_embedding(texts)
        query_embedding = np.array(embeddings[0])
        option_embeddings = np.array(embeddings[1:])
        similarities = cosine_similarity([query_embedding], option_embeddings)
        top_k_indices = np.argsort(similarities[0])[-top_k:][::-1]
        top_k_options = [options[i] for i in top_k_indices]
        # corresponding_neighbors = [neighbors[i] for i in top_k_indices]
        
        return top_k_options
        # return [f"{i+1}: {option} -> {neighbor}" for i, (option, neighbor) in enumerate(zip(top_k_options, corresponding_neighbors))]
    
    if retrieval_type == "vector_rag":
        """
        create embedding of query and options; semantic search top-k related options; select the next step reasoning path based on the top-k options
        """
        retrieved_options = vector_rag_engine(query, raw_options) # de-duplicate the same options
        
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
    
    processed_options = []
    for i, option in enumerate(retrieved_options):
        processed_options.append(f"{i+1}: {option}")

    return processed_options

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
    final_table = wandb.Table(columns=["id", "question", "q_entities", "reasoning_path", "prediction", "ground_truth", "w_o_extra"])
    
    save_list = []

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
        pred_list = []
        reasoning_path_list = []
        reasoning_options_list = []
        w_o_extra_list = []
        
        for q_entity in data['q_entity']:
            # start MCQ reasoning
            reasoning_path = []
            reasoning_options = []
            flag = True
            w_o_extra = False # without extra LLM calling
            while flag == True and len(reasoning_path) < 10:
                options = prepare_options_for_each_step(q_entity, reasoning_path, question, graph, args.retrieval_type)
                reasoning_options.append(options)
                _new_line_char = "\n"
                if len(reasoning_path) == 0:
                    prompt = (
                        f"Your goal is to find a path from a knowledge graph that is useful for answering the following question:  {question} \n"
                        f"To proceed, the starting entity is {q_entity}. \n"
                        f"Now your goal is: choose the next step from the following reasoning paths candidates that is most likely to lead to useful reasoning paths for answering the question. \n"
                        f"{_new_line_char.join(options)} \n"
                        f"Please only return the index of the selected reasoning path."
                    )
                else:
                    prompt = (
                        f"Your goal is to find a path from a knowledge graph that is useful for answering the following question:  {question} \n"
                        f"The current reasoning path that has been constructed so far is {process_str(reasoning_path)}. \n"
                        f"Now your goal is: examine this reasoning path to see whether the reasoning path can lead to useful paths for answering the question; If none of the provided options are suitable for answering the query, respond with 'EOS'."
                        f"If not, you need to choose the next step in the reasoning: from the following triples starting from the last entity from the reasoning path, select one of them that is likely to lead to useful paths for answering the question. \n"
                        f"{_new_line_char.join(options)} \n"
                        f"After evaluating the options, please provide only the index of the selected reasoning path."
                    )  
                # If the final entity from the current reasoning path directly answers the query, respond with 'EOS'"    
                try:
                    response = query_api(args, prompt)['response'].strip()
                    prediction_table.add_data(id, question, q_entity, prompt, response)
                    
                    if "EOS" in response:
                        logging.info(f"END of SELECTION: {process_str(reasoning_path)}")
                        logging.info(f"FINAL ENTITY: {extract_a_entity(process_str(reasoning_path))}")
                        logging.info(f"GROUND TRUTH: {answer}")
                        logging.info(f"\n\n")
                        flag = False
                        w_o_extra = True
                    else:
                        index = int(re.findall(r"\b\d+\b", response)[0]) - 1
                        # logging.info(f"RESPONSE: {response}; INDEX: {index}")
                        
                        # # path = options[index] 
                        # # neighbor = utils.get_next_entity(entity, path, graph)
                        # path = path_candidates[index]
                        # neighbor = neighbors[index]
                        
                        # neighbor = neighbors[index] # tail entity may have multiple candidates
                        # reasoning_path.append(f"-> {path} -> {neighbor}")
                        
                        reasoning_path.append(options[index].replace(f"{index+1}: ", ""))
                        # entity = options[index].split(" -> ")[-1]
                        
                except Exception as e:
                    logging.error("Error response: {}".format(e))
                    logging.error(
                        f"Failed to get response for query for error {e}: {question}"
                    )
                    break

            reasoning_path_list.append(process_str(reasoning_path))
            w_o_extra_list.append(w_o_extra)
            reasoning_options_list.append(reasoning_options)
            
            if w_o_extra == False:
                # reasoning based on the final MCQ reasoning path
                prompt_extra = f"""
                Your goal is to answer the following question: {question} based on the reasoning path: {process_str(reasoning_path)}. Only return the answer without any additional information.
                """
                pred_list.append(query_api(args, prompt_extra)['response'].strip())
            else:
                # directly answer the question based on the final entity from the reasoning path
                if len(reasoning_path) > 0:
                    entire_path = apply_rules(graph, reasoning_path, [q_entity])
                    for p in entire_path:
                        if len(p) > 0:
                            pred_list.append(p[-1][-1])
        
        # save the results to a jsonl file
        save_list.append(
            {
                "id": id,
                "prompt": prompt,
                "question": question,
                "q_entities": data['q_entity'],
                "reasoning_path": reasoning_path_list,
                "reasoning_options": reasoning_options_list,
                "prediction": "\n".join(set(pred_list)), # remove duplicate predictions
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
    parser.add_argument("--retrieval_type", type=str, default="vector_rag", choices=["vector_rag", "graph_rag", "graph_vector_rag", "NA"])
    args = parser.parse_args()
    main(args)
