import sys
import os
import argparse
import os
import json
import networkx as nx
import re
import logging
import multiprocessing as mp
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


def query_api(prompt):
    client = OpenAI(api_key=config["OPENAI_API_KEY"])
    response = (
        client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
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


def build_graph(graph: list) -> nx.Graph:
    G = nx.Graph()
    for triplet in graph:
        h, r, t = triplet
        G.add_edge(h, t, relation=r.strip())
    return G


def main():
    input_file = os.path.join("rmanluo", "RoG-webqsp")
    output_dir = os.path.join("datasets/AlignData", "RoG-webqsp")

    # print("Save results to: ", output_dir)
    if os.path.exists(output_dir) == False:
        os.makedirs(output_dir)
        logging.info("Created directory: {}".format(output_dir))

    # Load dataset, select first 10 examples
    dataset = load_dataset(input_file, split="train").select(range(10))

    for data in tqdm(dataset):
        id = data['id']
        question = data['question']
        graph = build_graph(data['graph'])
        starting_entity = data['q_entity'][0]

        # start MCQ reasoning
        reasoning_path = []
        flag = True
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
                response = query_api(prompt)['response'].strip()
            except Exception as e:
                logging.error("Error response: {}".format(e))
                logging.error(
                    f"Failed to get response for query for error {e}: {question}"
                )
                break

            if "EOS" in response:
                logging.info(f"END of SELECTION: {process_str(reasoning_path)}")
                flag = False
            else:
                index = int(re.findall(r"[-+]?\d*\.\d+|\d+", response)[0]) - 1
                logging.info(f"RESPONSE: {response}; INDEX: {index}")

                path = path_candidates[index]
                neighbor = neighbors[index]
                reasoning_path.append(f"{starting_entity} -> {path} -> {neighbor}")
                starting_entity = neighbor

        with open(os.path.join(output_dir, f"{id}.json"), "w") as f:
            json.dump(
                {"question": question, "reasoning_path": process_str(reasoning_path)}, f
            )


if __name__ == "__main__":
    main()
