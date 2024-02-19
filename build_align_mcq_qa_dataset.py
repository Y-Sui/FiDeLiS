import sys
import os
import argparse
import os
import json
from datasets import load_dataset
import multiprocessing as mp
from tqdm import tqdm
from functools import partial
import networkx as nx
import re


def get_truth_paths(q_entity: list, a_entity: list, graph: nx.Graph) -> list:
    '''
    Get shortest paths connecting question and answer entities.
    '''
    # Select paths
    paths = []
    for h in q_entity:
        if h not in graph:
            continue
        for t in a_entity:
            if t not in graph:
                continue
            try:
                for p in nx.all_shortest_paths(graph, h, t):
                    paths.append(p)
            except:
                pass
    # Add relation to paths
    result_paths = []
    for p in paths:
        tmp = []
        for i in range(len(p) - 1):
            u = p[i]
            v = p[i + 1]
            tmp.append((u, graph[u][v]['relation'], v))
        result_paths.append(tmp)
    return result_paths


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

            if len(reasoning_path) > 0:
                reasoning_path_str = process_str(reasoning_path)
                prompt = f"""
                    User query: {question} \n
                    
                    To proceed, you must identify the most relevant reasoning path based on the current reasoning steps: {reasoning_path_str} \n
                    
                    Please review the following options and select the most appropriate reasoning path for the query, also including the corresponding entity where applicable: \n
                    
                    {options_str} \n
                    
                    After evaluating the options, please provide only the index of the selected reasoning path. If the final entity from the current reasoning steps directly answers the query, respond with option 0: EOS, End of Selection.
                """
            else:
                prompt = f"""
                    User query: {question} \n
                    
                    To proceed, the starting entity is {starting_entity}. \n
                    
                    Please review the following options and select the most appropriate reasoning path for the query, also including the corresponding entity where applicable: \n
                    
                    {options_str} \n
                    
                    After evaluating the options, please provide only the index of the selected reasoning path. If the final entity from the current reasoning steps directly answers the query, respond with option 0: EOS, End of Selection.
                """

            try:
                response = query_api(prompt)['response'].strip()
            except Exception as e:
                print(e)
                print(f"Failed to get response for query: {question}")
                break

            if "EOS" in response:
                # print(f"END of SELECTION: {process_str(reasoning_path)}")
                flag = False
            else:
                index = int(re.findall(r"[-+]?\d*\.\d+|\d+", response)[0]) - 1
                # print(f"RESPONSE: {response}; INDEX: {index}")

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
