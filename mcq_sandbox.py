import os
import argparse
import os
import json
import logging
import multiprocessing as mp
import wandb
import numpy as np
import datetime
import copy
import re
import time
import litellm
import networkx as nx
from src.qa_prediction.evaluate_results import eval_result
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from openai import OpenAI
from litellm import completion, embedding, batch_completion # import litellm for calling multiple llms using the same input/output format 
from datasets import load_dataset, load_from_disk
from src import utils
from src.utils import prompt_list_cwq, prompt_list_webqsp

litellm.set_verbose=False
set_verbose = False
now = datetime.datetime.now()
timestamp = now.strftime(f"%Y_%m_%d_%H_%M")

with open("config.json", "r") as f:
    config = json.load(f)
    
os.environ["OPENAI_API_KEY"] = config["OPENAI_API_KEY"]
client = OpenAI()

def get_embeddings(texts: list, model="text-embedding-3-small"):
    attempt = 0
    while attempt < 5:
        try:
            _ = client.embeddings.create(model=model, input=texts)
            # return [item['embedding'] for item in _['data']]
            return [item.embedding for item in _.data]
        except Exception as e:
            logging.error(f"Error occurred: {e}")
            attempt += 1
            time.sleep(1)
            

def get_completion(args, prompt: dict):
    model = args.model_name
    messages = [
        {"role": "system", "content": prompt["system"]},
        *prompt["examples"],
        {"role": "user", "content": prompt["prompt"]}
    ]
        
    attempt = 0
    while attempt < 5:
        try:
            _ = client.chat.completions.create(model=model, messages=messages, temperature=0, top_p=0, logprobs=False)
            # return _['choices'][0]['message']['content']
            return _.choices[0].message.content
        except Exception as e:
            logging.error(f"Error occurred: {e}")
            attempt += 1
            time.sleep(1)


def get_log_probs(log_probs: list):
    scores = []
    for item in log_probs:
        top_logprobs = item[0]["top_logprobs"]
        match = False
        for i in range(len(top_logprobs)):
            if top_logprobs[i]["token"] in [" A", "A", "A "]:
                scores.append(top_logprobs[i]["logprob"])
                match = True
                break
        if not match:
            scores.append(-10000.0)
    return scores


def get_batch_completion(args, prompt: dict, input_batch: list):
    """
    for item in log_probs:
        if item["token"] == "A":
            print(item['logprob'])
    """
    
    model = args.model_name
    messages = []
    for item in input_batch:
        messages.append(
            [
                {"role": "system", "content": prompt["system"]},
                *prompt["examples"],
                {"role": "user", "content": item}
            ]
        )
    attempt = 0
    while attempt < 5: 
        try:
            _ = batch_completion(
                model=model, 
                messages=messages, 
                temperature=0, 
                top_p=0, 
                logprobs=True,
                top_logprobs=5
                )
            contents = [_[i]['choices'][0]['message']['content'] for i in range(len(_))]
            log_probs = [_[i]['choices'][0]['logprobs']['content'] for i in range(len(_))]
            return contents, log_probs
            
        except Exception as e:
            logging.error(f"Error occurred: {e}")
            attempt += 1
            time.sleep(1)


def prepare_options_for_each_step(
    args,
    starting_node: str, 
    reasoning_path: str, 
    query: str, 
    graph: nx.Graph, 
    prompt_list: object
    ) -> list:
    """
    prepare options for each step of the reasoning path propagation
    """
    if reasoning_path:
        raw_options, neighbors = utils.get_entity_edges([starting_node], graph)
    else:
        next_entity = reasoning_path.split("->")[-1]
        raw_options, neighbors = utils.get_entity_edges([next_entity], graph) # get edges of the entities 
    
    if args.squeeze:
        texts = [query] + [opt + "->" + neighbor for opt, neighbor in zip(raw_options, neighbors)]
        embeddings = get_embeddings(texts)
        query_embedding = np.array(embeddings[0])
        option_embeddings = np.array(embeddings[1:])
        similarities = cosine_similarity([query_embedding], option_embeddings)
        top_n_indices = np.argsort(similarities[0])[-args.top_n:][::-1] # index of the top-n similar options

        retrieved_options = [raw_options[i] for i in top_n_indices]
        corresponding_neighbors = [neighbors[i] for i in top_n_indices]
        processed_path_candidates = [f"{option}->{neighbor}" for option, neighbor in zip(retrieved_options, corresponding_neighbors)]
        processed_path_candidates = [reasoning_path + "->" + candidate if reasoning_path else starting_node + "->" + candidate for candidate in processed_path_candidates]
    
    else:
        processed_path_candidates = [f"{option}->{neighbor}" for option, neighbor in zip(raw_options, neighbors)]
        processed_path_candidates = [reasoning_path + "->" + candidate if reasoning_path else starting_node + "->" + candidate for candidate in processed_path_candidates]
    
    if args.strategy == "continuous_rating":
        deductive_prompts, self_confidence_prompts = [], []
        for candidate in processed_path_candidates:
            deductive_prompts.append(
                prompt_list.deductive_verifier_prompt["prompt"].format(
                    question=query,
                    reasoning_path=reasoning_path if reasoning_path else starting_node,
                    reasoning_step=candidate
                )
            )
            self_confidence_prompts.append(
                prompt_list.self_confidence_prompt["prompt"].format(
                    question=query,
                    reasoning_path=reasoning_path + "->" + candidate if reasoning_path else starting_node + "->" + candidate,
                )
            )
            
        # call the batch completion function to get the log probabilities of the reasoning paths
        _, log_probs_deductive_scores = get_batch_completion(
            args=args,
            prompt=prompt_list.deductive_verifier_prompt,
            input_batch=deductive_prompts
        )
        
        _, log_probs_self_confidence_scores = get_batch_completion(
            args=args,
            prompt=prompt_list.self_confidence_prompt,
            input_batch=self_confidence_prompts
        )
        
        deductive_scores = get_log_probs(log_probs_deductive_scores)
        self_confidence_scores = get_log_probs(log_probs_self_confidence_scores)
        
        return processed_path_candidates, deductive_scores, self_confidence_scores

    elif args.strategy == "discrete_rating":
        return processed_path_candidates
    


def find_top_k_candidates(
    args,
    next_step_candidates: list, 
    question: str = "", 
    plan_context: str = "",
    cur_rpth: str = ""
):
    if args.strategy == "continuous_rating":
        combined_scores = [0.5 * (item[1] + item[2]) for item in next_step_candidates]
        candidates_with_scores = list(zip([candidates[0] for candidates in next_step_candidates], combined_scores))
        sorted_candidates = sorted(candidates_with_scores, key=lambda x: x[1], reverse=True)
        top_k_candidates = sorted_candidates[:args.top_k]
        
        return top_k_candidates
        
    elif args.strategy == "discrete_rating":
        if args.d == "RoG-webqsp":
            prompt_list = prompt_list_webqsp
        elif args.d == "RoG-cwq":
            prompt_list = prompt_list_cwq
            
        _new_line_char = "\n" # for formatting the prompt
        formatted_next_step_candidates = [f"{i+1}: {item}" for i, item in enumerate(next_step_candidates)]
        rating_prompt = copy.copy(prompt_list.beam_search_prompt)
        rating_prompt["prompt"] = prompt_list.beam_search_prompt["prompt"].format(
            beam_width=args.top_k,
            plan_context=plan_context,
            current_reasoning_path=cur_rpth,
            question=question,
            reasoning_paths=_new_line_char.join(formatted_next_step_candidates)
        )
        
        logging.info("<<<<<<<<")
        logging.info("Beam Search Prompt: {}".format(rating_prompt["prompt"]))
        logging.info(">>>>>>>>")
        
        attempt = 0
        while attempt < 5:
            try:
                rating_index = get_completion(args, rating_prompt)
                rating_index = rating_index.replace("Answer: ", "").strip()
                _ = re.findall(r'\d+', rating_index)
                matched_indices = [int(i)-1 for i in _]
                
                logging.info("<<<<<<<<")
                logging.info("Top-k Indices: {}".format(matched_indices))
                logging.info(">>>>>>>>")
                top_k_candidates = [[next_step_candidates[i]] for i in matched_indices]
                return top_k_candidates
    
            except Exception as e:
                logging.error(f"Error occurred: {e}")
                attempt += 1
                time.sleep(1)


def meets_condition(
    args, 
    reasoning_path: str, 
    question: str, 
    planning_context: str = ""
):
    if args.d == "RoG-webqsp":
        prompt_list = prompt_list_webqsp
    elif args.d == "RoG-cwq":
        prompt_list = prompt_list_cwq
    
    if args.verifier == "enough":    
        condition_prompt = copy.copy(prompt_list.terminals_prune_single_prompt)
        condition_prompt["prompt"] = condition_prompt["prompt"].format(
            question=question,
            reasoning_path=reasoning_path,
            plan_context=planning_context
        )
    #TODO: add more verifiers
    elif args.verifier == "enough+planning": 
        pass
    elif args.verifier == "enough+planning+confidence":
        pass
    elif args.verifier == "deductive+planning":   
        pass
        
    res = get_completion(args, condition_prompt).replace("Answer: ", "").strip()
    if "Yes" in res:
        return True
    elif "No" in res:
        return False
    else:
        return False


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


def data_processing(args):
    input_file = os.path.join(args.data_path, args.d)
    output_file = os.path.join(args.save_cache, f"{args.d}_processed")
    dataset = load_dataset(input_file, split=args.split, cache_dir=args.save_cache)
    dataset = dataset.map(
            prepare_dataset,
            num_proc=args.N_CPUS,
        )
    dataset = dataset.filter(
            lambda x: x.get("hop") > 0, 
            num_proc=args.N_CPUS
        )
    dataset = dataset.filter(
            lambda x: x.get("q_entity") != None, 
            num_proc=args.N_CPUS
        )
    if os.path.exists(output_file) == False:
        os.makedirs(output_file)
        
    dataset.save_to_disk(output_file)
    return dataset


def beam_search(data, args):
    id = data['id']
    question = data['question']
    hop = data['hop']
    graph = utils.build_graph(data["graph"])
    answer = data['a_entity']
    starting_nodes = data['q_entity']
    pred_list_direct_answer = []
    pred_list_llm_reasoning = []
    reasoning_path_list = []
    ground_reasoning_path_list = data['ground_paths'] # shortest reasoning paths from q_entity to a_entity
    _new_line_char = "\n" # for formatting the prompt
    
    logging.info(f"Processing ID: {id}")
    logging.info(f"Question: {question}")
    logging.info(f"Ground Truth: {answer}")
    logging.info(f"Starting Nodes: {starting_nodes}")
    
    if args.d == "RoG-webqsp":
        prompt_list = prompt_list_webqsp
    elif args.d == "RoG-cwq":
        prompt_list = prompt_list_cwq
    
    # --------------
    # BEAM SEARCH
    # --------------
    for node in starting_nodes:    
        
        # --------------
        # PLANNING FIRST
        # --------------
        plan_prompt = copy.copy(prompt_list.plan_prompt)
        plan_prompt["prompt"] = plan_prompt["prompt"].format(
            question=question,
            starting_node=node
        )
        plan_res = get_completion(args, plan_prompt)
        
        logging.info("Plan Prompt: {}".format(plan_prompt["prompt"]))
        logging.info("Plan Response: {}".format(plan_res))
                
        if args.strategy == "continuous_rating":         
            reasoning_paths = [[node, 1.0]] # store the reasoning paths for each step, the length of the list is equal to the number of top-k
            for step in tqdm(range(args.max_length), desc="Beam searching...", delay=0.5, leave=False):
                all_candidates = []
        
                for rpth in reasoning_paths:
                    next_step_candidates, deductive_scores, self_confidence_scores = prepare_options_for_each_step(
                        args=args,
                        starting_node=node,
                        reasoning_path=rpth[0],
                        query=question,
                        graph=graph,
                        prompt_list=prompt_list
                    )
                    
                    self_confidence_probs = np.exp(np.array(self_confidence_scores)) 
                    normalized_self_confidence_probs = self_confidence_probs / np.sum(self_confidence_probs)
                    for i, candidate in enumerate(next_step_candidates):
                        # only consider the candidates with self-confidence score > 0.5
                        if normalized_self_confidence_probs[i] > 0.5 or step == 0:
                            all_candidates.append([candidate, deductive_scores[i], self_confidence_scores[i]])
                
                # if there are no candidates fit the criteria, break the loop
                if not all_candidates:
                    print("beam search stopped at step {}".format(step))
                    break
                
                reasoning_paths = find_top_k_candidates(
                        args=args,
                        next_step_candidates=all_candidates,
                    )
                
        elif args.strategy == "discrete_rating":
            reasoning_paths = [] # final reasoning paths 
            active_beam_raesoning_paths = [[node]] # store the reasoning paths for each step, the length of the list is equal to the number of top-k
            for step in tqdm(range(args.max_length), desc="Beam searching...", delay=0.5, leave=False):
                all_candidates = []
                
                for rpth in active_beam_raesoning_paths:
                
                    # if meet the condition, skip the current step
                    if step != 0:
                        flag = meets_condition(args, reasoning_path=rpth[0], question=question, planning_context=plan_res)
                        if flag:
                            reasoning_paths.append(rpth)
                            continue
                    
                    next_step_candidates = prepare_options_for_each_step(
                        args=args,
                        starting_node=node,
                        reasoning_path=rpth[0],
                        query=question,
                        graph=graph,
                        prompt_list=prompt_list
                    )
                    all_candidates.extend(next_step_candidates)
                    
                    if not all_candidates:
                        break
                
                # logging.info("<<<<<<<<")
                # logging.info("Step: {}, Avaiable Candidates: \n{}".format(step, _new_line_char.join(all_candidates)))
                # logging.info(">>>>>>>>")
                
                active_beam_raesoning_paths = find_top_k_candidates(
                        args=args,
                        next_step_candidates=all_candidates,
                        question=question,
                        plan_context=plan_res
                    )
                
                logging.info("<<<<<<<<")
                logging.info("Active Beam Reasoning Paths: {}".format(active_beam_raesoning_paths))
                logging.info(">>>>>>>>")
            
            # if there are no candidates fit the criteria, return the active_beam_raesoning_paths
            if not reasoning_paths:
                reasoning_paths = active_beam_raesoning_paths
        
        # --------------
        # LLM REASONING
        # --------------
        reasoning_pmt = copy.copy(prompt_list.reasoning_prompt)
        reasoning_pmt["prompt"] = reasoning_pmt["prompt"].format(
            question=question,
            reasoning_path = _new_line_char.join([item[0] for item in reasoning_paths])
        )
        res = get_completion(args, reasoning_pmt).replace("Answer: ", "").strip()
        # logging.info("Reasoning Response: {}".format(res))
        # print("Reasoning Response: {}".format(res))
        
        logging.info("<<<<<<<<")
        logging.info("Reasoning Prompt: {}".format(reasoning_pmt["prompt"]))
        logging.info("Reasoning Paths: \n{}".format(reasoning_paths))
        logging.info("Prediction: \n{}".format(res))
        logging.info(">>>>>>>>")

        for item in res.split(", "):
            pred_list_llm_reasoning.append(item)
        
        for item in reasoning_paths:
            pred_list_direct_answer.append(item[0].split("->")[-1])
            reasoning_path_list.append(item[0])     
        
    # save the results to a jsonl file
    res =  {
            "id": id,
            "question": question,
            "hop": hop,
            "q_entities": starting_nodes,
            "reasoning_path": reasoning_path_list,
            "ground_path": ground_reasoning_path_list,
            "prediction_llm": "\n".join(set(pred_list_llm_reasoning)), # remove duplicate predictions
            "prediction_direct_answer": "\n".join(set(pred_list_direct_answer)),
            "ground_truth": answer,
        }
    return res


def main(args):
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
    
    # load the dataset
    cached_dataset_path = os.path.join(args.save_cache, f"{args.d}_processed")
    if os.path.exists(cached_dataset_path):
        dataset = load_from_disk(cached_dataset_path)
    else:
        print("Processing data...")
        dataset = data_processing(args)
        print("Data processing completed!")
        
    if args.sample != -1:
        dataset = dataset.select(range(args.sample))
       
    # error analysis (case study)    
    # dataset = dataset.select([11, 12, 13])
        
    for data in tqdm(dataset, desc="Data Processing...", delay=0.5):
        try:
            res = beam_search(data, args) # run the beam search for each sample
            
        except Exception as e:
            logging.error("Error occurred: {}".format(e))
            f = open(os.path.join(output_dir, "error_sample.jsonl"), "a")
            json_str = json.dumps({"id": data['id'], "error": str(e)})
            f.write(json_str + "\n")
            continue
        
        # res = beam_search(data, args) # run the beam search for each sample
        
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
        
    # evaluate
    llm_res, direct_ans_res = eval_result(os.path.join(output_dir, f"{args.d}-{args.model_name}-{args.sample}.jsonl"), cal_f1=True)
    
    wandb.log(
        {
            "llm_result": llm_res,
            "direct_ans_result": direct_ans_res,
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
    parser.add_argument("--save_cache", type=str, default="/data/shared/yuansui/rog/.cache/huggingface/datasets")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output_path", type=str, default="results")
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo-0125")
    parser.add_argument("--top_n", type=int, default=30)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=3)
    parser.add_argument("--strategy", type=str, default="discrete_rating")
    parser.add_argument("--squeeze", type=bool, default=True)
    parser.add_argument("--verifier", type=str, default="enough")
    parser.add_argument("--embedding_model", type=str, default="text-embedding-3-small")
    parser.add_argument("--add_hop_information", type=bool, default=True)
    args = parser.parse_args()
    main(args)