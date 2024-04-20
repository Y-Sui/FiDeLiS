import os
import argparse
import os
import json
import logging
import multiprocessing as mp
import wandb
import datetime
import litellm
from concurrent.futures import ProcessPoolExecutor
from src.evaluate_results import eval_result
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
from src import utils
from src.llm_navigator import LLM_Navigator
from src.utils.data_types import Graph

litellm.set_verbose=False
set_verbose = False
now = datetime.datetime.now()
timestamp = now.strftime(f"%Y_%m_%d_%H_%M")

with open("config.json", "r") as f:
    config = json.load(f)
    
os.environ["OPENAI_API_KEY"] = config["OPENAI_API_KEY"]


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
         lambda x: x.get("hop") > 0 and x.get("question") != "" and len(x.get("q_entity")) > 0 and len(x.get("a_entity")) > 0 and len(x.get("ground_paths")) > 0, 
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


def init_embedding(data):
   id = data["id"]
   graph = Graph(
      args=args,
      graph=utils.build_graph(data["graph"]),
      cache_path=args.save_cache,
      id=id,
      embedding_method=args.embedding_model,
      replace=False
   )
   print(f"Embedding for {id} completed")


def main(args):
   output_dir = os.path.join(args.output_path, args.model_name, timestamp)
   
   if os.path.exists(output_dir) == False:
      os.makedirs(output_dir)
      
   logging.basicConfig(
      level=logging.INFO,
      format='%(asctime)s - %(levelname)s - %(message)s',
      filename=os.path.join(output_dir,'detailed_process.log'),
      filemode='w',
   )
   
   # load the dataset
   cached_dataset_path = os.path.join(args.save_cache, f"{args.d}_processed")
   
   if os.path.exists(cached_dataset_path):
      dataset = load_from_disk(cached_dataset_path)
   else:
      print("Processing data...")
      dataset = data_processing(args)
      print("Data processing completed!")

   # generate embeddings
   if args.generate_embeddings:
      with ProcessPoolExecutor(max_workers=args.N_CPUS) as executor:
         # Using list to consume the results as they come in for tqdm to track
         executor.map(init_embedding, dataset)
      print("Embedding completed!")
      return
   
   # sample the dataset
   if args.sample != -1:
      dataset = dataset.select(range(args.sample))
   llm_navigator = LLM_Navigator(args)
   
   for data in tqdm(dataset, desc="Data Processing...", delay=0.5):
      # try:
      #    res = llm_navigator.beam_search(data) # run the beam search for each sample
         
      # except Exception as e:
      #    logging.error("Error occurred: {}".format(e))
      #    print("Error occurred: {}".format(e))
      #    f = open(os.path.join(output_dir, "error_sample.jsonl"), "a")
      #    json_str = json.dumps({"id": data['id'], "error": str(e)})
      #    f.write(json_str + "\n")
      #    continue
      
      res = llm_navigator.beam_search(data)
      
      f = open(os.path.join(output_dir, f"{args.d}-{args.model_name}-{args.sample}.jsonl"), "a")
      json_str = json.dumps(res)
      f.write(json_str + "\n")
      
   # evaluate
   llm_res, direct_ans_res = eval_result(os.path.join(output_dir, f"{args.d}-{args.model_name}-{args.sample}.jsonl"), cal_f1=True)

if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument("--N_CPUS", type=int, default=mp.cpu_count())
   parser.add_argument("--sample", type=int, default=-1)
   parser.add_argument("--data_path", type=str, default="rmanluo")
   parser.add_argument("--d", "-d", type=str, default="RoG-webqsp")
   parser.add_argument("--save_cache", type=str, default="/data/yuansui/rog/datasets")
   parser.add_argument("--split", type=str, default="test")
   parser.add_argument("--output_path", type=str, default="results")
   parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo-0125")
   parser.add_argument("--top_n", type=int, default=30)
   parser.add_argument("--top_k", type=int, default=3)
   parser.add_argument("--max_length", type=int, default=3)
   parser.add_argument("--strategy", type=str, default="discrete_rating")
   parser.add_argument("--squeeze", type=bool, default=True)
   parser.add_argument("--verifier", type=str, default="deductive+planning")
   parser.add_argument("--embedding_model", type=str, default="text-embedding-3-small")
   parser.add_argument("--add_hop_information", action="store_true")
   parser.add_argument("--generate_embeddings", action="store_false")
   parser.add_argument("--alpha", type=float, default=0.3)
   args = parser.parse_args()
   main(args)