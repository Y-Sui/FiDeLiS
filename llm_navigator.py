import logging
import copy
import re
import json
import time
from tqdm import tqdm
from src import utils
from src.utils import prompt_list_cwq, prompt_list_webqsp
from path_rag import Path_RAG, LLM_Backbone

class LLM_Navigator():
   def __init__(self, args) -> None:
      self.llm_backbone = LLM_Backbone(args)
      self.path_rag_engine = Path_RAG(args)
      self.args = args
      if args.d == "RoG-webqsp":
         self.prompt_list = prompt_list_webqsp
      elif args.d == "RoG-cwq":
         self.prompt_list = prompt_list_cwq
      self._new_line_char = "\n" # for formatting the prompt
      
   def rpth_parser(
      self, 
      state: dict
   ):
      """
      Reformulate the reasoning path from the agent state
      """
      reasoning_path = state.get("rpth", "")
      reformulate_prompt = copy.copy(self.prompt_list.reasoning_path_parser_prompt)
      reformulate_prompt["prompt"] = reformulate_prompt["prompt"].format(
         reasoning_path=reasoning_path
      )
      reformulate_res = self.llm_backbone.get_completion(reformulate_prompt)
      state["parsed_rpth"] = reformulate_res
      
   def deductive_termination(
      self,
      state: dict
   ):
      self.rpth_parser(state) # reformulate the reasoning path
      
      reasoning_path = state.get("rpth", "")
      parsed_reasoning_path = state.get("parsed_rpth", "")
      question = state.get("question", "")
      planning_steps = state.get("planning_steps", "")
      declarative_statement = state.get("declarative_statement", "")
      
      placeholder_entity = reasoning_path.split(" -> ")[-1]
      declarative_statement = declarative_statement.replace("*placeholder*", placeholder_entity).strip(".")
      
      # print(self.args.verifier)
      
      if self.args.verifier == "enough":    
        condition_prompt = copy.copy(self.prompt_list.terminals_prune_single_prompt)
        condition_prompt["prompt"] = condition_prompt["prompt"].format(
            question=question,
            reasoning_path=reasoning_path,
            plan_context=planning_steps,
        )
      
      #TODO: add more verifiers
      elif self.args.verifier == "enough+planning": 
         pass
      elif self.args.verifier == "enough+planning+confidence":
         pass
      elif self.args.verifier == "deductive+planning":   
         condition_prompt = copy.copy(self.prompt_list.deductive_verifier_prompt)
         condition_prompt["prompt"] = condition_prompt["prompt"].format(
            parsed_reasoning_path=parsed_reasoning_path,
            declarative_statement=declarative_statement
         )

      res = self.llm_backbone.get_completion(condition_prompt).replace("Answer: ", "").strip()
      # print("Condition Prompt: ", condition_prompt["prompt"], "Deductive Termination: ", res)
      
      logging.info("<<<<<<<<")
      logging.info("Deductive Termination Prompt: {}".format(condition_prompt["prompt"]))
      logging.info("Prediction: {}".format(res))
      logging.info(">>>>>>>>")
      
      if "Yes" in res:
         return True
      elif "No" in res:
         return False
      else:
         return False
      
   def decide_top_k_candidates(
      self,
      state: dict
   ):
      
      next_step_candidates = state.get("next_step_candidates", [])
      question = state.get("question", "")
      planning_steps = state.get("planning_steps", "")
      
      formatted_next_step_candidates = [f"{i+1}: {item}" for i, item in enumerate(next_step_candidates)]
      rating_prompt = copy.copy(self.prompt_list.beam_search_prompt)
      rating_prompt["prompt"] = self.prompt_list.beam_search_prompt["prompt"].format(
         beam_width=self.args.top_k,
         plan_context=planning_steps,
         question=question,
         reasoning_paths=self._new_line_char.join(formatted_next_step_candidates)
      )
      
      logging.info("<<<<<<<<")
      logging.info("Beam Search Prompt: {}".format(rating_prompt["prompt"]))
      logging.info(">>>>>>>>")
      
      attempt = 0
      while attempt < 5: # try 5 times if the index is not found or not as expected
         try:
            rating_index = self.llm_backbone.get_completion(rating_prompt)
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
               
   def reasoning(
      self, 
      state: dict
   ):
      reasoning_paths = state.get("reasoning_paths", [])
      question = state.get("question", "")
      reasoning_prompt = copy.copy(self.prompt_list.reasoning_prompt)
      reasoning_prompt["prompt"] = reasoning_prompt["prompt"].format(
         question=question,
         reasoning_path=self._new_line_char.join([item[0] for item in reasoning_paths])
      )
      reasoning_res = self.llm_backbone.get_completion(reasoning_prompt)
      
      logging.info("<<<<<<<<")
      logging.info("Reasoning Prompt: {}".format(reasoning_prompt["prompt"]))
      logging.info("Reasoning Paths: \n{}".format(reasoning_paths))
      logging.info("Prediction: \n{}".format(reasoning_res))
      logging.info(">>>>>>>>")
      
      reasoning_res = reasoning_res.replace("Answer: ", "").strip()
      
      return reasoning_res
   
   def planning(
      self,
      state: dict
   ):
      """
      Generate the planning steps for the Beam Search, and the keywords for the Path-RAG
      """
      entity = state.get("entity", "")
      question = state.get("question", "")
      plan_prompt = copy.copy(self.prompt_list.plan_prompt)
      plan_prompt["prompt"] = plan_prompt["prompt"].format(
         question=question,
         starting_node=entity
      )
      plan_res = self.llm_backbone.get_completion(plan_prompt)
      key_words = ", ".join(json.loads(plan_res)["keywords"])
      planning_steps = ", ".join(json.loads(plan_res)["planning_steps"])
      declarative_statement = json.loads(plan_res)["declarative_statement"]
      
      logging.info("Plan Prompt: {}".format(plan_prompt["prompt"]))
      logging.info("Planning Keywords: {}".format(key_words))
      logging.info("Planning Steps: {}".format(planning_steps))
      logging.info("Declarative Statement: {}".format(declarative_statement))
      
      state["key_words"] = key_words
      state["planning_steps"] = planning_steps
      state["declarative_statement"] = declarative_statement

      # print("planning_steps: ", state["planning_steps"])
      # print("key_words: ", state["key_words"])
      # print("declarative_statement: ", state["declarative_statement"])

   def beam_search(
      self,
      data
   ):
      id  = data['id']
      question = data['question']
      hop = data['hop']
      graph = utils.build_graph(data["graph"])
      answer = data['a_entity']
      starting_entities = data['q_entity']
      pred_list_direct_answer = []
      pred_list_llm_reasoning = []
      reasoning_path_list = []
      ground_reasoning_path_list = data['ground_paths'] # shortest reasoning paths from q_entity to a_entity
      llm_states = {} #TODO modify the agentstate to store the states of the llm
      llm_states["question"] = question
      llm_states["hop"] = hop
      llm_states["graph"] = graph
      llm_states["answer"] = answer
      llm_states["starting_entities"] = starting_entities
      
      logging.info(f"Processing ID: {id}")
      logging.info(f"Question: {question}")
      logging.info(f"Ground Truth: {answer}")
      logging.info(f"Starting Nodes: {starting_entities}")
      
      for node in starting_entities:
         llm_states["entity"] = node
         self.planning(llm_states)
         
         reasoning_paths = [] # final reasoning paths
         active_beam_reasoning_paths = [[node]] # store the reasoning paths for each step, the the length of the list is equal to the number of top-k
         
         for step in tqdm(range(self.args.max_length), desc="Beam searching...", delay=0.5, leave=False):
         
            all_candidates = []
            
            for rpth in active_beam_reasoning_paths:
               
               llm_states["rpth"] = rpth[0]
            
               # if meet the condition, skip the current step
               if step != 0:
                  flag = self.deductive_termination(
                     state=llm_states
                  )
                  if flag:
                     reasoning_paths.append(rpth)
                     continue
               
               next_step_candidates = self.path_rag_engine.get_path(
                  state=llm_states
               )
               all_candidates.extend(next_step_candidates)
            
            if not all_candidates:
               break
            
            llm_states["next_step_candidates"] = all_candidates
            active_beam_reasoning_paths = self.decide_top_k_candidates(
                  state=llm_states
               )
            
            logging.info("<<<<<<<<")
            logging.info("Active Beam Reasoning Paths: {}".format(active_beam_reasoning_paths))
            logging.info(">>>>>>>>")
            
         # if there are no candidates fit the criteria, return the active_beam_raesoning_paths
         if not reasoning_paths:
            reasoning_paths = active_beam_reasoning_paths
            
         llm_states["reasoning_paths"] = reasoning_paths
               
         # --------------
         # LLM REASONING
         # --------------
         reasoning_res = self.reasoning(llm_states)
         
         for item in reasoning_res.split(", "):
            pred_list_llm_reasoning.append(item)
         
         for item in reasoning_paths:
            pred_list_direct_answer.append(item[0].split(" -> ")[-1])
            reasoning_path_list.append(item[0])    
            
      # save the results to a jsonl file
      res =  {
               "id": id,
               "question": question,
               "hop": hop,
               "q_entities": starting_entities,
               "reasoning_path": reasoning_path_list,
               "ground_path": ground_reasoning_path_list,
               "prediction_llm": "\n".join(set(pred_list_llm_reasoning)), # remove duplicate predictions
               "prediction_direct_answer": "\n".join(set(pred_list_direct_answer)),
               "ground_truth": answer,
         }
      return res
      


