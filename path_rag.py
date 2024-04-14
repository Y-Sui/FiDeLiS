import os
import logging
import multiprocessing as mp
import numpy as np
import time
import networkx as nx
from tqdm import tqdm
from openai import OpenAI
from litellm import completion, embedding, batch_completion # import litellm for calling multiple llms using the same input/output format


class LLM_Backbone():
   def __init__(self, args):
      self.client = OpenAI()
      self.embedding_model = args.embedding_model
      self.completion_model = args.model_name
      self.max_attempt = 5 # number of attempts to get the completion
      
   def get_embeddings(self, texts: list):
      embeddings = []
      texts_per_batch = 2000
      text_chunks = [texts[i:i + texts_per_batch] for i in range(0, len(texts), texts_per_batch)]
      
      attempt = 0
      while attempt < self.max_attempt:
         try:
            for chunk in text_chunks:
               chunk_embeddings = self.client.embeddings.create(
                  model=self.embedding_model, 
                  input=chunk
                  ) # return [item['embedding'] for item in _['data']]
               embeddings.extend([item.embedding for item in chunk_embeddings.data])
            return embeddings
         except Exception as e:
            logging.error(f"Error occurred: {e}")
            attempt += 1
            time.sleep(1)
            
   def get_completion(self, prompt: dict):
      messages = [
         {"role": "system", "content": prompt["system"]},
         *prompt["examples"],
         {"role": "user", "content": prompt["prompt"]}
      ]
         
      attempt = 0
      while attempt < self.max_attempt:
         try:
               _ = self.client.chat.completions.create(
                  model=self.completion_model, 
                  messages=messages, 
                  temperature=0, 
                  top_p=0, 
                  logprobs=False
                  )
               # return _['choices'][0]['message']['content']
               return _.choices[0].message.content
         except Exception as e:
               logging.error(f"Error occurred: {e}")
               attempt += 1
               time.sleep(1)
               
   def get_log_probs(self, log_probs: list):
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
   
   def get_batch_completion(self, prompt: dict, input_batch: list):
      """
      for item in log_probs:
         if item["token"] == "A":
               print(item['logprob'])
      """
      
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
                  model=self.completion_model, 
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


class Path_RAG():
   def __init__(self, args):
      self.llm_backbone = LLM_Backbone(args)
      self.args = args
      
   def cos_simiarlity(self, a: np.array, b: np.array):
      """
      calculate cosine similarity between two vectors
      Parameters:
         a: np.array, representing a single vector
         b: np.array, shape (n_vectors, vector_length), representing multiple vectors
      """
      a = a.reshape(1, -1)
      dot_product = np.dot(a, b.T).flatten()
      norm_a = np.linalg.norm(a)
      norm_b = np.linalg.norm(b, axis=1)
      
      epsilon = 1e-9
      cos_similarities = dot_product / (norm_a * norm_b + epsilon)
      return cos_similarities
   
   def get_entity_edges(
      self, 
      entity: str, 
      graph: nx.Graph
   ) -> list:
      """
      given an entity, find all edges and neighbors
      """
      edges = []
      neighbors = []
      
      if graph.has_node(entity):
         for neighbor in graph.neighbors(entity):
            relation = graph[entity][neighbor]['relation']
            if relation not in edges or neighbor not in neighbors: # remove the duplicates
               edges.append(relation)
               neighbors.append(neighbor)
               
      return edges, neighbors
   
   def has_relation(
      self, 
      graph: nx.Graph,
      entity: str,
      relation: str,
      neighbor: str
   ) -> bool:
      """
      check if the relation exists in the graph
      """
      if graph.has_edge(entity, neighbor):
         if graph[entity][neighbor]['relation'] == relation:
            return True
      return False
   
   def get_relations_neighbors_set_with_ratings(
      self,
      relations: list,
      neighbors: list,
      embeddings: list,
   ) -> list:
      """
      given a list of relations and neighbors, return top-n relations and neighbors with the corresponding ratings [(relation, 0.9), (relation, 0.8), ...]
      """
      query_embedding = np.array(embeddings[0])
      relations_embeddings = np.array(embeddings[1:len(relations)+1])
      neighbors_embeddings = np.array(embeddings[len(relations)+1:])
      
      # calculate cosine similarity
      query_relation_similarity = self.cos_simiarlity(query_embedding, relations_embeddings)
      query_neighbor_similarity = self.cos_simiarlity(query_embedding, neighbors_embeddings)
      
      # sort the neighbors by similarity
      relations = [(relations[i], query_relation_similarity[i]) for i in np.argsort(query_relation_similarity)[::-1]]
      neighbors = [(neighbors[i], query_neighbor_similarity[i]) for i in np.argsort(query_neighbor_similarity)[::-1]]
      
      return relations, neighbors
   
   def scoring_path(
      self,
      keywords: str,
      rated_relations: list,
      rated_neighbors: list,
      hub_node: str,
      reasoning_path: str,
      graph: nx.Graph
   ) -> list:
      """
      given a list of relations and neighbors with ratings, return top-k relations and neighbors with the corresponding ratings
      """
      # concatenate the relations and neighbors
      rated_paths = [] # [(path, score)]
      seen_paths = [] # store the seen paths [path, path]
      for relation, relation_score in rated_relations:
         for neighbor, neighbor_score in rated_neighbors:
            new_rpth = f"{reasoning_path} -> {relation} -> {neighbor}"
            if self.has_relation(
               graph=graph, 
               entity=hub_node, 
               relation=relation,
               neighbor=neighbor
            ) and new_rpth not in seen_paths:            
               
               if self.args.add_hop_information:
                  #TODO using vectorspace to store the embeddings, otherwise the efficiency is pretty low
                  # 1-hop neighbors = relation + neighbor
                  one_hop_relations, one_hop_neighbors = self.get_entity_edges(neighbor, graph)
                  texts = [keywords] + one_hop_relations + one_hop_neighbors
                  embeddings = self.llm_backbone.get_embeddings(texts)
                  one_hop_rated_relations, one_hop_rated_neighbors = self.get_relations_neighbors_set_with_ratings(one_hop_relations, one_hop_neighbors, embeddings)
                  
                  # score function for path_rag
                  rpth_score = relation_score + neighbor_score + self.args.alpha * (one_hop_rated_relations[0][1] + one_hop_rated_neighbors[0][1])
                  
               else:
                  rpth_score = relation_score + neighbor_score
                  
               rated_paths.append((new_rpth, rpth_score))
               seen_paths.append(new_rpth)
               
      rated_paths = sorted(rated_paths, key=lambda x: x[1], reverse=True)[:self.args.top_n]
      
      # only return the path
      paths = [path[0] for path in rated_paths]
            
      return paths
   
   def get_path(
      self, 
      state: dict
   ) -> list:
      """
      given a starting entity, find top-k one-step path to the query (keywords)
      """
      graph = state.get("graph", nx.Graph())
      keywords = state.get("key_words", "") # using the keywords generated from llm to represent the query
      reasoning_path = state.get("rpth", "")
      
      hub_node = reasoning_path.split(" -> ")[-1]
      
      relations, neighbors = self.get_entity_edges(hub_node, graph)
      
      #TODO load the embeddings from the vectorspace
      # get embeddings
      texts = [keywords] + relations + neighbors
      embeddings = self.llm_backbone.get_embeddings(texts)
      
      # get relations and neighbors with the corresponding ratings
      rated_relations, rated_neighbors = self.get_relations_neighbors_set_with_ratings(relations, neighbors, embeddings)
               
      # top-n scoring paths
      paths = self.scoring_path(keywords=keywords, reasoning_path=reasoning_path, rated_relations=rated_relations, rated_neighbors=rated_neighbors, hub_node=hub_node, graph=graph)
      
      return paths

   
# import unittest
# import networkx as nx
# import numpy as np
# from unittest.mock import Mock, patch

# with open("config.json", "r") as f:
#     config = json.load(f)
    
# os.environ["OPENAI_API_KEY"] = config["OPENAI_API_KEY"]

# class TestPathRAG(unittest.TestCase):
#    def setUp(self):
#       self.args = Mock()
#       self.args.top_n = 5
#       self.path_rag = Path_RAG(self.args)
#       self.graph = nx.Graph()
#       self.graph.add_edge('A', 'B', relation='relation1')
#       self.graph.add_edge('A', 'C', relation='relation2')

#    @patch.object(LLM_Backbone, 'get_embeddings')
#    def test_get_path(self, mock_get_embeddings):
#       mock_get_embeddings.return_value = [np.array([1, 0]), np.array([0, 1]), np.array([0, -1])]
#       paths = self.path_rag.get_path('A', self.graph, 'query')
#       self.assertEqual(len(paths), self.args.top_n)
#       self.assertTrue(all(isinstance(path, tuple) for path in paths))

#    def test_cos_similarity(self):
#       a = np.array([1, 0])
#       b = np.array([[0, 1], [0, -1]])
#       result = self.path_rag.cos_simiarlity(a, b)
#       self.assertEqual(result.shape, (2,))

#    def test_get_entity_edges(self):
#       edges, neighbors = self.path_rag.get_entity_edges('A', self.graph)
#       self.assertEqual(edges, ['relation1', 'relation2'])
#       self.assertEqual(neighbors, ['B', 'C'])

#    def test_has_relation(self):
#       self.assertTrue(self.path_rag.has_relation(self.graph, 'A', 'relation1', 'B'))
#       self.assertFalse(self.path_rag.has_relation(self.graph, 'A', 'relation3', 'B'))

# if __name__ == '__main__':
#    unittest.main()