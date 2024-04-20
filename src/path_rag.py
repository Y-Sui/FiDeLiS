import os
import logging
import multiprocessing as mp
import numpy as np
import time
import networkx as nx
from tqdm import tqdm
from src.utils.data_types import Graph, Node, Edge
from src.utils.llm_backbone import LLM_Backbone

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
      graph: Graph
   ) -> list:
      """
      given an entity, find all edges and neighbors
      """
      edges = [] # each edge is an instance of Edge, the attribute can be accessed through edge.attribute, and the embedding can be accessed through edge.embedding
      neighbors = []
      
      if graph.graph.has_node(entity):
         for neighbor in graph.graph.neighbors(entity):
            # relation = graph.edges.get((entity, neighbor), Edge(None, None))
            # neighbor = graph.nodes.get(neighbor, Node(None))
            try:
               relation = graph.edges[(entity, neighbor)]
               # print(f"x: {relation}")
            except KeyError:
               relation = graph.graph[entity][neighbor]['relation']
               print(f"entity: {entity}, neighbor: {neighbor}, relation: {relation}")
               print(f"real_entity and real_neighbor: {[k for k, v, in graph.edges.items()]}")
               # print(f"y: {relation}, {relation in [e.attribute for e in graph.edges.values()]}")
            neighbor = graph.nodes[neighbor]
            if relation not in edges or neighbor not in neighbors: # remove the duplicates
               edges.append(relation)
               neighbors.append(neighbor)

      return edges, neighbors
   
   def has_relation(
      self, 
      graph: Graph,
      entity: str,
      relation: str,
      neighbor: str
   ) -> bool:
      """
      check if the relation exists in the graph
      """
      if graph.graph.has_edge(entity, neighbor):
         if graph.graph[entity][neighbor]['relation'] == relation:
            return True
      return False
   
   def get_relations_neighbors_set_with_ratings(
      self,
      relations: list,
      neighbors: list,
      query_embedding: list,
   ) -> list:
      """
      given a list of relations and neighbors, return top-n relations and neighbors with the corresponding ratings [(relation, 0.9), (relation, 0.8), ...]
      """
      query_embedding = np.array(query_embedding)
      
      relations_embeddings = np.array([relation.embedding for relation in relations])
      neighbors_embeddings = np.array([neighbor.embedding for neighbor in neighbors])
      
      try:
         # calculate cosine similarity
         query_relation_similarity = self.cos_simiarlity(query_embedding, relations_embeddings)
         query_neighbor_similarity = self.cos_simiarlity(query_embedding, neighbors_embeddings)
         
      except Exception as e:
         print(f"query_embeddiong: {query_embedding}")
         print(f"relations_embeddings: {relations_embeddings}")
         print(f"neighbors_embeddings: {neighbors_embeddings}")
         print(query_embedding.shape, relations_embeddings.shape, neighbors_embeddings.shape)
      
      # sort the neighbors by similarity
      relations = [(relations[i].attribute, query_relation_similarity[i]) for i in np.argsort(query_relation_similarity)[::-1]]
      
      neighbors = [(neighbors[i].attribute, query_neighbor_similarity[i]) for i in np.argsort(query_neighbor_similarity)[::-1]]
      
      return relations, neighbors
   
   def scoring_path(
      self,
      keyword_embeddings: list,
      rated_relations: list,
      rated_neighbors: list,
      hub_node: str,
      reasoning_path: str,
      graph: Graph
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
                  
                  if one_hop_relations and one_hop_neighbors:
                     one_hop_rated_relations, one_hop_rated_neighbors = self.get_relations_neighbors_set_with_ratings(one_hop_relations, one_hop_neighbors, keyword_embeddings)
                     
                  else:
                     # if there is no one-hop neighbors, set the score to 0
                     one_hop_rated_relations, one_hop_rated_neighbors = [(None, 0)], [(None, 0)]
                  
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
      graph = state.get("graph", Graph)
      keywords = state.get("key_words", "") # using the keywords generated from llm to represent the query
      reasoning_path = state.get("rpth", "")
      
      hub_node = reasoning_path.split(" -> ")[-1]
      
      relations, neighbors = self.get_entity_edges(hub_node, graph)
      
      #TODO load the embeddings from the vectorspace
      # get embeddings
      embeddings = self.llm_backbone.get_embeddings(keywords)
      
      # get relations and neighbors with the corresponding ratings
      rated_relations, rated_neighbors = self.get_relations_neighbors_set_with_ratings(relations, neighbors, embeddings)
               
      # top-n scoring paths
      paths = self.scoring_path(keyword_embeddings=embeddings, reasoning_path=reasoning_path, rated_relations=rated_relations, rated_neighbors=rated_neighbors, hub_node=hub_node, graph=graph)
      
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