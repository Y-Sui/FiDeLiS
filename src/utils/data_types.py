import networkx as nx
import os
import numpy as np
from src.utils.llm_backbone import LLM_Backbone

class Node:
   def __init__(self, attribute=None):
      self.embedding = None # node embedding
      self.attribute = attribute # node attributes (dict)
      
   def set_embedding(self, embedding):
      self.embedding = embedding
      
   def __str__(self):
      return str(self.attribute)

class Edge:
   def __init__(self, src, des, attribute=None):
      self.src = src # source node
      self.des = des # destination node
      self.attribute = attribute # edge attributes (dict)
      self.embedding = None # edge embedding
      
   def set_embedding(self, embedding):
      self.embedding = embedding
      
   def __str__(self):
      return str(self.attribute)

class Graph:
   def __init__(
      self, 
      args, 
      id, 
      graph=None, 
      cache_path: str = "", 
      embedding_method: str = "text-embedding-3-small", 
      replace = False
   ):
      self.graph = graph if graph is not None else nx.DiGraph() # networkx graph object
      self.nodes = {n: Node(n) for n in self.graph.nodes()}
      self.edges = {(e[0], e[1]): Edge(e[0], e[1], e[2]) for e in self.graph.edges(data="relation")}
      nodes_embedding_dir = os.path.join(cache_path, f"{args.d}_{embedding_method}", "entity")
      edges_embedding_dir = os.path.join(cache_path, f"{args.d}_{embedding_method}", "relation")
      self.nodes_embedding_path = os.path.join(nodes_embedding_dir, f"node_embeddings_{id}.npy")
      self.edges_embedding_path = os.path.join(edges_embedding_dir, f"edge_embeddings_{id}.npy")
      
      # generate embeddings
      self.embedder = LLM_Backbone(args)
      
      if not os.path.exists(nodes_embedding_dir) or not os.path.exists(edges_embedding_dir):
        os.makedirs(nodes_embedding_dir)
        os.makedirs(edges_embedding_dir)
        
      if os.path.exists(self.nodes_embedding_path) and os.path.exists(self.edges_embedding_path) and replace != True:
         self.load_embedddings()
         
      else:
         self.generate_embeddings()
         self.save_embeddings()
      
   def generate_embeddings(self):
      # print("Generating embeddings...")
      nodes_attributes = [node.attribute for node in self.nodes.values()]
      edges_attributes = [edge.attribute for edge in self.edges.values()]
      embeddings = self.embedder.get_embeddings(nodes_attributes)
      embeddings_edges = self.embedder.get_embeddings(edges_attributes)
      
      for i, node in enumerate(self.graph.nodes()):
         self.nodes[node].set_embedding(embeddings[i])
         
      for i, edge in enumerate(self.graph.edges()):
         self.edges[(edge[0], edge[1])].set_embedding(embeddings_edges[i])
   
   def load_embedddings(self):
      # print("Loading embeddings...")
      nodes_embeddings = np.load(self.nodes_embedding_path)
      edges_embeddings = np.load(self.edges_embedding_path)
      
      # if issue occurs during embedding model
      if len(nodes_embeddings) != len(self.graph.nodes()) or len(edges_embeddings) != len(self.graph.edges()):
         self.generate_embeddings()
         self.save_embeddings()
         nodes_embeddings = np.load(self.nodes_embedding_path)
         edges_embeddings = np.load(self.edges_embedding_path)
      
      for i, node in enumerate(self.graph.nodes()):
         self.nodes[node].set_embedding(nodes_embeddings[i])
         
      for i, edge in enumerate(self.graph.edges()):
         self.edges[(edge[0], edge[1])].set_embedding(edges_embeddings[i])
   
   def save_embeddings(self):
      np.save(self.nodes_embedding_path, np.array([node.embedding for node in self.nodes.values()]))
      np.save(self.edges_embedding_path, np.array([edge.embedding for edge in self.edges.values()]))      
   
         
   def __str__(self) -> str:
      return f"Graph with {len(self.nodes)} nodes and {len(self.edges)} edges"