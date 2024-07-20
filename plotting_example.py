import datasets
import os
from src.utils import build_graph
from datasets import load_from_disk
import networkx as nx
import matplotlib.pyplot as plt

cached_dataset_path = os.path.join("/data/yuansui/rog/datasets", "RoG-cwq_processed")
dataset = load_from_disk(cached_dataset_path)

sample = dataset[0]

starting_node = sample["q_entity"]
print(f"Starting Node: {starting_node}")
graph = build_graph(sample["graph"])

# Get neighbors of the starting node
neighbors_1_hop = list(graph.neighbors(starting_node[0]))

# Get 2-hop neighbors
neighbors_2_hop = set()
for neighbor in neighbors_1_hop:
    neighbors_2_hop.update(graph.neighbors(neighbor))

# Remove the starting node from the 2-hop neighbors if present
neighbors_2_hop.discard(starting_node[0])

print(f"Graph: {graph}")
print(f"1-hop Neighbors: {neighbors_1_hop}")
print(f"2-hop Neighbors: {neighbors_2_hop}")

# Plot the graph
pos = nx.spring_layout(graph, k=0.5, iterations=50)  # positions for all nodes
plt.figure(figsize=(16, 16))

# Draw nodes
nx.draw_networkx_nodes(
    graph,
    pos,
    nodelist=[starting_node[0]],
    node_color='grey',
    node_size=300,
    label='Starting Node',
)
nx.draw_networkx_nodes(
    graph,
    pos,
    nodelist=neighbors_1_hop,
    node_color='blue',
    node_size=300,
    label='1-hop Neighbors',
)
nx.draw_networkx_nodes(
    graph,
    pos,
    nodelist=list(neighbors_2_hop),
    node_color='green',
    node_size=300,
    label='2-hop Neighbors',
)

# Draw edges
nx.draw_networkx_edges(graph, pos, edgelist=graph.edges(), alpha=0.5)

# Draw labels
nx.draw_networkx_labels(graph, pos, font_size=4)

# Set legend
plt.legend(scatterpoints=1)

# Show plot
plt.show()

plt.savefig("graph.pdf")
