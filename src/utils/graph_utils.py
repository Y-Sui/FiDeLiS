import networkx as nx
from collections import deque
from typing import List as list
from typing import Tuple as tuple
import walker


def build_graph(graph: list) -> nx.Graph:
    G = nx.Graph()
    for triplet in graph:
        h, r, t = triplet
        G.add_edge(h, t, relation=r.strip())
    return G


# 定义一个函数来进行宽度优先搜索
def bfs_with_rule(graph, start_node, target_rule, max_p=10):
    result_paths = []
    queue = deque([(start_node, [])])  # 使用队列存储待探索节点和对应路径
    while queue:
        current_node, current_path = queue.popleft()

        # 如果当前路径符合规则，将其添加到结果列表中
        if len(current_path) == len(target_rule):
            result_paths.append(current_path)
            # if len(result_paths) >= max_p:
            #     break

        # 如果当前路径长度小于规则长度，继续探索
        if len(current_path) < len(target_rule):
            if current_node not in graph:
                continue
            for neighbor in graph.neighbors(current_node):
                # 剪枝：如果当前边类型与规则中的对应位置不匹配，不继续探索该路径
                rel = graph[current_node][neighbor]['relation']
                if rel != target_rule[len(current_path)] or len(current_path) > len(
                    target_rule
                ):
                    continue
                queue.append((neighbor, current_path + [(current_node, rel, neighbor)]))

    return result_paths


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


def get_simple_paths(q_entity: list, a_entity: list, graph: nx.Graph, hop=2) -> list:
    '''
    Get all simple paths connecting question and answer entities within given hop
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
                for p in nx.all_simple_edge_paths(graph, h, t, cutoff=hop):
                    paths.append(p)
            except:
                pass
    # Add relation to paths
    result_paths = []
    for p in paths:
        result_paths.append([(e[0], graph[e[0]][e[1]]['relation'], e[1]) for e in p])
    return result_paths


def get_negative_paths(
    q_entity: list, a_entity: list, graph: nx.Graph, n_neg: int, hop=2
) -> list:
    '''
    Get negative paths for question witin hop
    '''
    # sample paths
    start_nodes = []
    end_nodes = []
    node_idx = list(graph.nodes())
    for h in q_entity:
        if h in graph:
            start_nodes.append(node_idx.index(h))
    for t in a_entity:
        if t in graph:
            end_nodes.append(node_idx.index(t))
    paths = walker.random_walks(
        graph, n_walks=n_neg, walk_len=hop, start_nodes=start_nodes, verbose=False
    )
    # Add relation to paths
    result_paths = []
    for p in paths:
        tmp = []
        # remove paths that end with answer entity
        if p[-1] in end_nodes:
            continue
        for i in range(len(p) - 1):
            u = node_idx[p[i]]
            v = node_idx[p[i + 1]]
            tmp.append((u, graph[u][v]['relation'], v))
        result_paths.append(tmp)
    return result_paths


def get_random_paths(q_entity: list, graph: nx.Graph, n=3, hop=2) -> tuple[list, list]:
    '''
    Get negative paths for question witin hop
    '''
    # sample paths
    start_nodes = []
    node_idx = list(graph.nodes())
    for h in q_entity:
        if h in graph:
            start_nodes.append(node_idx.index(h))
    paths = walker.random_walks(
        graph, n_walks=n, walk_len=hop, start_nodes=start_nodes, verbose=False
    )
    # Add relation to paths
    result_paths = []
    rules = []
    for p in paths:
        tmp = []
        tmp_rule = []
        for i in range(len(p) - 1):
            u = node_idx[p[i]]
            v = node_idx[p[i + 1]]
            tmp.append((u, graph[u][v]['relation'], v))
            tmp_rule.append(graph[u][v]['relation'])
        result_paths.append(tmp)
        rules.append(tmp_rule)
    return result_paths, rules


def get_entity_edges_with_neighbors(entity: list, graph: nx.Graph) -> list:
    '''
    given an entity, find all edges and neighbors
    '''
    results = []
    for h in entity:
        neighbors = []
        edges = []
        if graph.has_node(h):
            for neighbor in graph.neighbors(h):
                neighbors.append(neighbor)
                edges.append(graph[h][neighbor]['relation'])
            results.append(h, neighbors, edges)
    return results


def get_entity_edges_with_neighbors_single(entity: str, graph: nx.Graph) -> list:
    neighbors = []
    edges = []
    if graph.has_node(entity):
        for neighbor in graph.neighbors(entity):
            neighbors.append(neighbor)
            edges.append(graph[entity][neighbor]['relation'])
    return entity, edges, neighbors


def get_next_entity(entity: str, relation: str, graph: nx.Graph) -> list:
    '''
    given an entity and relation, find the next entity
    '''
    if entity in graph:
        for neighbor in graph.neighbors(entity):
            if graph[entity][neighbor]['relation'] == relation:
                return neighbor


def get_mcq_paths(
    q_entity: list, a_entity: list, graph: nx.Graph, shortest_paths: list[list]
) -> list:
    '''
    Get multiple choice question paths, hop number is the shortest path length (using shortest path as supervision)
    '''
    mcq_paths = []
    candidate_entities = []

    # get all entities in the shortest paths
    for path in shortest_paths:
        for p in path:
            if p[0] in q_entity and p[0] not in candidate_entities:
                candidate_entities.append(p[0])
            if p[2] not in a_entity:
                candidate_entities.append(p[2])

    # get all paths between question entities and candidate entities
    for e in candidate_entities:
        if e not in graph:
            continue
        edges, neighbors = get_entity_edges_with_neighbors(e, graph)
        for i in range(len(edges)):
            mcq_paths.append((e, edges[i], neighbors[i]))

    return mcq_paths
