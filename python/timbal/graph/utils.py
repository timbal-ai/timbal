"""
Utilities for working with Directed Acyclic Graphs (DAGs).

This module provides functions to analyze and traverse DAGs represented as adjacency lists.
Each graph is represented as a dictionary mapping node IDs to sets of their direct successors.

Example:
    dag = {
        'A': {'B', 'C'},
        'B': {'D'},
        'C': {'D'},
        'D': set()
    }
"""

# Type alias for a Directed Acyclic Graph (stored in adjacency list format).
Dag = dict[str, set[str]]


def reverse_dag(dag: Dag) -> Dag:
    """Function to reverse a Directed Acyclic Graph (DAG)."""
    return {
        node_id: {prev_node_id for prev_node_id in dag if node_id in dag[prev_node_id]} 
        for node_id in dag
    }


def is_dag(graph: Dag) -> bool:
    """Function to check if a graph is a Directed Acyclic Graph (DAG)."""
    # States: 0 = unvisited, 1 = visiting, 2 = visited
    state = {node_id: 0 for node_id in graph}
    def dfs(node_id):
        if state[node_id] == 1:
            return False
        if state[node_id] == 2:
            return True
        state[node_id] = 1
        for next_node_id in graph[node_id]:
            if not dfs(next_node_id):
                return False
        state[node_id] = 2
        return True
    for node_id in graph:
        if state[node_id] == 0:
            if not dfs(node_id):
                return False
    return True


def get_ancestors(node_id: str, rev_dag: Dag) -> set[str]:
    """Function to get the ancestors of a node in a (reversed) Directed Acyclic Graph (DAG)."""
    def dfs(node_id, visited):
        if node_id in visited:
            return
        visited.add(node_id)
        for prev_node_id in rev_dag[node_id]:
            dfs(prev_node_id, visited)
        return visited
    ancestors = dfs(node_id, set()) - {node_id}
    return ancestors


def get_successors(node_id: str, dag: Dag) -> set[str]:
    """Function to get the successors of a node in a Directed Acyclic Graph (DAG)."""
    def dfs(node_id, visited):
        if node_id in visited:
            return
        visited.add(node_id)
        for prev_node_id in dag[node_id]:
            dfs(prev_node_id, visited)
        return visited
    successors = dfs(node_id, set()) - {node_id}
    return successors


def get_sources(dag: Dag) -> set[str]:
    """Function to retrieve the nodes with no incoming edges of a Directed Acyclic Graph (DAG)."""
    all_nodes = set(dag)
    nodes_with_incoming_edges = set()
    for next_nodes_ids in dag.values():
        nodes_with_incoming_edges.update(next_nodes_ids)
    initial_steps_ids = all_nodes - nodes_with_incoming_edges
    return initial_steps_ids
