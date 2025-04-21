import pytest
from timbal.core.flow.utils import (
    get_ancestors,
    get_sources,
    get_successors,
    is_dag,
    reverse_dag,
)


@pytest.fixture
def simple_dag():
    return {
        "A": {"B", "C"},
        "B": {"D"},
        "C": {"D"},
        "D": set()
    }


@pytest.fixture
def cyclic_graph():
    return {
        "A": {"B"},
        "B": {"C"},
        "C": {"A"}
    }


@pytest.fixture
def complex_dag():
    return {
        "A": {"B", "C", "D"},
        "B": {"E", "F"},
        "C": {"F", "G"},
        "D": {"G"},
        "E": {"H"},
        "F": {"H", "I"},
        "G": {"I"},
        "H": {"J"},
        "I": {"J"},
        "J": set()
    }


@pytest.fixture
def diamond_dag():
    return {
        "A": {"B", "C"},
        "B": {"D"},
        "C": {"D"},
        "D": {"E"},
        "E": set()
    }


def test_is_dag_with_valid_dag(simple_dag):
    assert is_dag(simple_dag) is True


def test_is_dag_with_cyclic_graph(cyclic_graph):
    assert is_dag(cyclic_graph) is False


def test_get_ancestors(simple_dag):
    reversed_dag = reverse_dag(simple_dag)
    assert get_ancestors("D", reversed_dag) == {"A", "B", "C"}
    assert get_ancestors("B", reversed_dag) == {"A"}
    assert get_ancestors("A", reversed_dag) == set()


def test_get_successors(simple_dag):
    assert get_successors("A", simple_dag) == {"B", "C", "D"}
    assert get_successors("B", simple_dag) == {"D"}
    assert get_successors("D", simple_dag) == set()


def test_get_sources(simple_dag):
    assert get_sources(simple_dag) == {"A"}


def test_get_sources_multiple_sources():
    dag = {
        "A": {"C"},
        "B": {"C"},
        "C": set()
    }
    assert get_sources(dag) == {"A", "B"}


def test_complex_paths(complex_dag):
    reversed_dag = reverse_dag(complex_dag)
    assert get_ancestors("F", reversed_dag) == {"A", "B", "C"}
    assert get_ancestors("J", reversed_dag) == {"A", "B", "C", "D", "E", "F", "G", "H", "I"}
    assert get_successors("A", complex_dag) == {"B", "C", "D", "E", "F", "G", "H", "I", "J"}
    assert get_successors("F", complex_dag) == {"H", "I", "J"}
    assert get_successors("J", complex_dag) == set()


def test_disconnected_dag():
    dag = {
        "A": {"B"},
        "B": set(),
        "C": {"D"},
        "D": set(),
        "E": set()
    }
    assert get_sources(dag) == {"A", "C", "E"}
    assert get_successors("A", dag) == {"B"}
    assert get_successors("C", dag) == {"D"}
    assert get_successors("E", dag) == set()


def test_single_node():
    dag = {"A": set()}
    assert is_dag(dag) is True
    assert get_sources(dag) == {"A"}
    assert get_successors("A", dag) == set()
    assert get_ancestors("A", dag) == set()


def test_empty_graph():
    dag = {}
    assert is_dag(dag) is True
    assert get_sources(dag) == set()


def test_diamond_paths(diamond_dag):
    reversed_dag = reverse_dag(diamond_dag)
    assert get_successors("A", diamond_dag) == {"B", "C", "D", "E"}
    assert get_successors("B", diamond_dag) == {"D", "E"}
    assert get_ancestors("D", reversed_dag) == {"A", "B", "C"}
    assert get_ancestors("E", reversed_dag) == {"A", "B", "C", "D"}


def test_invalid_node():
    dag = {"A": {"B"}, "B": set()}
    with pytest.raises(KeyError):
        get_successors("C", dag)
    with pytest.raises(KeyError):
        get_ancestors("C", dag)


def test_complex_cycles():
    # Self-cycle
    assert is_dag({"A": {"A"}}) is False
    
    # Complex cycle
    complex_cycle = {
        "A": {"B"},
        "B": {"C", "D"},
        "C": {"E"},
        "D": {"F"},
        "E": {"G"},
        "F": {"G"},
        "G": {"B"}  # Creates cycle B -> C/D -> ... -> G -> B
    }
    assert is_dag(complex_cycle) is False
