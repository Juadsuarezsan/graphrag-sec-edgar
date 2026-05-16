from src.graph.store import InMemoryGraph


def test_upsert_and_count():
    g = InMemoryGraph()
    g.upsert_node(id="a", type="Company", name="A")
    g.upsert_node(id="b", type="Company", name="B")
    g.upsert_node(id="p", type="Person", name="P")
    assert g.n_nodes == 3
    assert g.count_by_type(type="Company") == 2
    assert g.count_by_type(type="Person") == 1


def test_traverse_finds_neighbors():
    g = InMemoryGraph()
    for n in ("a", "b", "c"):
        g.upsert_node(id=n, type="Company", name=n)
    g.upsert_edge(from_id="a", to_id="b", type="REL")
    g.upsert_edge(from_id="b", to_id="c", type="REL")
    sub = g.traverse(seed_ids=["a"], max_hops=2)
    ids = {n["id"] for n in sub["nodes"]}
    assert ids == {"a", "b", "c"}


def test_vector_search_self_similar_is_top():
    g = InMemoryGraph()
    g.upsert_node(id="apple", type="Company", name="Apple Inc")
    g.upsert_node(id="microsoft", type="Company", name="Microsoft Corp")
    results = g.vector_search(query="Apple Inc", k=2)
    assert results[0]["id"] == "apple"
