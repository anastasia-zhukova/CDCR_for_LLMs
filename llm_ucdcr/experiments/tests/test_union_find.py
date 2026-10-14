from llm_ucdcr.experiments.union_find import UnionFind


def test_union_makes_nodes_transitively_connected() -> None:
    uf = UnionFind(["a", "b", "c", "d"])
    uf.union("a", "b")
    uf.union("b", "c")

    assert uf.connected("a", "c")  # transitive: a~b, b~c => a~c
    assert not uf.connected("a", "d")


def test_groups_partition_every_node_once() -> None:
    uf = UnionFind(["a", "b", "c", "d"])
    uf.union("a", "b")

    groups = {frozenset(group) for group in uf.groups()}
    assert groups == {frozenset({"a", "b"}), frozenset({"c"}), frozenset({"d"})}


def test_contains_reports_known_nodes() -> None:
    uf = UnionFind(["a", "b"])
    assert "a" in uf
    assert "z" not in uf
