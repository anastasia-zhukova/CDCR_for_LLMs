from collections import defaultdict
from collections.abc import Iterable


class UnionFind:

    """Disjoint-set forest over mention ids, with path halving on find.

    Used both to build predicted clusters from positive pairwise predictions
    (CoreferenceEvaluator) and to drive dynamic transitive grouping during a run:
    once A~B and B~C are predicted coreferent, A and C share a root, so the A-C
    pair can be inferred coreferent without spending an LLM request on it.
    """

    def __init__(self, nodes: Iterable[str]) -> None:
        self._parent: dict[str, str] = {node: node for node in nodes}

    def __contains__(self, node: str) -> bool:
        return node in self._parent

    def find(self, node: str) -> str:
        parent = self._parent
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(self, a: str, b: str) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self._parent[root_a] = root_b

    def connected(self, a: str, b: str) -> bool:
        return self.find(a) == self.find(b)

    def groups(self) -> list[list[str]]:
        clusters: dict[str, list[str]] = defaultdict(list)
        for node in self._parent:
            clusters[self.find(node)].append(node)
        return list(clusters.values())
