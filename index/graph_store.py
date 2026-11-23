"""Graph adjacency index (Entity -> Relation -> Target) with k-hop traversal.

GraphStore is an interface; NetworkXGraph is the default implementation. A Neo4j
implementation can be dropped in by implementing the same methods.
"""
from abc import ABC, abstractmethod

import networkx as nx
import numpy as np

from ingest.entities import normalize


class GraphStore(ABC):
    @abstractmethod
    def add_extraction(self, ex: dict): ...
    @abstractmethod
    def link_query(self, query: str) -> list[str]: ...
    @abstractmethod
    def k_hop(self, seeds: list[str], k: int) -> tuple[set[str], list[tuple]]: ...
    @abstractmethod
    def chunks_for(self, nodes) -> list[str]: ...


class NetworkXGraph(GraphStore):
    def __init__(self):
        self.g = nx.MultiDiGraph()
        self.disabled_nodes: set[str] = set()   # UI filter (Tab 2)
        self.disabled_edges: set[tuple] = set()

    def add_extraction(self, ex: dict):
        cid = ex["chunk_id"]
        for e in ex["entities"]:
            n = e["name"]
            if n not in self.g:
                self.g.add_node(n, type=e["type"], chunks=set())
            self.g.nodes[n]["chunks"].add(cid)
        for r in ex["relations"]:
            for n in (r["source"], r["target"]):
                if n not in self.g:
                    self.g.add_node(n, type="Concept", chunks=set())
                self.g.nodes[n]["chunks"].add(cid)
            self.g.add_edge(r["source"], r["target"], relation=r["relation"], chunk_id=cid)

    def _active(self):
        keep = [n for n in self.g if n not in self.disabled_nodes]
        return self.g.subgraph(keep)

    def link_query(self, query: str) -> list[str]:
        """Entity linking: alias-normalised substring match of node names in the query."""
        q = " " + query.lower() + " "
        found = []
        for n in self._active():
            if f" {n.lower()} " in q or n.lower() in q:
                found.append(n)
        # alias match (e.g. "Income Tax Dept" -> CBDT)
        from ingest.entities import _ALIAS_LOOKUP
        for alias, canon in _ALIAS_LOOKUP.items():
            if alias in q and canon in self.g and canon not in found and canon not in self.disabled_nodes:
                found.append(canon)
        return found

    def k_hop(self, seeds, k):
        sub = self._active()
        und = sub.to_undirected(as_view=True)
        seen = {s for s in seeds if s in sub}
        frontier = set(seen)
        for _ in range(k):
            nxt = set()
            for n in frontier:
                nxt.update(und.neighbors(n))
            frontier = nxt - seen
            seen |= nxt
        edges = [(u, v, d["relation"], d["chunk_id"]) for u, v, d in sub.edges(data=True)
                 if u in seen and v in seen and (u, v, d["relation"]) not in self.disabled_edges]
        return seen, edges

    def chunks_for(self, nodes):
        ids = []
        for n in nodes:
            ids.extend(self.g.nodes[n]["chunks"])
        return ids

    def similar_nodes(self, seeds: list[str], top_n: int = 2, dim: int = 16) -> list[str]:
        """Entities structurally closest (cosine, random-walk embedding) to the seeds."""
        emb = self.node_embeddings(dim=dim)
        act = [n for n in emb if n not in self.disabled_nodes and n not in seeds]
        out = {}
        for s in seeds:
            if s not in emb:
                continue
            v = emb[s]
            for n in act:
                den = np.linalg.norm(v) * np.linalg.norm(emb[n])
                sim = float(v @ emb[n] / den) if den else 0.0
                out[n] = max(out.get(n, -1), sim)
        return [n for n, _ in sorted(out.items(), key=lambda x: -x[1])[:top_n]]

    def node_embeddings(self, dim=16, walks=10, length=8, seed=0) -> dict[str, np.ndarray]:
        """Lightweight Node2Vec-style structural embeddings: random-walk co-occurrence + SVD."""
        rng = np.random.default_rng(seed)
        nodes = list(self.g)
        if not nodes:
            return {}
        idx = {n: i for i, n in enumerate(nodes)}
        und = self.g.to_undirected()
        M = np.zeros((len(nodes), len(nodes)))
        for n in nodes:
            for _ in range(walks):
                cur = n
                for _ in range(length):
                    nb = list(und.neighbors(cur))
                    if not nb:
                        break
                    cur = nb[rng.integers(len(nb))]
                    M[idx[n], idx[cur]] += 1
        u, s, _ = np.linalg.svd(np.log1p(M), full_matrices=False)
        d = min(dim, len(nodes))
        emb = u[:, :d] * s[:d]
        return {n: emb[idx[n]] for n in nodes}
