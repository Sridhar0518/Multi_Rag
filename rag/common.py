import time
from dataclasses import dataclass, field

from config import Hyperparams


@dataclass
class RagContext:
    index: object            # VectorIndex
    graph: object            # GraphStore
    llm: object
    hp: Hyperparams = field(default_factory=Hyperparams)


class Timer:
    def __init__(self, t: dict, key: str):
        self.t, self.key = t, key
    def __enter__(self):
        self.s = time.perf_counter()
    def __exit__(self, *a):
        self.t[self.key] = self.t.get(self.key, 0) + (time.perf_counter() - self.s) * 1000
