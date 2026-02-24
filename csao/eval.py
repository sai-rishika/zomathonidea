import math
from typing import Iterable, List, Sequence, Set


def precision_at_k(pred: Sequence[str], truth: Set[str], k: int) -> float:
    p = pred[:k]
    if not p:
        return 0.0
    hit = sum(1 for x in p if x in truth)
    return hit / len(p)


def recall_at_k(pred: Sequence[str], truth: Set[str], k: int) -> float:
    if not truth:
        return 0.0
    p = pred[:k]
    hit = sum(1 for x in p if x in truth)
    return hit / len(truth)


def ndcg_at_k(pred: Sequence[str], truth: Set[str], k: int) -> float:
    p = pred[:k]
    dcg = 0.0
    for i, item in enumerate(p, start=1):
        rel = 1.0 if item in truth else 0.0
        dcg += rel / math.log2(i + 1)

    ideal_hits = min(k, len(truth))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def batch_mean(values: Iterable[float]) -> float:
    vals: List[float] = list(values)
    if not vals:
        return 0.0
    return sum(vals) / len(vals)
