from typing import Dict, List, Set, Tuple

from .data import CAT_NEXT, Item


def cooccurrence_candidates(
    cart_item_ids: List[str],
    cooccurrence: Dict[Tuple[str, str], float],
    limit: int,
) -> List[Tuple[str, str]]:
    # Returns (candidate_id, reason)
    scores: Dict[str, float] = {}
    for source in cart_item_ids:
        for (a, b), s in cooccurrence.items():
            if a != source:
                continue
            scores[b] = max(scores.get(b, 0.0), s)

    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(iid, "co_occurrence") for iid, _ in ordered[:limit]]


def meal_graph_candidates(cart_item_ids: List[str], items: Dict[str, Item], limit: int) -> List[Tuple[str, str]]:
    existing_categories = {items[i].category for i in cart_item_ids if i in items}
    targets: Set[str] = set()

    for c in existing_categories:
        for nxt in CAT_NEXT.get(c, []):
            if nxt not in existing_categories:
                targets.add(nxt)

    candidates: List[Tuple[str, str, float]] = []
    for iid, item in items.items():
        if item.category in targets and iid not in cart_item_ids:
            candidates.append((iid, "meal_completion", item.popularity))

    candidates.sort(key=lambda x: x[2], reverse=True)
    return [(iid, reason) for iid, reason, _ in candidates[:limit]]


def popularity_candidates(cart_item_ids: List[str], items: Dict[str, Item], limit: int) -> List[Tuple[str, str]]:
    remaining = [(iid, item.popularity) for iid, item in items.items() if iid not in cart_item_ids]
    remaining.sort(key=lambda x: x[1], reverse=True)
    return [(iid, "popularity") for iid, _ in remaining[:limit]]
