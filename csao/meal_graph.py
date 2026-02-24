from typing import Dict, List

from .data import CAT_NEXT, Item


def completion_gap_score(cart_item_ids: List[str], candidate_id: str, items: Dict[str, Item]) -> float:
    if candidate_id not in items:
        return 0.0

    ccat = items[candidate_id].category
    existing = {items[i].category for i in cart_item_ids if i in items}

    # Reward candidate if it is the immediate next logical category.
    for cat in existing:
        if ccat in CAT_NEXT.get(cat, []):
            return 1.0

    # Mild reward for still being an unfilled meal component.
    needed = set()
    for cat in existing:
        needed.update(CAT_NEXT.get(cat, []))
    return 0.5 if ccat in needed else 0.0
