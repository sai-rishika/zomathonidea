from typing import Dict, List

from .data import Item, UserProfile, RestaurantContext


def cart_value(cart_item_ids: List[str], items: Dict[str, Item]) -> int:
    return sum(items[i].price for i in cart_item_ids if i in items)


def category_distribution(cart_item_ids: List[str], items: Dict[str, Item]) -> Dict[str, int]:
    dist: Dict[str, int] = {}
    for iid in cart_item_ids:
        if iid not in items:
            continue
        c = items[iid].category
        dist[c] = dist.get(c, 0) + 1
    return dist


def budget_fit_score(user: UserProfile, cart_total: int, candidate: Item) -> float:
    # Encourage add-ons that don't drastically overshoot user's typical spend.
    target = user.avg_cart_value
    new_total = cart_total + candidate.price
    gap = abs(target - new_total)
    norm = max(target, 1)
    return max(0.0, 1.0 - (gap / (norm * 1.25)))


def user_preference_score(user: UserProfile, candidate: Item) -> float:
    if user.veg_only and not candidate.veg:
        return 0.0
    cuisine_boost = 1.0 if candidate.cuisine in user.preferred_cuisines else 0.6
    veg_boost = 1.0 if (not user.veg_only or candidate.veg) else 0.0
    return cuisine_boost * veg_boost


def context_score(context: RestaurantContext, candidate: Item, time_of_day: str) -> float:
    score = 0.5
    if candidate.cuisine == context.cuisine or candidate.cuisine == "indian":
        score += 0.2

    # Simple time heuristic
    if time_of_day in {"lunch", "dinner"} and candidate.category in {"side", "dessert", "beverage"}:
        score += 0.2
    if time_of_day == "breakfast" and candidate.category == "dessert":
        score -= 0.1

    return min(max(score, 0.0), 1.0)
