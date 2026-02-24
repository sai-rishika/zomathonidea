import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .config import CSAOConfig
from .data import Item, RestaurantContext, UserProfile
from .features import budget_fit_score, cart_value, context_score, user_preference_score
from .meal_graph import completion_gap_score
from .ml_model import LogisticModel, train_logistic_sgd


@dataclass(frozen=True)
class RankedRecommendation:
    item_id: str
    score: float
    reason: str


def _max_cooccurrence(cart_item_ids: List[str], candidate_id: str, cooccurrence_strength: Dict[Tuple[str, str], float]) -> float:
    max_co = 0.0
    for src in cart_item_ids:
        max_co = max(max_co, cooccurrence_strength.get((src, candidate_id), 0.0))
    return max_co


def feature_vector(
    *,
    cart_item_ids: List[str],
    candidate_id: str,
    reason: str,
    items: Dict[str, Item],
    user: UserProfile,
    context: RestaurantContext,
    time_of_day: str,
    cooccurrence_strength: Dict[Tuple[str, str], float],
) -> List[float]:
    cand = items[candidate_id]
    cart_total = cart_value(cart_item_ids, items)

    max_co = _max_cooccurrence(cart_item_ids, candidate_id, cooccurrence_strength)
    meal = completion_gap_score(cart_item_ids, candidate_id, items)
    pref = user_preference_score(user, cand)
    budget = budget_fit_score(user, cart_total, cand)
    pop = cand.popularity
    ctx = context_score(context, cand, time_of_day)

    is_side = 1.0 if cand.category == "side" else 0.0
    is_dessert = 1.0 if cand.category == "dessert" else 0.0
    is_beverage = 1.0 if cand.category == "beverage" else 0.0
    is_main = 1.0 if cand.category == "main" else 0.0

    reason_co = 1.0 if reason == "co_occurrence" else 0.0
    reason_meal = 1.0 if reason == "meal_completion" else 0.0
    reason_pop = 1.0 if reason == "popularity" else 0.0

    return [
        max_co,
        meal,
        pref,
        budget,
        pop,
        ctx,
        is_main,
        is_side,
        is_dessert,
        is_beverage,
        reason_co,
        reason_meal,
        reason_pop,
    ]


def build_synthetic_training_rows(
    *,
    items: Dict[str, Item],
    users: Dict[str, UserProfile],
    cooccurrence_strength: Dict[Tuple[str, str], float],
    n_samples: int = 500,
    seed: int = 123,
) -> Tuple[List[List[float]], List[int]]:
    rnd = random.Random(seed)

    contexts = [
        RestaurantContext("r_10", cuisine="hyderabadi", price_level="mid", city="Hyderabad"),
        RestaurantContext("r_20", cuisine="indian", price_level="mid", city="Bengaluru"),
    ]
    times = ["lunch", "dinner"]

    x_rows: List[List[float]] = []
    y_rows: List[int] = []

    item_ids = list(items.keys())
    user_ids = list(users.keys())

    for _ in range(n_samples):
        user = users[rnd.choice(user_ids)]
        context = rnd.choice(contexts)
        time_of_day = rnd.choice(times)

        mains = [iid for iid in item_ids if items[iid].category == "main"]
        cart_item_ids = [rnd.choice(mains)]
        if rnd.random() < 0.35:
            cart_item_ids.append(rnd.choice([iid for iid in item_ids if iid != cart_item_ids[0]]))

        for candidate_id in item_ids:
            if candidate_id in cart_item_ids:
                continue

            max_co = _max_cooccurrence(cart_item_ids, candidate_id, cooccurrence_strength)
            meal = completion_gap_score(cart_item_ids, candidate_id, items)
            pref = user_preference_score(user, items[candidate_id])

            accept_prob = 0.05 + 0.60 * max_co + 0.25 * meal + 0.10 * pref
            accept_prob = max(0.0, min(0.95, accept_prob))
            y = 1 if rnd.random() < accept_prob else 0

            reason = "co_occurrence" if max_co > 0.0 else ("meal_completion" if meal > 0.0 else "popularity")
            x = feature_vector(
                cart_item_ids=cart_item_ids,
                candidate_id=candidate_id,
                reason=reason,
                items=items,
                user=user,
                context=context,
                time_of_day=time_of_day,
                cooccurrence_strength=cooccurrence_strength,
            )
            x_rows.append(x)
            y_rows.append(y)

    return x_rows, y_rows


def train_default_rank_model(
    *,
    items: Dict[str, Item],
    users: Dict[str, UserProfile],
    cooccurrence_strength: Dict[Tuple[str, str], float],
) -> LogisticModel:
    x_rows, y_rows = build_synthetic_training_rows(
        items=items,
        users=users,
        cooccurrence_strength=cooccurrence_strength,
        n_samples=500,
        seed=123,
    )
    return train_logistic_sgd(x_rows, y_rows)


def rank_candidates(
    cart_item_ids: List[str],
    candidate_with_reason: List[Tuple[str, str]],
    items: Dict[str, Item],
    user: UserProfile,
    context: RestaurantContext,
    time_of_day: str,
    cooccurrence_strength: Dict[Tuple[str, str], float],
    cfg: CSAOConfig,
    model: LogisticModel,
) -> List[RankedRecommendation]:
    results: List[RankedRecommendation] = []
    for item_id, reason in candidate_with_reason:
        if item_id in cart_item_ids or item_id not in items:
            continue

        x = feature_vector(
            cart_item_ids=cart_item_ids,
            candidate_id=item_id,
            reason=reason,
            items=items,
            user=user,
            context=context,
            time_of_day=time_of_day,
            cooccurrence_strength=cooccurrence_strength,
        )
        score = model.predict_proba(x)
        results.append(RankedRecommendation(item_id=item_id, score=score, reason=reason))

    results.sort(key=lambda r: r.score, reverse=True)
    return results
