import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from .bandit import UCBBandit
from .candidate_generation import cooccurrence_candidates, meal_graph_candidates, popularity_candidates
from .config import CSAOConfig
from .data import Item, RestaurantContext, UserProfile
from .ml_model import LogisticModel
from .ranker import RankedRecommendation, rank_candidates, train_default_rank_model


@dataclass
class RecommendationRequest:
    user_id: str
    restaurant_id: str
    city: str
    time_of_day: str
    cart_item_ids: List[str]
    top_k: int = 8


class CSAOEngine:
    def __init__(
        self,
        items: Dict[str, Item],
        users: Dict[str, UserProfile],
        cooccurrence: Dict[Tuple[str, str], float],
        cfg: CSAOConfig | None = None,
        model_path: str | None = None,
    ):
        self.items = items
        self.users = users
        self.cooccurrence = cooccurrence
        self.cfg = cfg or CSAOConfig()
        self.bandit = UCBBandit(alpha=self.cfg.ucb_alpha)

        if model_path and Path(model_path).exists():
            self.model = LogisticModel.load(model_path)
        else:
            self.model = train_default_rank_model(
                items=self.items,
                users=self.users,
                cooccurrence_strength=self.cooccurrence,
            )

    def recommend(self, req: RecommendationRequest, context: RestaurantContext) -> Dict[str, object]:
        t0 = time.perf_counter()
        user = self.users.get(req.user_id)
        if user is None:
            user = UserProfile(user_id=req.user_id, veg_only=False, avg_cart_value=300, preferred_cuisines={context.cuisine})

        pool_n = self.cfg.candidate_pool_size
        c1 = cooccurrence_candidates(req.cart_item_ids, self.cooccurrence, limit=pool_n // 2)
        c2 = meal_graph_candidates(req.cart_item_ids, self.items, limit=pool_n // 3)
        c3 = popularity_candidates(req.cart_item_ids, self.items, limit=pool_n // 3)

        seen = set()
        merged: List[Tuple[str, str]] = []
        for bucket in [c1, c2, c3]:
            for iid, reason in bucket:
                if iid in seen:
                    continue
                seen.add(iid)
                merged.append((iid, reason))

        ranked = rank_candidates(
            cart_item_ids=req.cart_item_ids,
            candidate_with_reason=merged,
            items=self.items,
            user=user,
            context=context,
            time_of_day=req.time_of_day,
            cooccurrence_strength=self.cooccurrence,
            cfg=self.cfg,
            model=self.model,
        )

        if self.cfg.bandit_enabled:
            ranked = self._apply_bandit(ranked)

        top = ranked[: req.top_k]
        for rec in top:
            self.bandit.log_impression(rec.item_id)

        latency_ms = int((time.perf_counter() - t0) * 1000)
        return {
            "recommendations": [
                {
                    "item_id": r.item_id,
                    "name": self.items[r.item_id].name,
                    "score": round(r.score, 4),
                    "reason": r.reason,
                }
                for r in top
            ],
            "latency_ms": latency_ms,
        }

    def _apply_bandit(self, ranked: List[RankedRecommendation]) -> List[RankedRecommendation]:
        reranked: List[RankedRecommendation] = []
        for rec in ranked:
            boost = self.bandit.score_boost(rec.item_id)
            reranked.append(RankedRecommendation(item_id=rec.item_id, score=rec.score + boost, reason=rec.reason))
        reranked.sort(key=lambda r: r.score, reverse=True)
        return reranked

    def log_accept(self, item_id: str) -> None:
        self.bandit.log_accept(item_id)
