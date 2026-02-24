from dataclasses import dataclass
from typing import Dict, List

from .data import RestaurantContext
from .engine import CSAOEngine, RecommendationRequest
from .storage import FeedbackStore


@dataclass
class RecoRecord:
    item_id: str
    reason: str


class CSAOService:
    def __init__(self, engine: CSAOEngine, store: FeedbackStore):
        self.engine = engine
        self.store = store

    def recommend(self, payload: Dict[str, object]) -> Dict[str, object]:
        req = RecommendationRequest(
            user_id=str(payload["user_id"]),
            restaurant_id=str(payload["restaurant_id"]),
            city=str(payload["city"]),
            time_of_day=str(payload["time_of_day"]),
            cart_item_ids=list(payload["cart_item_ids"]),
            top_k=int(payload.get("top_k", 8)),
        )
        context = RestaurantContext(
            restaurant_id=req.restaurant_id,
            cuisine=str(payload.get("restaurant_cuisine", "indian")),
            price_level=str(payload.get("restaurant_price_level", "mid")),
            city=req.city,
        )

        response = self.engine.recommend(req, context)

        # Log impressions for future retraining.
        cart_csv = ",".join(req.cart_item_ids)
        for rec in response["recommendations"]:
            self.store.log_event(
                user_id=req.user_id,
                restaurant_id=req.restaurant_id,
                city=req.city,
                time_of_day=req.time_of_day,
                cart_item_ids_csv=cart_csv,
                item_id=rec["item_id"],
                reason=rec["reason"],
                accepted=False,
            )

        return response

    def accept(self, payload: Dict[str, object]) -> Dict[str, object]:
        user_id = str(payload["user_id"])
        restaurant_id = str(payload["restaurant_id"])
        city = str(payload["city"])
        time_of_day = str(payload["time_of_day"])
        cart_item_ids = list(payload["cart_item_ids"])
        item_id = str(payload["item_id"])
        reason = str(payload.get("reason", "unknown"))

        self.engine.log_accept(item_id)

        self.store.log_event(
            user_id=user_id,
            restaurant_id=restaurant_id,
            city=city,
            time_of_day=time_of_day,
            cart_item_ids_csv=",".join(cart_item_ids),
            item_id=item_id,
            reason=reason,
            accepted=True,
        )
        return {"ok": True}
