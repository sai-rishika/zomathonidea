import math
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class BanditState:
    impressions: Dict[str, int] = field(default_factory=dict)
    accepts: Dict[str, int] = field(default_factory=dict)


class UCBBandit:
    def __init__(self, alpha: float = 0.25):
        self.alpha = alpha
        self.state = BanditState()

    def score_boost(self, item_id: str) -> float:
        n = self.state.impressions.get(item_id, 0)
        r = self.state.accepts.get(item_id, 0)
        total = sum(self.state.impressions.values()) + 1

        if n == 0:
            return self.alpha  # explore unseen items

        ctr = r / n
        ucb = ctr + math.sqrt((2.0 * math.log(total)) / n)
        return self.alpha * min(ucb, 1.5)

    def log_impression(self, item_id: str) -> None:
        self.state.impressions[item_id] = self.state.impressions.get(item_id, 0) + 1

    def log_accept(self, item_id: str) -> None:
        self.state.accepts[item_id] = self.state.accepts.get(item_id, 0) + 1
