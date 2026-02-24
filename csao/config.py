from dataclasses import dataclass


@dataclass(frozen=True)
class CSAOConfig:
    candidate_pool_size: int = 50
    top_k: int = 8

    # Ranking weights
    w_cooccurrence: float = 0.30
    w_meal_completion: float = 0.25
    w_user_preference: float = 0.15
    w_budget_fit: float = 0.10
    w_popularity: float = 0.10
    w_context: float = 0.10

    # Bandit exploration
    bandit_enabled: bool = True
    ucb_alpha: float = 0.25

    # Safety/UX
    max_repeats_per_session: int = 2
