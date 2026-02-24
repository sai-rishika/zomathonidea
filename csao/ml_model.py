import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence


@dataclass(frozen=True)
class LogisticModel:
    weights: List[float]
    bias: float

    def predict_proba(self, x: Sequence[float]) -> float:
        z = self.bias
        for w, xi in zip(self.weights, x):
            z += w * xi
        # Numerical stability for sigmoid
        if z >= 0:
            ez = math.exp(-z)
            return 1.0 / (1.0 + ez)
        ez = math.exp(z)
        return ez / (1.0 + ez)

    def to_dict(self) -> dict:
        return {"weights": self.weights, "bias": self.bias}

    @staticmethod
    def from_dict(payload: dict) -> "LogisticModel":
        return LogisticModel(weights=list(payload["weights"]), bias=float(payload["bias"]))

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict()), encoding="utf-8")

    @staticmethod
    def load(path: str) -> "LogisticModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return LogisticModel.from_dict(payload)


def train_logistic_sgd(
    x_rows: List[List[float]],
    y_rows: List[int],
    *,
    epochs: int = 250,
    lr: float = 0.08,
    l2: float = 1e-4,
    seed: int = 42,
) -> LogisticModel:
    if not x_rows or not y_rows or len(x_rows) != len(y_rows):
        raise ValueError("Invalid training data for logistic model")

    dim = len(x_rows[0])
    w = [0.0] * dim
    b = 0.0

    idx = list(range(len(x_rows)))
    rnd = random.Random(seed)

    for _ in range(epochs):
        rnd.shuffle(idx)
        for i in idx:
            x = x_rows[i]
            y = y_rows[i]

            z = b + sum(wj * xj for wj, xj in zip(w, x))
            if z >= 0:
                ez = math.exp(-z)
                p = 1.0 / (1.0 + ez)
            else:
                ez = math.exp(z)
                p = ez / (1.0 + ez)

            err = p - y

            # SGD step with L2 regularization.
            for j in range(dim):
                grad = err * x[j] + l2 * w[j]
                w[j] -= lr * grad
            b -= lr * err

    return LogisticModel(weights=w, bias=b)
