from csao.data import sample_cooccurrence, sample_items, sample_users
from csao.ranker import build_synthetic_training_rows
from csao.ml_model import train_logistic_sgd


def auc_score(y_true: list[int], y_score: list[float]) -> float:
    pos = [(s, y) for s, y in zip(y_score, y_true) if y == 1]
    neg = [(s, y) for s, y in zip(y_score, y_true) if y == 0]
    if not pos or not neg:
        return 0.5

    wins = 0.0
    total = 0.0
    neg_scores = [s for s, _ in neg]
    for ps, _ in pos:
        for ns in neg_scores:
            total += 1.0
            if ps > ns:
                wins += 1.0
            elif ps == ns:
                wins += 0.5
    return wins / total if total else 0.5


def main() -> None:
    items = sample_items()
    users = sample_users()
    cooc = sample_cooccurrence()

    x_rows, y_rows = build_synthetic_training_rows(
        items=items,
        users=users,
        cooccurrence_strength=cooc,
        n_samples=900,
        seed=42,
    )

    split = int(0.8 * len(x_rows))
    x_train, x_test = x_rows[:split], x_rows[split:]
    y_train, y_test = y_rows[:split], y_rows[split:]

    model = train_logistic_sgd(x_train, y_train, epochs=350, lr=0.06, l2=1e-4, seed=42)

    probs = [model.predict_proba(x) for x in x_test]
    auc = auc_score(y_test, probs)

    out = "artifacts/csao_logistic.json"
    model.save(out)

    print(f"Trained rows: {len(x_train)}")
    print(f"Test rows: {len(x_test)}")
    print(f"Test AUC: {auc:.4f}")
    print(f"Saved model: {out}")


if __name__ == "__main__":
    main()
