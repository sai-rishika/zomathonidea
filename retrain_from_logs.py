from csao.data import RestaurantContext, sample_cooccurrence, sample_items, sample_users
from csao.ml_model import train_logistic_sgd
from csao.ranker import feature_vector
from csao.storage import FeedbackStore


def main() -> None:
    items = sample_items()
    users = sample_users()
    cooc = sample_cooccurrence()
    store = FeedbackStore("artifacts/csao.db")

    rows = store.fetch_training_rows(limit=50000)
    if len(rows) < 50:
        print("Not enough feedback rows yet. Need at least 50.")
        return

    x_rows = []
    y_rows = []

    for user_id, restaurant_id, city, time_of_day, cart_csv, item_id, reason, accepted in rows:
        if item_id not in items:
            continue
        user = users.get(user_id)
        if user is None:
            # skip unknown users for now; can be replaced with user feature store lookup
            continue

        cart_item_ids = [x for x in cart_csv.split(",") if x in items]
        context = RestaurantContext(
            restaurant_id=restaurant_id,
            cuisine="indian",
            price_level="mid",
            city=city,
        )

        x = feature_vector(
            cart_item_ids=cart_item_ids,
            candidate_id=item_id,
            reason=reason,
            items=items,
            user=user,
            context=context,
            time_of_day=time_of_day,
            cooccurrence_strength=cooc,
        )
        x_rows.append(x)
        y_rows.append(int(accepted))

    if len(x_rows) < 50:
        print("Not enough usable rows after filtering.")
        return

    model = train_logistic_sgd(x_rows, y_rows, epochs=250, lr=0.05, l2=1e-4, seed=7)
    model.save("artifacts/csao_logistic.json")

    pos = sum(y_rows)
    print(f"Trained on {len(y_rows)} rows, positives={pos}, negatives={len(y_rows)-pos}")
    print("Saved model: artifacts/csao_logistic.json")


if __name__ == "__main__":
    main()
