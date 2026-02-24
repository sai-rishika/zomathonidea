import argparse

from csao import CSAOConfig, CSAOEngine, RecommendationRequest
from csao.data import RestaurantContext, sample_cooccurrence, sample_items, sample_users


def print_recs(stage: str, response: dict) -> None:
    print(f"\\n{stage}")
    print(f"Latency: {response['latency_ms']} ms")
    for i, rec in enumerate(response["recommendations"], start=1):
        print(f"{i}. {rec['name']} ({rec['item_id']}) score={rec['score']} reason={rec['reason']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CSAO demo")
    parser.add_argument("--model", default=None, help="Path to saved ML model JSON")
    args = parser.parse_args()

    items = sample_items()
    users = sample_users()
    cooc = sample_cooccurrence()

    cfg = CSAOConfig(top_k=4)
    engine = CSAOEngine(items=items, users=users, cooccurrence=cooc, cfg=cfg, model_path=args.model)

    context = RestaurantContext(
        restaurant_id="r_10",
        cuisine="hyderabadi",
        price_level="mid",
        city="Hyderabad",
    )

    req1 = RecommendationRequest(
        user_id="u_1",
        restaurant_id="r_10",
        city="Hyderabad",
        time_of_day="dinner",
        cart_item_ids=["m_biryani"],
        top_k=4,
    )
    res1 = engine.recommend(req1, context)
    print_recs("Cart: [Biryani]", res1)

    engine.log_accept("s_salan")

    req2 = RecommendationRequest(
        user_id="u_1",
        restaurant_id="r_10",
        city="Hyderabad",
        time_of_day="dinner",
        cart_item_ids=["m_biryani", "s_salan"],
        top_k=4,
    )
    res2 = engine.recommend(req2, context)
    print_recs("Cart: [Biryani, Salan]", res2)

    req3 = RecommendationRequest(
        user_id="u_1",
        restaurant_id="r_10",
        city="Hyderabad",
        time_of_day="dinner",
        cart_item_ids=["m_biryani", "s_salan", "d_gulab_jamun"],
        top_k=4,
    )
    res3 = engine.recommend(req3, context)
    print_recs("Cart: [Biryani, Salan, Gulab Jamun]", res3)


if __name__ == "__main__":
    main()
