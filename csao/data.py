from dataclasses import dataclass
from typing import Dict, List, Set, Tuple


@dataclass(frozen=True)
class Item:
    item_id: str
    name: str
    category: str  # main, side, dessert, beverage
    cuisine: str
    price: int
    veg: bool
    popularity: float


@dataclass(frozen=True)
class UserProfile:
    user_id: str
    veg_only: bool
    avg_cart_value: int
    preferred_cuisines: Set[str]


@dataclass(frozen=True)
class RestaurantContext:
    restaurant_id: str
    cuisine: str
    price_level: str  # low, mid, high
    city: str


CAT_NEXT = {
    "main": ["side"],
    "side": ["dessert"],
    "dessert": ["beverage"],
}


def sample_items() -> Dict[str, Item]:
    items = [
        Item("m_biryani", "Chicken Biryani", "main", "hyderabadi", 280, False, 0.95),
        Item("m_paneer_biryani", "Paneer Biryani", "main", "hyderabadi", 260, True, 0.78),
        Item("s_salan", "Mirchi Ka Salan", "side", "hyderabadi", 70, True, 0.80),
        Item("s_raita", "Raita", "side", "indian", 60, True, 0.72),
        Item("d_gulab_jamun", "Gulab Jamun", "dessert", "indian", 90, True, 0.85),
        Item("d_khubani", "Khubani Ka Meetha", "dessert", "hyderabadi", 120, True, 0.67),
        Item("b_coke", "Coke", "beverage", "global", 50, True, 0.88),
        Item("b_lassi", "Sweet Lassi", "beverage", "indian", 80, True, 0.63),
        Item("u_kebab_platter", "Kebab Platter", "main", "mughlai", 350, False, 0.74),
    ]
    return {i.item_id: i for i in items}


def sample_cooccurrence() -> Dict[Tuple[str, str], float]:
    # directional affinity: P(b | a)-like normalized strength
    return {
        ("m_biryani", "s_salan"): 0.82,
        ("m_biryani", "s_raita"): 0.54,
        ("s_salan", "d_gulab_jamun"): 0.58,
        ("s_raita", "d_gulab_jamun"): 0.46,
        ("d_gulab_jamun", "b_coke"): 0.51,
        ("m_paneer_biryani", "s_raita"): 0.63,
        ("m_paneer_biryani", "d_gulab_jamun"): 0.42,
    }


def sample_users() -> Dict[str, UserProfile]:
    return {
        "u_1": UserProfile("u_1", veg_only=False, avg_cart_value=320, preferred_cuisines={"hyderabadi", "indian"}),
        "u_2": UserProfile("u_2", veg_only=True, avg_cart_value=250, preferred_cuisines={"indian"}),
    }
