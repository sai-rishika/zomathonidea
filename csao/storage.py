import sqlite3
from pathlib import Path
from typing import List, Tuple


class FeedbackStore:
    def __init__(self, db_path: str = "artifacts/csao.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    restaurant_id TEXT NOT NULL,
                    city TEXT NOT NULL,
                    time_of_day TEXT NOT NULL,
                    cart_item_ids TEXT NOT NULL,
                    item_id TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    accepted INTEGER NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    def log_event(
        self,
        *,
        user_id: str,
        restaurant_id: str,
        city: str,
        time_of_day: str,
        cart_item_ids_csv: str,
        item_id: str,
        reason: str,
        accepted: bool,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO feedback_events(
                    user_id, restaurant_id, city, time_of_day, cart_item_ids,
                    item_id, reason, accepted
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    restaurant_id,
                    city,
                    time_of_day,
                    cart_item_ids_csv,
                    item_id,
                    reason,
                    1 if accepted else 0,
                ),
            )
            conn.commit()

    def fetch_training_rows(self, limit: int = 50000) -> List[Tuple[str, str, str, str, str, str, str, int]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT user_id, restaurant_id, city, time_of_day, cart_item_ids, item_id, reason, accepted
                FROM feedback_events
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return rows
