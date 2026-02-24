# Cart Super Add-On (CSAO) - Real Backend Starter

A real, app-integratable recommendation backend for Zomato-like apps.

## What is real now

- Real-time recommendation engine with ML scoring
- API server endpoints:
  - `GET /health`
  - `POST /recommend`
  - `POST /feedback/accept`
- Persistent feedback store in SQLite (`artifacts/csao.db`)
- Retraining pipeline from logged feedback
- Saved model artifact (`artifacts/csao_logistic.json`)

## 1) Train initial model

```bash
cd /Users/sairishika/Downloads/zomathon
python3 train_model.py
```

Expected output includes `Test AUC` and saved model path.

## 2) Run API server

```bash
python3 app.py
```

## 3) Call recommend endpoint

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "u_1",
    "restaurant_id": "r_10",
    "city": "Hyderabad",
    "time_of_day": "dinner",
    "restaurant_cuisine": "hyderabadi",
    "restaurant_price_level": "mid",
    "cart_item_ids": ["m_biryani"],
    "top_k": 4
  }'
```

## 4) Log accepted recommendation

```bash
curl -X POST http://localhost:8000/feedback/accept \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "u_1",
    "restaurant_id": "r_10",
    "city": "Hyderabad",
    "time_of_day": "dinner",
    "cart_item_ids": ["m_biryani"],
    "item_id": "s_salan",
    "reason": "co_occurrence"
  }'
```

## 5) Retrain from real logs

```bash
python3 retrain_from_logs.py
```

This updates `artifacts/csao_logistic.json` with behavior from actual accepts/impressions.

## Important files

- `app.py` - HTTP server
- `csao/service.py` - request handling and feedback logging
- `csao/storage.py` - SQLite storage
- `csao/engine.py` - candidate generation + ML ranking + bandit
- `csao/ranker.py` - feature engineering + training data builder
- `csao/ml_model.py` - logistic model + save/load
- `train_model.py` - initial training script
- `retrain_from_logs.py` - incremental retraining from live feedback
