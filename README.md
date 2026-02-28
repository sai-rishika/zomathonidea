# Cart Super Add-On (CSAO) 



## What is real now

- Real-time recommendation engine with ML scoring
- API server endpoints:
  - `GET /health`
  - `POST /recommend`
  - `POST /feedback/accept`


## 1) Train initial model



## 2) Run API server


## 3) Call recommend endpoint


## 4) Log accepted recommendation


## 5) Retrain from real logs



## Important files

- `app.py` - HTTP server
- `csao/service.py` - request handling and feedback logging
- `csao/storage.py` - SQLite storage
- `csao/engine.py` - candidate generation + ML ranking + bandit
- `csao/ranker.py` - feature engineering + training data builder
- `csao/ml_model.py` - logistic model + save/load
- `train_model.py` - initial training script
- `retrain_from_logs.py` - incremental retraining from live feedback
