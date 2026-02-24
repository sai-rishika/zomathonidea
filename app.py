import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict

from csao import CSAOConfig, CSAOEngine
from csao.data import sample_cooccurrence, sample_items, sample_users
from csao.service import CSAOService
from csao.storage import FeedbackStore


def build_service() -> CSAOService:
    items = sample_items()
    users = sample_users()
    cooc = sample_cooccurrence()
    engine = CSAOEngine(
        items=items,
        users=users,
        cooccurrence=cooc,
        cfg=CSAOConfig(top_k=8),
        model_path="artifacts/csao_logistic.json",
    )
    store = FeedbackStore("artifacts/csao.db")
    return CSAOService(engine=engine, store=store)


SERVICE = build_service()


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(200, {"ok": True})
            return
        self._send_json(404, {"error": "not_found"})

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8") or "{}")
        except Exception:
            self._send_json(400, {"error": "invalid_json"})
            return

        if self.path == "/recommend":
            try:
                out = SERVICE.recommend(payload)
                self._send_json(200, out)
                return
            except KeyError as exc:
                self._send_json(400, {"error": "missing_field", "field": str(exc)})
                return

        if self.path == "/feedback/accept":
            try:
                out = SERVICE.accept(payload)
                self._send_json(200, out)
                return
            except KeyError as exc:
                self._send_json(400, {"error": "missing_field", "field": str(exc)})
                return

        self._send_json(404, {"error": "not_found"})


def run() -> None:
    server = ThreadingHTTPServer(("0.0.0.0", 8000), Handler)
    print("CSAO API running on http://0.0.0.0:8000")
    print("Endpoints: GET /health, POST /recommend, POST /feedback/accept")
    server.serve_forever()


if __name__ == "__main__":
    run()
