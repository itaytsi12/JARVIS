from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer


HOST = "127.0.0.1"
PORT = 5050

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "intent_classifier"
EMBEDDING_CACHE_PATH = PROJECT_ROOT / "models" / "embedding_cache"


print("=" * 60)
print("JARVIS LOCAL INTENT SERVICE")
print("=" * 60)

if not MODEL_PATH.exists():
    print("ERROR: Intent model not found:")
    print(MODEL_PATH)
    sys.exit(1)

config = json.loads((MODEL_PATH / "config.json").read_text(encoding="utf-8"))
print(f"Loading embedding model: {config['embedding_model']}")
embedder = SentenceTransformer(
    config["embedding_model"], device="cpu", cache_folder=str(EMBEDDING_CACHE_PATH)
)
print(f"Loading classifier: {MODEL_PATH / 'classifier.joblib'}")
classifier = joblib.load(MODEL_PATH / "classifier.joblib")

print("Intent model loaded successfully.")
print(f"Listening on http://{HOST}:{PORT}")
print("=" * 60)


def predict_intent(text: str) -> dict:
    text = text.strip()

    if not text:
        return {
            "success": False,
            "error": "Empty text",
        }

    started = time.perf_counter()
    embedding = embedder.encode(
        [text], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
    )
    probabilities = classifier.predict_proba(embedding)[0]
    index = int(np.argmax(probabilities))
    intent = str(classifier.classes_[index])
    confidence = float(probabilities[index])
    elapsed_ms = (time.perf_counter() - started) * 1000

    print(
        f'[INTENT] "{text}" -> '
        f'{intent} '
        f'({confidence:.4f}, {elapsed_ms:.2f} ms)'
    )

    threshold = float(config["confidence_threshold"])
    if confidence < threshold:
        return {
            "success": False,
            "intent": intent,
            "confidence": confidence,
            "error": f"Low confidence (threshold={threshold:.2f})",
        }

    return {
        "success": True,
        "intent": intent,
        "confidence": confidence,
    }


class IntentRequestHandler(BaseHTTPRequestHandler):

    def _send_json(self, status_code: int, payload: dict):
        body = json.dumps(
            payload,
            ensure_ascii=False,
        ).encode("utf-8")

        self.send_response(status_code)

        self.send_header(
            "Content-Type",
            "application/json; charset=utf-8",
        )

        self.send_header(
            "Content-Length",
            str(len(body)),
        )

        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(
                200,
                {
                    "success": True,
                    "status": "ready",
                },
            )
            return

        self._send_json(
            404,
            {
                "success": False,
                "error": "Not found",
            },
        )

    def do_POST(self):
        if self.path != "/predict":
            self._send_json(
                404,
                {
                    "success": False,
                    "error": "Not found",
                },
            )
            return

        try:
            content_length = int(
                self.headers.get(
                    "Content-Length",
                    "0",
                )
            )

            raw_body = self.rfile.read(content_length)

            data = json.loads(
                raw_body.decode("utf-8")
            )

            text = str(
                data.get("text", "")
            ).strip()

            result = predict_intent(text)

            self._send_json(
                200 if result["success"] else 400,
                result,
            )

        except Exception as exc:
            print(f"[ERROR] {exc}")

            self._send_json(
                500,
                {
                    "success": False,
                    "error": str(exc),
                },
            )

    def log_message(self, format, *args):
        return


def main():
    server = ThreadingHTTPServer(
        (HOST, PORT),
        IntentRequestHandler,
    )

    try:
        server.serve_forever()

    except KeyboardInterrupt:
        print("\nStopping intent service...")

    finally:
        server.server_close()


if __name__ == "__main__":
    main()
