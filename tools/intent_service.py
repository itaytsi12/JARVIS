from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys

from setfit import SetFitModel


HOST = "127.0.0.1"
PORT = 5050

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "intent"


print("=" * 60)
print("JARVIS LOCAL INTENT SERVICE")
print("=" * 60)

if not MODEL_PATH.exists():
    print(f"ERROR: Intent model not found:")
    print(MODEL_PATH)
    sys.exit(1)

print(f"Loading model from: {MODEL_PATH}")

model = SetFitModel.from_pretrained(str(MODEL_PATH))

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

    prediction = model.predict([text])

    if hasattr(prediction, "tolist"):
        prediction = prediction.tolist()

    intent = str(prediction[0])

    confidence = None

    try:
        probabilities = model.predict_proba([text])

        if hasattr(probabilities, "cpu"):
            probabilities = probabilities.cpu().numpy()
        elif hasattr(probabilities, "numpy"):
            probabilities = probabilities.numpy()

        confidence = float(max(probabilities[0]))

    except Exception as exc:
        print(f"[WARNING] Could not calculate confidence: {exc}")

    print(
        f'[INTENT] "{text}" -> '
        f'{intent} '
        f'({confidence if confidence is not None else "unknown"})'
    )

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
        # Prevent noisy HTTP logs.
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