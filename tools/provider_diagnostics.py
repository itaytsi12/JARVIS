"""Manual, quota-conscious provider reachability/model discovery report."""
from __future__ import annotations

import argparse
import time

from providers.bootstrap import build_multi_model_provider


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect configured JARVIS AI providers without sending chat prompts.")
    parser.add_argument("--provider", help="Only inspect one provider")
    args = parser.parse_args()
    pool = build_multi_model_provider()
    print("provider\treachable\tmodels_discovered\tlatency_ms\tstatus")
    for name, provider in sorted(pool.providers.items()):
        if args.provider and name != args.provider:
            continue
        started = time.perf_counter()
        if not provider.is_available():
            print(f"{name}\tno\t0\t0\t{provider.unavailable_reason()}")
            continue
        routes = provider.discover_models(timeout=5.0) if hasattr(provider, "discover_models") else []
        elapsed = (time.perf_counter() - started) * 1000
        status = "ok" if routes else "configured; discovery unavailable/empty"
        print(f"{name}\t{'yes' if routes else 'unknown'}\t{len(routes)}\t{elapsed:.1f}\t{status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
