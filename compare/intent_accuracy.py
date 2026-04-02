from __future__ import annotations

import os

import httpx

from benchmarks.common import asset_data_path, load_tab_samples, random_trace_id, service_url


URL = service_url("intent")
DATASET = asset_data_path("train", "data", "intent", "test.txt")
MAX_SAMPLES = int(os.getenv("INTENT_MAX_SAMPLES", "0") or 0)


async def main() -> int:
    rows = load_tab_samples(DATASET)
    if MAX_SAMPLES > 0:
        rows = rows[:MAX_SAMPLES]
    right = 0
    total = 0
    async with httpx.AsyncClient(timeout=30.0, trust_env=False) as client:
        for text, label in rows:
            response = await client.post(URL, json={"query": text, "trace_id": random_trace_id("intent")})
            response.raise_for_status()
            payload = response.json()
            if int(str(payload["data"]).split(",")[0]) == int(label):
                right += 1
            total += 1
    print(f"intent samples: {total}")
    print(f"intent acc@1: {right / total:.6f} ({right}/{total})")
    return 0


if __name__ == "__main__":
    import asyncio

    raise SystemExit(asyncio.run(main()))

