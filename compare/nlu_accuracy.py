from __future__ import annotations

import json
import os

import httpx

from benchmarks.common import benchmark_data_path, load_tab_samples, random_trace_id, service_url


URL = service_url("nlu")
DATASET = benchmark_data_path("single_slots_new.txt")
MAX_SAMPLES = int(os.getenv("NLU_MAX_SAMPLES", "0") or 0)


async def main() -> int:
    rows = load_tab_samples(DATASET)
    if MAX_SAMPLES > 0:
        rows = rows[:MAX_SAMPLES]
    intent_right = 0
    slots_right = 0
    total = 0
    async with httpx.AsyncClient(timeout=60.0, trust_env=False) as client:
        for text, label, slots_raw in rows:
            response = await client.post(
                URL,
                json={"query": text, "trace_id": random_trace_id("nlu"), "enable_dm": False},
            )
            response.raise_for_status()
            payload = response.json()
            expected_slots = json.loads(slots_raw)
            if payload.get("slots") == expected_slots:
                slots_right += 1
            if str(payload.get("intent_id")) == label:
                intent_right += 1
            total += 1
    print(f"nlu samples: {total}")
    print(f"nlu intent acc: {intent_right / total:.6f} ({intent_right}/{total})")
    print(f"nlu slots  acc: {slots_right / total:.6f} ({slots_right}/{total})")
    return 0


if __name__ == "__main__":
    import asyncio

    raise SystemExit(asyncio.run(main()))

