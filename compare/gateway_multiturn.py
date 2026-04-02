from __future__ import annotations

import json
import os
import queue
import random

import requests
import socketio

from benchmarks.common import benchmark_data_path, random_trace_id, result_path, service_base_url


URL = service_base_url("gateway")
DATASET = benchmark_data_path(os.getenv("MULTITURN_DATASET", "multi_test.txt"))
OUTPUT = result_path(os.getenv("MULTITURN_OUTPUT", "multi_test_output.txt"))

HTTP_SESSION = requests.Session()
HTTP_SESSION.trust_env = False
sio = socketio.Client(http_session=HTTP_SESSION)
responses: "queue.Queue[dict]" = queue.Queue(2000)


def rand_sender(size: int = 9) -> str:
    population = "1234567890zyxwvutsrqponmlkjihgfedcba"
    return "".join(random.sample(population, size))


@sio.on("request_nlu")
def on_response(data):
    payload = json.loads(data) if isinstance(data, str) else data
    responses.put(payload)


def collect_single_result() -> list[dict]:
    items: list[dict] = []
    while True:
        response = responses.get(timeout=30)
        items.append(response)
        if response.get("intent") != "闲聊百科":
            return items
        if response.get("status") == 2:
            return items


def main() -> int:
    sessions = []
    with DATASET.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                sessions.append([item.strip() for item in line.split("\t") if item.strip()])

    with OUTPUT.open("w", encoding="utf-8") as writer:
        for session in sessions:
            sender_id = rand_sender()
            sio.connect(URL)
            try:
                for query in session:
                    payload = {
                        "sender_id": sender_id,
                        "trace_id": random_trace_id("gateway"),
                        "query": query,
                        "enable_dm": False,
                    }
                    sio.emit("request_nlu", json.dumps(payload, ensure_ascii=False))
                    result = {"query": query, "res": collect_single_result()}
                    writer.write(json.dumps(result, ensure_ascii=False) + "\n")
                    writer.flush()
            finally:
                sio.disconnect()
    print(f"gateway output written to: {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
