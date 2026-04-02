from __future__ import annotations

import random

from benchmarks.common import (
    asset_data_path,
    ensure_local_no_proxy,
    load_tab_samples,
    random_trace_id,
    service_base_url,
    service_route,
)

ensure_local_no_proxy()

from locust import HttpUser, between, task


SAMPLES = [row[0] for row in load_tab_samples(asset_data_path("train", "data", "intent", "test.txt"))]


class NluUser(HttpUser):
    wait_time = between(3, 5)
    host = service_base_url("nlu")

    def on_start(self) -> None:
        if hasattr(self.client, "trust_env"):
            self.client.trust_env = False

    @task
    def predict_nlu(self):
        self.client.post(
            service_route("nlu"),
            json={"query": random.choice(SAMPLES), "trace_id": random_trace_id("nlu-load"), "enable_dm": False},
            headers={"Content-Type": "application/json"},
        )
