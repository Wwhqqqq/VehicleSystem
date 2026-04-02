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


SAMPLES = [row[0] for row in load_tab_samples(asset_data_path("train", "data", "reject", "test.txt"))]


class RejectUser(HttpUser):
    wait_time = between(1, 1.5)
    host = service_base_url("reject")

    def on_start(self) -> None:
        if hasattr(self.client, "trust_env"):
            self.client.trust_env = False

    @task
    def predict_reject(self):
        self.client.post(
            service_route("reject"),
            json={"query": random.choice(SAMPLES), "thres": 0.5, "trace_id": random_trace_id("reject-load")},
            headers={"Content-Type": "application/json"},
        )
