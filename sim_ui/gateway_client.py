from __future__ import annotations

import json
import threading
import uuid
from typing import Any

import requests
import socketio
from PySide6.QtCore import QObject, Signal

from benchmarks.common import ensure_local_no_proxy


class GatewayClient(QObject):
    """Socket.IO client; callbacks run on engineio thread, signals marshal to Qt GUI thread."""

    connected_ok = Signal(bool, str)
    payload_received = Signal(dict)
    client_error = Signal(str)

    def __init__(self, gateway_base_url: str) -> None:
        super().__init__()
        self._url = gateway_base_url.rstrip("/")
        self._http_session = requests.Session()
        self._http_session.trust_env = False
        self._sio = socketio.Client(
            reconnection=True,
            reconnection_attempts=0,
            logger=False,
            engineio_logger=False,
            http_session=self._http_session,
        )

        @self._sio.on("request_nlu")
        def _on_nlu(data: Any) -> None:
            payload = json.loads(data) if isinstance(data, str) else dict(data)
            self.payload_received.emit(payload)

        @self._sio.event
        def connect() -> None:
            self.connected_ok.emit(True, f"已连接 {self._url}")

        @self._sio.event
        def disconnect() -> None:
            self.connected_ok.emit(False, "已断开")

        self._sio.on("connect_error", lambda *_: None)

    @property
    def is_connected(self) -> bool:
        return bool(self._sio.connected)

    def connect_async(self) -> None:
        def runner() -> None:
            ensure_local_no_proxy()
            try:
                self._sio.connect(self._url, wait_timeout=12, transports=["websocket", "polling"])
            except Exception as exc:
                self.client_error.emit(str(exc))

        threading.Thread(target=runner, daemon=True).start()

    def disconnect(self) -> None:
        try:
            if self._sio.connected:
                self._sio.disconnect()
        except Exception:
            pass

    def send_query(self, query: str, sender_id: str, enable_dm: bool = True) -> None:
        if not self._sio.connected:
            self.client_error.emit("未连接网关，请先连接")
            return
        payload = {
            "query": query.strip(),
            "sender_id": sender_id,
            "trace_id": f"sim-{uuid.uuid4().hex[:12]}",
            "enable_dm": enable_dm,
        }
        self._sio.emit("request_nlu", json.dumps(payload, ensure_ascii=False))
