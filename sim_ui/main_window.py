from __future__ import annotations

import json
from typing import Any

from PySide6.QtCore import QLocale, Qt, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from shared.settings import get_settings

from sim_ui.cabin_vehicle_widget import CabinVehicleWidget
from sim_ui.car_state import VehicleState
from sim_ui.gateway_client import GatewayClient
from sim_ui.voice_input import SpeechListenWorker, speech_available


def _console_text_edit() -> QTextEdit:
    w = QTextEdit()
    w.setReadOnly(True)
    f = QFont("Consolas", 10)
    if not f.exactMatch():
        f = QFont("monospace", 10)
    w.setFont(f)
    w.setStyleSheet(
        "QTextEdit { background-color:#0a0d0a; color:#b8e8b8; "
        "border:1px solid #1e3a1e; border-radius:4px; padding:6px; }",
    )
    return w


class MainWindow(QMainWindow):
    def __init__(self, gateway_url: str) -> None:
        super().__init__()
        self.setWindowTitle("车载智能助手")
        self.resize(1320, 820)

        self._settings = get_settings()
        self._vehicle = VehicleState()
        self._gateway = GatewayClient(gateway_url)
        self._sender_default = "pyside_sim_user"
        self._listen_thread: SpeechListenWorker | None = None

        self._tts = None
        try:
            from PySide6.QtTextToSpeech import QTextToSpeech

            self._tts = QTextToSpeech(self)
            for loc in self._tts.availableLocales():
                if loc.language() == QLocale.Language.Chinese:
                    self._tts.setLocale(loc)
                    break
        except Exception:
            pass

        central = QWidget()
        self.setCentralWidget(central)
        grid = QGridLayout(central)
        grid.setRowStretch(1, 1)
        grid.setColumnStretch(1, 1)

        # --- Top bar: gateway + pipeline (console header strip) ---
        top = QWidget()
        top_l = QHBoxLayout(top)
        top_l.setContentsMargins(0, 0, 0, 0)
        self._url_edit = QLineEdit(gateway_url)
        self._url_edit.setPlaceholderText("网关 Base URL")
        self._connect_btn = QPushButton("连接")
        self._connect_btn.clicked.connect(self._on_connect)
        self._sender_edit = QLineEdit(self._sender_default)
        self._sender_edit.setMaximumWidth(160)
        self._enable_dm = QCheckBox("DM 工具链")
        self._enable_dm.setChecked(True)
        self._conn_label = QLabel("未连接")
        self._conn_label.setMinimumWidth(140)

        top_l.addWidget(QLabel("网关"))
        top_l.addWidget(self._url_edit, 1)
        top_l.addWidget(self._connect_btn)
        top_l.addWidget(QLabel("sender"))
        top_l.addWidget(self._sender_edit)
        top_l.addWidget(self._enable_dm)
        top_l.addWidget(self._conn_label)

        self._pipe_labels: dict[str, QLabel] = {}
        for key, text in (
            ("gw", "Gateway"),
            ("task", "任务/NLU+DM"),
            ("chat", "闲聊流式"),
            ("reject", "拒识"),
        ):
            lab = QLabel(text)
            lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lab.setStyleSheet(
                "padding:4px 8px;border:1px solid #555;border-radius:3px;"
                "background:#1a1f1a;color:#8a9a8a;font-size:11px;",
            )
            lab.setMinimumWidth(92)
            self._pipe_labels[key] = lab
            top_l.addWidget(lab)
        self._reset_pipeline()
        grid.addWidget(top, 0, 0, 1, 3)

        # --- Left: raw log (vertical console) ---
        log_box = QGroupBox("链路 / 原始回包")
        log_box.setStyleSheet("QGroupBox { font-weight:600; color:#9abf9a; border:1px solid #2a3a2a; }")
        lg = QVBoxLayout(log_box)
        self._log = _console_text_edit()
        self._log.setPlaceholderText("网关 JSON、发送记录与车态日志…")
        lg.addWidget(self._log)
        log_box.setMinimumWidth(300)
        log_box.setMaximumWidth(420)
        grid.addWidget(log_box, 1, 0)

        # --- Center: vehicle simulation ---
        center_box = QGroupBox("座舱仿真")
        center_box.setStyleSheet("QGroupBox { font-weight:600; color:#aac8ff; border:1px solid #2a3550; }")
        cv = QVBoxLayout(center_box)
        cv.setContentsMargins(8, 12, 8, 8)
        self._vehicle_scene = CabinVehicleWidget(self._vehicle)
        self._vehicle_scene.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        cv.addWidget(self._vehicle_scene, 1)
        grid.addWidget(center_box, 1, 1)

        # --- Right: NLG + redis (side console) ---
        right_col = QWidget()
        right_l = QVBoxLayout(right_col)
        right_l.setContentsMargins(0, 0, 0, 0)
        nlg_box = QGroupBox("NLG / 播报文案")
        nlg_box.setStyleSheet("QGroupBox { font-weight:600; color:#c8b8e8; border:1px solid #3a2a4a; }")
        ng = QVBoxLayout(nlg_box)
        self._nlg_view = _console_text_edit()
        self._nlg_view.setPlaceholderText("任务 NLG、拒识或闲聊流式文本…")
        ng.addWidget(self._nlg_view)
        right_l.addWidget(nlg_box, 1)

        redis_box = QGroupBox("Redis 会话键")
        redis_box.setStyleSheet("QGroupBox { font-weight:600; color:#a8a8c8; border:1px solid #2a2a40; }")
        rg = QVBoxLayout(redis_box)
        self._redis_view = _console_text_edit()
        self._redis_view.setFont(QFont("Consolas", 9))
        self._redis_view.setMaximumHeight(120)
        rg.addWidget(self._redis_view)
        right_l.addWidget(redis_box)
        right_col.setMinimumWidth(280)
        right_col.setMaximumWidth(400)
        grid.addWidget(right_col, 1, 2)

        # --- Bottom: command line + presets ---
        bottom = QGroupBox("指令输入")
        bottom.setStyleSheet("QGroupBox { font-weight:600; color:#d0d0d0; border:1px solid #444; }")
        bg = QVBoxLayout(bottom)

        row = QHBoxLayout()
        self._query_edit = QLineEdit()
        self._query_edit.setPlaceholderText("输入指令后回车，或使用语音 / 快捷按钮")
        self._query_edit.returnPressed.connect(self._on_send)
        send_btn = QPushButton("发送")
        send_btn.clicked.connect(self._on_send)
        self._mic_btn = QPushButton("麦克风")
        self._mic_btn.clicked.connect(self._on_mic)
        if not speech_available():
            self._mic_btn.setToolTip("安装 SpeechRecognition + PyAudio 后可用")
            self._mic_btn.setEnabled(False)
        row.addWidget(self._query_edit, 1)
        row.addWidget(send_btn)
        row.addWidget(self._mic_btn)
        bg.addLayout(row)

        preset_scroll = QScrollArea()
        preset_scroll.setWidgetResizable(True)
        preset_scroll.setMaximumHeight(120)
        preset_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        preset_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        pw_host = QWidget()
        ph = QHBoxLayout(pw_host)
        ph.setContentsMargins(4, 4, 4, 4)
        presets = [
            "导航到最近的充电站",
            "打开主驾驶车窗",
            "关闭所有车窗",
            "主驾驶车窗开一半",
            "把空调调到24度",
            "音量大一点",
            "今天上海天气怎么样",
            "播放周杰伦七里香",
            "现在几号",
        ]
        for text in presets:
            btn = QPushButton(text)
            btn.setMinimumHeight(36)
            btn.clicked.connect(lambda _, t=text: self._run_preset(t))
            ph.addWidget(btn)
        ph.addStretch(1)
        preset_scroll.setWidget(pw_host)
        bg.addWidget(preset_scroll)

        grid.addWidget(bottom, 2, 0, 1, 3)

        self._gateway.connected_ok.connect(self._on_conn_signal)
        self._gateway.payload_received.connect(self._on_payload)
        self._gateway.client_error.connect(self._on_client_error)

        self._redis_timer = QTimer(self)
        self._redis_timer.setInterval(2000)
        self._redis_timer.timeout.connect(self._poll_redis)
        self._redis_timer.start()

        self._chat_stream = ""
        QTimer.singleShot(300, self._gateway.connect_async)

    def closeEvent(self, event) -> None:
        self._gateway.disconnect()
        super().closeEvent(event)

    def _reset_pipeline(self) -> None:
        for lab in self._pipe_labels.values():
            lab.setStyleSheet(
                "padding:4px 8px;border:1px solid #444;border-radius:3px;"
                "background:#151515;color:#555;font-size:11px;",
            )

    def _flash_pipe(self, key: str) -> None:
        self._reset_pipeline()
        if key in self._pipe_labels:
            self._pipe_labels[key].setStyleSheet(
                "padding:4px 8px;border:1px solid #4caf50;border-radius:3px;"
                "background:#142614;color:#c8ffc8;font-size:11px;",
            )

    def _on_connect(self) -> None:
        url = self._url_edit.text().strip()
        if not url:
            QMessageBox.warning(self, "网关", "请填写网关地址")
            return
        old = self._gateway
        old.disconnect()
        for sig, slot in (
            (old.connected_ok, self._on_conn_signal),
            (old.payload_received, self._on_payload),
            (old.client_error, self._on_client_error),
        ):
            try:
                sig.disconnect(slot)
            except (TypeError, RuntimeError):
                pass
        self._gateway = GatewayClient(url)
        self._gateway.connected_ok.connect(self._on_conn_signal)
        self._gateway.payload_received.connect(self._on_payload)
        self._gateway.client_error.connect(self._on_client_error)
        self._gateway.connect_async()

    def _on_conn_signal(self, ok: bool, msg: str) -> None:
        self._conn_label.setText(msg)
        if ok:
            self._conn_label.setStyleSheet("color:#6c6;")
            self._flash_pipe("gw")
        else:
            self._conn_label.setStyleSheet("color:#c66;")

    def _on_client_error(self, msg: str) -> None:
        self._log.append(f"[错误] {msg}")

    def _sender_id(self) -> str:
        return self._sender_edit.text().strip() or self._sender_default

    def _on_send(self) -> None:
        q = self._query_edit.text().strip()
        if not q:
            return
        self._gateway.send_query(q, self._sender_id(), self._enable_dm.isChecked())
        self._log.append(f">>> 发送: {q}")
        self._query_edit.clear()

    def _run_preset(self, text: str) -> None:
        self._query_edit.setText(text)
        self._on_send()

    def _on_mic(self) -> None:
        if self._listen_thread and self._listen_thread.isRunning():
            return
        self._mic_btn.setEnabled(False)
        self._log.append("… 正在聆听（请对着麦克风说话）")
        self._listen_thread = SpeechListenWorker()
        self._listen_thread.finished_text.connect(self._on_heard)
        self._listen_thread.failed.connect(self._on_hear_failed)
        self._listen_thread.finished.connect(lambda: self._mic_btn.setEnabled(True))
        self._listen_thread.start()

    def _on_heard(self, text: str) -> None:
        self._log.append(f"<<< 语音识别: {text}")
        self._query_edit.setText(text)
        self._gateway.send_query(text, self._sender_id(), self._enable_dm.isChecked())

    def _on_hear_failed(self, msg: str) -> None:
        self._log.append(f"[语音] {msg}")
        QMessageBox.information(self, "语音识别", msg)

    def _on_payload(self, payload: dict[str, Any]) -> None:
        self._log.append(json.dumps(payload, ensure_ascii=False, indent=2))
        func = str(payload.get("func") or "")

        if func == "CHAT":
            self._flash_pipe("chat")
            st = payload.get("status")
            frame = str(payload.get("frame") or "")
            if st == 0:
                self._chat_stream = ""
                self._nlg_view.clear()
            elif st == 1 and frame:
                self._chat_stream += frame
                self._nlg_view.setPlainText(self._chat_stream)
            elif st == 2:
                self._speak(self._chat_stream)
                self._vehicle_scene.pulse_action()
            return

        if func == "REJECT":
            self._flash_pipe("reject")
            frame = str(payload.get("frame") or "")
            if frame:
                self._nlg_view.setPlainText(frame)
                self._speak(frame)
            self._vehicle_scene.pulse_action()
            return

        self._flash_pipe("task")
        lines = self._vehicle.apply_nlu_payload(payload)
        for line in lines:
            self._log.append(f"[车态] {line}")

        nlg = str(payload.get("nlg") or "").strip()
        if nlg:
            self._nlg_view.setPlainText(nlg)
            self._speak(nlg)

        if lines or nlg:
            self._vehicle_scene.pulse_action()

    def _speak(self, text: str) -> None:
        if not self._tts or not text.strip():
            return
        self._tts.say(text.strip())

    def _poll_redis(self) -> None:
        try:
            import redis
        except ImportError:
            self._redis_view.setPlainText("（未安装 redis 包，跳过）")
            return

        sid = self._sender_id()
        key_tpl = self._settings.runtime.redis_keys.last_service
        key = key_tpl.format(sender_id=sid)
        try:
            client = redis.Redis.from_url(self._settings.runtime.redis_url, decode_responses=True)
            val = client.get(key)
        except Exception as exc:
            self._redis_view.setPlainText(f"Redis 不可用: {exc}")
            return

        self._redis_view.setPlainText(f"{key}\n{val or '（空）'}")
