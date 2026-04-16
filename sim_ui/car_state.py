from __future__ import annotations

import re
from dataclasses import dataclass, field


def _first_int(text: str) -> int | None:
    m = re.search(r"-?\d+", str(text))
    if not m:
        return None
    try:
        return int(m.group(0))
    except ValueError:
        return None


def _window_keys_from_position(position: str) -> list[str]:
    p = str(position or "").strip()
    if not p or "全" in p or "所有" in p or "四" in p:
        return ["fl", "fr", "rl", "rr"]
    mapping: list[tuple[tuple[str, ...], str]] = [
        (("主驾", "主驾驶", "左前", "前排左", "驾驶"), "fl"),
        (("副驾", "副驾驶", "右前", "前排右"), "fr"),
        (("左后", "后排左"), "rl"),
        (("右后", "后排右"), "rr"),
    ]
    for needles, key in mapping:
        if any(n in p for n in needles):
            return [key]
    return []


@dataclass
class VehicleState:
    """Simplified cabin state driven by NLU + dm_tool control payloads."""

    window_fl: int = 0
    window_fr: int = 0
    window_rl: int = 0
    window_rr: int = 0
    volume: int = 35
    ac_temp_c: int = 24
    nav_destination: str = ""
    now_playing: str = ""
    weather_line: str = ""

    _chat_buffer: str = field(default="", repr=False)

    def window_label(self, key: str) -> str:
        return {"fl": "主驾", "fr": "副驾", "rl": "左后", "rr": "右后"}.get(key, key)

    def set_windows(self, keys: list[str], value: int) -> None:
        value = max(0, min(100, int(value)))
        for k in keys:
            if k == "fl":
                self.window_fl = value
            elif k == "fr":
                self.window_fr = value
            elif k == "rl":
                self.window_rl = value
            elif k == "rr":
                self.window_rr = value

    def apply_nlu_payload(self, payload: dict) -> list[str]:
        """Update state from one gateway `request_nlu` message. Returns human log lines."""
        lines: list[str] = []
        func = str(payload.get("func") or "")

        if func == "CHAT":
            st = payload.get("status")
            frame = str(payload.get("frame") or "")
            if st == 0:
                self._chat_buffer = ""
            elif st == 1 and frame:
                self._chat_buffer += frame
            elif st == 2:
                lines.append("闲聊流结束")
            return lines

        if func == "REJECT":
            frame = str(payload.get("frame") or "")
            if frame:
                lines.append(f"拒识/模板: {frame[:200]}")
            else:
                lines.append("拒识（无播报文案）")
            return lines

        function_name = str(payload.get("function") or "")
        slots = payload.get("slots") if isinstance(payload.get("slots"), dict) else {}
        tool = payload.get("tool")

        if isinstance(tool, dict) and tool.get("service") == "maps":
            pois = tool.get("pois") or []
            if pois and isinstance(pois[0], dict):
                name = str(pois[0].get("name") or "").strip()
                if name:
                    self.nav_destination = name
                    lines.append(f"导航目标: {name}")
            err = tool.get("error")
            if err:
                lines.append(f"地图: {err}")
            return lines

        if isinstance(tool, dict) and tool.get("service") == "weather":
            if tool.get("error"):
                lines.append(f"天气: {tool['error']}")
            else:
                parts = [
                    str(tool.get("城市") or ""),
                    str(tool.get("日期") or ""),
                    str(tool.get("天气") or ""),
                    str(tool.get("温度") or ""),
                ]
                self.weather_line = " ".join(p for p in parts if p).strip() or "（天气已更新）"
                lines.append(self.weather_line)
            return lines

        if isinstance(tool, list) and tool:
            first = tool[0] if isinstance(tool[0], dict) else {}
            title = str(first.get("title") or first.get("name") or "").strip()
            if title:
                self.now_playing = title
                lines.append(f"音乐: {title}")
            return lines

        if isinstance(tool, dict) and tool.get("domain"):
            slots = tool.get("slots") if isinstance(tool.get("slots"), dict) else slots
            return self._apply_control(function_name, slots, tool, lines)

        if function_name and function_name != "Unknown":
            return self._apply_control(function_name, slots, {}, lines)

        return lines

    def _apply_control(
        self,
        function_name: str,
        slots: dict,
        tool: dict,
        lines: list[str],
    ) -> list[str]:
        pos = str(
            slots.get("Position")
            or slots.get("位置")
            or tool.get("target")
            or "",
        )

        if function_name in {"Open_Window", "Open_Window_Diagonal"}:
            keys = _window_keys_from_position(pos) or ["fl"]
            self.set_windows(keys, 100)
            lines.append(f"车窗打开 → {','.join(self.window_label(k) for k in keys)}")
            return lines

        if function_name in {"Close_Window", "Close_Window_Diagonal"}:
            keys = _window_keys_from_position(pos) or ["fl", "fr", "rl", "rr"]
            self.set_windows(keys, 0)
            lines.append(f"车窗关闭 → {','.join(self.window_label(k) for k in keys)}")
            return lines

        if function_name == "Set_Window":
            keys = _window_keys_from_position(pos) or ["fl"]
            ratio = slots.get("Ratio") or slots.get("ratio")
            val = _first_int(str(ratio)) if ratio is not None else 50
            if val is None:
                val = 50
            self.set_windows(keys, val)
            lines.append(
                f"车窗开度 {val}% → {','.join(self.window_label(k) for k in keys)}",
            )
            return lines

        if function_name == "Inc_Sound_Volume":
            self.volume = min(100, self.volume + 8)
            lines.append(f"音量 ↑ {self.volume}")
            return lines
        if function_name == "Dec_Sound_Volume":
            self.volume = max(0, self.volume - 8)
            lines.append(f"音量 ↓ {self.volume}")
            return lines
        if function_name in {"Set_Sound_Volume", "Set_Sound_Volume_Max", "Set_Sound_Volume_Min"}:
            for key in ("Value", "音量", "ratio", "Ratio"):
                if key in slots:
                    v = _first_int(str(slots[key]))
                    if v is not None:
                        self.volume = max(0, min(100, v))
                        lines.append(f"音量设为 {self.volume}")
                        return lines
            if "Max" in function_name:
                self.volume = 100
            elif "Min" in function_name:
                self.volume = 0
            lines.append(f"音量 {self.volume}")
            return lines

        if function_name in {"Inc_Air_Condition_Temperature", "Dec_Air_Condition_Temperature"}:
            delta = 1 if "Inc" in function_name else -1
            self.ac_temp_c = max(16, min(32, self.ac_temp_c + delta))
            lines.append(f"空调温度 → {self.ac_temp_c}°C")
            return lines
        if function_name == "Set_Air_Condition_Temperature":
            for key in ("Temperature", "温度", "Value", "ratio"):
                if key in slots:
                    v = _first_int(str(slots[key]))
                    if v is not None:
                        self.ac_temp_c = max(16, min(32, v))
                        lines.append(f"空调温度设为 {self.ac_temp_c}°C")
                        return lines

        msg = str(tool.get("message") or "").strip()
        if msg:
            lines.append(f"车控: {msg}")
        elif function_name:
            lines.append(f"车控: {function_name}")
        return lines

    def consume_chat_text(self) -> str:
        t = self._chat_buffer
        self._chat_buffer = ""
        return t
