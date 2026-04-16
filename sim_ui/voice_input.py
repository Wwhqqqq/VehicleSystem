from __future__ import annotations

from PySide6.QtCore import QObject, QThread, Signal


class SpeechListenWorker(QThread):
    finished_text = Signal(str)
    failed = Signal(str)

    def __init__(self, language: str = "zh-CN", phrase_time_limit: float = 12.0) -> None:
        super().__init__()
        self._language = language
        self._phrase_time_limit = phrase_time_limit

    def run(self) -> None:
        try:
            import speech_recognition as sr
        except ImportError:
            self.failed.emit("未安装 SpeechRecognition：pip install SpeechRecognition PyAudio")
            return

        r = sr.Recognizer()
        try:
            mic = sr.Microphone()
        except OSError as exc:
            self.failed.emit(f"无法打开麦克风: {exc}（Windows 可尝试 pip install PyAudio）")
            return

        try:
            with mic as source:
                r.adjust_for_ambient_noise(source, duration=0.4)
                audio = r.listen(source, timeout=6, phrase_time_limit=self._phrase_time_limit)
        except sr.WaitTimeoutError:
            self.failed.emit("没有听到语音，请重试")
            return
        except Exception as exc:
            self.failed.emit(f"录音失败: {exc}")
            return

        try:
            text = r.recognize_google(audio, language=self._language)
        except sr.UnknownValueError:
            self.failed.emit("未能识别语音内容")
        except sr.RequestError as exc:
            self.failed.emit(f"识别服务不可用: {exc}")
        else:
            self.finished_text.emit(text.strip())


def speech_available() -> bool:
    try:
        import speech_recognition  # noqa: F401

        return True
    except ImportError:
        return False
