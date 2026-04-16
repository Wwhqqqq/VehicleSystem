from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from benchmarks.common import ensure_local_no_proxy, service_base_url

from sim_ui.main_window import MainWindow


def main() -> int:
    ensure_local_no_proxy()
    url = service_base_url("gateway")
    if len(sys.argv) > 1:
        url = sys.argv[1].rstrip("/")

    app = QApplication(sys.argv)
    win = MainWindow(url)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
