from __future__ import annotations

import math
from typing import Sequence

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import (
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
    QRadialGradient,
)
from PySide6.QtWidgets import QSizePolicy, QWidget

from sim_ui.car_state import VehicleState


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _lerp_pt(a: QPointF, b: QPointF, t: float) -> QPointF:
    return QPointF(_lerp(a.x(), b.x(), t), _lerp(a.y(), b.y(), t))


def _roof_uv(
    p0: QPointF,
    p1: QPointF,
    p2: QPointF,
    p3: QPointF,
    u: float,
    v: float,
) -> QPointF:
    """Bilinear point on roof quad (p0=前左, p1=前右, p2=后右, p3=后左)."""
    top = _lerp_pt(p0, p1, u)
    bot = _lerp_pt(p3, p2, u)
    return _lerp_pt(top, bot, v)


def _quad_path(a: QPointF, b: QPointF, c: QPointF, d: QPointF) -> QPainterPath:
    path = QPainterPath()
    path.moveTo(a)
    path.lineTo(b)
    path.lineTo(c)
    path.lineTo(d)
    path.closeSubpath()
    return path


class CabinVehicleWidget(QWidget):
    """Pseudo-3D cabin view: isometric-style body, four roof glass panels, HUD."""

    def __init__(self, vehicle: VehicleState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._v = vehicle
        self.setMinimumSize(480, 420)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._disp_fl = 0.0
        self._disp_fr = 0.0
        self._disp_rl = 0.0
        self._disp_rr = 0.0
        self._disp_vol = float(vehicle.volume)
        self._disp_ac = float(vehicle.ac_temp_c)
        self._route_phase = 0.0
        self._action_pulse = 0.0

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(33)

    def pulse_action(self) -> None:
        self._action_pulse = 1.0

    def _tick(self) -> None:
        v = self._v
        rate = 0.18
        self._disp_fl = _lerp(self._disp_fl, float(v.window_fl), rate)
        self._disp_fr = _lerp(self._disp_fr, float(v.window_fr), rate)
        self._disp_rl = _lerp(self._disp_rl, float(v.window_rl), rate)
        self._disp_rr = _lerp(self._disp_rr, float(v.window_rr), rate)
        self._disp_vol = _lerp(self._disp_vol, float(v.volume), rate)
        self._disp_ac = _lerp(self._disp_ac, float(v.ac_temp_c), rate)

        if v.nav_destination:
            self._route_phase += 0.06
        self._action_pulse = max(0.0, self._action_pulse - 0.04)
        self.update()

    def _car_geometry(self, rect: QRectF) -> tuple[QPointF, QPointF, QPointF, QPointF, QPointF, float]:
        """Roof quad + extrusion offset (screen space). Returns p0..p3, depth_vec, scale."""
        s = min(rect.width(), rect.height()) / 520.0
        cx = rect.center().x()
        cy = rect.center().y() - 18 * s
        w = 118 * s
        lf = 108 * s
        lb = 92 * s
        p0 = QPointF(cx - w, cy - lf * 0.55)
        p1 = QPointF(cx + w, cy - lf * 0.42)
        p2 = QPointF(cx + w * 0.78, cy + lb * 0.72)
        p3 = QPointF(cx - w * 0.78, cy + lb * 0.68)
        depth = QPointF(22 * s, 36 * s)
        return p0, p1, p2, p3, depth, s

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = QRectF(self.rect())
        g = QLinearGradient(rect.topLeft(), rect.bottomRight())
        g.setColorAt(0.0, QColor(10, 14, 24))
        g.setColorAt(1.0, QColor(5, 7, 12))
        p.fillRect(rect, g)

        p0, p1, p2, p3, depth, s = self._car_geometry(rect)
        roof = [p0, p1, p2, p3]

        if self._action_pulse > 0.01:
            hull = QPolygonF(roof + [p2 + depth, p3 + depth, p0 + depth, p1 + depth])
            br = hull.boundingRect().adjusted(-16, -16, 16, 16)
            glow = QRadialGradient(br.center(), max(br.width(), br.height()) * 0.55)
            glow.setColorAt(0.0, QColor(80, 200, 255, int(70 * self._action_pulse)))
            glow.setColorAt(1.0, QColor(80, 200, 255, 0))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(glow)
            p.drawRoundedRect(br, 20, 20)

        self._draw_ground_shadow(p, p0, p1, p2, p3, depth, s)
        self._draw_wheels(p, p0, p1, p2, p3, depth, s)
        self._draw_prism_body(p, p0, p1, p2, p3, depth, s)
        self._draw_windows_roof(p, p0, p1, p2, p3, s)
        self._draw_lights_and_grille(p, p0, p1, depth, s)
        self._draw_route(p, p0, p1, s)
        self._draw_hud(p, rect)

    def _draw_ground_shadow(
        self,
        p: QPainter,
        p0: QPointF,
        p1: QPointF,
        p2: QPointF,
        p3: QPointF,
        depth: QPointF,
        s: float,
    ) -> None:
        base = QPolygonF([p0 + depth, p1 + depth, p2 + depth, p3 + depth])
        br = base.boundingRect()
        center = br.center()
        rad = max(br.width(), br.height()) * 0.55
        sh = QRadialGradient(center, rad)
        sh.setColorAt(0.0, QColor(0, 0, 0, 110))
        sh.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.save()
        p.translate(10 * s, 14 * s)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(sh)
        p.drawEllipse(QRectF(center.x() - rad * 1.05, center.y() - rad * 0.42, rad * 2.1, rad * 0.84))
        p.restore()

    def _wheel(
        self,
        p: QPainter,
        center: QPointF,
        radius: float,
        tilt_deg: float,
    ) -> None:
        p.save()
        p.translate(center)
        p.rotate(tilt_deg)
        rg = QRadialGradient(-radius * 0.25, -radius * 0.25, radius * 1.4)
        rg.setColorAt(0.0, QColor(90, 95, 105))
        rg.setColorAt(0.45, QColor(35, 38, 48))
        rg.setColorAt(1.0, QColor(15, 16, 22))
        p.setPen(QPen(QColor(25, 28, 35), 1))
        p.setBrush(rg)
        p.drawEllipse(QRectF(-radius * 1.05, -radius * 0.52, radius * 2.1, radius * 1.04))
        p.setPen(QPen(QColor(180, 190, 210, 90), 1))
        p.setBrush(QColor(0, 0, 0, 0))
        p.drawEllipse(QRectF(-radius * 0.55, -radius * 0.28, radius * 1.1, radius * 0.56))
        p.restore()

    def _draw_wheels(
        self,
        p: QPainter,
        p0: QPointF,
        p1: QPointF,
        p2: QPointF,
        p3: QPointF,
        depth: QPointF,
        s: float,
    ) -> None:
        r = 26 * s
        fl = _lerp_pt(p0, p3, 0.12) + depth * 0.35
        fr = _lerp_pt(p0, p1, 0.88) + depth * 0.35
        rl = _lerp_pt(p0, p3, 0.82) + depth * 0.55
        rr = _lerp_pt(p1, p2, 0.82) + depth * 0.55
        for c in (rl, rr, fl, fr):
            self._wheel(p, c, r, -18.0)

    def _shade_face(
        self,
        p: QPainter,
        pts: Sequence[QPointF],
        c0: QColor,
        c1: QColor,
        ax: QPointF,
    ) -> None:
        poly = QPolygonF(list(pts))
        br = poly.boundingRect()
        lg = QLinearGradient(br.topLeft(), br.bottomLeft() + ax)
        lg.setColorAt(0.0, c0)
        lg.setColorAt(1.0, c1)
        p.setPen(QPen(QColor(20, 24, 32), 1))
        p.setBrush(lg)
        p.drawPolygon(poly)

    def _draw_prism_body(
        self,
        p: QPainter,
        p0: QPointF,
        p1: QPointF,
        p2: QPointF,
        p3: QPointF,
        depth: QPointF,
        s: float,
    ) -> None:
        b0, b1, b2, b3 = p0 + depth, p1 + depth, p2 + depth, p3 + depth

        self._shade_face(
            p,
            (b3, b2, p2, p3),
            QColor(28, 32, 44),
            QColor(18, 21, 30),
            QPointF(0, 30 * s),
        )
        self._shade_face(
            p,
            (b1, b2, p2, p1),
            QColor(34, 40, 56),
            QColor(22, 26, 38),
            QPointF(-16 * s, 22 * s),
        )
        self._shade_face(
            p,
            (b0, b3, p3, p0),
            QColor(40, 48, 66),
            QColor(26, 31, 46),
            QPointF(18 * s, 20 * s),
        )

        hood = _quad_path(p0, p1, _lerp_pt(p1, p2, 0.22), _lerp_pt(p0, p3, 0.22))
        hg = QLinearGradient(p0, p1)
        hg.setColorAt(0.0, QColor(52, 60, 82))
        hg.setColorAt(0.5, QColor(68, 78, 102))
        hg.setColorAt(1.0, QColor(48, 55, 74))
        p.setPen(QPen(QColor(30, 36, 50), 1))
        p.setBrush(hg)
        p.drawPath(hood)

        roof_path = _quad_path(p0, p1, p2, p3)
        rg = QLinearGradient(p0, p2)
        rg.setColorAt(0.0, QColor(72, 82, 108))
        rg.setColorAt(0.35, QColor(98, 110, 138))
        rg.setColorAt(0.7, QColor(62, 72, 96))
        rg.setColorAt(1.0, QColor(48, 56, 76))
        p.setPen(QPen(QColor(55, 65, 88), 1))
        p.setBrush(rg)
        p.drawPath(roof_path)

        hl = QLinearGradient(p0, p3)
        hl.setColorAt(0.0, QColor(255, 255, 255, 35))
        hl.setColorAt(0.55, QColor(255, 255, 255, 0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(hl)
        p.drawPath(roof_path)

    def _window_quad(
        self,
        p0: QPointF,
        p1: QPointF,
        p2: QPointF,
        p3: QPointF,
        u0: float,
        u1: float,
        v0: float,
        v1: float,
    ) -> QPainterPath:
        a = _roof_uv(p0, p1, p2, p3, u0, v0)
        b = _roof_uv(p0, p1, p2, p3, u1, v0)
        c = _roof_uv(p0, p1, p2, p3, u1, v1)
        d = _roof_uv(p0, p1, p2, p3, u0, v1)
        return _quad_path(a, b, c, d)

    def _draw_windows_roof(
        self,
        p: QPainter,
        p0: QPointF,
        p1: QPointF,
        p2: QPointF,
        p3: QPointF,
        s: float,
    ) -> None:
        specs: list[tuple[str, float, float, float, float, float]] = [
            ("主驾", self._disp_fl, 0.02, 0.46, 0.02, 0.46),
            ("副驾", self._disp_fr, 0.54, 0.98, 0.02, 0.46),
            ("左后", self._disp_rl, 0.02, 0.46, 0.54, 0.98),
            ("右后", self._disp_rr, 0.54, 0.98, 0.54, 0.98),
        ]
        inset = 2.2 * s
        for label, open_pct, u0, u1, v0, v1 in specs:
            win_path = self._window_quad(p0, p1, p2, p3, u0, u1, v0, v1)
            br = win_path.boundingRect()
            g = QLinearGradient(br.topLeft(), br.bottomRight())
            o = open_pct / 100.0
            g.setColorAt(0.0, QColor(160, 210, 255, int(35 + 130 * o)))
            g.setColorAt(0.45, QColor(80, 140, 210, int(50 + 160 * o)))
            g.setColorAt(1.0, QColor(20, 50, 90, int(120 + 100 * o)))
            p.setPen(QPen(QColor(70, 90, 120), 1))
            p.setBrush(g)
            p.drawPath(win_path)

            p.setPen(QPen(QColor(200, 230, 255, int(70 + 150 * o)), max(1, int(1.2 + o * 2))))
            p.setBrush(QColor(0, 0, 0, 0))
            p.drawPath(win_path)

            p.setPen(QColor(220, 235, 255))
            tf = QFont(self.font())
            tf.setPointSize(max(7, int(8 * max(0.85, s))))
            tf.setBold(True)
            p.setFont(tf)
            lp = _roof_uv(p0, p1, p2, p3, (u0 + u1) * 0.5, v0 + 0.02)
            p.drawText(QRectF(lp.x() - 40, lp.y() - inset - 14, 80, 14), Qt.AlignmentFlag.AlignHCenter, label)
            p.setPen(QColor(180, 220, 255))
            nf = QFont(tf)
            nf.setBold(False)
            nf.setPointSize(max(6, tf.pointSize() - 1))
            p.setFont(nf)
            cp = _roof_uv(p0, p1, p2, p3, (u0 + u1) * 0.5, (v0 + v1) * 0.5)
            p.drawText(QRectF(cp.x() - 22, cp.y() - 10, 44, 20), Qt.AlignmentFlag.AlignCenter, f"{int(round(open_pct))}%")

    def _draw_lights_and_grille(
        self,
        p: QPainter,
        p0: QPointF,
        p1: QPointF,
        depth: QPointF,
        s: float,
    ) -> None:
        mid = _lerp_pt(p0, p1, 0.5)
        n = QPointF(-(p1.y() - p0.y()), (p1.x() - p0.x()))
        ln = math.hypot(n.x(), n.y()) or 1.0
        n = QPointF(n.x() / ln * (8 * s), n.y() / ln * (8 * s))
        lp = _lerp_pt(p0, p1, 0.18) - n
        rp = _lerp_pt(p0, p1, 0.82) - n
        for pt, warm in ((lp, QColor(255, 248, 220)), (rp, QColor(240, 245, 255))):
            rg = QRadialGradient(pt, 14 * s)
            rg.setColorAt(0.0, warm)
            rg.setColorAt(0.55, QColor(200, 210, 230, 120))
            rg.setColorAt(1.0, QColor(80, 90, 110, 0))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(rg)
            p.drawEllipse(QRectF(pt.x() - 12 * s, pt.y() - 8 * s, 24 * s, 16 * s))

        g0 = _lerp_pt(p0, p1, 0.32) - n * 0.4
        g1 = _lerp_pt(p0, p1, 0.68) - n * 0.4
        gg = QLinearGradient(g0, g1)
        gg.setColorAt(0.0, QColor(25, 28, 36))
        gg.setColorAt(0.5, QColor(40, 44, 56))
        gg.setColorAt(1.0, QColor(22, 25, 34))
        gw = math.hypot(p1.x() - p0.x(), p1.y() - p0.y()) * 0.22
        perp = QPointF(-(p1.y() - p0.y()), p1.x() - p0.x())
        pl = math.hypot(perp.x(), perp.y()) or 1.0
        perp = QPointF(perp.x() / pl * (4 * s), perp.y() / pl * (4 * s))
        gp = _quad_path(g0 - perp, g1 - perp, g1 + perp, g0 + perp)
        p.setPen(QPen(QColor(15, 18, 24), 1))
        p.setBrush(gg)
        p.drawPath(gp)

        logo = mid - n * 1.2
        p.setPen(QPen(QColor(130, 150, 190, 160), 2))
        p.setBrush(QColor(0, 0, 0, 0))
        p.drawEllipse(QRectF(logo.x() - 5 * s, logo.y() - 5 * s, 10 * s, 10 * s))

    def _draw_route(self, p: QPainter, p0: QPointF, p1: QPointF, s: float) -> None:
        if not self._v.nav_destination:
            return
        mid = _lerp_pt(p0, p1, 0.5)
        n = QPointF(-(p1.y() - p0.y()), (p1.x() - p0.x()))
        ln = math.hypot(n.x(), n.y()) or 1.0
        n = QPointF(n.x() / ln, n.y() / ln)
        dash_len = 10 + 4 * math.sin(self._route_phase)
        pen = QPen(QColor(110, 240, 170, 220))
        pen.setWidthF(max(2.0, 2.8 * s))
        pen.setStyle(Qt.PenStyle.CustomDashLine)
        pen.setDashPattern([dash_len, 9])
        pen.setDashOffset(self._route_phase * 22)
        p.setPen(pen)
        start = mid - n * (115 * s)
        end = mid - n * (38 * s)
        p.drawLine(start, end)
        tip = start + n * (8 * s)
        side = QPointF(-n.y(), n.x()) * (12 * s)
        tri = QPolygonF([start + side, start - side, tip])
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(110, 240, 170))
        p.drawPolygon(tri)

    def _draw_hud(self, p: QPainter, rect: QRectF) -> None:
        v = self._v
        pad = 12
        hud = QRectF(rect.left() + pad, rect.bottom() - 118, rect.width() - pad * 2, 106)
        p.setPen(QPen(QColor(60, 200, 140, 120), 1))
        p.setBrush(QColor(8, 14, 22, 210))
        p.drawRoundedRect(hud, 8, 8)

        mono = QFont("Consolas", max(8, min(10, int(self.height() / 72))))
        if not mono.exactMatch():
            mono = QFont(self.font())
            mono.setFamily("monospace")
            mono.setPointSize(max(8, min(10, int(self.height() / 72))))
        p.setFont(mono)

        t = 16.0
        line_h = 18.0
        x = hud.left() + 14
        y = hud.top() + t

        t_ac = max(0.0, min(1.0, (self._disp_ac - 16) / 16.0))
        cold, hot = QColor(70, 140, 255), QColor(255, 110, 70)
        ac_c = QColor(
            int(_lerp(cold.red(), hot.red(), t_ac)),
            int(_lerp(cold.green(), hot.green(), t_ac)),
            int(_lerp(cold.blue(), hot.blue(), t_ac)),
        )
        p.setPen(QColor(140, 200, 255))
        p.drawText(QPointF(x, y), f"空调 {self._disp_ac:.0f} °C")
        p.fillRect(QRectF(x + 120, y - 12, 80, 8), ac_c)

        y += line_h
        p.setPen(QColor(200, 200, 120))
        vol_w = min(220.0, hud.width() * 0.35)
        p.drawText(QPointF(x, y), f"媒体音量 {int(round(self._disp_vol))}%")
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(50, 60, 80))
        p.drawRoundedRect(QRectF(x + 110, y - 13, vol_w, 9), 2, 2)
        p.setBrush(QColor(220, 180, 60))
        p.drawRoundedRect(QRectF(x + 110, y - 13, vol_w * (self._disp_vol / 100.0), 9), 2, 2)

        y += line_h
        p.setPen(QColor(160, 210, 255))
        nav = (v.nav_destination or "—").strip()
        if len(nav) > 42:
            nav = nav[:40] + "…"
        p.drawText(QPointF(x, y), f"导航 {nav}")

        y += line_h
        p.setPen(QColor(220, 160, 255))
        mus = (v.now_playing or "—").strip()
        if len(mus) > 42:
            mus = mus[:40] + "…"
        p.drawText(QPointF(x, y), f"音乐 {mus}")

        y += line_h
        p.setPen(QColor(160, 200, 180))
        wx = (v.weather_line or "—").strip()
        if len(wx) > 52:
            wx = wx[:50] + "…"
        p.drawText(QPointF(x, y), f"天气 {wx}")
