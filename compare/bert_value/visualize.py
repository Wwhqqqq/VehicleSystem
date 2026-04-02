from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np


def _label_for_name(name: str) -> str:
    return {
        "bert_only": "仅 BERT",
        "full_llm": "全 LLM 拒识",
        "bert_prescreen_llm_deep": "BERT 前筛 + LLM 深判",
    }.get(name, name)


def _palette() -> dict[str, tuple[float, float, float]]:
    return {
        "bert_only": (0.22, 0.46, 0.69),
        "full_llm": (0.94, 0.33, 0.31),
        "bert_prescreen_llm_deep": (0.13, 0.55, 0.35),
    }


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#f7f8fa",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#d7dce5",
            "axes.labelcolor": "#2b2f36",
            "axes.titlecolor": "#1b1f24",
            "text.color": "#2b2f36",
            "xtick.color": "#2b2f36",
            "ytick.color": "#2b2f36",
            "grid.color": "#e6e9ef",
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "PingFang SC",
                "Noto Sans CJK SC",
                "Arial Unicode MS",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
        }
    )


def _records(block: list[dict]) -> tuple[list[str], list[dict]]:
    names = [str(item["name"]) for item in block]
    return names, block


def build_figures(payload: dict, out_dir: Path) -> list[Path]:
    _apply_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    palette = _palette()
    written: list[Path] = []

    single = payload.get("single_turn") or []
    multi = payload.get("multi_turn") or []

    if len(single) == 1:
        row = single[0]
        name = str(row["name"])
        fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
        color = palette.get(name, (0.4, 0.4, 0.4))
        xs = np.arange(2)
        ax.bar(
            xs,
            [float(row["latency_ms_mean"]), float(row["latency_ms_p95"])],
            color=[color, color],
            alpha=[0.9, 0.55],
            edgecolor="#ffffff",
            linewidth=0.6,
        )
        ax.set_xticks(xs, ["平均", "P95"])
        ax.set_ylabel("毫秒 (ms)")
        ax.set_title(f"基线延迟：{_label_for_name(name)}（完整对比需配置 LLM 并去掉 --skip-llm）")
        ax.grid(axis="y", linestyle="--", linewidth=0.8)
        path = out_dir / "fig0_bert_only_baseline.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path)

    if len(single) >= 2:
        names, rows = _records(single)
        labels = [_label_for_name(name) for name in names]
        x = np.arange(len(labels))
        width = 0.34

        fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), constrained_layout=True)
        means = [float(row["latency_ms_mean"]) for row in rows]
        p95s = [float(row["latency_ms_p95"]) for row in rows]
        colors = [palette.get(name, (0.4, 0.4, 0.4)) for name in names]

        axes[0].bar(x - width / 2, means, width, label="平均", color=colors, alpha=0.88, edgecolor="#ffffff", linewidth=0.6)
        axes[0].bar(x + width / 2, p95s, width, label="P95", color=colors, alpha=0.55, edgecolor="#ffffff", linewidth=0.6)
        axes[0].set_xticks(x, labels, rotation=12, ha="right")
        axes[0].set_ylabel("毫秒 (ms)")
        axes[0].set_title("拒识链路延迟（单轮测试集）")
        axes[0].legend(frameon=False, loc="upper left")
        axes[0].yaxis.set_major_locator(mtick.MaxNLocator(integer=True))
        axes[0].grid(axis="y", linestyle="--", linewidth=0.8)

        llm_calls = [float(row["llm_calls"]) for row in rows]
        tokens = [float(row["estimated_total_tokens"]) for row in rows]
        axes[1].bar(x - width / 2, llm_calls, width, label="LLM 调用次数", color=colors, alpha=0.88, edgecolor="#ffffff", linewidth=0.6)
        ax2 = axes[1].twinx()
        ax2.bar(x + width / 2, tokens, width, label="估算 tokens", color=colors, alpha=0.45, edgecolor="#ffffff", linewidth=0.6)
        axes[1].set_xticks(x, labels, rotation=12, ha="right")
        axes[1].set_ylabel("调用次数")
        ax2.set_ylabel("估算 tokens（提示+输出）")
        axes[1].set_title("成本对比（单轮测试集）")
        axes[1].legend(loc="upper left", frameon=False)
        ax2.legend(loc="upper right", frameon=False)
        axes[1].grid(axis="y", linestyle="--", linewidth=0.8)

        path = out_dir / "fig1_latency_cost_single_turn.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path)

        fig, ax = plt.subplots(figsize=(9.6, 4.6), constrained_layout=True)
        acc = [float(row["reject_accuracy"]) for row in rows]
        far = [float(row["false_accept_rate_on_should_reject"]) for row in rows]
        ax.bar(x - width / 2, acc, width, label="拒识任务总体准确率", color=colors, alpha=0.9, edgecolor="#ffffff", linewidth=0.6)
        ax.bar(x + width / 2, far, width, label="误召回率（应拒而放行）", color=colors, alpha=0.55, edgecolor="#ffffff", linewidth=0.6)
        ax.set_xticks(x, labels, rotation=12, ha="right")
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_title("关键指标：准确率 vs 误召回（单轮测试集）")
        ax.legend(frameon=False, loc="lower right")
        ax.grid(axis="y", linestyle="--", linewidth=0.8)
        path = out_dir / "fig2_accuracy_false_accept_single_turn.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path)

    if len(multi) == 1:
        row = multi[0]
        name = str(row["name"])
        fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
        color = palette.get(name, (0.4, 0.4, 0.4))
        rate = float(row.get("multiturn_session_success_rate", 0.0))
        ax.bar([_label_for_name(name)], [rate], color=color, alpha=0.9, edgecolor="#ffffff", linewidth=0.6)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_title("多轮会话成功率（单策略基线）")
        ax.grid(axis="y", linestyle="--", linewidth=0.8)
        path = out_dir / "fig3_multiturn_session_success.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path)

    if len(multi) >= 2:
        names, rows = _records(multi)
        labels = [_label_for_name(name) for name in names]
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(8.8, 4.6), constrained_layout=True)
        colors = [palette.get(name, (0.4, 0.4, 0.4)) for name in names]
        mts = [float(row["multiturn_session_success_rate"]) for row in rows]
        ax.bar(x, mts, color=colors, alpha=0.9, edgecolor="#ffffff", linewidth=0.6)
        ax.set_xticks(x, labels, rotation=12, ha="right")
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_title("多轮会话：逐轮拒识标注全对率（会话级成功率）")
        ax.grid(axis="y", linestyle="--", linewidth=0.8)
        path = out_dir / "fig3_multiturn_session_success.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path)

    readme = out_dir / "figures_index.txt"
    with readme.open("w", encoding="utf-8") as handle:
        handle.write("BERT routing experiment figures\n")
        for item in written:
            handle.write(f"- {item.name}\n")
    written.append(readme)

    return written


def load_results(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def replot_from_file(results_json: Path, out_dir: Path | None = None) -> list[Path]:
    payload = load_results(results_json)
    target = out_dir or results_json.parent
    return build_figures(payload, target)
