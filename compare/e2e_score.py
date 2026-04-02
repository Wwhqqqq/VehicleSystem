from __future__ import annotations

import json

from benchmarks.common import result_path


OUTPUT = result_path("multi_test_output.txt")
LABELED = result_path("multi_test_output_labeled.txt")


def summarize_predictions() -> None:
    with OUTPUT.open("r", encoding="utf-8") as handle:
        for line in handle:
            info = json.loads(line)
            query = info["query"]
            result = ""
            for msg in info["res"]:
                if msg.get("intent") == "闲聊百科":
                    result += str(msg.get("frame", "")).replace("\n", " ")
                else:
                    result += f"{msg.get('intent')}\t{json.dumps(msg.get('slots', {}), ensure_ascii=False)}"
            print(query + "############" + result)


def summarize_labeled_scores() -> None:
    if not LABELED.exists():
        print(f"labeled file not found: {LABELED}")
        return
    right = 0
    total = 0
    with LABELED.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith("1"):
                right += 1
            total += 1
    if total == 0:
        print("labeled file is empty")
        return
    print(f"e2e accuracy: {right / total * 100:.3f}% ({right}/{total})")


if __name__ == "__main__":
    summarize_predictions()
    print("=" * 50)
    summarize_labeled_scores()
    print("=" * 50)
