import json
import os


def main():
    baseline_path = "baseline/metrics.json"
    current_path = "reports/metrics.json"

    if not os.path.exists(baseline_path):
        print("Baseline metrics not found.")
        return

    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline = json.load(f)
    with open(current_path, "r", encoding="utf-8") as f:
        current = json.load(f)

    markdown = "| Метрика | Baseline (main) | Current (PR) | $\Delta$ (Дельта) |\n"
    markdown += "|---|---|---|---|\n"

    for key in current.keys():
        b_val = baseline.get(key, 0.0)
        c_val = current.get(key, 0.0)
        delta = c_val - b_val

        markdown += f"| **{key}** | {b_val:.4f} | {c_val:.4f} | {delta:+.4f} |\n"

    with open("reports/comparison_report.md", "w", encoding="utf-8") as f:
        f.write(markdown)


if __name__ == "__main__":
    main()
