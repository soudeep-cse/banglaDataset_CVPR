from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def _load_metrics(results_dir: Path) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for path in sorted(results_dir.glob("*_metrics.json")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["_path"] = str(path)
        entries.append(payload)
    return entries


def _load_predictions(results_dir: Path) -> dict[str, list[dict[str, object]]]:
    outputs: dict[str, list[dict[str, object]]] = {}
    for path in sorted(results_dir.glob("*_predictions.json")):
        with path.open("r", encoding="utf-8") as handle:
            outputs[path.stem.replace("_predictions", "")] = json.load(handle)
    return outputs


def _ensure_output(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def plot_reliability_diagram(results_dir: str | Path, output_dir: str | Path) -> Path | None:
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    _ensure_output(output_dir)
    predictions = _load_predictions(results_dir)
    if not predictions:
        return None

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    for model_name, rows in predictions.items():
        confidences = []
        accuracies = []
        bins = 10
        for index in range(bins):
            start = index / bins
            end = (index + 1) / bins
            bin_rows = [row for row in rows if row.get("confidence") is not None and start <= (float(row["confidence"]) / 100.0 if float(row["confidence"]) > 1 else float(row["confidence"])) < end]
            if not bin_rows:
                continue
            avg_conf = sum((float(row["confidence"]) / 100.0 if float(row["confidence"]) > 1 else float(row["confidence"])) for row in bin_rows) / len(bin_rows)
            acc = sum(1 for row in bin_rows if row.get("correct")) / len(bin_rows)
            confidences.append(avg_conf)
            accuracies.append(acc)
        if confidences:
            ax.plot(confidences, accuracies, marker="o", label=model_name)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability Diagram")
    ax.legend(fontsize=8)
    path = output_dir / "reliability_diagram.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_benchmark_table(results_dir: str | Path, output_dir: str | Path) -> Path | None:
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    _ensure_output(output_dir)
    metrics = _load_metrics(results_dir)
    if not metrics:
        return None

    rows = []
    for entry in metrics:
        model = entry.get("model", "unknown")
        values = entry.get("metrics", {})
        rows.append([
            model,
            f"{float(values.get('accuracy', 0.0)):.3f}",
            f"{float(values.get('ece', 0.0)):.3f}",
            f"{float(values.get('hallucination_rate', 0.0)):.3f}",
            f"{float(values.get('logical_consistency', 0.0)):.3f}",
        ])

    fig, ax = plt.subplots(figsize=(10, max(2.5, 0.45 * len(rows) + 1.5)))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Model", "Acc", "ECE", "Hallucination", "Consistency"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)
    path = output_dir / "benchmark_table.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_hallucination_rates(results_dir: str | Path, output_dir: str | Path) -> Path | None:
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    _ensure_output(output_dir)
    metrics = _load_metrics(results_dir)
    if not metrics:
        return None

    models = [entry.get("model", "unknown") for entry in metrics]
    hallucination_rates = [float(entry.get("metrics", {}).get("hallucination_rate", 0.0)) for entry in metrics]
    accuracy = [float(entry.get("metrics", {}).get("accuracy", 0.0)) for entry in metrics]

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.2), 5))
    x = range(len(models))
    ax.bar(x, accuracy, width=0.4, label="Accuracy")
    ax.bar([value + 0.4 for value in x], hallucination_rates, width=0.4, label="Hallucination")
    ax.set_xticks([value + 0.2 for value in x])
    ax.set_xticklabels(models, rotation=25, ha="right")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.set_title("Accuracy and Hallucination Rate")
    path = output_dir / "hallucination_rates.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_category_heatmap(results_dir: str | Path, output_dir: str | Path) -> Path | None:
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    _ensure_output(output_dir)
    predictions = _load_predictions(results_dir)
    if not predictions:
        return None

    labels = sorted({row.get("answer_type", "unknown") for rows in predictions.values() for row in rows})
    models = sorted(predictions)
    matrix = []
    for model in models:
        rows = predictions[model]
        values = []
        for label in labels:
            subset = [row for row in rows if row.get("answer_type", "unknown") == label]
            if not subset:
                values.append(0.0)
            else:
                values.append(sum(1 for row in subset if row.get("correct")) / len(subset))
        matrix.append(values)

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), max(4, len(models) * 0.7)))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    fig.colorbar(image, ax=ax, label="Accuracy")
    ax.set_title("Per-Type Accuracy Heatmap")
    path = output_dir / "category_heatmap.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def generate_all_figures(results_dir: str | Path, output_dir: str | Path) -> list[Path]:
    paths = [
        plot_reliability_diagram(results_dir, output_dir),
        plot_benchmark_table(results_dir, output_dir),
        plot_hallucination_rates(results_dir, output_dir),
        plot_category_heatmap(results_dir, output_dir),
    ]
    return [path for path in paths if path is not None]
