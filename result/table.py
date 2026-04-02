import json
import math

from typing import Any
from pathlib import Path

from rfm.utils.utils import ensure_dir, save_json

CLASSIFIER_DIRS: dict[str, str] = {
    "NTK": "../ntk/results",
    "RFM": "../rfm/results",
    "NN": "../nn_jf/results",
}

OUTPUT_PATH = "table.json"

def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON file."""
    with open(path, "r") as f:
        return json.load(f)

def load_dataset_accuracies(classifier_dirs: dict[str, str]) -> dict[str, dict[str, float]]:
    """
    Load dataset-level average accuracies.

    Returns:
        per_dataset[dataset_name][classifier_name] = avg_test_accuracy
    """
    per_dataset: dict[str, dict[str, float]] = {}

    for classifier_name, result_dir_str in classifier_dirs.items():
        result_dir = Path(result_dir_str)
        if not result_dir.is_dir():
            continue

        for dataset_dir in sorted(result_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue

            summary_path = dataset_dir / "eval_summary.json"
            if not summary_path.is_file():
                continue

            payload = load_json(summary_path)
            dataset_name = payload["dataset_name"]
            avg_test_accuracy = float(payload["avg_test_accuracy"])

            if dataset_name not in per_dataset:
                per_dataset[dataset_name] = {}
            per_dataset[dataset_name][classifier_name] = avg_test_accuracy

    return per_dataset

def average_ranks_desc(values: list[float]) -> list[float]:
    """
    Rank values in descending order with average ranks for ties.

    Best value gets rank 1.
    """
    indexed = list(enumerate(values))
    indexed.sort(key=lambda item: item[1], reverse=True)

    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1

        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            original_idx = indexed[k][0]
            ranks[original_idx] = avg_rank

        i = j + 1

    return ranks

def mean(values: list[float]) -> float:
    """Compute arithmetic mean."""
    return sum(values) / len(values)

def std(values: list[float]) -> float:
    """Compute sample standard deviation."""
    if len(values) <= 1:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((x - mu) ** 2 for x in values) / (len(values) - 1))

def compute_metrics(per_dataset: dict[str, dict[str, float]]) -> dict[str, Any]:
    """
    Compute comparison metrics over datasets that have results for all classifiers.
    """
    classifier_names = sorted({
        classifier_name
        for dataset_scores in per_dataset.values()
        for classifier_name in dataset_scores
    })

    complete_dataset_names = sorted(
        dataset_name
        for dataset_name, dataset_scores in per_dataset.items()
        if all(classifier_name in dataset_scores for classifier_name in classifier_names)
    )

    metrics: dict[str, dict[str, Any]] = {
        classifier_name: {
            "accuracies": [],
            "ranks": [],
            "pma_values": [],
            "p90_count": 0,
            "p95_count": 0,
        }
        for classifier_name in classifier_names
    }

    for dataset_name in complete_dataset_names:
        dataset_scores = per_dataset[dataset_name]
        values = [dataset_scores[classifier_name] for classifier_name in classifier_names]
        ranks = average_ranks_desc(values)
        dataset_max = max(values)

        for classifier_idx, classifier_name in enumerate(classifier_names):
            value = values[classifier_idx]
            metrics[classifier_name]["accuracies"].append(value)
            metrics[classifier_name]["ranks"].append(ranks[classifier_idx])

            pma = 100.0 * value / dataset_max if dataset_max > 0.0 else 0.0
            metrics[classifier_name]["pma_values"].append(pma)

            if value >= 0.90 * dataset_max:
                metrics[classifier_name]["p90_count"] += 1
            if value >= 0.95 * dataset_max:
                metrics[classifier_name]["p95_count"] += 1

    num_datasets = len(complete_dataset_names)

    classifier_summary: dict[str, Any] = {}
    for classifier_name in classifier_names:
        accuracies = metrics[classifier_name]["accuracies"]
        ranks = metrics[classifier_name]["ranks"]
        pma_values = metrics[classifier_name]["pma_values"]

        classifier_summary[classifier_name] = {
            "friedman_rank": mean(ranks) if ranks else None,
            "average_accuracy_mean": mean(accuracies) if accuracies else None,
            "average_accuracy_std": std(accuracies) if accuracies else None,
            "p90": metrics[classifier_name]["p90_count"] / num_datasets if num_datasets > 0 else None,
            "p95": metrics[classifier_name]["p95_count"] / num_datasets if num_datasets > 0 else None,
            "pma_mean": mean(pma_values) if pma_values else None,
            "pma_std": std(pma_values) if pma_values else None,
        }

    return {
        "dataset_names": complete_dataset_names,
        "num_datasets": num_datasets,
        "classifiers": classifier_summary,
        "per_dataset": {
            dataset_name: {
                classifier_name: per_dataset[dataset_name][classifier_name]
                for classifier_name in classifier_names
            }
            for dataset_name in complete_dataset_names
        },
    }

def main() -> None:
    """Load classifier eval summaries, compute aggregate metrics, and save them."""
    per_dataset = load_dataset_accuracies(CLASSIFIER_DIRS)
    summary = compute_metrics(per_dataset)

    ensure_dir(Path(OUTPUT_PATH).parent)
    save_json(OUTPUT_PATH, summary)

if __name__ == "__main__":
    main()