"""Shared detection metrics used by the LR6 container detector."""
from __future__ import annotations
import math
from typing import Any
import numpy as np

def collate(batch):
    return tuple(zip(*batch))

def iou(a: list[float], b: list[float]) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    union += max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0

def average_precision(
    predictions: list[dict[str, Any]],
    truths: dict[str, list[dict[str, Any]]],
    label: int,
    iou_threshold: float,
) -> tuple[float, list[float], list[float], int]:
    gt = {
        image_id: [item for item in values if item["label"] == label]
        for image_id, values in truths.items()
    }
    support = sum(map(len, gt.values()))
    if support == 0:
        return math.nan, [], [], 0
    matched = {image_id: set() for image_id in gt}
    ranked = []
    for prediction in predictions:
        for box, class_id, score in zip(
            prediction["boxes"], prediction["labels"], prediction["scores"]
        ):
            if class_id == label:
                ranked.append((score, prediction["image_id"], box))
    ranked.sort(key=lambda item: item[0], reverse=True)
    tp, fp, recalls, precisions = 0, 0, [], []
    for _, image_id, box in ranked:
        candidates = [
            (iou(box, truth["box"]), index)
            for index, truth in enumerate(gt.get(image_id, []))
            if index not in matched.get(image_id, set())
        ]
        overlap, index = max(candidates, default=(0.0, -1))
        if overlap >= iou_threshold:
            tp += 1
            matched[image_id].add(index)
        else:
            fp += 1
        recalls.append(tp / support)
        precisions.append(tp / (tp + fp))
    envelope = np.maximum.accumulate(np.asarray(precisions[::-1]))[::-1] if precisions else []
    ap = (
        sum(
            max(
                (
                    float(envelope[index])
                    for index, recall in enumerate(recalls)
                    if recall >= target
                ),
                default=0.0,
            )
            for target in np.linspace(0, 1, 101)
        )
        / 101.0
    )
    return float(ap), recalls, precisions, support

def best_threshold(
    predictions: list[dict[str, Any]],
    truths: dict[str, list[dict[str, Any]]],
    labels: list[int],
) -> tuple[float, dict[str, float]]:
    best_threshold = 0.0
    best = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    for threshold in np.arange(0.05, 0.951, 0.025):
        tp = fp = fn = 0
        for prediction in predictions:
            gt = truths[prediction["image_id"]]
            found: set[int] = set()
            candidate_predictions = sorted(
                [
                    (score, box, label)
                    for box, label, score in zip(
                        prediction["boxes"], prediction["labels"], prediction["scores"]
                    )
                    if score >= threshold
                ],
                reverse=True,
            )
            for _, box, label in candidate_predictions:
                options = [
                    (iou(box, item["box"]), i)
                    for i, item in enumerate(gt)
                    if item["label"] == label and i not in found
                ]
                overlap, index = max(options, default=(0.0, -1))
                if overlap >= 0.5:
                    tp += 1
                    found.add(index)
                else:
                    fp += 1
            fn += len(gt) - len(found)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if f1 > best["f1"]:
            best_threshold = float(threshold)
            best = {"precision": precision, "recall": recall, "f1": f1}
    return best_threshold, best
