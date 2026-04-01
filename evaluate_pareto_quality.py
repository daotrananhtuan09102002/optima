from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np


def dominates(candidate: np.ndarray, competitor: np.ndarray) -> bool:
    return bool(np.all(candidate <= competitor) and np.any(candidate < competitor))


def non_dominated(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points

    keep_mask = np.ones(len(points), dtype=bool)
    for i in range(len(points)):
        if not keep_mask[i]:
            continue
        for j in range(len(points)):
            if i == j or not keep_mask[j]:
                continue
            if dominates(points[j], points[i]):
                keep_mask[i] = False
                break
    return points[keep_mask]


def hypervolume_2d_min(points_2d: np.ndarray, reference_2d: np.ndarray) -> float:
    if points_2d.size == 0:
        return 0.0

    valid = points_2d[
        (points_2d[:, 0] < reference_2d[0]) & (points_2d[:, 1] < reference_2d[1])
    ]
    if valid.size == 0:
        return 0.0

    order = np.argsort(valid[:, 0])
    ordered = valid[order]

    area = 0.0
    best_z = float(reference_2d[1])
    ref_y = float(reference_2d[0])

    for y, z in ordered:
        if z >= best_z:
            continue
        width = max(0.0, ref_y - float(y))
        height = max(0.0, best_z - float(z))
        area += width * height
        best_z = float(z)

    return area


def hypervolume_3d_min(points_3d: np.ndarray, reference_3d: np.ndarray) -> float:
    if points_3d.size == 0:
        return 0.0

    valid = points_3d[
        (points_3d[:, 0] < reference_3d[0])
        & (points_3d[:, 1] < reference_3d[1])
        & (points_3d[:, 2] < reference_3d[2])
    ]
    if valid.size == 0:
        return 0.0

    x_breaks = sorted(set(valid[:, 0].tolist() + [float(reference_3d[0])]))
    if len(x_breaks) < 2:
        return 0.0

    hv = 0.0
    for idx in range(len(x_breaks) - 1):
        x_left = x_breaks[idx]
        x_right = x_breaks[idx + 1]
        width = x_right - x_left
        if width <= 0:
            continue

        active = valid[valid[:, 0] <= x_left][:, 1:3]
        if active.size == 0:
            continue

        area = hypervolume_2d_min(active, reference_3d[1:3])
        hv += width * area

    return hv


def gd_igd(
    approx_front: np.ndarray,
    reference_front: np.ndarray,
    scale_min: np.ndarray,
    scale_span: np.ndarray,
) -> tuple[float, float]:
    if approx_front.size == 0 or reference_front.size == 0:
        return math.nan, math.nan

    normalized_approx = (approx_front - scale_min) / scale_span
    normalized_ref = (reference_front - scale_min) / scale_span

    pairwise = np.linalg.norm(
        normalized_approx[:, np.newaxis, :] - normalized_ref[np.newaxis, :, :],
        axis=2,
    )
    gd = float(np.mean(np.min(pairwise, axis=1)))
    igd = float(np.mean(np.min(pairwise, axis=0)))
    return gd, igd


def read_pareto_front(run_dir: Path) -> np.ndarray:
    path = run_dir / "pareto_front.json"
    if not path.exists():
        return np.empty((0, 3), dtype=float)

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return np.empty((0, 3), dtype=float)

    vectors: list[tuple[float, float, float]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        objectives = item.get("objectives", {})
        if not isinstance(objectives, dict):
            continue

        performance = objectives.get("performance")
        length = objectives.get("length")
        perplexity = objectives.get("perplexity")
        if performance is None or length is None or perplexity is None:
            continue

        performance = float(performance)
        length = float(length)
        perplexity = float(perplexity)
        if not (
            math.isfinite(performance) and math.isfinite(length) and math.isfinite(perplexity)
        ):
            continue

        vectors.append((performance, length, perplexity))

    if not vectors:
        return np.empty((0, 3), dtype=float)
    return np.asarray(vectors, dtype=float)


def aggregate(values: Iterable[float]) -> dict[str, float | None]:
    valid = [value for value in values if math.isfinite(value)]
    if not valid:
        return {"mean": None, "std": None}

    mean = float(np.mean(valid))
    std = float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0
    return {"mean": mean, "std": std}


def infer_reference_point(points: np.ndarray, margin_ratio: float) -> np.ndarray:
    maxima = np.max(points, axis=0)
    minima = np.min(points, axis=0)
    span = maxima - minima

    # Keep a strictly dominated box boundary, even for degenerate ranges.
    eps = np.where(span > 0, span * margin_ratio, 1e-6)
    return maxima + eps


def evaluate(artifact_root: Path, reference_point: np.ndarray | None, margin: float) -> dict[str, object]:
    run_dirs = sorted(path for path in artifact_root.glob("run_*") if path.is_dir())

    run_fronts: list[tuple[int, np.ndarray]] = []
    all_points: list[np.ndarray] = []

    for run_dir in run_dirs:
        try:
            run_index = int(run_dir.name.split("_", maxsplit=1)[1])
        except (IndexError, ValueError):
            continue
        front = non_dominated(read_pareto_front(run_dir))
        if front.size == 0:
            continue
        run_fronts.append((run_index, front))
        all_points.append(front)

    if not all_points:
        raise ValueError(f"No Pareto points found under {artifact_root}.")

    all_points_matrix = np.vstack(all_points)
    reference_front = non_dominated(all_points_matrix)

    if reference_point is None:
        reference_point = infer_reference_point(all_points_matrix, margin)

    scale_min = np.min(reference_front, axis=0)
    scale_max = np.max(reference_front, axis=0)
    scale_span = np.where(scale_max > scale_min, scale_max - scale_min, 1e-9)

    run_metrics: list[dict[str, float | int]] = []
    for run_index, front in run_fronts:
        hv = hypervolume_3d_min(front, reference_point)
        gd, igd = gd_igd(front, reference_front, scale_min, scale_span)
        run_metrics.append(
            {
                "run_index": run_index,
                "front_size": int(len(front)),
                "hv": float(hv),
                "gd": float(gd),
                "igd": float(igd),
            }
        )

    return {
        "artifact_root": str(artifact_root),
        "objective_space": "minimization(performance,length,perplexity)",
        "reference_point": reference_point.tolist(),
        "reference_front_size": int(len(reference_front)),
        "run_metrics": run_metrics,
        "aggregate": {
            "hv": aggregate(item["hv"] for item in run_metrics),
            "gd": aggregate(item["gd"] for item in run_metrics),
            "igd": aggregate(item["igd"] for item in run_metrics),
        },
    }


def print_report(result: dict[str, object]) -> None:
    print("Pareto-quality metrics (lower is better for GD/IGD, higher is better for HV)")
    print(f"Artifact root: {result['artifact_root']}")
    print(f"Reference front size: {result['reference_front_size']}")
    print(f"Reference point: {result['reference_point']}")
    print("")
    print(f"{'Run':>5} {'Size':>6} {'HV':>14} {'GD':>12} {'IGD':>12}")

    run_metrics = result.get("run_metrics", [])
    if isinstance(run_metrics, list):
        for item in run_metrics:
            if not isinstance(item, dict):
                continue
            print(
                f"{int(item['run_index']):>5} "
                f"{int(item['front_size']):>6} "
                f"{float(item['hv']):>14.6f} "
                f"{float(item['gd']):>12.6f} "
                f"{float(item['igd']):>12.6f}"
            )

    aggregate_metrics = result.get("aggregate", {})
    if isinstance(aggregate_metrics, dict):
        print("")
        for key in ("hv", "gd", "igd"):
            metric = aggregate_metrics.get(key, {})
            if not isinstance(metric, dict):
                continue
            mean = metric.get("mean")
            std = metric.get("std")
            if mean is None:
                print(f"{key.upper()}: n/a")
            else:
                print(f"{key.upper()}: mean={float(mean):.6f}, std={float(std):.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Pareto-front quality (HV, GD, IGD) from saved run logs.",
    )
    parser.add_argument(
        "artifact_root",
        type=Path,
        help="Artifact root folder containing run_*/pareto_front.json",
    )
    parser.add_argument(
        "--reference-point",
        type=float,
        nargs=3,
        metavar=("PERFORMANCE", "LENGTH", "PERPLEXITY"),
        default=None,
        help="Optional manual reference point for hypervolume.",
    )
    parser.add_argument(
        "--reference-margin",
        type=float,
        default=0.05,
        help="Relative margin to infer the reference point from observed maxima.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output JSON path (default: <artifact_root>/pareto_quality.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_root = args.artifact_root.resolve()

    if not artifact_root.exists():
        raise FileNotFoundError(f"Artifact root does not exist: {artifact_root}")

    ref_point = (
        np.asarray(args.reference_point, dtype=float)
        if args.reference_point is not None
        else None
    )
    result = evaluate(
        artifact_root=artifact_root,
        reference_point=ref_point,
        margin=float(args.reference_margin),
    )

    print_report(result)

    output_path = args.output or artifact_root / "pareto_quality.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("")
    print(f"Saved report: {output_path}")


if __name__ == "__main__":
    main()
