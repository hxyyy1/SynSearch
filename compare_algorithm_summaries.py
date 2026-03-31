from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_SUMMARIES = {
    "MCTSyn": Path(".alphasyn_work/summary.csv"),
    "SASyn": Path(".sasyn_work/summary.csv"),
    "MABSyn": Path(".mabsyn_work/results/baseline_mab/summary.csv"),
}
DEFAULT_OUTPUT = Path(".compare_results/aggregate.csv")
DEFAULT_DETAILS_OUTPUT = Path(".compare_results/details.csv")

METRIC_COLUMNS = {
    "design": ("design_name", "file"),
    "and": ("final_and", "and"),
    "lev": ("final_lev", "lev"),
    "runtime": ("total_runtime_sec", "runtime_sec"),
    "memory": ("peak_memory_mb", "peak_memory_kb", "peak_memory", "memory_mb", "memory_kb", "memory"),
}


@dataclass(frozen=True)
class SummarySpec:
    name: str
    path: Path


@dataclass(frozen=True)
class LoadedSummary:
    spec: SummarySpec
    design_column: str
    metric_columns: dict[str, str]
    rows: dict[str, dict[str, float | None]]
    variant_label: str | None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare algorithm summaries by normalizing each metric with the per-design "
            "maximum across algorithms, then aggregating with weighted sums."
        )
    )
    parser.add_argument(
        "--summary",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help=(
            "Algorithm summary to compare. Can be passed multiple times. "
            "Default: MCTSyn/SASyn/MABSyn project summaries."
        ),
    )
    parser.add_argument(
        "--missing-policy",
        choices=("common", "error"),
        default="common",
        help="How to handle designs not shared by every summary.",
    )
    parser.add_argument("--and-weight", type=float, default=0.5)
    parser.add_argument("--lev-weight", type=float, default=0.2)
    parser.add_argument("--runtime-weight", type=float, default=0.2)
    parser.add_argument("--memory-weight", type=float, default=0.1)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="CSV path for algorithm-level aggregate scores.",
    )
    parser.add_argument(
        "--details-output",
        type=Path,
        default=DEFAULT_DETAILS_OUTPUT,
        help="CSV path for per-design normalized details.",
    )
    parser.add_argument(
        "--print-report",
        action="store_true",
        help="Also print the comparison tables to stdout.",
    )
    return parser


def _parse_summary_specs(raw_specs: list[str]) -> list[SummarySpec]:
    if not raw_specs:
        return [SummarySpec(name, path) for name, path in DEFAULT_SUMMARIES.items()]

    specs: list[SummarySpec] = []
    for raw_spec in raw_specs:
        if "=" not in raw_spec:
            raise SystemExit(f"Invalid --summary '{raw_spec}'. Expected NAME=PATH.")
        name, raw_path = raw_spec.split("=", 1)
        name = name.strip()
        raw_path = raw_path.strip()
        if not name or not raw_path:
            raise SystemExit(f"Invalid --summary '{raw_spec}'. Expected NAME=PATH.")
        specs.append(SummarySpec(name, Path(raw_path)))
    return specs


def _pick_column(headers: Iterable[str], candidates: tuple[str, ...]) -> str | None:
    header_set = {header.strip() for header in headers}
    for candidate in candidates:
        if candidate in header_set:
            return candidate
    return None


def _parse_float(value: str, field_name: str, design_name: str, source: Path) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise SystemExit(
            f"Failed to parse '{field_name}'='{value}' for design '{design_name}' in {source}"
        ) from exc


def load_summary(spec: SummarySpec) -> LoadedSummary:
    path = spec.path.resolve()
    if not path.exists():
        raise SystemExit(f"Summary file not found for {spec.name}: {path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        design_column = _pick_column(headers, METRIC_COLUMNS["design"])
        if design_column is None:
            raise SystemExit(
                f"Could not find a design column in {path}. Expected one of: {METRIC_COLUMNS['design']}"
            )

        metric_columns: dict[str, str] = {}
        for metric_name in ("and", "lev", "runtime", "memory"):
            picked = _pick_column(headers, METRIC_COLUMNS[metric_name])
            if picked is not None:
                metric_columns[metric_name] = picked

        required = ("and", "lev", "runtime")
        missing_required = [metric for metric in required if metric not in metric_columns]
        if missing_required:
            raise SystemExit(
                f"Summary {path} is missing required columns for metrics: {', '.join(missing_required)}"
            )

        rows: dict[str, dict[str, float | None]] = {}
        variant_labels: set[str] = set()
        for raw_row in reader:
            design_name = (raw_row.get(design_column) or "").strip()
            if not design_name:
                continue
            variant_label = (raw_row.get("variant_label") or "").strip()
            if variant_label:
                variant_labels.add(variant_label)
            metrics: dict[str, float | None] = {}
            for metric_name, column_name in metric_columns.items():
                raw_value = (raw_row.get(column_name) or "").strip()
                if not raw_value:
                    metrics[metric_name] = None
                    continue
                metrics[metric_name] = _parse_float(raw_value, column_name, design_name, path)
            rows[design_name] = metrics

    if not rows:
        raise SystemExit(f"Summary {path} does not contain any usable rows.")
    if len(variant_labels) > 1:
        raise SystemExit(
            f"Summary {path} contains multiple variant_label values; please split different parameter runs into separate summaries."
        )

    return LoadedSummary(
        spec=SummarySpec(spec.name, path),
        design_column=design_column,
        metric_columns=metric_columns,
        rows=rows,
        variant_label=next(iter(variant_labels)) if variant_labels else None,
    )


def _natural_key(text: str) -> tuple[object, ...]:
    parts: list[object] = []
    current = ""
    in_digits = False
    for char in text:
        if char.isdigit():
            if not in_digits and current:
                parts.append(current.lower())
                current = ""
            current += char
            in_digits = True
        else:
            if in_digits and current:
                parts.append(int(current))
                current = ""
            current += char
            in_digits = False
    if current:
        parts.append(int(current) if in_digits else current.lower())
    return tuple(parts)


def _format_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def render(cells: list[str]) -> str:
        return " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(cells))

    separator = "-+-".join("-" * width for width in widths)
    return "\n".join([render(headers), separator, *(render(row) for row in rows)])


def _shared_designs(summaries: list[LoadedSummary], policy: str) -> list[str]:
    design_sets = [set(summary.rows) for summary in summaries]
    shared = set.intersection(*design_sets)
    if not shared:
        raise SystemExit("No common designs were found across the provided summaries.")

    if policy == "error":
        union = set.union(*design_sets)
        if shared != union:
            missing_lines: list[str] = []
            for summary in summaries:
                missing = sorted(union - set(summary.rows), key=_natural_key)
                if missing:
                    missing_lines.append(f"{summary.spec.name}: missing {', '.join(missing)}")
            detail = "\n".join(missing_lines)
            raise SystemExit(f"Summaries do not cover the same designs:\n{detail}")

    return sorted(shared, key=_natural_key)


def _summary_has_metric(summary: LoadedSummary, design_name: str, metric_name: str) -> bool:
    if metric_name not in summary.metric_columns:
        return False
    value = summary.rows[design_name].get(metric_name)
    return isinstance(value, (int, float))


def _active_metrics(
    summaries: list[LoadedSummary],
    weights: dict[str, float],
) -> list[str]:
    active: list[str] = []
    for metric_name in ("and", "lev", "runtime", "memory"):
        if weights[metric_name] <= 0:
            continue
        if all(metric_name in summary.metric_columns for summary in summaries):
            active.append(metric_name)
    if not active:
        raise SystemExit("No metrics are available after applying the provided weights and summary columns.")
    return active


def compare_summaries(
    summaries: list[LoadedSummary],
    *,
    weights: dict[str, float],
    missing_policy: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[str], list[str], float]:
    designs = _shared_designs(summaries, missing_policy)
    metrics = _active_metrics(summaries, weights)
    active_weight_sum = sum(weights[metric_name] for metric_name in metrics)
    if active_weight_sum <= 0:
        raise SystemExit("The sum of active metric weights must be positive.")

    detail_rows: list[dict[str, object]] = []
    aggregate_rows: list[dict[str, object]] = []
    scores_by_algorithm: dict[str, list[dict[str, float | None]]] = {summary.spec.name: [] for summary in summaries}

    for design_name in designs:
        design_metrics = [
            metric_name
            for metric_name in metrics
            if all(_summary_has_metric(summary, design_name, metric_name) for summary in summaries)
        ]
        if not design_metrics:
            continue

        denominators = {
            metric_name: max(float(summary.rows[design_name][metric_name]) for summary in summaries)
            for metric_name in design_metrics
        }
        design_weight_sum = sum(weights[metric_name] for metric_name in design_metrics)
        for summary in summaries:
            raw_metrics = summary.rows[design_name]
            normalized_metrics = {
                metric_name: (
                    float(raw_metrics[metric_name]) / denominators[metric_name]
                    if denominators[metric_name] > 0
                    else 0.0
                )
                for metric_name in design_metrics
            }
            weighted_cost = sum(
                weights[metric_name] * normalized_metrics[metric_name]
                for metric_name in design_metrics
            )
            final_score = 100.0 * (1.0 - (weighted_cost / design_weight_sum))
            scores_by_algorithm[summary.spec.name].append(
                {
                    "weighted_cost": weighted_cost,
                    "final_score": final_score,
                    "weight_sum": design_weight_sum,
                    **{
                        f"raw_{metric_name}": float(raw_metrics[metric_name])
                        if metric_name in design_metrics
                        else None
                        for metric_name in metrics
                    },
                    **{
                        f"norm_{metric_name}": normalized_metrics.get(metric_name)
                        for metric_name in metrics
                    },
                }
            )
            detail_row: dict[str, object] = {
                "algorithm": summary.spec.name,
                "variant_label": summary.variant_label,
                "design": design_name,
                "weighted_cost": weighted_cost,
                "final_score": final_score,
                "active_weight_sum": design_weight_sum,
            }
            for metric_name in metrics:
                detail_row[f"{metric_name}"] = raw_metrics.get(metric_name)
                detail_row[f"max_{metric_name}"] = denominators.get(metric_name)
                detail_row[f"norm_{metric_name}"] = normalized_metrics.get(metric_name)
            detail_rows.append(detail_row)

    for summary in summaries:
        algorithm_scores = scores_by_algorithm[summary.spec.name]
        if not algorithm_scores:
            continue
        row: dict[str, object] = {
            "algorithm": summary.spec.name,
            "variant_label": summary.variant_label,
            "design_count": len(algorithm_scores),
            "weighted_cost": sum(item["weighted_cost"] for item in algorithm_scores) / len(algorithm_scores),
            "final_score": sum(item["final_score"] for item in algorithm_scores) / len(algorithm_scores),
            "summary_path": str(summary.spec.path),
        }
        for metric_name in metrics:
            raw_values = [item[f"raw_{metric_name}"] for item in algorithm_scores if item[f"raw_{metric_name}"] is not None]
            norm_values = [item[f"norm_{metric_name}"] for item in algorithm_scores if item[f"norm_{metric_name}"] is not None]
            row[f"avg_{metric_name}"] = (sum(raw_values) / len(raw_values)) if raw_values else None
            row[f"avg_norm_{metric_name}"] = (sum(norm_values) / len(norm_values)) if norm_values else None
        aggregate_rows.append(row)

    aggregate_rows.sort(
        key=lambda row: (
            -float(row["final_score"]),
            float(row["weighted_cost"]),
            _natural_key(str(row["algorithm"])),
            _natural_key(str(row.get("variant_label") or "")),
        )
    )
    detail_rows.sort(
        key=lambda row: (
            _natural_key(str(row["design"])),
            _natural_key(str(row["algorithm"])),
            _natural_key(str(row.get("variant_label") or "")),
        )
    )
    return aggregate_rows, detail_rows, designs, metrics, active_weight_sum


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _format_optional_float(value: object, digits: int = 6) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def _print_report(
    aggregate_rows: list[dict[str, object]],
    detail_rows: list[dict[str, object]],
    *,
    designs: list[str],
    metrics: list[str],
    active_weight_sum: float,
    requested_weights: dict[str, float],
) -> None:
    print(f"Compared designs: {len(designs)}")
    print(f"Active metrics: {', '.join(metrics)}")
    if active_weight_sum < sum(requested_weights.values()):
        print(
            "Note: some weighted metrics were unavailable in the summaries and were excluded "
            f"(active weight sum = {active_weight_sum:.3f})."
        )

    aggregate_headers = ["rank", "algorithm", "variant_label", "designs", "weighted_cost", "final_score"]
    for metric_name in metrics:
        aggregate_headers.append(f"avg_{metric_name}")
        aggregate_headers.append(f"avg_norm_{metric_name}")
    aggregate_rows_text: list[list[str]] = []
    for rank, row in enumerate(aggregate_rows, start=1):
        cells = [
            str(rank),
            str(row["algorithm"]),
            str(row.get("variant_label") or "-"),
            str(row["design_count"]),
            f"{float(row['weighted_cost']):.6f}",
            f"{float(row['final_score']):.4f}",
        ]
        for metric_name in metrics:
            cells.append(_format_optional_float(row[f"avg_{metric_name}"]))
            cells.append(_format_optional_float(row[f"avg_norm_{metric_name}"]))
        aggregate_rows_text.append(cells)
    print()
    print(_format_table(aggregate_headers, aggregate_rows_text))

    detail_headers = ["design", "algorithm", "variant_label", "weighted_cost", "final_score", "active_weight_sum"]
    for metric_name in metrics:
        detail_headers.extend((metric_name, f"max_{metric_name}", f"norm_{metric_name}"))
    detail_rows_text: list[list[str]] = []
    for row in detail_rows:
        cells = [
            str(row["design"]),
            str(row["algorithm"]),
            str(row.get("variant_label") or "-"),
            f"{float(row['weighted_cost']):.6f}",
            f"{float(row['final_score']):.4f}",
            f"{float(row['active_weight_sum']):.3f}",
        ]
        for metric_name in metrics:
            cells.append(_format_optional_float(row[metric_name]))
            cells.append(_format_optional_float(row[f"max_{metric_name}"]))
            cells.append(_format_optional_float(row[f"norm_{metric_name}"]))
        detail_rows_text.append(cells)
    print()
    print(_format_table(detail_headers, detail_rows_text))


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    specs = _parse_summary_specs(args.summary)
    loaded = [load_summary(spec) for spec in specs]
    weights = {
        "and": args.and_weight,
        "lev": args.lev_weight,
        "runtime": args.runtime_weight,
        "memory": args.memory_weight,
    }

    aggregate_rows, detail_rows, designs, metrics, active_weight_sum = compare_summaries(
        loaded,
        weights=weights,
        missing_policy=args.missing_policy,
    )
    _write_csv(args.output, aggregate_rows)
    _write_csv(args.details_output, detail_rows)
    if args.print_report:
        _print_report(
            aggregate_rows,
            detail_rows,
            designs=designs,
            metrics=metrics,
            active_weight_sum=active_weight_sum,
            requested_weights=weights,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
