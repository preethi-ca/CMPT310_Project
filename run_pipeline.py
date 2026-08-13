"""Run the CMPT 310 project model scripts from one entry point."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_CLASSIFICATION_DATA = "data/location-information-with-competitors.csv"


@dataclass(frozen=True)
class PipelineStep:
    name: str
    command: tuple[str, ...]


def build_steps(
    classification_data: str,
    include_visualizations: bool,
    skip_xgboost: bool,
) -> list[PipelineStep]:
    steps = [
        PipelineStep("Ridge regression", ("src/linear-regression.py",)),
        PipelineStep("Decision Tree regression", ("src/decision-tree.py",)),
        PipelineStep("KNN classification", ("src/knn_classification.py", classification_data)),
    ]

    if not skip_xgboost:
        steps.append(
            PipelineStep(
                "XGBoost classification",
                ("src/xgboost_classification.py", classification_data),
            )
        )

    if include_visualizations:
        steps.extend(
            [
                PipelineStep("Regression visualizations", ("src/regression-visualization.py",)),
                PipelineStep("KNN visualizations", ("src/knn-visualization.py",)),
            ]
        )

    return steps


def validate_step_files(step: PipelineStep) -> None:
    for value in step.command:
        if not value.endswith((".py", ".csv")):
            continue

        file_path = ROOT / value
        if not file_path.exists():
            raise FileNotFoundError(f"{step.name} needs missing file: {value}")


def run_step(step: PipelineStep, dry_run: bool) -> None:
    validate_step_files(step)
    command = [sys.executable, *step.command]
    printable_command = " ".join(command)

    if dry_run:
        print(f"[dry-run] {step.name}: {printable_command}")
        return

    print(f"\n=== {step.name} ===", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CMPT 310 model pipeline.")
    parser.add_argument(
        "--classification-data",
        default=DEFAULT_CLASSIFICATION_DATA,
        help="CSV dataset passed to the KNN and XGBoost classification scripts.",
    )
    parser.add_argument(
        "--include-visualizations",
        action="store_true",
        help="Also run the existing scripts that regenerate the PNG visualizations.",
    )
    parser.add_argument(
        "--skip-xgboost",
        action="store_true",
        help="Skip the XGBoost step for a faster local smoke check.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would run without executing them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    steps = build_steps(
        classification_data=args.classification_data,
        include_visualizations=args.include_visualizations,
        skip_xgboost=args.skip_xgboost,
    )

    for step in steps:
        run_step(step, args.dry_run)


if __name__ == "__main__":
    main()
