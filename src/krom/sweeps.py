from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

import matplotlib.pyplot as plt


@dataclass
class SweepResult:
    parameter_name: str
    parameter_values: List[float]
    metric_name: str
    ylabel: str
    title: str
    empirical_values: List[float]
    matern_values: List[float]
    raw_results: List[Dict[str, object]]


ComparisonEvaluator = Callable[[float], Dict[str, object]]


def run_comparison_sweep(
    parameter_values: Sequence[float],
    evaluator: ComparisonEvaluator,
    metric_name: str,
    parameter_name: str,
    ylabel: str,
    title: str,
) -> SweepResult:
    empirical_values: List[float] = []
    matern_values: List[float] = []
    raw_results: List[Dict[str, object]] = []

    for value in parameter_values:
        result = evaluator(float(value))
        raw_results.append(result)
        empirical_values.append(float(result['empirical'][metric_name]))
        matern_values.append(float(result['matern'][metric_name]))

    return SweepResult(
        parameter_name=parameter_name,
        parameter_values=[float(v) for v in parameter_values],
        metric_name=metric_name,
        ylabel=ylabel,
        title=title,
        empirical_values=empirical_values,
        matern_values=matern_values,
        raw_results=raw_results,
    )


def plot_sweep_result(
    sweep: SweepResult,
    ax=None,
    empirical_label: str = 'Empirical',
    matern_label: str = 'Matern',
):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sweep.parameter_values, sweep.empirical_values, marker='o', label=empirical_label)
    ax.plot(sweep.parameter_values, sweep.matern_values, marker='s', label=matern_label)
    ax.set_xlabel(sweep.parameter_name)
    ax.set_ylabel(sweep.ylabel)
    ax.set_title(sweep.title)
    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.legend()
    return ax
