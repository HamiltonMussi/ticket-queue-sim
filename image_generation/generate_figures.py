"""
Generate analysis figures for the ticket-queue-sim report.

Creates PNGs with side-by-side comparisons across scenarios and
illustrative time-series for the bottleneck scenario.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from simulation.run_simulation import run_experiment, analyze_results
from simulation.config import NORMAL_SCENARIO, EXTREME_DEGRADATION_200W, EXTREME_DEGRADATION_1W
from simulation.models import JobType


def summarize(results) -> Dict[str, float]:
    """Return means for key metrics from a list of Metrics objects."""
    W = np.mean([r.get_W() for r in results])
    X = np.mean([r.get_X() for r in results])
    rho = np.mean([r.get_rho() for r in results])
    L = np.mean([r.get_L() for r in results])
    return {"W": W, "X": X, "rho": rho, "L": L}


def ci_bounds(values: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """Return half-width symmetric CI using t distribution."""
    import scipy.stats as stats

    a = np.array(values, dtype=float)
    n = len(a)
    if n < 2:
        return (0.0, 0.0)
    se = stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    mean = np.mean(a)
    return (mean - h, mean + h)


def bar_with_ci(ax, labels, means, lows, highs, title, ylabel):
    x = np.arange(len(labels))
    means = np.asarray(means)
    yerr = np.vstack([means - np.asarray(lows), np.asarray(highs) - means])
    ax.bar(x, means, yerr=yerr, capsize=6, alpha=0.8)
    ax.set_xticks(x, labels)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle=":", alpha=0.5)


def generate_comparison_figures(num_replications: int = 20, simulation_time: int = 3600):
    ROOT = Path(__file__).resolve().parent.parent
    OUT = ROOT / "outputs"
    OUT.mkdir(parents=True, exist_ok=True)
    labels = ["Normal", "Extreme 200w", "Extreme 1w"]
    scenarios = [NORMAL_SCENARIO, EXTREME_DEGRADATION_200W, EXTREME_DEGRADATION_1W]

    all_results = []
    ci_data = {"W": [], "X": [], "rho": [], "L": []}
    means_data = {"W": [], "X": [], "rho": [], "L": []}

    import csv
    rows = []

    for scen, name in zip(scenarios, labels):
        results = run_experiment(scen, num_replications=num_replications, simulation_time=simulation_time)
        all_results.append(results)

        W_vals = [r.get_W() for r in results]
        X_vals = [r.get_X() for r in results]
        rho_vals = [r.get_rho() for r in results]
        L_vals = [r.get_L() for r in results]

        for k, vals in zip(["W", "X", "rho", "L"], [W_vals, X_vals, rho_vals, L_vals]):
            m = float(np.mean(vals))
            lo, hi = ci_bounds(vals)
            means_data[k].append(m)
            ci_data[k].append((lo, hi))

        print(f"Summary for {name}:")
        analyze_results(results, name)

        # Persist summary rows
        import numpy as _np
        import scipy.stats as _stats
        def _ci(arr):
            a = _np.asarray(arr)
            n = len(a)
            se = _stats.sem(a)
            h = se * _stats.t.ppf(0.975, n - 1)
            return float(_np.mean(a) - h), float(_np.mean(a) + h)

        for metric, vals in [("W", W_vals), ("X", X_vals), ("rho", rho_vals), ("L", L_vals)]:
            mean = float(_np.mean(vals))
            std = float(_np.std(vals, ddof=1))
            lo, hi = _ci(vals)
            rows.append({
                "scenario": name,
                "metric": metric,
                "mean": mean,
                "std": std,
                "ci95_lo": lo,
                "ci95_hi": hi,
                "n": len(vals),
                "sim_time": simulation_time,
            })

    # Build 2x2 figure with error bars
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for (k, title, ylabel), ax in zip(
        [("W", "Avg Response Time (W)", "seconds"),
         ("X", "Throughput (X)", "jobs/s"),
         ("rho", "Utilization (ρ)", "fraction"),
         ("L", "Avg Queue Length (L)", "jobs")],
        axes.flatten()
    ):
        means = means_data[k]
        lows = [ci_data[k][i][0] for i in range(len(labels))]
        highs = [ci_data[k][i][1] for i in range(len(labels))]
        bar_with_ci(ax, labels, means, lows, highs, title, ylabel)

    fig.suptitle("Performance Metrics by Scenario (mean ± 95% CI)")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(OUT / "scenario_metrics_comparison.png", dpi=220)
    print("Saved scenario_metrics_comparison.png")

    # Queue length time-series for the 1-worker extreme scenario (single run)
    from simulation.simulator import TicketQueueSimulator
    sim = TicketQueueSimulator(EXTREME_DEGRADATION_1W)
    metrics = sim.run(simulation_time)
    times = [t for t, _ in metrics.queue_samples]
    lengths = [l for _, l in metrics.queue_samples]
    plt.figure(figsize=(10, 4))
    plt.plot(times, lengths, label="Queue length")
    plt.xlabel("Time (s)")
    plt.ylabel("Queue length")
    plt.title("Queue Buildup Over Time (Extreme 1 worker)")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "queue_buildup_extreme_1w.png", dpi=220)
    print("Saved queue_buildup_extreme_1w.png")

    # Save CSV summary
    with open(OUT / "metrics_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scenario", "metric", "mean", "std", "ci95_lo", "ci95_hi", "n", "sim_time"])
        w.writeheader()
        w.writerows(rows)
    print("Saved metrics_summary.csv")


def _collect_responses(results) -> np.ndarray:
    """Flatten response times across replications."""
    all_rt = []
    for r in results:
        all_rt.extend([j.response_time for j in r.completed_jobs if j.finish_time is not None])
    return np.array(all_rt, dtype=float)


def generate_response_time_histograms(num_replications: int = 10, simulation_time: int = 3600):
    ROOT = Path(__file__).resolve().parent.parent
    OUT = ROOT / "outputs"
    OUT.mkdir(parents=True, exist_ok=True)
    labels = ["Normal", "Extreme 200w", "Extreme 1w"]
    scenarios = [NORMAL_SCENARIO, EXTREME_DEGRADATION_200W, EXTREME_DEGRADATION_1W]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, scen, name in zip(axes, scenarios, labels):
        res = run_experiment(scen, num_replications=num_replications, simulation_time=simulation_time)
        rt = _collect_responses(res)
        if len(rt) == 0:
            continue
        # Log-scale bins for wide ranges
        rt = rt[rt > 0]
        bins = np.logspace(np.log10(max(rt.min(), 1e-3)), np.log10(rt.max()), 30)
        ax.hist(rt, bins=bins, alpha=0.8)
        ax.set_xscale('log')
        ax.set_title(f"{name}")
        ax.set_xlabel("Tempo de resposta (s)")
        ax.set_ylabel("Frequência")
        ax.grid(True, which='both', linestyle=':', alpha=0.5)
    fig.suptitle("Histogramas de tempos de resposta (escala log)")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(OUT / "response_time_histograms.png", dpi=220)
    print("Saved response_time_histograms.png")


def generate_response_time_percentiles(simulation_time: int = 3*3600):
    ROOT = Path(__file__).resolve().parent.parent
    OUT = ROOT / "outputs"
    OUT.mkdir(parents=True, exist_ok=True)
    labels = ["Normal", "Extreme 200w", "Extreme 1w"]
    scenarios = [NORMAL_SCENARIO, EXTREME_DEGRADATION_200W, EXTREME_DEGRADATION_1W]
    pcts = {"p50": [], "p95": [], "p99": []}

    from simulation.simulator import TicketQueueSimulator
    for scen, name in zip(scenarios, labels):
        sim = TicketQueueSimulator(scen)
        metrics = sim.run(simulation_time)
        rt = np.array([j.response_time for j in metrics.completed_jobs if j.finish_time is not None], dtype=float)
        if len(rt) == 0:
            pcts["p50"].append(0.0)
            pcts["p95"].append(0.0)
            pcts["p99"].append(0.0)
        else:
            pcts["p50"].append(float(np.percentile(rt, 50)))
            pcts["p95"].append(float(np.percentile(rt, 95)))
            pcts["p99"].append(float(np.percentile(rt, 99)))

    x = np.arange(len(labels))
    width = 0.25
    plt.figure(figsize=(10, 4))
    plt.bar(x - width, pcts["p50"], width, label="P50")
    plt.bar(x, pcts["p95"], width, label="P95")
    plt.bar(x + width, pcts["p99"], width, label="P99")
    plt.xticks(x, labels)
    plt.ylabel("Tempo de resposta (s)")
    plt.title("Percentis de tempos de resposta (simulação longa)")
    plt.legend()
    plt.grid(axis='y', linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUT / "response_time_percentiles.png", dpi=220)
    print("Saved response_time_percentiles.png")


def generate_capacity_sweep_extreme(num_replications: int = 10, simulation_time: int = 3600):
    ROOT = Path(__file__).resolve().parent.parent
    OUT = ROOT / "outputs"
    OUT.mkdir(parents=True, exist_ok=True)
    workers = [1, 2, 5, 10, 20, 50, 100, 200]
    W_means, X_means = [], []
    import copy, csv
    rows = []
    for n in workers:
        scen = copy.deepcopy(EXTREME_DEGRADATION_200W)
        scen['num_workers'] = n
        scen['lambda_total'] = 0.63
        res = run_experiment(scen, num_replications=num_replications, simulation_time=simulation_time)
        W_vals = [r.get_W() for r in res]
        X_vals = [r.get_X() for r in res]
        W_means.append(float(np.mean(W_vals)))
        X_means.append(float(np.mean(X_vals)))
        rows.append({"workers": n, "W": W_means[-1], "X": X_means[-1]})

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(workers, W_means, marker='o')
    axes[0].set_xlabel('Número de trabalhadores')
    axes[0].set_ylabel('W (s)')
    axes[0].set_title('Tempo de resposta vs capacidade')
    axes[0].grid(True, linestyle=':', alpha=0.5)

    axes[1].plot(workers, X_means, marker='o', color='green')
    axes[1].set_xlabel('Número de trabalhadores')
    axes[1].set_ylabel('X (jobs/s)')
    axes[1].set_title('Vazão vs capacidade')
    axes[1].grid(True, linestyle=':', alpha=0.5)

    fig.suptitle('Varredura de capacidade sob degradação extrema (λ=0,63)')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(OUT / 'capacity_sweep_extreme.png', dpi=220)
    print('Saved capacity_sweep_extreme.png')

    with open(OUT / 'capacity_sweep_extreme.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['workers', 'W', 'X'])
        w.writeheader()
        w.writerows(rows)
    print('Saved capacity_sweep_extreme.csv')


def generate_arrival_sweep_1w(num_replications: int = 8, simulation_time: int = 3600):
    ROOT = Path(__file__).resolve().parent.parent
    OUT = ROOT / "outputs"
    OUT.mkdir(parents=True, exist_ok=True)
    lambdas = [0.05, 0.1, 0.2, 0.4, 0.63, 0.8, 1.0]
    import copy, csv
    X_means, W_means, L_means = [], [], []
    rows = []
    for lam in lambdas:
        scen = copy.deepcopy(EXTREME_DEGRADATION_1W)
        scen['lambda_total'] = lam
        res = run_experiment(scen, num_replications=num_replications, simulation_time=simulation_time)
        X_vals = [r.get_X() for r in res]
        W_vals = [r.get_W() for r in res]
        L_vals = [r.get_L() for r in res]
        X_means.append(float(np.mean(X_vals)))
        W_means.append(float(np.mean(W_vals)))
        L_means.append(float(np.mean(L_vals)))
        rows.append({"lambda": lam, "X": X_means[-1], "W": W_means[-1], "L": L_means[-1]})

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(lambdas, X_means, marker='o')
    axes[0].set_xlabel('λ (jobs/s)')
    axes[0].set_ylabel('X (jobs/s)')
    axes[0].set_title('Saturação de vazão')
    axes[0].grid(True, linestyle=':', alpha=0.5)

    axes[1].plot(lambdas, W_means, marker='o', color='orange')
    axes[1].set_xlabel('λ (jobs/s)')
    axes[1].set_ylabel('W (s)')
    axes[1].set_title('Tempo de resposta vs carga')
    axes[1].grid(True, linestyle=':', alpha=0.5)

    axes[2].plot(lambdas, L_means, marker='o', color='red')
    axes[2].set_xlabel('λ (jobs/s)')
    axes[2].set_ylabel('L (jobs)')
    axes[2].set_title('Comprimento médio da fila')
    axes[2].grid(True, linestyle=':', alpha=0.5)

    fig.suptitle('Saturação (1 trabalhador): varredura de chegada')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(OUT / 'arrival_sweep_1w.png', dpi=220)
    print('Saved arrival_sweep_1w.png')

    with open(OUT / 'arrival_sweep_1w.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['lambda', 'X', 'W', 'L'])
        w.writeheader()
        w.writerows(rows)
    print('Saved arrival_sweep_1w.csv')


def generate_w_by_type_bars(num_replications: int = 12, simulation_time: int = 3600):
    ROOT = Path(__file__).resolve().parent.parent
    OUT = ROOT / "outputs"
    OUT.mkdir(parents=True, exist_ok=True)
    labels = ["Normal", "Extreme 200w", "Extreme 1w"]
    scenarios = [NORMAL_SCENARIO, EXTREME_DEGRADATION_200W, EXTREME_DEGRADATION_1W]
    Wp, We = [], []
    for scen in scenarios:
        res = run_experiment(scen, num_replications=num_replications, simulation_time=simulation_time)
        Wp_vals = [r.get_W_by_type(JobType.PURCHASE) for r in res if r.get_W_by_type(JobType.PURCHASE) > 0]
        We_vals = [r.get_W_by_type(JobType.EXPIRE) for r in res if r.get_W_by_type(JobType.EXPIRE) > 0]
        Wp.append(float(np.mean(Wp_vals)))
        We.append(float(np.mean(We_vals)))

    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(10, 4))
    plt.bar(x - width/2, Wp, width, label='purchase')
    plt.bar(x + width/2, We, width, label='expire')
    plt.xticks(x, labels)
    plt.ylabel('W por classe (s)')
    plt.title('Tempo médio de resposta por tipo de trabalho')
    plt.legend()
    plt.grid(axis='y', linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUT / 'w_by_type.png', dpi=220)
    print('Saved w_by_type.png')


if __name__ == "__main__":
    generate_comparison_figures()
    generate_response_time_histograms()
    generate_response_time_percentiles()
    generate_capacity_sweep_extreme()
    generate_arrival_sweep_1w()
    generate_w_by_type_bars()
