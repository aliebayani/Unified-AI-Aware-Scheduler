import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

from sim import (
    gen_throughput_matrix, jps_sampling, matching_markets,
    evaluate_assignment, enumerate_categories
)
from auto_tune import auto_tune_search

# NEW evaluation tools
from evaluation import (
    analyze_jct_vs_workers, analyze_fairness,
    worker_heatmap, analyze_entropy
)


# -------------------------------------------------------------------------
# Utility: ensure clean folder structure
# -------------------------------------------------------------------------
def ensure_dirs():
    os.makedirs("plots", exist_ok=True)
    os.makedirs("results", exist_ok=True)

ensure_dirs()


# -------------------------------------------------------------------------
# Global simulation parameters
# -------------------------------------------------------------------------
S = 4
K = 15
Pis = [100000] * S
Tis = [10, 8, 12, 6]
Dis = [50e6, 20e6, 40e6, 10e6]


# -------------------------------------------------------------------------
# FIGURE 1 — Grid Search Heatmap
# -------------------------------------------------------------------------
def generate_grid_heatmap():
    print("[1/10] Generating grid heatmap...")
    rho = gen_throughput_matrix(S, K, seed=3)

    grid_size = 10
    alpha_vals = np.linspace(0, 0.9, grid_size)
    beta_vals = np.linspace(0, 1.0, grid_size)
    heat = np.zeros((grid_size, grid_size))

    for i, a in enumerate(alpha_vals):
        for j, b in enumerate(beta_vals):
            best, _ = jps_sampling(S, K, rho, Pis, Tis, Dis, N=30, alpha=a, beta=b)
            heat[j, i] = best['avgJ']

    plt.figure(figsize=(8, 6))
    plt.imshow(
        heat, origin="lower", cmap="viridis",
        extent=[alpha_vals.min(), alpha_vals.max(), beta_vals.min(), beta_vals.max()],
        aspect="auto"
    )
    plt.colorbar(label="avgJ")
    plt.xlabel("alpha")
    plt.ylabel("beta")
    plt.title("Grid Search Heatmap (avgJ)")
    plt.savefig("plots/grid_heatmap.png", dpi=300)
    plt.close()


# -------------------------------------------------------------------------
# FIGURE 2 — Fairness/JCT Tradeoff
# -------------------------------------------------------------------------
def generate_fairness_tradeoff():
    print("[2/10] Generating fairness tradeoff curve...")
    rho = gen_throughput_matrix(S, K, seed=4)
    alpha_vals = np.linspace(0, 0.9, 20)

    avgJs = []
    fairnesses = []

    for a in alpha_vals:
        res, _ = jps_sampling(S, K, rho, Pis, Tis, Dis, N=30, alpha=a, beta=1.0)
        avgJs.append(res['avgJ'])
        fairnesses.append(res['fairness'])

    plt.figure(figsize=(8, 5))
    plt.plot(alpha_vals, avgJs, label="AvgJ")
    plt.plot(alpha_vals, fairnesses, label="Fairness")
    plt.xlabel("alpha")
    plt.grid(True)
    plt.legend()
    plt.title("Fairness–JCT Tradeoff")
    plt.savefig("plots/fairness_tradeoff.png", dpi=300)
    plt.close()


# -------------------------------------------------------------------------
# FIGURE 3 — JCT CDF
# -------------------------------------------------------------------------
def generate_jct_cdf(alpha=0.2578, beta=0.0272):
    print("[3/10] Generating JCT CDF...")
    rho = gen_throughput_matrix(S, K, seed=5)
    best, _ = jps_sampling(S, K, rho, Pis, Tis, Dis, N=50, alpha=alpha, beta=beta)

    sel = matching_markets(S, K, rho, best['cat'])
    Js, _ = evaluate_assignment(S, K, Tis, Dis, Pis, sel, rho)
    Js = sorted(Js)

    N = len(Js)
    y = np.arange(1, N + 1) / N

    plt.figure(figsize=(8, 5))
    plt.plot(Js, y)
    plt.xlabel("JCT")
    plt.ylabel("CDF")
    plt.grid(True)
    plt.title("CDF of Job Completion Times")
    plt.savefig("plots/jct_cdf.png", dpi=300)
    plt.close()


# -------------------------------------------------------------------------
# FIGURE 4 — Category Size Variance vs JCT
# -------------------------------------------------------------------------
def generate_category_variance_plot():
    print("[4/10] Generating category size variance plot...")
    rho = gen_throughput_matrix(S, K, seed=7)
    cats = enumerate_categories(K, S)

    avgJs = []
    variances = []
    sizes = []

    for cat in cats:
        sel = matching_markets(S, K, rho, cat)
        Js, avgJ = evaluate_assignment(S, K, Tis, Dis, Pis, sel, rho)
        avgJs.append(avgJ)
        variances.append(np.var(Js))
        sizes.append(sum(cat))

    plt.figure(figsize=(8, 5))
    plt.scatter(sizes, variances, s=20)
    plt.xlabel("Total Category Size")
    plt.ylabel("JCT Variance")
    plt.title("JCT Variance vs Category Size")
    plt.grid(True)
    plt.savefig("plots/category_variance.png", dpi=300)
    plt.close()


# -------------------------------------------------------------------------
# FIGURE 5 — Sampler Comparison
# -------------------------------------------------------------------------
def generate_sampler_comparison():
    print("[5/10] Generating sampler comparison...")
    rho = gen_throughput_matrix(S, K, seed=8)

    # Random
    cat_random = tuple(random.randint(1, 5) for _ in range(S))
    sel_r = matching_markets(S, K, rho, cat_random)
    Js_r, avgJ_r = evaluate_assignment(S, K, Tis, Dis, Pis, sel_r, rho)

    # Uniform
    Ki = K // S
    cat_uniform = tuple([Ki] * S)
    sel_u = matching_markets(S, K, rho, cat_uniform)
    Js_u, avgJ_u = evaluate_assignment(S, K, Tis, Dis, Pis, sel_u, rho)

    # JPS
    best, _ = jps_sampling(S, K, rho, Pis, Tis, Dis, N=40, alpha=0.2578, beta=0.0272)
    avgJ_j = best['avgJ']

    methods = ["Random", "Uniform", "JPS Optimal"]
    values = [avgJ_r, avgJ_u, avgJ_j]

    plt.figure(figsize=(7, 5))
    plt.bar(methods, values)
    plt.ylabel("Average JCT")
    plt.title("Sampler Comparison")
    plt.savefig("plots/sampler_comparison.png", dpi=300)
    plt.close()


# -------------------------------------------------------------------------
#  NEW FIGURE 6 — JCT vs Workers
# -------------------------------------------------------------------------
def generate_jct_vs_workers():
    print("[6/10] Generating JCT vs Worker Count plot...")
    rho = gen_throughput_matrix(S, K, seed=10)
    cats = enumerate_categories(K, S)
    analyze_jct_vs_workers(S, K, rho, Pis, Tis, Dis, cats,
                           save="plots/jct_vs_workers.png")


# -------------------------------------------------------------------------
# NEW FIGURE 7 — Worker Utilization Heatmap
# -------------------------------------------------------------------------
def generate_worker_heatmap():
    print("[7/10] Generating worker utilization heatmap...")
    rho = gen_throughput_matrix(S, K, seed=11)
    best, _ = jps_sampling(S, K, rho, Pis, Tis, Dis, N=40,
                           alpha=0.2578, beta=0.0272)
    sel = matching_markets(S, K, rho, best['cat'])

    worker_heatmap(S, K, sel, save="plots/worker_heatmap.png")


# -------------------------------------------------------------------------
# NEW FIGURE 8 — Fairness Metrics (Jain + Spread)
# -------------------------------------------------------------------------
def generate_fairness_metrics():
    print("[8/10] Computing fairness metrics...")
    rho = gen_throughput_matrix(S, K, seed=12)
    best, _ = jps_sampling(S, K, rho, Pis, Tis, Dis,
                           N=40, alpha=0.2578, beta=0.0272)
    sel = matching_markets(S, K, rho, best['cat'])
    Js, _ = evaluate_assignment(S, K, Tis, Dis, Pis, sel, rho)

    # Equal baseline
    Ki_equal = K // S
    equal_cat = tuple([Ki_equal] * S)
    sel_eq = matching_markets(S, K, rho, equal_cat)
    Je, _ = evaluate_assignment(S, K, Tis, Dis, Pis, sel_eq, rho)

    analyze_fairness(Js, Je, save="results/fairness_metrics.txt")


# -------------------------------------------------------------------------
# NEW FIGURE 9 — Category Entropy Analysis
# -------------------------------------------------------------------------
def generate_entropy_plot():
    print("[9/10] Generating entropy plot...")
    cats = enumerate_categories(K, S)
    analyze_entropy(cats, save="plots/category_entropy.png")


# -------------------------------------------------------------------------
# Auto Tune
# -------------------------------------------------------------------------
def run_autotune():
    print("[10/10] Running autotune (alpha, beta optimization)...")
    best_params, best_res = auto_tune_search(S, K, trials=60, N=40)
    return best_params, best_res


# -------------------------------------------------------------------------
# MASTER EXECUTION
# -------------------------------------------------------------------------
if __name__ == "__main__":
    print("===============================================")
    print("       Running FULL Experiment Suite           ")
    print("===============================================")

    generate_grid_heatmap()
    generate_fairness_tradeoff()
    generate_jct_cdf()
    generate_category_variance_plot()
    generate_sampler_comparison()

    generate_jct_vs_workers()
    generate_worker_heatmap()
    generate_fairness_metrics()
    generate_entropy_plot()

    best_params, best_res = run_autotune()

    print("\nAll experiments completed successfully!")
    print("Plots saved in ./plots/")
    print("Tables saved in ./results/")
