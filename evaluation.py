import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sim import matching_markets, evaluate_assignment

# --------------------------------------------------------
# 1. JCT vs Worker Count (max category size or median)
# --------------------------------------------------------

def analyze_jct_vs_workers(S, K, rho, Pis, Tis, Dis, categories, save="jct_vs_workers.png"):
    data = []
    for cat in categories:
        selected = matching_markets(S, K, rho, cat)
        Js, avgJ = evaluate_assignment(S, K, Tis, Dis, Pis, selected, rho)

        max_w = max(cat)              # most demanding job
        med_w = int(np.median(cat))   # typical load

        data.append([max_w, med_w, avgJ])

    data = np.array(data)
    plt.figure(figsize=(8,5))
    plt.scatter(data[:,0], data[:,2], label="Max Workers")
    plt.scatter(data[:,1], data[:,2], label="Median Workers")
    plt.xlabel("Workers Allocated")
    plt.ylabel("Avg JCT")
    plt.title("JCT vs Workers per Job")
    plt.legend()
    plt.grid()
    plt.savefig(save)
    plt.close()


# --------------------------------------------------------
# 2. More Sensitive Fairness Metrics
# --------------------------------------------------------

def jain_index(values):
    v = np.array(values)
    return (v.sum()**2) / (len(v) * (v**2).sum() + 1e-9)

def fairness_spread(values):
    """Max/min ratio (more sensitive)."""
    v = np.array(values)
    return v.max() / (v.min() + 1e-9)

def analyze_fairness(Js, Je, save="fairness_metrics.txt"):
    ratios = [Js[i] / Je[i] if Je[i] > 0 else 1 for i in range(len(Js))]

    jain = jain_index(ratios)
    spread = fairness_spread(ratios)

    with open(save, "w") as f:
        f.write("Fairness Analysis\n")
        f.write("=================\n")
        f.write(f"Jain Index: {jain:.4f}\n")
        f.write(f"Spread (max/min): {spread:.4f}\n")

    return jain, spread


# --------------------------------------------------------
# 3. Worker Utilization Heatmap
# --------------------------------------------------------

def worker_heatmap(S, K, selected_map, save="worker_heatmap.png"):
    # Matrix: jobs × workers → 1 if used, 0 otherwise
    mat = np.zeros((S, K))

    for i in range(S):
        for w in selected_map.get(i, []):
            mat[i][w] = 1

    plt.figure(figsize=(10,5))
    sns.heatmap(mat, cmap="YlGnBu", annot=False)
    plt.xlabel("Worker ID")
    plt.ylabel("Job ID")
    plt.title("Worker Utilization Heatmap")
    plt.savefig(save)
    plt.close()


# --------------------------------------------------------
# 4. Allocation Pattern Entropy
# --------------------------------------------------------

def allocation_entropy(cat):
    """Entropy of category distribution."""
    p = np.array(cat) / sum(cat)
    return -np.sum(p * np.log2(p + 1e-12))

def analyze_entropy(categories, save="entropy_plot.png"):
    ent = [allocation_entropy(c) for c in categories]

    plt.figure(figsize=(8,5))
    plt.plot(ent)
    plt.title("Allocation Entropy Across Categories")
    plt.xlabel("Category Index")
    plt.ylabel("Entropy (bits)")
    plt.grid()
    plt.savefig(save)
    plt.close()

    return ent
