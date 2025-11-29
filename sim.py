import random
import math
import numpy as np

def gen_throughput_matrix(S, K, seed=1):
    """Generate synthetic throughput matrix ρ_{i,k} (samples/sec)."""
    random.seed(seed); np.random.seed(seed)
    base = {'T4':300, 'P100':700, 'V100':1500, 'H100':2500}
    types = ['T4', 'P100', 'V100']
    mat = np.zeros((S, K))
    for i in range(S):
        job_factor = 0.8 + 0.8 * random.random()
        for k in range(K):
            t = types[k % len(types)]
            noise = 0.9 + 0.2 * random.random()
            mat[i, k] = int(base.get(t, 500) * job_factor * noise)
    return mat

def enumerate_categories(K, S):
    """Generate all positive compositions of K into S parts (feasible solution categories)."""
    def compositions(n, k):
        if k == 1:
            yield (n,)
        else:
            for i in range(1, n - (k - 1) + 1):
                for tail in compositions(n - i, k - 1):
                    yield (i,) + tail
    return list(compositions(K, S))

def proportional_assignment(selected_workers, rho_row, Pi):
    """Return l_p and per-worker assigned samples for a job with proportional workload assignment."""
    if len(selected_workers) == 0:
        return float('inf'), {}
    rhos = rho_row[selected_workers]
    denom = rhos.sum()
    if denom == 0:
        return float('inf'), {k: 0 for k in selected_workers}
    l_p = Pi / denom
    assigned = {k: int(round(Pi * rho / denom)) for k, rho in zip(selected_workers, rhos)}
    # fix rounding
    total = sum(assigned.values())
    diff = Pi - total
    idx = 0
    while diff != 0 and len(selected_workers) > 0:
        assigned[selected_workers[idx % len(selected_workers)]] += 1 if diff > 0 else -1
        diff = Pi - sum(assigned.values())
        idx += 1
    return l_p, assigned

def evaluate_assignment(S, K, Tis, Dis, Pis, selected_map, rho):
    """Compute per-job JCTs and average JCT given a selected_map: job->list(workers)."""
    Js = []
    for i in range(S):
        sel = selected_map.get(i, [])
        if len(sel) == 0:
            Js.append(float('inf')); continue
        l_p, assigned = proportional_assignment(sel, rho[i], Pis[i])
        # Simplified communication latency model using eq (6) but with large rmin to keep focus on computation
        rmin = 1e9
        l_c = 2 * (len(sel) - 1) * Dis[i] / (rmin * len(sel))
        Ji = Tis[i] * (l_p + l_c)
        Js.append(Ji)
    return Js, sum(Js) / len(Js)

def matching_markets(S, K, rho, category, max_iter=500):
    """
    Simplified implementation of Algorithm 2 (matching markets).
    Returns selected_map: job -> list of workers indices
    """
    prices = np.zeros(K)
    Gamma = rho.copy()
    for _ in range(max_iter):
        prefs = []
        for i in range(S):
            payoffs = Gamma[i, :] - prices
            topk = np.argsort(-payoffs)[:category[i]]
            prefs.append(set(topk))
        worker_to_jobs = {}
        for i, topk in enumerate(prefs):
            for w in topk:
                worker_to_jobs.setdefault(w, []).append(i)
        constricted = {w: js for w, js in worker_to_jobs.items() if len(js) > 1}
        if not constricted:
            return {i: sorted(list(prefs[i])) for i in range(S)}
        w = next(iter(constricted.keys()))
        comps = constricted[w]
        best_inc = 1.0
        for i in comps:
            vals = np.sort(-Gamma[i, :])
            Vi = -vals[:category[i]].sum()
            arr = Gamma[i, :].copy(); arr[w] = -1e9
            Vki = -np.sort(-arr)[:category[i]].sum()
            inc = max(1.0, Vi - Vki)
            if inc > best_inc:
                best_inc = inc
        prices[w] += best_inc
    # fallback greedy if not converged
    sel = {}
    for i in range(S):
        sel[i] = list(np.argsort(-rho[i, :])[:category[i]])
    return sel

def jps_sampling(S, K, rho, Pis, Tis, Dis, N, alpha, beta):
    """
    Job Prioritizing for Sampling (Algorithm 3 variant).
    - alpha: fraction to skip (start sampling in tail of categories)
    - beta: tradeoff weight between JCT (beta=1 focuses on JCT) and fairness (beta=0 focuses on fairness)
    """
    all_cat = enumerate_categories(K, S)
    L = len(all_cat)
    start = int(math.floor(alpha * L))
    candidates = all_cat[start:] if len(all_cat[start:]) > 0 else all_cat
    samples = random.sample(candidates, min(N, len(candidates)))
    best = None; best_score = float('inf'); results = []
    for cat in samples:
        selected_map = matching_markets(S, K, rho, cat)
        Js, avgJ = evaluate_assignment(S, K, Tis, Dis, Pis, selected_map, rho)
        # fairness measure: compare to equal-allocation baseline
        Ki_equal = max(1, K // S)
        equal_cat = tuple([Ki_equal] * S)
        sel_equal = matching_markets(S, K, rho, equal_cat)
        Je, _ = evaluate_assignment(S, K, Tis, Dis, Pis, sel_equal, rho)
        ratios = [Js[i] / Je[i] if Je[i] > 0 else 1.0 for i in range(S)]
        numer = (sum(ratios))**2; denom = S * sum([r*r for r in ratios])
        fairness = numer / denom if denom > 0 else 1.0
        score = beta * avgJ + (1 - beta) * (1 - fairness) * avgJ
        results.append({'cat': cat, 'avgJ': avgJ, 'fairness': fairness, 'score': score})
        if score < best_score:
            best_score = score; best = results[-1]
    return best, results

# Example main-run snippet you can invoke externally
def example_run():
    S = 4; K = 15
    rho = gen_throughput_matrix(S, K, seed=2)
    Pis = [100000] * S; Tis = [10, 8, 12, 6]; Dis = [50e6, 20e6, 40e6, 10e6]
    best, results = jps_sampling(S, K, rho, Pis, Tis, Dis, N=40, alpha=0.7, beta=1.0)
    return best, results

if __name__ == "__main__":
    print("This file contains simulation functions. Import them or call example_run().")
