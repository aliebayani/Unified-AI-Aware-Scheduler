import random, csv
import numpy as np
from sim import gen_throughput_matrix, jps_sampling
import pandas as pd
import matplotlib.pyplot as plt

def auto_tune_search(S, K, trials=60, N=40, seed=2):
    random.seed(seed); np.random.seed(seed)
    rho = gen_throughput_matrix(S, K, seed=seed)
    Pis = [100000] * S; Tis = [10, 8, 12, 6]; Dis = [50e6, 20e6, 40e6, 10e6]
    history = []
    best_obj = 1e18; best_params=None; best_res=None
    for t in range(trials):
        a = random.random() * 0.9
        b = random.random()
        res, _ = jps_sampling(S, K, rho, Pis, Tis, Dis, N, a, b)
        obj = res['score']
        history.append({'alpha': a, 'beta': b, 'obj': obj, 'avgJ': res['avgJ'], 'fairness': res['fairness']})
        if obj < best_obj:
            best_obj = obj; best_params=(a,b); best_res=res
    df = pd.DataFrame(history)
    df.to_csv("auto_tune_history.csv", index=False)
    print("Best params:", best_params, "best result:", best_res)
    # quick plots
    plt.scatter(df['alpha'], df['obj'], c=df['avgJ'], cmap='viridis')
    plt.colorbar(label='avgJ'); plt.xlabel('alpha'); plt.ylabel('obj')
    plt.title('Auto-tune Trials'); plt.savefig("autotune_trials.png"); plt.clf()
    return best_params, best_res

if __name__ == "__main__":
    best_params, best_res = auto_tune_search(S=4, K=15, trials=60, N=40)
