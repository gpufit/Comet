import matplotlib.pyplot as plt
import numpy as np


def plot_q_with_baseline(q_obs, q_null_mean, pairs=None, window=None, ax=None, title=None):
    T = len(q_obs); t = np.arange(T)
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))
    else:
        fig = ax.figure

    ax.plot(t, q_obs, lw=1.2, label="q (observed)",)
    ax.plot(t, q_null_mean, lw=1.0, ls="--", label="q null mean")
    q_null_std = np.std(q_null_mean)
    ax.fill_between(t, q_null_mean - q_null_std, q_null_mean + q_null_std,
                    alpha=0.2, step=None, label="null ±1σ")
    ax.set_xlabel("time index t")
    ax.set_ylabel("normalized overlap")
    ax.set_title(title or f"Windowed normalized overlap (window={window})")
    ax.grid(True, alpha=0.3)

    idx_flaw = np.where(q_obs < q_null_mean + q_null_std)[0]
    ax.scatter(t[idx_flaw], q_obs[idx_flaw], color="r", s=10, label="pot. flawed est.")

    if pairs is not None:
        ax2 = ax.twinx()
        ax2.plot(t, pairs, lw=0.8, alpha=0.6, color="k")
        ax2.set_ylim(0, np.max(pairs)*1.1)
        ax2.set_ylabel("#pairs (windowed)")
    ax.legend(loc="best")

    return idx_flaw
