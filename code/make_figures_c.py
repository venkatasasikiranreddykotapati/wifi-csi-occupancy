"""Thesis figure: parameter sensitivity (final 3-room dataset)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.titlesize": 11,
    "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 8, "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.3,
})
os.makedirs("figures", exist_ok=True)

wins = [1, 2, 3, 5, 8, 10]
w_pres = [88.6, 87.2, 87.1, 86.2, 89.3, 87.3]
w_std  = [3.4, 7.0, 5.4, 4.7, 1.9, 5.4]
w_lvl  = [61.4, 62.9, 60.0, 61.3, 68.4, 67.0]

rates = [5, 10, 15, 30]
r_pres = [85.2, 86.2, 88.8, 86.2]
r_std  = [3.4, 8.2, 4.0, 4.7]
r_lvl  = [63.3, 61.9, 66.2, 61.3]

subs = [8, 16, 32, 64, 166]
s_pres = [85.8, 87.0, 86.9, 86.2, 86.2]
s_std  = [4.7, 5.8, 5.3, 4.2, 4.7]
s_lvl  = [61.3, 63.6, 60.2, 59.5, 61.3]

fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
panels = [
    (ax[0], wins,  w_pres, w_std, w_lvl, "Window length [s]",   "(a) Window length"),
    (ax[1], rates, r_pres, r_std, r_lvl, "Packet rate [Hz]",    "(b) Packet rate"),
    (ax[2], subs,  s_pres, s_std, s_lvl, "Subcarriers used",    "(c) Subcarrier count"),
]
for a, xs, pres, sd, lvl, xlabel, title in panels:
    a.errorbar(xs, pres, yerr=sd, marker="o", capsize=3, color="#1f77b4",
               label="Presence (0 vs 1+)")
    a.plot(xs, lvl, marker="s", color="#ff7f0e", label="Level (0/1/2+)")
    a.set_xlabel(xlabel); a.set_ylabel("Balanced accuracy [%]")
    a.set_title(title); a.set_ylim(50, 100); a.legend(loc="lower right")
ax[2].set_xscale("log", base=2)
fig.tight_layout()
fig.savefig("figures/fig_parameter_sensitivity.png")
fig.savefig("figures/fig_parameter_sensitivity.pdf")
print("saved figures/fig_parameter_sensitivity")