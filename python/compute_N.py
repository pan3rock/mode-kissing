#!/usr/bin/env python
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from functools import partial

params = {
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 14,
    "font.family": "serif",
}
plt.rcParams.update(params)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("freqs", type=float, nargs="*")
    parser.add_argument("--model", default="model.txt")
    parser.add_argument("-c", "--config")
    parser.add_argument("--disp", default="disp.txt")
    parser.add_argument("--unit_m", action="store_true", help="yaxis in m")
    args = parser.parse_args()
    freqs = args.freqs
    file_model = args.model
    file_disp = args.disp
    unit_m = args.unit_m

    if unit_m:
        km2m = 1.0e3
        unit = "m"
    else:
        km2m = 1.0
        unit = "km"

    model = np.loadtxt(file_model)
    h = np.diff(model[:, 1])
    vs = model[:, 3]
    vp = model[:, 4]

    cmin = np.amin(vs) * 0.8
    cmax = vs[-1]
    nc = 10000
    c = np.linspace(cmin, cmax, nc)

    colors = ["tab:brown", "tab:blue", "tab:orange", "tab:red"]

    disp = np.loadtxt(file_disp)

    fig, ax = plt.subplots(layout="constrained")
    for i, freq in enumerate(freqs):
        func = partial(evaluate, freq, h, vs, vp)
        vfunc = np.vectorize(func)
        N = vfunc(c)
        ax.plot(c * km2m, N, "k-", alpha=0.8)

        c_find = []
        modes = disp[:, 2].astype(int)
        for m in sorted(list(set(modes))):
            d = disp[modes == m]
            intp = interp1d(d[:, 0], d[:, 1], bounds_error=False)
            c1 = intp(freq)
            if c1:
                c_find.append(c1)
        c_find = np.asarray(c_find)
        ax.plot(
            c_find * km2m,
            np.arange(len(c_find)) + 1,
            "o",
            color=colors[i],
            label="{} Hz".format(freq),
        )
    ax.legend()
    ax.set_xlabel("Phase velocity ({:s}/s)".format(unit))
    ax.set_ylabel("Number of modes")

    plt.savefig("N.pdf", dpi=300)
    plt.show()


def evaluate(freq, h, vs, vp, c):
    val = 0.0
    nl = vs.shape[0]
    c_2 = c ** (-2)
    for i in range(nl - 1):
        if c > vs[i]:
            val += np.sqrt(vs[i] ** (-2) - c_2) * h[i]
        if c > vp[i]:
            val += np.sqrt(vp[i] ** (-2) - c_2) * h[i]
    val *= 2.0 * freq
    return val


if __name__ == "__main__":
    main()
