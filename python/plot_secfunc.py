#!/usr/bin/env python
import argparse
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_sfunc")
    parser.add_argument("--sign", action="store_true")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--disp")
    parser.add_argument("--N", action="store_true")
    args = parser.parse_args()
    file_sfunc = args.file_sfunc
    file_disp = args.disp
    show_sign = args.sign
    show_sample = args.sample
    show_N = args.N

    fh5 = h5py.File(file_sfunc, "r")
    f = fh5["f"][()]
    c = fh5["c"][()]
    sfunc = fh5["sfunc"][()]
    samples = fh5["samples"][()]
    N = fh5["N"][()]
    roots = fh5["roots"][()]
    fh5.close()

    if show_sign:
        sfunc = np.sign(sfunc)

    fig, ax = plt.subplots(layout="constrained")
    ax.axhline(0, c="k", linestyle="-", alpha=0.6)
    if show_sample:
        for i in range(samples.shape[0]):
            ax.axvline(samples[i], c="k", alpha=0.6, linewidth=0.5)
    if file_disp:
        plot_disp(ax, file_disp, f)
    for i in range(roots.shape[0]):
        ax.axvline(roots[i], c="r", alpha=0.6)
    if show_N:
        ax.plot(samples, N, "-", c="brown")
    ax.plot(c, sfunc, "k-", alpha=0.8)
    ax.set_xlim([c[0] - 0.01, c[-1] + 0.01])
    plt.show()


def plot_disp(ax, file_disp, f):
    disp = np.loadtxt(file_disp)
    modes = disp[:, 2].astype(int)
    for m in set(modes):
        d = disp[modes == m]
        intp = interp1d(d[:, 0], d[:, 1], bounds_error=False)
        c = intp(f)
        ax.axvline(c, c="b", alpha=0.6)


if __name__ == "__main__":
    main()
