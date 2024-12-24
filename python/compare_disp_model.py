#!/usr/bin/env python
import numpy as np
import argparse
import matplotlib.pyplot as plt

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
    parser.add_argument(
        "files_disp_model", nargs=2, help="file_disp file_model"
    )
    parser.add_argument("--ref", nargs=2)
    parser.add_argument("--zmax", type=float)
    parser.add_argument("--depth_m", action="store_true")
    parser.add_argument("-o", "--out")
    args = parser.parse_args()
    files_disp_model = args.files_disp_model
    files_ref = args.ref
    zmax = args.zmax
    depth_m = args.depth_m
    file_out = args.out

    file_disp, file_model = files_disp_model
    file_disp_ref, file_model_ref = files_ref

    fig, ax = plt.subplots(1, 2, width_ratios=[5, 1], layout="constrained")
    plot_disp(ax[0], file_disp_ref, "-", "k")
    plot_disp(ax[0], file_disp, "--", "r")

    plot_model(ax[1], file_model_ref, zmax, depth_m, "-", "k")
    plot_model(ax[1], file_model, zmax, depth_m, "--", "r")

    ax[0].set_xlabel("Frequency (Hz)")
    ax[0].set_ylabel("Phase velocity (km/s)")
    ax[1].set_xlabel("Vs (km/s)")
    if depth_m:
        unit = "m"
        zmax *= 1000
    else:
        unit = "km"
    ax[1].set_ylim([0, zmax])
    ax[1].set_ylabel(f"Depth ({unit})")
    ax[1].invert_yaxis()
    if file_out:
        fig.savefig(file_out, dpi=300)
    plt.show()


def plot_disp(ax, file_disp, linestyle, color):
    disp = np.loadtxt(file_disp)
    modes = disp[:, 2].astype(int)
    for m in set(modes):
        d = disp[modes == m]
        ax.plot(
            d[:, 0],
            d[:, 1],
            linestyle=linestyle,
            c=color,
            alpha=0.7,
            linewidth=1.5,
        )
    f = disp[:, 0]
    ax.set_xlim([f.min(), f.max()])


def plot_model(ax, file_model, zmax, depth_m, linestyle, color):
    model = np.loadtxt(file_model)
    z, vs = model[:, 1], model[:, 3]
    if zmax is None:
        zmax = z[-1] * 2 - z[-2]
    z = np.append(z, zmax)
    vs = np.append(vs, vs[-1])
    if depth_m:
        z *= 1000
    ax.step(vs, z, linestyle=linestyle, c=color, alpha=0.7, linewidth=1.5)


if __name__ == "__main__":
    main()
