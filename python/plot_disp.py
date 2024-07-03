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


if __name__ == "__main__":
    msg = "plot dispersion curves"
    parser = argparse.ArgumentParser(description=msg)
    parser.add_argument(
        "file_disp", default=None, help="file of dispersion curves"
    )
    parser.add_argument("--file_ref")
    parser.add_argument(
        "--color", action="store_true", help="using different color for modes"
    )
    parser.add_argument("--unit_m", action="store_true", help="yaxis in m")
    parser.add_argument("-o", "--out", default=None, help=" output figure name")
    args = parser.parse_args()
    file_disp = args.file_disp
    file_ref = args.file_ref
    use_color = args.color
    unit_m = args.unit_m
    file_out = args.out

    disp = np.loadtxt(file_disp)
    modes = set(disp[:, 2].astype(int))

    if unit_m:
        km2m = 1.0e3
        unit = "m"
    else:
        km2m = 1.0
        unit = "km"

    if use_color:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    else:
        colors = ["k"] * 10000

    fig, ax = plt.subplots(layout="constrained")
    for i, m in enumerate(modes):
        d = disp[disp[:, 2] == m]
        ax.plot(
            d[:, 0],
            d[:, 1] * km2m,
            "-",
            c=colors[i],
            label=str(m),
            linewidth=1,
            alpha=0.8,
        )

    if use_color:
        if len(modes) < 5:
            ax.legend()

    if file_ref:
        disp_ref = np.loadtxt(file_ref)
        modes = set(disp_ref[:, 2].astype(int))
        for i, m in enumerate(modes):
            d = disp_ref[disp_ref[:, 2] == m]
            ax.plot(
                d[:, 0],
                d[:, 1] * km2m,
                linestyle="--",
                c="r",
                linewidth=1,
                alpha=0.8,
            )

    ax.set_xlim([np.min(disp[:, 0]), np.max(disp[:, 0])])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Phase velocity ({:s}/s)".format(unit))

    if file_out:
        plt.savefig(file_out, dpi=300)
    plt.show()
