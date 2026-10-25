from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter

ROOT = Path(__file__).resolve().parent
source = ROOT / "bwf_4n.csv"

lines = source.read_text(encoding="utf-8-sig").splitlines()
header = next(i for i, line in enumerate(lines) if line.startswith("Frequency"))
data = np.loadtxt(source, delimiter=",", skiprows=header + 1)
f, gain = data[:, 0], data[:, 1]


def crossing(level):
    i = np.flatnonzero((gain[:-1] >= level) & (gain[1:] <= level))[0]
    return 10 ** np.interp(
        level, gain[i:i + 2][::-1], np.log10(f[i:i + 2])[::-1]
    )


f3, f10 = crossing(-3), crossing(-10)
given = 2662.5
g_given = np.interp(np.log10(given), np.log10(f), gain)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "text.usetex": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "svg.fonttype": "none"
})

fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")

ax.semilogx(
    f, gain, color="#185b86", lw=2, marker="o", ms=3,
    label="Målt amplituderespons"
)

ax.set(
    xlim=(1000, 10000),
    ylim=(-67, 5),
    xlabel="Frekvens (Hz)",
    ylabel="Forsterkning (dB)"
)
ax.xaxis.label.set_fontsize(11)

ax.xaxis.set_major_locator(
    FixedLocator([1000, 2000, 3000, 4000, 5000, 7000, 10000])
)
ax.xaxis.set_major_formatter(
    FuncFormatter(lambda x, p: f"{x:,.0f}".replace(",", " "))
)
ax.minorticks_off()
ax.set_yticks(np.arange(-60, 1, 10))
ax.grid(True, color="#dde2e6", linewidth=0.65)
ax.set_axisbelow(True)

for freq, level, color, offset in [
    (f3, -3, "#b65220", (42, 14)),
    (f10, -10, "#7160a1", (50, 10))
]:
    ax.hlines(level, 1000, freq, color=color, ls="--", lw=1)
    ax.vlines(freq, -67, level, color=color, ls="--", lw=0.9, alpha=0.65)
    ax.scatter(freq, level, s=55, color=color, edgecolor="white", zorder=5)

    label = f"−{abs(level)} dB\n{freq:.1f} Hz"
    
    ax.annotate(
        label, (freq, level),
        xytext=offset,
        textcoords="offset points",
        color=color,
        fontsize=10,
        va="center",
        bbox=dict(boxstyle="round,pad=.35", fc="white", ec=color, lw=0.7),
        arrowprops=dict(arrowstyle="-", color=color, lw=0.9)
    )

ax.scatter(
    given, g_given, marker="D", s=38,
    facecolor="white", edgecolor="#525b63", zorder=6
)
ax.annotate(
    r"$f_c = 2662.5\,\mathrm{Hz}$" + f"\nMålt respons ≈ {g_given:.2f} dB",
    (given, g_given),
    xytext=(-30, -85),
    textcoords="offset points",
    ha="left",
    fontsize=9,
    color="#414b54",
    bbox=dict(boxstyle="round,pad=.4", fc="white", ec="#aeb6bc", lw=0.6),
    arrowprops=dict(arrowstyle="-", color="#68737c", lw=0.8)
)

ax.legend(
    loc="lower left", frameon=True,
    facecolor="white", edgecolor="#dde2e6"
)

for ext in ["pdf", "svg", "png"]:
    fig.savefig(
        ROOT / f"filter_amplituderespons.{ext}",
        dpi=300, bbox_inches="tight"
    )

plt.show()