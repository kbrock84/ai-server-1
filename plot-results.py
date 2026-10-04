#!/usr/bin/env python3
"""Plot power draw and token output over time from a gpu-stress.py results file.

    ./plot-results.py results/full-power_10min_20261004.json    # writes the .png next to it
    ./plot-results.py stress_*.json

Requires matplotlib (pip install matplotlib).
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def gpu_label(gpu, summary):
    """e.g. 'GPU 0 (MSI, PCIE7)'; older results files have no slot info"""
    parts = [p for p in (summary.get("card_vendor"), summary.get("slot_name")) if p]
    return f"GPU {gpu}" + (f" ({', '.join(parts)})" if parts else "")


def style_axis(ax, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=10)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.set_ylim(bottom=0)
    ax.margins(x=0)


def plot(path):
    with open(path) as f:
        data = json.load(f)
    samples, summary, config = data["samples"], data["summary"], data["configuration"]
    if not samples:
        print(f"{path}: no samples, skipping")
        return
    gpus = sorted(summary, key=int)
    minutes = [s["elapsed_s"] / 60 for s in samples]

    fig, (ax_power, ax_tokens) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, facecolor=SURFACE)

    total_power = total_rate = 0.0
    for i, gpu in enumerate(gpus):
        color = SERIES_COLORS[i]  # fixed order: the color follows the GPU index
        power = [s["gpus"][gpu]["power_draw_w"] for s in samples]
        rate = [s["gpus"][gpu]["output_tokens_per_second"] for s in samples]
        avg_power, avg_rate = sum(power) / len(power), sum(rate) / len(rate)
        total_power += avg_power
        total_rate += avg_rate
        label = gpu_label(gpu, summary[gpu])
        ax_power.plot(minutes, power, color=color, linewidth=2, solid_capstyle="round",
                      label=f"{label}: avg {avg_power:.0f} W")
        ax_tokens.plot(minutes, rate, color=color, linewidth=2, solid_capstyle="round",
                       label=f"{label}: avg {avg_rate:,.0f} tok/s")

    # Power panel, with the configured limit as a reference line
    style_axis(ax_power, "Power draw (W)")
    limits = {summary[g]["idle"].get("power_limit_w") for g in gpus} - {None}
    if len(limits) == 1:
        limit = limits.pop()
        ax_power.axhline(limit, color=MUTED, linewidth=1)
        ax_power.set_ylim(top=limit * 1.15)
        ax_power.text(minutes[-1], limit * 1.02, f"{limit:.0f} W power limit", color=TEXT_SECONDARY,
                      fontsize=9, ha="right", va="bottom")
    ax_power.set_title(f"Power draw per GPU (all cards: avg {total_power:.0f} W)", color=TEXT,
                       fontsize=11, loc="left")

    style_axis(ax_tokens, "Output tokens / second")
    ax_tokens.set_ylim(top=ax_tokens.get_ylim()[1] * 1.15)
    ax_tokens.set_xlabel("Minutes under load", color=TEXT_SECONDARY, fontsize=10)
    ax_tokens.set_title(f"Token output per GPU (all cards: avg {total_rate:,.0f} tok/s)", color=TEXT,
                        fontsize=11, loc="left")

    for ax in (ax_power, ax_tokens):
        legend = ax.legend(loc="lower right", frameon=True, fontsize=9, ncols=len(gpus),
                           facecolor=SURFACE, edgecolor=GRID)
        for text in legend.get_texts():
            text.set_color(TEXT)

    prompt = config.get("prompt_tokens_per_request")
    prompt = f"{prompt:,}-token prompts" if prompt else "short prompts"
    fig.suptitle(
        f"{data['model']}\n{config['concurrency_per_gpu']} concurrent requests per GPU, {prompt}, "
        f"{config['max_tokens_per_request']} output tokens each",
        color=TEXT, fontsize=12, x=0.065, ha="left",
    )
    fig.tight_layout()

    output = os.path.splitext(path)[0] + ".png"
    fig.savefig(output, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {output}")


def main():
    parser = argparse.ArgumentParser(description="Plot power and token output from gpu-stress.py results")
    parser.add_argument("results", nargs="+", help="results JSON file(s)")
    for path in parser.parse_args().results:
        plot(path)


if __name__ == "__main__":
    main()
