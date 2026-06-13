#!/usr/bin/env python3
"""
plot_results.py
Reads results/processing_log.csv and generates a performance bar chart.
"""
import csv
import os

LOG_FILE = "results/processing_log.csv"
OUT_FILE = "results/performance_plot.txt"

if not os.path.exists(LOG_FILE):
    print(f"Log file {LOG_FILE} not found. Run the processor first.")
    exit(1)

rows = []
with open(LOG_FILE) as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

if not rows:
    print("No data in log file.")
    exit(1)

# Group by kernel type
from collections import defaultdict
kernel_times = defaultdict(list)
for row in rows:
    kernel_times[row['kernel']].append(float(row['elapsed_ms']))

print("\n=== Performance Summary ===")
print(f"{'Kernel':<15} {'Count':>6} {'Avg (ms)':>10} {'Min (ms)':>10} {'Max (ms)':>10}")
print("-" * 55)

with open(OUT_FILE, 'w') as out:
    out.write("=== CUDA Image Processor Performance Summary ===\n\n")
    out.write(f"{'Kernel':<15} {'Count':>6} {'Avg (ms)':>10} {'Min (ms)':>10} {'Max (ms)':>10}\n")
    out.write("-" * 55 + "\n")
    for kernel, times in sorted(kernel_times.items()):
        avg = sum(times) / len(times)
        mn  = min(times)
        mx  = max(times)
        line = f"{kernel:<15} {len(times):>6} {avg:>10.4f} {mn:>10.4f} {mx:>10.4f}"
        print(line)
        out.write(line + "\n")

    out.write(f"\nTotal images processed: {len(rows)}\n")
    out.write(f"Log: {LOG_FILE}\n")

print(f"\nSummary saved to {OUT_FILE}")

# Try matplotlib if available
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    kernels = sorted(kernel_times.keys())
    avgs = [sum(kernel_times[k]) / len(kernel_times[k]) for k in kernels]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(kernels, avgs, color=['#4C9BE8', '#E87B4C', '#4CE87B'])
    ax.set_xlabel('Kernel Type')
    ax.set_ylabel('Avg GPU Time (ms)')
    ax.set_title('CUDA Image Processor - Avg Execution Time per Kernel')
    for bar, val in zip(bars, avgs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}ms', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig('results/performance_chart.png', dpi=150)
    print("Chart saved to results/performance_chart.png")
except ImportError:
    print("matplotlib not available - skipping chart (text summary saved)")
