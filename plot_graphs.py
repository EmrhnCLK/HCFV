#!/usr/bin/env python3
# plot_graphs.py

import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def apply_optimal_hatch(ax, pivot):
    hatches = {
        'Gurobi': '///',
        'Mod-Gurobi': 'xxx'
    }
    solvers = list(pivot.columns)
    for idx, solver in enumerate(solvers):
        if solver in hatches:
            for bar in ax.containers[idx]:
                bar.set_hatch(hatches[solver])
                bar.set_edgecolor('black')
                bar.set_linewidth(0.8)


def plot_makespan(df, output_dir):
    pivot = df.pivot(index='instance', columns='solver', values='makespan')
    ax = pivot.plot(kind='bar', figsize=(8,5))
    apply_optimal_hatch(ax, pivot)
    ax.set_ylabel('Makespan')
    ax.set_xlabel('Instance')
    ax.set_title('Makespan by Solver and Instance\n(hatched = optimal çözücüler)')
    plt.xticks(rotation=0)
    plt.tight_layout()
    path = os.path.join(output_dir, 'makespan.png')
    plt.savefig(path)
    print(f"* Makespan grafiği kaydedildi: {path}")
    plt.show()


def plot_runtime(df, output_dir):
    # 1) Tüm instancelar, 15x15 hariç
    mask = df['instance'] != 'hurink/edata/la36.txt_15x15.txt'
    pivot1 = df[mask].pivot(index='instance', columns='solver', values='runtime_s')
    ax1 = pivot1.plot(kind='bar', figsize=(8,5))
    # Symlog scale for main plot (except 15x15)
    ax1.set_yscale('symlog', linthresh=1, base=10)
    ax1.set_ylabel('Runtime (s)')
    ax1.set_xlabel('Instance')
    ax1.set_title('Runtime by Solver and Instance (except 15×15, symlog)')
    plt.xticks(rotation=0)
    plt.tight_layout()
    path1 = os.path.join(output_dir, 'runtime.png')
    plt.savefig(path1)
    print(f"* Runtime grafiği (15×15 hariç) kaydedildi: {path1}")
    plt.show()

    # 2) Sadece 15×15 instancelar (linear scale)
    mask2 = df['instance'] == 'hurink/edata/la36.txt_15x15.txt'
    pivot2 = df[mask2].pivot(index='instance', columns='solver', values='runtime_s')
    ax2 = pivot2.plot(kind='bar', figsize=(6,4))
    ax2.set_ylabel('Runtime (s)')
    ax2.set_xlabel('Instance')
    ax2.set_title('Runtime for 15×15 Instance')
    plt.xticks(rotation=0)
    plt.tight_layout()
    path2 = os.path.join(output_dir, 'runtime_15x15.png')
    plt.savefig(path2)
    print(f"* Runtime grafiği (sadece 15×15) kaydedildi: {path2}")
    plt.show()


def plot_gap(df, output_dir):
    pivot = df.pivot(index='instance', columns='solver', values='gap_%')
    ax = pivot.plot(kind='bar', figsize=(8,5))
    ax.set_ylabel('Gap (%)')
    ax.set_xlabel('Instance')
    ax.set_title('Optimal Gap by Solver and Instance')
    plt.xticks(rotation=0)
    plt.tight_layout()
    path = os.path.join(output_dir, 'gap.png')
    plt.savefig(path)
    print(f"* Gap grafiği kaydedildi: {path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Solvers karşılaştırma grafikleri (makespan, runtime, gap) üretir.')
    parser.add_argument('csv_file', help='Veri içeren CSV dosyası')
    parser.add_argument('-o', '--outdir',
                        default='.',
                        help='Grafikleri kaydedeceği klasör (default: çalışma dizini)')
    args = parser.parse_args()

    df = pd.read_csv(args.csv_file)
    os.makedirs(args.outdir, exist_ok=True)

    plot_makespan(df, args.outdir)
    plot_runtime(df, args.outdir)
    plot_gap(df, args.outdir)

if __name__ == '__main__':
    main()
