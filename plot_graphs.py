#!/usr/bin/env python3
# plot_graphs.py

import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def plot_makespan(df, output_dir):
    pivot_ms = df.pivot(index='instance', columns='solver', values='makespan')
    ax = pivot_ms.plot(kind='bar', figsize=(8,5))
    ax.set_ylabel('Makespan')
    ax.set_xlabel('Instance')
    ax.set_title('Makespan by Solver and Instance')
    plt.xticks(rotation=0)
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'makespan.png')
    plt.savefig(out_path)
    print(f"* Makespan grafiği kaydedildi: {out_path}")
    plt.show()

def plot_runtime(df, output_dir):
    pivot_rt = df.pivot(index='instance', columns='solver', values='runtime_s')
    ax = pivot_rt.plot(kind='bar', figsize=(8,5))
    ax.set_ylabel('Runtime (s)')
    ax.set_xlabel('Instance')
    ax.set_title('Runtime by Solver and Instance')
    plt.xticks(rotation=0)
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'runtime.png')
    plt.savefig(out_path)
    print(f"* Runtime grafiği kaydedildi: {out_path}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(
        description='Solvers karşılama grafikleri (makespan ve runtime) üretir.')
    parser.add_argument('csv_file', help='Veri içeren CSV dosyası')
    parser.add_argument('-o', '--outdir',
                        default='.',
                        help='Grafikleri kaydedeceği klasör (default: çalışma dizini)')
    args = parser.parse_args()

    # CSV okuma
    df = pd.read_csv(args.csv_file)
    # Klasör varsa yoksa oluştur
    os.makedirs(args.outdir, exist_ok=True)

    # Grafik üretimler
    plot_makespan(df, args.outdir)
    plot_runtime(df, args.outdir)

if __name__ == '__main__':
    main()
