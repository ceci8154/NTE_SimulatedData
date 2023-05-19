import json
import time

import numpy as np
import pandas as pd
import plotly.express as px

from NTEpyechelle import simulator


def run_benchmark_cpu(N_CPU=None, T=None):
    if N_CPU is None:
        N_CPU = [1, 2, 4, 8]
    if T is None:
        T = [0.01, 0.1, 1.0, 10, 30]
    times = {}

    # dry run, so everything gets compiled...
    simulator.main(
        args=(["-s", "MaroonX", "--sources", "Constant", "-t", f"{0.01}", "--orders", "100-105", "--max_cpu", f"{1}",
               "--overwrite"]))
    simulator.main(
        args=(["-s", "MaroonX", "--sources", "Constant", "-t", f"{0.01}", "--orders", "100-105", "--max_cpu", f"{2}",
               "--overwrite"]))

    run = 0
    for n_cpu in N_CPU:
        for t in T:
            run += 1
            t1 = time.time()
            nphot = simulator.main(args=(
                ["-s", "MaroonX", "--sources", "Constant", "-t", f"{t}", "--orders", "100-105", "--max_cpu",
                 f"{n_cpu}", "--overwrite"]))
            t2 = time.time()
            times.update({f"run_{run}": {"cpu": n_cpu, "exp.time": t, "duration": t2 - t1, "nphot": nphot}})

    with open("benchmark_results.json", 'w') as fout:
        json_dumps_str = json.dumps(times, indent=4)
        print(json_dumps_str, file=fout)
    return times


def run_benchmark_cuda(T=None):
    if T is None:
        T = [0.01, 0.1, 1.0, 10, 30]
    times = {}

    # dry run, so everything gets compiled...
    simulator.main(
        args=(["-s", "MaroonX", "--sources", "Constant", "-t", f"{0.01}", "--orders", "100-105", "--cuda",
               "--overwrite"]))

    run = 0
    for t in T:
        run += 1
        t1 = time.time()
        nphot = simulator.main(
            args=(["-s", "MaroonX", "--sources", "Constant", "-t", f"{t}", "--orders", "100-105", "--cuda",
                   "--overwrite"]))
        t2 = time.time()
        times.update({f"cuda_run_{run}": {"cpu": 1, "exp.time": t, "duration": t2 - t1, "nphot": nphot}})

    with open("benchmark_results_cuda.json", 'w') as fout:
        json_dumps_str = json.dumps(times, indent=4)
        print(json_dumps_str, file=fout)
    return times


def plot_results(result_cpu: dict, result_cuda: dict):
    x = []
    y = []
    c = []
    for run, result in result_cpu.items():
        x.append(int(result['nphot']))
        y.append(float(result['duration']))
        c.append(int(result['cpu']))

    if result_cuda is not None:
        for run, result in result_cuda.items():
            x.append(int(result['nphot']))
            y.append(float(result['duration']))
            c.append('cuda Quadro RTX4000')
    df = pd.DataFrame(np.array([x, y, c]).T, columns=["N Photons", "duration [s]", "# Cores"])
    df['N Photons'] = pd.to_numeric(df['N Photons'])
    df['duration [s]'] = pd.to_numeric(df['duration [s]'])
    fig = px.line(df, x="N Photons", y="duration [s]", color="# Cores", markers=True, title='Benchmark AMD Ryzen 5950')
    # fig.write_html("../docs/source/_static/plots/benchmark.html")
    fig.show()


if __name__ == "__main__":
    run_benchmark_cpu()
    run_benchmark_cuda()

    with open('benchmark_results.json', 'r') as f:
        results_cpu = json.load(f)
    with open('benchmark_results_cuda.json', 'r') as f:
        results_cuda = json.load(f)
    plot_results(results_cpu, results_cuda)
