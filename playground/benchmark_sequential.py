# %%
import itertools
import json
import random
import time

import numpy as np
import scipy.optimize as opt
import torch
from benchmark_tensor_operations import TensorOperationsBenchmark
from vectorized_f19 import vectorized_f19, vectorized_f19_grad

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)


# %%
def solve(x0: np.ndarray, tensor_benchmark: TensorOperationsBenchmark | None):
    dimension = len(x0)
    bounds = [(-5, 5)] * dimension
    R = np.eye(dimension)  # For simplicity, use identity matrix
    num_opt = 1  # Single optimization problem
    x_opt, f_opt, res_dict = opt.fmin_l_bfgs_b(
        func=vectorized_f19,
        x0=x0,
        fprime=vectorized_f19_grad,
        bounds=bounds,
        args=(R, num_opt, dimension, tensor_benchmark),
        # maxiter=200,
    )
    # z_opt = compute_z_from_x(x_opt, R)
    print(
        f"number of iterations: {res_dict['nit']}, function evaluations: {res_dict['funcalls']}"
    )
    return x_opt, f_opt, res_dict


def solve_sequential(x0s) -> dict:
    assert x0s.ndim == 2, "x0s should be a 2D array"
    num_opt, dimension = x0s.shape
    assert num_opt > 0, "num_opt should be greater than 0"
    assert dimension > 0, "D should be greater than 0"
    x_opts = []
    f_opts = []
    total_nits = []
    total_funcalls = []
    tensor_benchmark = TensorOperationsBenchmark(
        n_trials=300, dimension=dimension, batch_size=1
    )
    tensor_benchmark = None
    start = time.time()
    for x0 in x0s:
        x_opt, f_opt, res_dict = solve(x0, tensor_benchmark)
        x_opts.append(x_opt)
        f_opts.append(f_opt)
        total_nits.append(res_dict["nit"])
        total_funcalls.append(res_dict["funcalls"])
    elapsed = time.time() - start

    median_nits = np.median(total_nits)
    median_funcalls = np.median(total_funcalls)
    print(f"Time taken: {elapsed} seconds")
    result = {
        "x_opts": x_opts,
        "f_opts": f_opts,
        "elapsed": elapsed,
        "total_nits": sum(total_nits),
        "total_funcalls": sum(total_funcalls),
        "median_nits": float(median_nits),
        "median_funcalls": float(median_funcalls),
    }
    return result


def save_jsonl(results: list[dict], filename="results.jsonl"):
    with open(filename, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    num_opts = [1]
    dimensions = [10]
    results = []
    elapsed_times_sequential = []
    for num_opt, dimension in itertools.product(num_opts, dimensions):
        print(f"Solving with {num_opt=} and {dimension=}...")
        x0s = np.random.uniform(-5, 5, size=(num_opt, dimension))
        result_seq = solve_sequential(x0s)

        print(f"Results for {num_opt=} and {dimension=}: {result_seq}")
        print("Best Results (Sequential):", np.min(result_seq["f_opts"]))
        elapsed_times_sequential.append(result_seq["elapsed"])
        result = {
            "num_opt": num_opt,
            "dimension": dimension,
            "best_f_opts": min(result_seq["f_opts"]),
            "elapsed (sec)": result_seq["elapsed"],
            "total_nits": result_seq["total_nits"],
            "total_funcalls": result_seq["total_funcalls"],
            "median_nits": result_seq["median_nits"],
            "median_funcalls": result_seq["median_funcalls"],
        }
        results.append(result)

    print("Elapsed Times (Sequential):", elapsed_times_sequential)
    save_jsonl(results, "results_sequential.jsonl")
