# %%
import itertools
import json
import random
import time

import numpy as np
import scipy.optimize as opt
import torch
from benchmark_tensor_operations import TensorOperationsBenchmark
from f19 import f19
from vectorized_f19 import vectorized_f19, vectorized_f19_grad

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)


# %%
def solve_stacking(x0s) -> dict:
    assert x0s.ndim == 2, "x0s should be a 2D array"
    batch_size, dimension = x0s.shape
    assert batch_size > 0, "batch_size should be greater than 0"
    assert dimension > 0, "D should be greater than 0"
    R = np.eye(dimension)
    bounds = [(-5, 5)] * dimension * batch_size

    print("x0s.shape:", x0s.shape)
    tensor_benchmark = TensorOperationsBenchmark(
        n_trials=300, dimension=dimension, batch_size=batch_size
    )

    start = time.time()
    res = opt.fmin_l_bfgs_b(
        func=vectorized_f19,
        x0=x0s.flatten(),
        fprime=vectorized_f19_grad,
        args=(R, batch_size, dimension, tensor_benchmark),
        bounds=bounds,
        # maxiter=200,
        # maxiter=150_000,
        # maxfun=150_000,
        # m=100,
    )
    elapsed = time.time() - start
    print(
        "Time taken:",
        elapsed,
        f"Optimization result with {dimension=}, {batch_size=}:",
        res,
    )

    x_opts, sum_f_opt, res_dict = res
    x_opts = x_opts.reshape(batch_size, dimension)
    check_sum = 0.0
    for i in range(batch_size):
        f_opt = f19(x_opts[i], R)
        print(f"Starting Point: {x0s[i]}, Optimal x: {x_opts[i]}, Optimal f: {f_opt}")
        check_sum += f_opt

    assert np.isclose(check_sum, sum_f_opt), "Check sum does not match expected sum"

    result = {
        "x_opts": x_opts,
        "f_opts": [f19(x, R) for x in x_opts],
        "elapsed": elapsed,
    }
    print("Result Keys:", res_dict.keys())
    for key, value in res_dict.items():
        result[key] = value
    print(
        f"number of iterations: {res_dict['nit']}, function evaluations: {res_dict['funcalls']}"
    )
    return result


# %%


def save_jsonl(results: list[dict], filename="results.jsonl"):
    with open(filename, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")


# %%
if __name__ == "__main__":
    batch_sizes = [1]  # , 3, 5]  # , 10, 30, 100]
    dimensions = [5, 10, 30, 50]
    results = []
    elapsed_times_stacking = []

    for batch_size, dimension in itertools.product(batch_sizes, dimensions):
        print(f"Solving with {batch_size=} and {dimension=}...")
        x0s = np.random.uniform(-5, 5, size=(batch_size, dimension))
        result_st = solve_stacking(x0s)
        print(f"Results for {batch_size=} and {dimension=}: {result_st}")
        print(
            f"number of iterations: {result_st['nit']}, function evaluations: {result_st['funcalls']}, funcalls per batch_size: {result_st['funcalls'] / batch_size if batch_size > 0 else 0:.2f}"
        )
        print("Best Results (Stacking):", np.min(result_st["f_opts"]))
        elapsed_times_stacking.append(result_st["elapsed"])

        result = {
            "batch_size": batch_size,
            "dimension": dimension,
            "best_f_opts": min(result_st["f_opts"]),
            "elapsed (sec)": result_st["elapsed"],
            "nit": result_st["nit"],
            "funcalls": result_st["funcalls"],
        }
        results.append(result)

    print("Elapsed Times (Stacking):", elapsed_times_stacking)
    save_jsonl(results, "results_stacking.jsonl")
