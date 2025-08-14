# %%
import random
import time

import numpy as np
import scipy.optimize as opt
import torch

from playground.f19 import f19, f19_grad

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)


# %%
def solve(x0):
    D = len(x0)
    bounds = [(-5, 5)] * D
    R = np.eye(D)  # For simplicity, use identity matrix
    x_opt, f_opt, res_dict = opt.fmin_l_bfgs_b(
        func=f19,
        x0=x0,
        fprime=f19_grad,
        bounds=bounds,
        args=(R,),
        maxiter=200,
    )
    # z_opt = compute_z_from_x(x_opt, R)
    print(
        f"number of iterations: {res_dict['nit']}, function evaluations: {res_dict['funcalls']}"
    )
    return x_opt, f_opt, res_dict


def solve_sequential(x0s) -> dict:
    assert x0s.ndim == 2, "x0s should be a 2D array"
    num_opt, D = x0s.shape
    assert num_opt > 0, "num_opt should be greater than 0"
    assert D > 0, "D should be greater than 0"
    x_opts = []
    f_opts = []
    start = time.time()
    total_nits = []
    total_funcalls = []
    for i, x0 in enumerate(x0s):
        x_opt, f_opt, res_dict = solve(x0)
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
        "total_nits": total_nits,
        "total_funcalls": total_funcalls,
        "median_nits": f"{median_nits:.2f}",
        "median_funcalls": f"{median_funcalls:.2f}",
    }
    return result


if __name__ == "__main__":
    num_opts = [1]
    dimensions = [10]
    results_sequential = []
    elapsed_times_sequential = []

    for num_opt in num_opts:
        for dimension in dimensions:
            print(f"Solving with {num_opt=} and {dimension=}...")
            x0s = np.random.uniform(-5, 5, size=(num_opt, dimension))
            result_seq = solve_sequential(x0s)
            results_sequential.append(result_seq)

            print(f"Results for {num_opt=} and {dimension=}: {result_seq}")
            print("Best Results (Sequential):", np.min(result_seq["f_opts"]))
            elapsed_times_sequential.append(result_seq["elapsed"])

    print("Elapsed Times (Sequential):", elapsed_times_sequential)
