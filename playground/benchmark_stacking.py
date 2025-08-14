# %%
import random
import time

import numpy as np
import scipy.optimize as opt
import torch

from playground.f19 import f19
from playground.vectorized_f19 import vectorized_f19, vectorized_f19_grad

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)


# %%
def solve_stacking(x0s) -> dict:
    assert x0s.ndim == 2, "x0s should be a 2D array"
    num_opt, D = x0s.shape
    assert num_opt > 0, "num_opt should be greater than 0"
    assert D > 0, "D should be greater than 0"
    R = np.eye(D)
    bounds = [(-5, 5)] * D * num_opt

    print("x0s.shape:", x0s.shape)
    start = time.time()
    res = opt.fmin_l_bfgs_b(
        func=vectorized_f19,
        x0=x0s.ravel(),
        fprime=vectorized_f19_grad,
        args=(R, num_opt, D),
        bounds=bounds,
        maxiter=200,
        # maxfun=150_000,
    )
    elapsed = time.time() - start
    print(
        "Time taken:",
        elapsed,
        f"Optimization result with {D=}, {num_opt=}:",
        res,
    )

    x_opts, sum_f_opt, res_dict = res
    x_opts = x_opts.reshape(num_opt, D)
    check_sum = 0.0
    for i in range(num_opt):
        f_opt = f19(x_opts[i], R, enable_torch_computations=False)
        print(f"Starting Point: {x0s[i]}, Optimal x: {x_opts[i]}, Optimal f: {f_opt}")
        check_sum += f_opt

    assert np.isclose(check_sum, sum_f_opt), "Check sum does not match expected sum"

    result = {
        "x_opts": x_opts,
        "f_opts": [f19(x, R, enable_torch_computations=False) for x in x_opts],
        "elapsed": elapsed,
    }
    for key, value in res_dict.items():
        result[key] = value
    print(
        f"number of iterations: {res_dict['nit']}, function evaluations: {res_dict['funcalls']}"
    )
    return result


# %%
if __name__ == "__main__":
    num_opts = [2]
    dimensions = [10]
    MULTI_PROCESSING = False
    results_stacking = []
    elapsed_times_stacking = []

    for num_opt in num_opts:
        for dimension in dimensions:
            print(f"Solving with {num_opt=} and {dimension=}...")
            x0s = np.random.uniform(-5, 5, size=(num_opt, dimension))
            result_st = solve_stacking(x0s)
            results_stacking.append(result_st)
            print(f"Results for {num_opt=} and {dimension=}: {result_st}")
            print(
                f"number of iterations: {result_st['nit']}, function evaluations: {result_st['funcalls']}, funcalls per num_opt: {result_st['funcalls'] / num_opt if num_opt > 0 else 0:.2f}"
            )
            print("Best Results (Stacking):", np.min(result_st["f_opts"]))
            elapsed_times_stacking.append(result_st["elapsed"])

    print("Elapsed Times (Stacking):", elapsed_times_stacking)
