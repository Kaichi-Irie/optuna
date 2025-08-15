# %%
import numpy as np
from benchmark_tensor_operations import TensorOperationsBenchmark


def vectorized_f19(
    x_flat: np.ndarray,
    R: np.ndarray,
    num_opt: int,
    dimension: int,
    tensor_benchmark: TensorOperationsBenchmark | None = None,
) -> float:
    """
    f19関数のベクトル化バージョン。
    """

    xs = x_flat.reshape(num_opt, dimension)
    if dimension < 2:
        raise ValueError("次元Dは2以上である必要があります。")

    # 定数cの計算
    c = max(1.0, np.sqrt(dimension) / 8.0)

    # zの計算をバッチ処理
    # (N, D) @ (D, D) -> (N, D)
    zs = c * (xs @ R.T) + 0.5

    # sの計算をベクトル化 (forループを排除)
    # zsの隣接要素を使ってRosenbrock-likeな関数を計算
    z_i = zs[:, :-1]  # 形状: (N, D-1)
    z_i_plus_1 = zs[:, 1:]  # 形状: (N, D-1)

    s = 100.0 * (z_i**2 - z_i_plus_1) ** 2 + (z_i - 1) ** 2

    if tensor_benchmark is not None:
        tensor_benchmark.execute()

    # 最終的な値をバッチごとに計算 (axis=1で各行ごとに合計)
    values = (10.0 / (dimension - 1)) * np.sum(s / 4000.0 - np.cos(s), axis=1) + 10.0
    return np.sum(values)


def vectorized_f19_grad(
    x_flat: np.ndarray,
    R: np.ndarray,
    num_opt: int,
    dimension: int,
    tensor_benchmark: TensorOperationsBenchmark | None = None,
) -> np.ndarray:
    """
    f19_grad関数のベクトル化バージョン。
    """
    xs = x_flat.reshape(num_opt, dimension)
    if dimension < 2:
        raise ValueError("次元Dは2以上である必要があります。")

    # ステップ1: 中間変数 z の計算 (バッチ処理)
    c = max(1.0, np.sqrt(dimension) / 8.0)
    zs = c * (xs @ R.T) + 0.5

    # ステップ2: 中間変数 s の計算 (ベクトル化)
    z_i = zs[:, :-1]
    z_i_plus_1 = zs[:, 1:]
    term1_rosen = z_i**2 - z_i_plus_1
    term2_rosen = z_i - 1
    s = 100 * (term1_rosen**2) + (term2_rosen**2)

    # ステップ3: f19 の s に関する偏微分 (df/ds)
    df_ds = (10.0 / (dimension - 1)) * (1.0 / 4000.0 + np.sin(s))

    # ステップ4: f19 の z に関する偏微分 (df/dz)
    df_dzs = np.zeros_like(zs)  # 形状: (N, D)

    dsi_dzi = 400 * z_i * term1_rosen + 2 * term2_rosen
    dsi_dzi_plus_1 = -200 * term1_rosen

    # df/dzへの寄与をスライシングで一括加算
    df_dzs[:, :-1] += df_ds * dsi_dzi
    df_dzs[:, 1:] += df_ds * dsi_dzi_plus_1

    # 最終的な勾配を連鎖律で計算 (バッチ処理)
    # (N, D) @ (D, D) -> (N, D)
    grads = c * (df_dzs @ R)

    return grads.ravel()


# %%
# double check

import numpy as np
from f19 import f19, f19_grad

np.random.seed(0)


def double_check(N=30, D=5):
    # --- 入力データの準備 ---
    # (N, D) の形状を持つランダムな入力ベクトル
    xs_batch = np.random.rand(N, D)
    # (D, D) の形状を持つ回転行列
    A = np.random.rand(D, D)
    R_matrix = A.T @ A

    # --- ベクトル化された関数の実行 ---
    # 1. 目的関数の値を一括で計算

    sum_f = vectorized_f19(xs_batch.ravel(), R_matrix, N, D)

    # 2. 勾配を一括で計算
    grads = vectorized_f19_grad(xs_batch.ravel(), R_matrix, N, D)
    grads = grads.reshape(N, D)

    print("--- 入力バッチの形状 ---")
    print(xs_batch.shape)

    print("\n--- 計算結果 (目的関数値) ---")
    print(sum_f)

    print("\n--- 計算結果 (勾配) ---")
    print(grads.shape)
    print(grads)
    f_values_check = []
    grads_check = []

    for i in range(N):
        x = xs_batch[i]
        f_value = f19(x, R_matrix)
        grad_value = f19_grad(x, R_matrix)
        f_values_check.append(f_value)
        grads_check.append(grad_value)
    sum_f_check = np.sum(f_values_check)

    if not np.isclose(sum_f_check, sum_f):
        print(f"Mismatch found in sum: {sum_f_check} vs {sum_f}")
        raise ValueError("目的関数の合計が一致しません。")

    for i in range(N):
        if not np.allclose(grads_check[i], grads[i]):
            print(f"Mismatch found at index {i}")
            print(f"f19_grad: {grads_check[i]} vs {grads[i]}")
            raise ValueError("勾配の不一致が見つかりました。")
    print("All checks passed!")


# double_check()
