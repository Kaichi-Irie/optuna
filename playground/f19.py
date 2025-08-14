import math

import numpy as np
import torch


def f19(x: np.ndarray, R: np.ndarray, enable_torch_computations=True) -> float:
    D = len(x)
    if D < 2:
        raise ValueError("D must be greater than 1")

    # R = np.eye(D)  # For simplicity, use identity matrix
    s = np.zeros(D - 1)
    z = max(1, math.sqrt(D) / 8.0) * (R @ x) + 0.5
    for i in range(D - 1):
        s[i] = 100.0 * (z[i] ** 2 - z[i + 1]) ** 2 + (z[i] - 1) ** 2
    if enable_torch_computations:
        torch.randn(300, 300) @ torch.randn(300, 300)  # some operation

    return 10.0 / (D - 1) * np.sum(s / 4000.0 - np.cos(s)) + 10.0


def f19_grad(x: np.ndarray, R: np.ndarray) -> np.ndarray:
    D = len(x)
    if D < 2:
        raise ValueError("次元Dは2以上である必要があります。")

    # R = np.eye(D)  # For simplicity, use identity matrix
    # ステップ1: 中間変数 z の計算
    c = max(1.0, np.sqrt(D) / 8.0)
    z = c * (R @ x) + 0.5

    # ステップ2: 中間変数 s の計算
    # s_i = 100*(z_i^2 - z_{i+1})^2 + (z_i - 1)^2
    z_i = z[:-1]
    z_i_plus_1 = z[1:]

    # この部分がRosenbrock関数の標準形
    term1_rosen = z_i**2 - z_i_plus_1
    term2_rosen = z_i - 1
    # 100が第1項のみにかかるように修正
    s = 100 * (term1_rosen**2) + (term2_rosen**2)

    # ステップ3: f19 の s に関する偏微分 (df/ds) の計算
    df_ds = (10.0 / (D - 1)) * (1.0 / 4000.0 + np.sin(s))

    # ステップ4: f19 の z に関する偏微分 (df/dz) の計算
    df_dz = np.zeros(D)

    # ds_i/dz_i の計算 (訂正後の式を反映)
    dsi_dzi = 400 * z_i * term1_rosen + 2 * term2_rosen

    # ds_i/dz_{i+1} の計算
    dsi_dzi_plus_1 = -200 * term1_rosen

    # df/dz_j の各成分への寄与を計算
    # z_jがs_jのz_i項として与える寄与 (df/ds_j * ds_j/dz_j)
    contribution_from_si = df_ds * dsi_dzi
    df_dz[:-1] += contribution_from_si

    # z_jがs_{j-1}のz_{i+1}項として与える寄与 (df/ds_{j-1} * ds_{j-1}/dz_j)
    contribution_from_si_minus_1 = df_ds * dsi_dzi_plus_1
    df_dz[1:] += contribution_from_si_minus_1

    grad_f19 = c * (R.T @ df_dz)

    return grad_f19


def compute_z_from_x(x: np.ndarray, R: np.ndarray) -> np.ndarray:
    D = len(x)
    if D < 2:
        raise ValueError("D must be greater than 1")

    c = max(1.0, np.sqrt(D) / 8.0)
    z = c * (R @ x) + 0.5
    return z
