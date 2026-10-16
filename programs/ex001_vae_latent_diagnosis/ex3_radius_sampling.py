"""
実験3: 半径を固定したサンプリングとの比較．

同じ u ~ N(0, I) から，ガウスサンプル z = u と，半径を固定したサンプル z = r * sqrt(d) * u / ||u||
(r は `common.RADIUS_FACTORS`) を作る．方向は同じで半径だけが異なる対応のあるサンプルである．
各 alpha について cos(v_tilde, v) を比較し，ガウスと r=1 の差を対応のある検定で評価する．
u は全ての alpha で共有するため，(alpha, r) と (alpha * r, 1) の結果は厳密に一致するはずであり，
半径の固定が alpha のスケーリングと等価であることを数値的に確認できる．

    python programs/ex001_vae_latent_diagnosis/ex3_radius_sampling.py
"""

import itertools
import os
from typing import Dict

import numpy as np
import pandas as pd
import torch
from scipy import stats
from tqdm import tqdm

import common
import metrics
from machine_learning_utils import set_seed

METHOD = "ex3_radius_sampling"
HYPERPARAMETER = f"K{common.NUM_SAMPLES}"
GAUSS = "gauss"


def variant_name(radius_factor: float) -> str:
    """
    半径の倍率から，サンプリング方法の名前を作る．

    引数:
        radius_factor (float): 半径の倍率 r (半径は r * sqrt(d))．
    戻り値:
        name (str): サンプリング方法の名前 (例: "shell_r1.0")．
    """
    return f"shell_r{radius_factor}"


def make_noise_variants(u: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    同じ u から，ガウスサンプルと半径を固定したサンプルを作る．

    引数:
        u (torch.Tensor): 形状 (N, K, d) の標準正規乱数．
    戻り値:
        variants (dict): サンプリング方法の名前をkey，形状 (N, K, d) のノイズをvalueとする辞書．
    """
    direction = u / u.norm(dim=2, keepdim=True)
    sqrt_d = u.shape[2] ** 0.5
    variants = {GAUSS: u}
    for radius_factor in common.RADIUS_FACTORS:
        variants[variant_name(radius_factor)] = radius_factor * sqrt_d * direction
    return variants


@torch.no_grad()
def compute_cosines(model: torch.nn.Module, v: torch.Tensor, mu: torch.Tensor, std: torch.Tensor,
                    variants: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    """
    全ての alpha・全てのサンプリング方法について cos(v_tilde, v) を計算する．

    引数:
        model (torch.nn.Module): 学習済みのVAE．
        v (torch.Tensor): 形状 (N, 512) のArcFaceの顔ベクトル．
        mu (torch.Tensor): 形状 (N, d) の潜在変数の平均．
        std (torch.Tensor): 形状 (N, d) の潜在変数の標準偏差．
        variants (dict): `make_noise_variants` の戻り値．
    戻り値:
        cosines (dict): サンプリング方法の名前をkey，形状 (alphaの数, N, K) の配列をvalueとする辞書．
    """
    cosines = {name: [] for name in variants}
    conditions = list(itertools.product(common.ALPHAS, variants.items()))
    for alpha, (name, z) in tqdm(conditions, desc="ex3", leave=False):
        v_tilde = metrics.decode_displaced(model, mu, alpha * std.unsqueeze(1) * z)
        cosines[name].append(metrics.cosine_to_reference(v_tilde, v).cpu().numpy().astype(np.float32))
    return {name: np.stack(values) for name, values in cosines.items()}


def paired_test(a: np.ndarray, b: np.ndarray, prefix: str) -> dict:
    """
    対応のある2群 a, b について，差 a - b の検定と効果量を計算する．

    引数:
        a (np.ndarray): 形状 (M,) の1群目の値．
        b (np.ndarray): 形状 (M,) の2群目の値 (a と対応している)．
        prefix (str): 列名の接頭辞．
    戻り値:
        result (dict): 差の平均，95%信頼区間，対応のあるt検定・Wilcoxonの符号順位検定のp値，
            効果量 (差の標準偏差で規格化した d_z，2群の標準偏差の平均で規格化した d_av) の辞書．
    """
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    diff = a - b
    if np.allclose(diff, 0.0):
        keys = ["diff_mean", "diff_ci95_low", "diff_ci95_high", "t_stat", "t_pvalue", "wilcoxon_pvalue", "cohen_dz", "cohen_dav"]
        return {f"{prefix}_{key}": (0.0 if key.startswith("diff") else float("nan")) for key in keys}

    sem = diff.std(ddof=1) / np.sqrt(diff.size)
    t_result = stats.ttest_rel(a, b)
    wilcoxon_result = stats.wilcoxon(diff)
    pooled_std = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
    return {
        f"{prefix}_diff_mean": float(diff.mean()),
        f"{prefix}_diff_ci95_low": float(diff.mean() - 1.96 * sem),
        f"{prefix}_diff_ci95_high": float(diff.mean() + 1.96 * sem),
        f"{prefix}_t_stat": float(t_result.statistic),
        f"{prefix}_t_pvalue": float(t_result.pvalue),
        f"{prefix}_wilcoxon_pvalue": float(wilcoxon_result.pvalue),
        f"{prefix}_cohen_dz": float(diff.mean() / diff.std(ddof=1)),
        f"{prefix}_cohen_dav": float(diff.mean() / pooled_std),
    }


def compare_gauss_and_shell(cosines: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    各 alpha について，ガウスサンプルと半径 sqrt(d) のサンプルの cos(v_tilde, v) を比較する．
    検定は，画像ごとの平均 (N 組) と，サンプル単位 (N*K 組) の両方で行う．

    引数:
        cosines (dict): `compute_cosines` の戻り値．
    戻り値:
        table (pd.DataFrame): alpha ごとに1行の比較表．
    """
    shell = cosines[variant_name(1.0)]
    rows = []
    for index, alpha in enumerate(common.ALPHAS):
        gauss_alpha, shell_alpha = cosines[GAUSS][index], shell[index]
        row = {
            "alpha": alpha,
            "gauss_cos_v_mean": float(gauss_alpha.mean()), "gauss_cos_v_std": float(gauss_alpha.std()),
            "shell_cos_v_mean": float(shell_alpha.mean()), "shell_cos_v_std": float(shell_alpha.std()),
        }
        row.update(paired_test(gauss_alpha.mean(axis=1), shell_alpha.mean(axis=1), "image"))
        row.update(paired_test(gauss_alpha.ravel(), shell_alpha.ravel(), "sample"))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_variants(cosines: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    全ての alpha・サンプリング方法について，cos(v_tilde, v) の平均・分位点をまとめる．

    引数:
        cosines (dict): `compute_cosines` の戻り値．
    戻り値:
        table (pd.DataFrame): (alpha, サンプリング方法) ごとに1行の要約表．
            effective_alpha は alpha * r (ガウスの場合は alpha) である．
    """
    rows = []
    for (index, alpha), name in itertools.product(enumerate(common.ALPHAS), cosines):
        radius_factor = 1.0 if name == GAUSS else float(name.split("_r")[1])
        row = {"alpha": alpha, "variant": name, "effective_alpha": alpha * radius_factor}
        row.update(metrics.quantile_summary(cosines[name][index], "cos_v"))
        rows.append(row)
    return pd.DataFrame(rows)


def check_scaling_equivalence(cosines: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    半径の固定が alpha のスケーリングと等価であることを確認する．alpha * r が別の alpha' と
    一致する組について，(alpha, r) と (alpha', r=1) の cos(v_tilde, v) の差の最大絶対値を求める．

    引数:
        cosines (dict): `compute_cosines` の戻り値．
    戻り値:
        table (pd.DataFrame): 一致する組ごとに1行の表．
    """
    reference = cosines[variant_name(1.0)]
    rows = []
    for (index, alpha), radius_factor in itertools.product(enumerate(common.ALPHAS), common.RADIUS_FACTORS):
        effective_alpha = alpha * radius_factor
        if radius_factor == 1.0 or alpha == 0.0 or effective_alpha not in common.ALPHAS:
            continue
        other = reference[common.ALPHAS.index(effective_alpha)]
        target = cosines[variant_name(radius_factor)][index]
        rows.append({
            "alpha": alpha, "radius_factor": radius_factor, "equivalent_alpha_at_r1": effective_alpha,
            "cos_v_mean": float(target.mean()), "equivalent_cos_v_mean": float(other.mean()),
            "max_abs_diff": float(np.abs(target - other).max()),
        })
    return pd.DataFrame(rows)


def run_seed(seed: int) -> None:
    """
    1つのseedについて実験3を実行し，要約表・検定結果・サンプルごとの cos を保存する．

    引数:
        seed (int): 乱数シード．
    戻り値:
        なし．
    """
    target_dir = common.condition_dir(METHOD, HYPERPARAMETER, seed)
    if common.is_finished(target_dir):
        print(f"実行済みのためスキップ: {target_dir}")
        return

    model, v, mu, std, _ = metrics.load_inputs()
    set_seed(seed)
    latent_dim = mu.shape[1]
    u = metrics.sample_standard_normal(mu.shape[0], common.NUM_SAMPLES, latent_dim, metrics.make_generator(seed))
    u_norm = u.norm(dim=2).cpu().numpy().astype(np.float64)

    cosines = compute_cosines(model, v, mu, std, make_noise_variants(u))
    comparison = compare_gauss_and_shell(cosines)
    comparison.to_csv(os.path.join(target_dir, "gauss_vs_shell_tests.csv"), index=False)
    summarize_variants(cosines).to_csv(os.path.join(target_dir, "summary_by_variant.csv"), index=False)
    check_scaling_equivalence(cosines).to_csv(os.path.join(target_dir, "scaling_equivalence.csv"), index=False)
    np.savez_compressed(os.path.join(target_dir, "cos_v_per_sample.npz"), alphas=np.array(common.ALPHAS), u_norm=u_norm, **cosines)

    common.save_json({
        "seed": seed, "alphas": common.ALPHAS, "radius_factors": common.RADIUS_FACTORS,
        "num_samples": common.NUM_SAMPLES, "num_images": int(v.shape[0]), "latent_dim": int(latent_dim),
        "sqrt_d": float(latent_dim ** 0.5),
        "z_norm_mean": float(u_norm.mean()), "z_norm_std": float(u_norm.std()),
        "z_norm_min": float(u_norm.min()), "z_norm_max": float(u_norm.max()),
        "z_norm_cv": float(u_norm.std() / u_norm.mean()),
    }, os.path.join(target_dir, "meta.json"))

    columns = ["alpha", "gauss_cos_v_mean", "shell_cos_v_mean", "image_diff_mean", "image_cohen_dz", "image_cohen_dav", "image_t_pvalue"]
    print(f"\n[seed={seed}] ||z||: mean={u_norm.mean():.4f}, std={u_norm.std():.4f}, sqrt(d)={latent_dim ** 0.5:.4f}")
    print(comparison[columns].to_string(index=False))


def run() -> None:
    """
    `common.SEEDS` の全てのseedについて実験3を実行する．

    引数:
        なし．
    戻り値:
        なし．
    """
    for seed in common.SEEDS:
        run_seed(seed)


if __name__ == "__main__":
    run()
