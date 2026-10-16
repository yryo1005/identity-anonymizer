"""
実験1: 潜在次元ごとの統計 (不活性次元の把握)．

全画像について潜在変数の平均 mu と標準偏差 std = exp(0.5 * logvar) を計算し，次元ごとに
std の平均・中央値，mu のデータ全体での分散 (Active Units の指標)，標準的なガウスKLの
データ平均を求める．あわせて，線形デコーダの重み行列から次元ごとの感度も求める．

サンプリングを伴わない決定論的な計算であるため，seed には依存しない．

    python programs/ex001_vae_latent_diagnosis/ex1_latent_stats.py
"""

import os

import numpy as np
import pandas as pd
import torch

import common
from data import load_arcface_cache
from model import describe_weights, encode_all, load_model

METHOD = "ex1_latent_stats"
HYPERPARAMETER = "all_images"


def per_dimension_statistics(mu: torch.Tensor, std: torch.Tensor, decoder_weight: torch.Tensor) -> pd.DataFrame:
    """
    潜在次元ごとの統計量を計算する．

    引数:
        mu (torch.Tensor): 形状 (N, d) の潜在変数の平均．
        std (torch.Tensor): 形状 (N, d) の潜在変数の標準偏差．
        decoder_weight (torch.Tensor): 形状 (512, d) の線形デコーダの重み行列．
    戻り値:
        table (pd.DataFrame): 行数 d の表．列は以下の通り．
            dim: 次元の番号，std_mean / std_median / std_min / std_max: 標準偏差の統計量，
            mu_mean / mu_var: mu のデータ全体での平均・分散，active: mu_var が閾値を超えるか，
            kl_mean: 標準的なガウスKLのデータ平均 [nat]，
            decoder_col_norm: デコーダの重みの列ノルム (その次元を1動かしたときの出力の移動量)，
            output_shift_per_alpha: std_mean * decoder_col_norm (alpha=1 のときの出力の移動量の目安)．
    """
    # 中央値などはGPU上で決定論的に計算できないため，CPU上で計算する
    mu, std, decoder_weight = mu.cpu(), std.cpu(), decoder_weight.cpu()
    logvar = 2.0 * torch.log(std)
    kl = 0.5 * (mu.pow(2) + std.pow(2) - 1.0 - logvar)
    column_norm = decoder_weight.norm(dim=0)

    table = pd.DataFrame({
        "dim": np.arange(mu.shape[1]),
        "std_mean": std.mean(dim=0).cpu().numpy(),
        "std_median": std.median(dim=0).values.cpu().numpy(),
        "std_min": std.min(dim=0).values.cpu().numpy(),
        "std_max": std.max(dim=0).values.cpu().numpy(),
        "mu_mean": mu.mean(dim=0).cpu().numpy(),
        "mu_var": mu.var(dim=0, unbiased=True).cpu().numpy(),
        "kl_mean": kl.mean(dim=0).cpu().numpy(),
        "decoder_col_norm": column_norm.cpu().numpy(),
    })
    table["active"] = table["mu_var"] > common.ACTIVE_UNIT_THRESHOLD
    table["output_shift_per_alpha"] = table["std_mean"] * table["decoder_col_norm"]
    return table


def summarize(table: pd.DataFrame, mu: torch.Tensor, std: torch.Tensor, decoder_weight: torch.Tensor) -> dict:
    """
    次元ごとの統計量から，全体の集計値を計算する．

    引数:
        table (pd.DataFrame): `per_dimension_statistics` の戻り値 (行数 d)．
        mu (torch.Tensor): 形状 (N, d) の潜在変数の平均．
        std (torch.Tensor): 形状 (N, d) の潜在変数の標準偏差．
        decoder_weight (torch.Tensor): 形状 (512, d) の線形デコーダの重み行列．
    戻り値:
        summary (dict): 活性次元数，KLの合計，標準偏差の分布，デコーダの特異値などの集計値．
    """
    singular_values = torch.linalg.svdvals(decoder_weight.cpu()).numpy()
    quantile_points = [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]
    flat_std = std.flatten().cpu().numpy()
    std_quantiles = np.quantile(flat_std, quantile_points)

    return {
        "num_images": int(mu.shape[0]),
        "latent_dim": int(mu.shape[1]),
        "num_active_units": int(table["active"].sum()),
        "num_inactive_units": int((~table["active"]).sum()),
        "active_unit_threshold": common.ACTIVE_UNIT_THRESHOLD,
        "kl_total_mean_nat": float(table["kl_mean"].sum()),
        "kl_per_dim_min": float(table["kl_mean"].min()),
        "kl_per_dim_median": float(table["kl_mean"].median()),
        "kl_per_dim_max": float(table["kl_mean"].max()),
        "num_dims_kl_below_0.01": int((table["kl_mean"] < 0.01).sum()),
        "mu_var_min": float(table["mu_var"].min()),
        "mu_var_median": float(table["mu_var"].median()),
        "mu_var_max": float(table["mu_var"].max()),
        "std_all_elements_mean": float(flat_std.mean()),
        "std_all_elements_quantiles": dict(zip(map(str, quantile_points), map(float, std_quantiles))),
        "std_dim_mean_min": float(table["std_mean"].min()),
        "std_dim_mean_max": float(table["std_mean"].max()),
        "mu_norm_mean": float(mu.norm(dim=1).mean()),
        "std_vector_norm_mean": float(std.norm(dim=1).mean()),
        "decoder_singular_values": [float(s) for s in singular_values],
        "decoder_col_norm_min": float(table["decoder_col_norm"].min()),
        "decoder_col_norm_mean": float(table["decoder_col_norm"].mean()),
        "decoder_col_norm_max": float(table["decoder_col_norm"].max()),
    }


def run() -> None:
    """
    実験1を実行し，次元ごとの統計のCSV，集計のJSON，図の描画に用いる配列を保存する．

    引数:
        なし．
    戻り値:
        なし．
    """
    target_dir = common.condition_dir(METHOD, HYPERPARAMETER, 0)
    if common.is_finished(target_dir):
        print(f"実行済みのためスキップ: {target_dir}")
        return

    embeddings, _, _ = load_arcface_cache()
    model = load_model()
    mu, std = encode_all(model, embeddings)
    decoder_weight = model.decoder[0].weight.detach()

    table = per_dimension_statistics(mu, std, decoder_weight)
    summary = summarize(table, mu, std, decoder_weight)

    table.to_csv(os.path.join(target_dir, "per_dim_stats.csv"), index=False)
    common.save_json(summary, os.path.join(target_dir, "summary.json"))
    # ヒストグラムの描画用に，標準偏差の全要素を間引いて保存する
    flat_std = std.flatten().cpu().numpy()
    picked = np.random.default_rng(0).choice(flat_std.size, size=min(500_000, flat_std.size), replace=False)
    np.save(os.path.join(target_dir, "std_sample.npy"), flat_std[picked])
    common.save_json({
        "weight_path": os.path.relpath(common.VAE_WEIGHT_PATH, common.REPO_ROOT),
        "weight_shapes": describe_weights(),
        "std_definition": "std = exp(0.5 * logvar)",
        "kl_definition": "0.5 * (mu^2 + std^2 - 1 - logvar)",
    }, os.path.join(target_dir, "meta.json"))

    printable = {k: v for k, v in summary.items() if k != "decoder_singular_values"}
    print(pd.Series(printable).to_string())


if __name__ == "__main__":
    run()
