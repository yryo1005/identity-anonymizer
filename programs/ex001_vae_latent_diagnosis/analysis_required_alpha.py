"""
実験2の補足: 目標とする cos(v_tilde, v) に到達するために必要な alpha を，線形デコーダに基づく
解析的な近似式から画像ごとに見積もる．

デコーダが線形であり，潜在空間のノイズによる出力の移動が decoder(mu) とほぼ直交することから，

    cos(v_tilde, v) ≈ cos(v0, v) * ||v0|| / sqrt(||v0||^2 + alpha^2 * s^2)

が成り立つ (v0 = decoder(mu)，s は alpha=1 のときの出力の移動量の二乗平均平方根)．
この式を alpha について解き，目標値ごとに必要な alpha の分布を求める．見積もりは近似式による
予測であり，実測値ではない (近似式の精度は実験2の `pred_cos_v_mean` と実測の比較で確認できる)．

あわせて，cos(v_tilde, v) のばらつきを同一画像内と画像間に分解する (`within_image_spread`)．

    python programs/ex001_vae_latent_diagnosis/analysis_required_alpha.py
"""

import os

import numpy as np
import pandas as pd

import common
import metrics

TARGET_COSINES = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]


def required_alpha(cos_v0_v: np.ndarray, v0_norm: np.ndarray, noise_out_rms: np.ndarray, target: float) -> np.ndarray:
    """
    画像ごとに，cos(v_tilde, v) の期待値が目標値になる alpha を近似式から求める．

    引数:
        cos_v0_v (np.ndarray): 形状 (N,) の cos(decoder(mu), v)．
        v0_norm (np.ndarray): 形状 (N,) の ||decoder(mu)||．
        noise_out_rms (np.ndarray): 形状 (N,) の alpha=1 のときの出力の移動量の二乗平均平方根．
        target (float): 目標とするコサイン類似度．
    戻り値:
        alpha (np.ndarray): 形状 (N,) の必要な alpha．alpha=0 の時点で既に目標値以下の画像は 0．
    """
    ratio_sq = np.maximum((cos_v0_v / target) ** 2 - 1.0, 0.0)
    return v0_norm / noise_out_rms * np.sqrt(ratio_sq)


def run(seed: int = 0) -> None:
    """
    目標値ごとに必要な alpha の分布を計算し，CSVとして保存する．

    引数:
        seed (int): 比較基準を読み込む実験2のseed (比較基準はseedに依存しない量のみを用いる)．
    戻り値:
        なし．
    """
    source_dir = os.path.join(common.OUTPUT_DIR, "ex2_displacement", f"K{common.NUM_SAMPLES}", str(seed))
    baselines = np.load(os.path.join(source_dir, "per_image_baselines.npz"))
    rows = []
    for target in TARGET_COSINES:
        alpha = required_alpha(baselines["cos_v0_v"], baselines["v0_norm"], baselines["noise_out_rms"], target)
        row = {"target_cos_v": target, "frac_already_below_at_alpha0": float((alpha == 0).mean())}
        row.update(metrics.quantile_summary(alpha, "required_alpha"))
        row["required_d_latent_mean"] = float((alpha * baselines["std_norm"]).mean())
        rows.append(row)

    table = pd.DataFrame(rows)
    target_dir = common.condition_dir("ex2_required_alpha", "closed_form", seed)
    table.to_csv(os.path.join(target_dir, "required_alpha.csv"), index=False)
    common.save_json({"seed": seed, "target_cosines": TARGET_COSINES, "note": "closed-form prediction, not measured"},
                     os.path.join(target_dir, "meta.json"))
    print(table.round(3).to_string(index=False))




def within_image_spread(seed: int = 0) -> None:
    """
    cos(v_tilde, v) のばらつきを，同一画像内 (z の違いによる) と画像間に分解して保存する．
    同一画像内のばらつきは，同じ alpha のままサンプルを選び直すことで下げられる cos の幅の目安になる．

    引数:
        seed (int): 読み込む実験2のseed．
    戻り値:
        なし．
    """
    source_dir = os.path.join(common.OUTPUT_DIR, "ex2_displacement", f"K{common.NUM_SAMPLES}", str(seed))
    samples = np.load(os.path.join(source_dir, "per_sample_metrics.npz"))
    rows = []
    for alpha, cos_v in zip(samples["alphas"], samples["cos_v"]):
        rows.append({
            "alpha": float(alpha),
            "cos_v_mean": float(cos_v.mean()),
            "within_image_std_mean": float(cos_v.std(axis=1, ddof=1).mean()),
            "between_image_std": float(cos_v.mean(axis=1).std(ddof=1)),
            "min_of_K_samples_mean": float(cos_v.min(axis=1).mean()),
        })
    table = pd.DataFrame(rows)
    target_dir = common.condition_dir("ex2_required_alpha", "closed_form", seed)
    table.to_csv(os.path.join(target_dir, "within_image_spread.csv"), index=False)
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    run()
    within_image_spread()
