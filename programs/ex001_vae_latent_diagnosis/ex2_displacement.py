"""
実験2: 潜在空間での移動量と出力の変化．

各画像・各 alpha・各サンプルについて，潜在空間での移動量 d_latent = ||alpha * std ⊙ z|| と，
デコード結果 v_tilde = decoder(mu + alpha * std ⊙ z) の元の顔ベクトル v に対するコサイン類似度を
計算する．比較基準として，mu 空間での他画像との最近傍距離と，ArcFace空間での他画像との
コサイン類似度 (最近傍・ランダム) を求める．

    python programs/ex001_vae_latent_diagnosis/ex2_displacement.py
"""

import os
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

import common
import metrics
from machine_learning_utils import set_seed

METHOD = "ex2_displacement"
HYPERPARAMETER = f"K{common.NUM_SAMPLES}"

#: mu 空間での最近傍距離がこの値以下の画像は，完全に重複した画像とみなす
DUPLICATE_DISTANCE = 1e-4
#: ArcFace空間での最近傍の cos がこの値以上の画像は，ほぼ同一の画像が存在するとみなす
NEAR_DUPLICATE_COSINE = 0.9

#: サンプルごとに保存する指標 (形状 (alphaの数, N, K))
SAMPLE_METRICS = ["d_latent", "d_latent_over_mu_norm", "d_latent_over_nn_dist", "d_out", "cos_v", "cos_dec"]


@torch.no_grad()
def compute_baselines(model: torch.nn.Module, v: torch.Tensor, mu: torch.Tensor, std: torch.Tensor, seed: int) -> Dict[str, torch.Tensor]:
    """
    サンプリングに依存しない，画像ごとの比較基準を計算する．

    引数:
        model (torch.nn.Module): 学習済みのVAE．
        v (torch.Tensor): 形状 (N, 512) のArcFaceの顔ベクトル．
        mu (torch.Tensor): 形状 (N, d) の潜在変数の平均．
        std (torch.Tensor): 形状 (N, d) の潜在変数の標準偏差．
        seed (int): ランダムな他画像の割り当てに用いる乱数シード．
    戻り値:
        baselines (dict): いずれも形状 (N,) のテンソルを値とする辞書．
            mu_norm: ||mu||，std_norm: ||std||，
            mu_nn_dist / mu_rand_dist: mu 空間での最近傍 / ランダムな他画像までの距離，
            v_norm: ||v||，v0_norm: ||decoder(mu)||，cos_v0_v: cos(decoder(mu), v)，
            arc_nn_cos / arc_rand_cos: ArcFace空間での最近傍 / ランダムな他画像との cos，
            dec_nn_cos / dec_rand_cos: decoder(mu) 同士での最近傍 / ランダムな他画像との cos，
            noise_out_rms: alpha=1 のときの出力の移動量の二乗平均平方根 (線形デコーダから解析的に計算)．
    """
    v0 = model.decode(mu)
    other = metrics.random_other_index(v.shape[0], seed)
    column_sq_norm = model.decoder[0].weight.pow(2).sum(dim=0)
    return {
        "mu_norm": mu.norm(dim=1),
        "std_norm": std.norm(dim=1),
        "mu_nn_dist": metrics.nearest_other_distance(mu),
        "mu_rand_dist": (mu - mu[other]).norm(dim=1),
        "v_norm": v.norm(dim=1),
        "v0_norm": v0.norm(dim=1),
        "cos_v0_v": F.cosine_similarity(v0, v, dim=1),
        "arc_nn_cos": metrics.nearest_other_cosine(v),
        "arc_rand_cos": F.cosine_similarity(v, v[other], dim=1),
        "dec_nn_cos": metrics.nearest_other_cosine(v0),
        "dec_rand_cos": F.cosine_similarity(v0, v0[other], dim=1),
        "noise_out_rms": (std.pow(2) * column_sq_norm).sum(dim=1).sqrt(),
    }


@torch.no_grad()
def compute_sample_metrics(model: torch.nn.Module, v: torch.Tensor, mu: torch.Tensor, std: torch.Tensor,
                           baselines: Dict[str, torch.Tensor], alpha: float, generator: torch.Generator) -> Dict[str, np.ndarray]:
    """
    1つの alpha について，全画像・全サンプルの指標を計算する．

    引数:
        model (torch.nn.Module): 学習済みのVAE．
        v (torch.Tensor): 形状 (N, 512) のArcFaceの顔ベクトル．
        mu (torch.Tensor): 形状 (N, d) の潜在変数の平均．
        std (torch.Tensor): 形状 (N, d) の潜在変数の標準偏差．
        baselines (dict): `compute_baselines` の戻り値．
        alpha (float): 標準偏差に乗じるスケール係数．
        generator (torch.Generator): 乱数生成器．
    戻り値:
        sample_metrics (dict): `SAMPLE_METRICS` の各指標をkey，形状 (N, K) の配列をvalueとする辞書．
    """
    z = metrics.sample_standard_normal(mu.shape[0], common.NUM_SAMPLES, mu.shape[1], generator)
    displacement = alpha * std.unsqueeze(1) * z
    v0 = model.decode(mu)
    v_tilde = metrics.decode_displaced(model, mu, displacement)

    d_latent = displacement.norm(dim=2)
    # 完全に重複した画像では最近傍距離が0になり比を定義できないため，NaNとして集計から除外する
    nn_dist = baselines["mu_nn_dist"].unsqueeze(1)
    nn_dist = torch.where(nn_dist > DUPLICATE_DISTANCE, nn_dist, torch.full_like(nn_dist, float("nan")))
    result = {
        "d_latent": d_latent,
        "d_latent_over_mu_norm": d_latent / baselines["mu_norm"].unsqueeze(1),
        "d_latent_over_nn_dist": d_latent / nn_dist,
        "d_out": (v_tilde - v0.unsqueeze(1)).norm(dim=2),
        "cos_v": metrics.cosine_to_reference(v_tilde, v),
        "cos_dec": metrics.cosine_to_reference(v_tilde, v0),
    }
    return {name: value.cpu().numpy() for name, value in result.items()}


def summarize_alpha(sample_metrics: Dict[str, np.ndarray], baselines: Dict[str, np.ndarray],
                    alpha: float, mask: np.ndarray, subset: str) -> dict:
    """
    1つの alpha・1つの部分集合について，指標の平均・分位点と比較基準に対する到達率を1行にまとめる．

    引数:
        sample_metrics (dict): 形状 (N, K) の配列を値とする指標の辞書．
        baselines (dict): 形状 (N,) の配列を値とする比較基準の辞書．
        alpha (float): 標準偏差に乗じるスケール係数．
        mask (np.ndarray): 形状 (N,) のbool配列．集計対象の画像が True．
        subset (str): 部分集合の名前 ("all"，"train"，"test")．
    戻り値:
        row (dict): 要約表の1行分の辞書．
    """
    row = {"subset": subset, "alpha": alpha, "num_images": int(mask.sum())}
    for name in SAMPLE_METRICS:
        row.update(metrics.quantile_summary(sample_metrics[name][mask], name))

    cos_v = sample_metrics["cos_v"][mask]
    d_latent = sample_metrics["d_latent"][mask]
    selected = {name: value[mask] for name, value in baselines.items()}
    arc_rand_q95 = np.quantile(baselines["arc_rand_cos"], 0.95)

    # 潜在空間・出力空間のそれぞれで，「他人」の基準に到達したサンプルの割合
    row["frac_d_latent_gt_nn_dist"] = float((d_latent > selected["mu_nn_dist"][:, None]).mean())
    row["frac_d_latent_gt_rand_dist"] = float((d_latent > selected["mu_rand_dist"][:, None]).mean())
    row["frac_cos_v_lt_arc_nn_cos"] = float((cos_v < selected["arc_nn_cos"][:, None]).mean())
    row["frac_cos_v_lt_arc_rand_q95"] = float((cos_v < arc_rand_q95).mean())

    # 線形デコーダと，ノイズと信号の直交性を仮定した解析的な予測値
    noise_sq = (alpha * selected["noise_out_rms"]) ** 2
    shrink = selected["v0_norm"] / np.sqrt(selected["v0_norm"] ** 2 + noise_sq)
    row["pred_d_out_rms"] = float(np.sqrt(noise_sq.mean()))
    row["pred_cos_dec_mean"] = float(shrink.mean())
    row["pred_cos_v_mean"] = float((selected["cos_v0_v"] * shrink).mean())
    row["gain_d_out_over_d_latent_mean"] = (
        float((sample_metrics["d_out"][mask] / d_latent).mean()) if alpha > 0 else float("nan")
    )
    return row


def run_seed(seed: int) -> None:
    """
    1つのseedについて実験2を実行し，サンプルごとの指標・比較基準・要約表を保存する．

    引数:
        seed (int): 乱数シード．
    戻り値:
        なし．
    """
    target_dir = common.condition_dir(METHOD, HYPERPARAMETER, seed)
    if common.is_finished(target_dir):
        print(f"実行済みのためスキップ: {target_dir}")
        return

    model, v, mu, std, is_test = metrics.load_inputs()
    set_seed(seed)
    generator = metrics.make_generator(seed)

    baselines = {name: value.cpu().numpy() for name, value in compute_baselines(model, v, mu, std, seed).items()}
    baselines_gpu = {name: torch.from_numpy(value).cuda() for name, value in baselines.items()}
    subsets = {"all": np.ones_like(is_test), "train": ~is_test, "test": is_test}

    stacked = {name: [] for name in SAMPLE_METRICS}
    rows = []
    for alpha in tqdm(common.ALPHAS, desc=f"ex2 seed={seed}", leave=False):
        sample_metrics = compute_sample_metrics(model, v, mu, std, baselines_gpu, alpha, generator)
        for name in SAMPLE_METRICS:
            stacked[name].append(sample_metrics[name].astype(np.float32))
        rows.extend(summarize_alpha(sample_metrics, baselines, alpha, mask, subset) for subset, mask in subsets.items())

    summary = pd.DataFrame(rows).sort_values(["subset", "alpha"]).reset_index(drop=True)
    summary.to_csv(os.path.join(target_dir, "summary_by_alpha.csv"), index=False)

    baseline_rows = [{"quantity": name, **metrics.quantile_summary(value, "value")} for name, value in baselines.items()]
    pd.DataFrame(baseline_rows).to_csv(os.path.join(target_dir, "baselines_summary.csv"), index=False)

    np.savez_compressed(os.path.join(target_dir, "per_image_baselines.npz"), is_test=is_test, **baselines)
    np.savez_compressed(
        os.path.join(target_dir, "per_sample_metrics.npz"),
        alphas=np.array(common.ALPHAS), **{name: np.stack(values) for name, values in stacked.items()},
    )
    common.save_json({
        "seed": seed, "alphas": common.ALPHAS, "num_samples": common.NUM_SAMPLES,
        "num_images": int(v.shape[0]), "latent_dim": int(mu.shape[1]),
        "num_test_images": int(is_test.sum()),
        "num_images_with_duplicate_mu": int((baselines["mu_nn_dist"] <= DUPLICATE_DISTANCE).sum()),
        "num_images_with_near_duplicate_face": int((baselines["arc_nn_cos"] >= NEAR_DUPLICATE_COSINE).sum()),
    }, os.path.join(target_dir, "meta.json"))

    columns = ["alpha", "d_latent_mean", "d_latent_over_nn_dist_q50", "frac_d_latent_gt_nn_dist", "cos_v_mean", "cos_dec_mean", "pred_cos_v_mean"]
    print(f"\n[seed={seed}]\n" + summary[summary["subset"] == "all"][columns].to_string(index=False))


def run() -> None:
    """
    `common.SEEDS` の全てのseedについて実験2を実行する．

    引数:
        なし．
    戻り値:
        なし．
    """
    for seed in common.SEEDS:
        run_seed(seed)


if __name__ == "__main__":
    run()
