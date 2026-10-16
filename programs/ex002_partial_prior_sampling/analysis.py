"""
実験Bの結果の集計．

1. 同じ類似度での比較表: 方式ごとの掃引点を cos(v_tilde, v) の平均について線形補間し，
   目標の類似度における各指標の値を求める (seedごと，およびseed間の平均と標準偏差)．
2. 提案方式と対照の差: 同じ類似度における M2 - M2r，M3 - M3u，M2 - M0 などの差．
3. 閉形式の精度: cos(v_tilde, v) の実測と閉形式による予測の差．
4. 分類器の重みの分解: 属性分類器の重みのうち，デコーダの列空間 (潜在空間の方式のノイズが
   存在しうる128次元の部分空間) に含まれる割合．等方的なノイズに対する分類器の感度の説明に用いる．

    python programs/ex002_partial_prior_sampling/analysis.py
"""

import os
from typing import List

import numpy as np
import pandas as pd
import torch

import common
from classifiers import VARIANTS, load_predictor
from data import load_dataset
from model import encode_all, load_model

METHOD = "expB_analysis"
SOURCE_METHOD = "expB_sampling"
HYPERPARAMETER = f"K{common.NUM_SAMPLES}"

CURVES = ["M0", "M1", "M1b", "M2", "M2r", "M3_k16", "M3_k32", "M3_k64", "M3u"]
METRICS = ["reid_top1", "reid_top10", "gender_agree", "race_agree", "age_diff", "gender_acc", "race_acc", "age_err",
           "l2_gender_agree", "l2_race_agree", "l2_age_diff", "norm", "nn_cos", "diversity_cos", "cos_v0"]
#: 同じ再識別率での比較表に用いる top-1 再識別率の目標値
TARGET_REID = [0.9, 0.5, 0.1]
#: (提案または比較する方式, 対照) の組
CONTRASTS = [("M2", "M2r"), ("M2", "M3u"), ("M3_k16", "M3u"), ("M3_k32", "M3u"), ("M3_k64", "M3u"),
             ("M2", "M0"), ("M3_k32", "M0"), ("M3u", "M0")]


def load_summary(seed: int) -> pd.DataFrame:
    """
    1つのseedの実験Bの要約表を読み込む．

    引数:
        seed (int): 乱数シード．
    戻り値:
        summary (pd.DataFrame): 設定ごとに1行の要約表．
    """
    return pd.read_csv(os.path.join(common.OUTPUT_DIR, SOURCE_METHOD, HYPERPARAMETER, str(seed), "summary.csv"))


def interpolate_curve(curve_rows: pd.DataFrame, target: float, metric: str) -> float:
    """
    1本の曲線について，cos(v_tilde, v) の平均が目標値になる点での指標の値を線形補間で求める．

    引数:
        curve_rows (pd.DataFrame): 1本の曲線に属する設定の要約表．
        target (float): cos(v_tilde, v) の目標値．
        metric (str): 指標の名前 (`{metric}_mean` の列を用いる)．
    戻り値:
        value (float): 補間した値．目標値が掃引の範囲外の場合は NaN．
    """
    ordered = curve_rows.sort_values("cos_v_mean")
    x, y = ordered["cos_v_mean"].to_numpy(), ordered[f"{metric}_mean"].to_numpy()
    if target < x[0] or target > x[-1]:
        return float("nan")
    return float(np.interp(target, x, y))


def matched_table(summary: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    1つのseedについて，同じ類似度での比較表を作る．

    引数:
        summary (pd.DataFrame): 実験Bの要約表．
        seed (int): 乱数シード．
    戻り値:
        table (pd.DataFrame): (目標の類似度, 曲線) ごとに1行，各指標を列とする表．
    """
    rows = []
    for target in common.TARGET_COSINES:
        for curve in CURVES:
            curve_rows = summary[summary["curve"] == curve]
            row = {"seed": seed, "target_cos_v": target, "curve": curve}
            row.update({metric: interpolate_curve(curve_rows, target, metric) for metric in METRICS})
            row["param_name"] = curve_rows["param_name"].iloc[0]
            ordered = curve_rows.sort_values("cos_v_mean")
            in_range = ordered["cos_v_mean"].iloc[0] <= target <= ordered["cos_v_mean"].iloc[-1]
            row["param_at_target"] = (
                float(np.interp(target, ordered["cos_v_mean"], ordered["param"])) if in_range else float("nan")
            )
            rows.append(row)
    return pd.DataFrame(rows)


def matched_reid_table(summary: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    1つのseedについて，同じ再識別率 (top-1) での比較表を作る．横軸を cos(v_tilde, v) ではなく
    再識別率に取り替えた `matched_table` に相当する．

    引数:
        summary (pd.DataFrame): 実験Bの要約表．
        seed (int): 乱数シード．
    戻り値:
        table (pd.DataFrame): (目標の再識別率, 曲線) ごとに1行，各指標を列とする表．
    """
    metrics = ["cos_v", "gender_agree", "race_agree", "age_diff", "l2_gender_agree", "l2_race_agree", "l2_age_diff",
               "norm", "nn_cos", "diversity_cos"]
    rows = []
    for target in TARGET_REID:
        for curve in CURVES:
            ordered = summary[summary["curve"] == curve].sort_values("cos_v_mean")
            x = ordered["reid_top1_mean"].to_numpy()
            row = {"seed": seed, "target_reid_top1": target, "curve": curve}
            # 再識別率は類似度に対して単調に増加するため，目標値を初めて超える区間で補間する
            above = np.nonzero(x >= target)[0]
            in_range = len(above) > 0 and above[0] > 0
            for metric in metrics:
                y = ordered[f"{metric}_mean"].to_numpy()
                row[metric] = float(np.interp(target, x[above[0] - 1:above[0] + 1], y[above[0] - 1:above[0] + 1])) if in_range else float("nan")
            rows.append(row)
    return pd.DataFrame(rows)


def contrast_table(matched: pd.DataFrame) -> pd.DataFrame:
    """
    同じ類似度における方式間の差を，seedごとに計算してからseed間の平均と標準偏差にまとめる．

    引数:
        matched (pd.DataFrame): 全seedの `matched_table` を連結した表．
    戻り値:
        table (pd.DataFrame): (目標の類似度, 方式の組) ごとに1行の表．
    """
    metrics = ["reid_top1", "gender_agree", "race_agree", "age_diff", "l2_gender_agree", "l2_race_agree", "l2_age_diff"]
    indexed = matched.set_index(["seed", "target_cos_v", "curve"])
    rows = []
    for target in common.TARGET_COSINES:
        for first, second in CONTRASTS:
            diffs = pd.DataFrame([
                indexed.loc[(seed, target, first), metrics] - indexed.loc[(seed, target, second), metrics]
                for seed in common.SEEDS
            ]).astype(float)
            row = {"target_cos_v": target, "contrast": f"{first} - {second}"}
            for metric in metrics:
                row[f"{metric}_diff_mean"] = float(diffs[metric].mean())
                row[f"{metric}_diff_std"] = float(diffs[metric].std(ddof=0))
            rows.append(row)
    return pd.DataFrame(rows)


def closed_form_error(summaries: List[pd.DataFrame]) -> pd.DataFrame:
    """
    方式ごとに，cos(v_tilde, v)・cos(v_tilde, v0)・ノルムの実測と閉形式による予測の差の最大絶対値を求める．

    引数:
        summaries (list[pd.DataFrame]): 各seedの実験Bの要約表．
    戻り値:
        table (pd.DataFrame): 方式ごとに1行の表．
    """
    merged = pd.concat(summaries)
    merged = merged[merged["method"] != "REF"].copy()
    merged["err_cos_v"] = (merged["cos_v_mean"] - merged["pred_cos_v"]).abs()
    merged["err_cos_v0"] = (merged["cos_v0_mean"] - merged["pred_cos_v0"]).abs()
    merged["rel_err_norm"] = ((merged["norm_mean"] - merged["pred_norm"]) / merged["norm_mean"]).abs()
    return merged.groupby("method", sort=False)[["err_cos_v", "err_cos_v0", "rel_err_norm"]].max().reset_index()


@torch.no_grad()
def classifier_weight_decomposition() -> pd.DataFrame:
    """
    属性分類器の各重みベクトルについて，デコーダの列空間に含まれる二乗ノルムの割合と，
    等方的なノイズおよび列空間内のノイズに対する予測値の感度を計算する．

    潜在空間の方式のノイズは全てデコーダの列空間 (128次元) に含まれる．一方，等方的なノイズは
    512次元の全方向に広がる．重みの大半が列空間の外にあれば，分類器は等方的なノイズにだけ強く反応する．

    引数:
        なし．
    戻り値:
        table (pd.DataFrame): (分類器の種類, 属性) ごとに1行の表．
            frac_in_decoder_span: 重みの二乗ノルムのうち列空間に含まれる割合，
            score_std_real: 評価用データの実ベクトルに対する判別スコアの標準偏差，
            score_std_iso_tau1: v と同じノルムの等方的なノイズ (tau = 1) によるスコアの標準偏差，
            score_std_m0_alpha1: M0 (alpha = 1) のノイズによるスコアの標準偏差．
    """
    dataset = load_dataset()
    model = load_model()
    evaluation = dataset.is_eval
    v = torch.from_numpy(dataset.v[evaluation]).cuda().double()
    _, std = encode_all(model, dataset.v[evaluation])
    decoder_weight = model.decoder[0].weight.double()
    basis = torch.linalg.qr(decoder_weight).Q
    mean_sq_norm = float(v.pow(2).sum(dim=1).mean())
    mean_noise_cov = (decoder_weight * std.double().pow(2).mean(dim=0)) @ decoder_weight.T

    rows = []
    for variant in VARIANTS:
        predictor = load_predictor(variant)
        for name, weight in predictor.weights.items():
            # 多クラスの場合は，クラス間の差を表す中心化した重みを用いる
            centered = weight - weight.mean(dim=0, keepdim=True) if weight.shape[0] > 1 else weight
            inside = (centered @ basis).pow(2).sum()
            scores = (torch.nn.functional.normalize(v, dim=1) if predictor.normalize else v) @ centered.T
            scale = 1.0 / mean_sq_norm if predictor.normalize else 1.0
            rows.append({
                "variant": variant, "attribute": name,
                "weight_norm": float(centered.norm()),
                "frac_in_decoder_span": float(inside / centered.pow(2).sum()),
                "score_std_real": float(scores.std(dim=0).mean()),
                "score_std_iso_tau1": float((centered.pow(2).sum(dim=1) * mean_sq_norm / 512 * scale).sqrt().mean()),
                "score_std_m0_alpha1": float((((centered @ mean_noise_cov) * centered).sum(dim=1) * scale).sqrt().mean()),
            })
    return pd.DataFrame(rows)


def run() -> None:
    """
    実験Bの集計を実行し，結果をCSVとして保存する．

    引数:
        なし．
    戻り値:
        なし．
    """
    target_dir = common.condition_dir(METHOD, HYPERPARAMETER, 0)
    summaries = [load_summary(seed) for seed in common.SEEDS]
    matched = pd.concat([matched_table(summary, seed) for summary, seed in zip(summaries, common.SEEDS)])
    grouped = matched.drop(columns=["seed", "param_name"]).groupby(["target_cos_v", "curve"], sort=False)
    matched_mean = grouped.mean().reset_index()
    matched_std = grouped.std(ddof=0).reset_index()

    matched.to_csv(os.path.join(target_dir, "matched_cosine_per_seed.csv"), index=False)
    matched_mean.to_csv(os.path.join(target_dir, "matched_cosine_seed_mean.csv"), index=False)
    matched_std.to_csv(os.path.join(target_dir, "matched_cosine_seed_std.csv"), index=False)
    matched_reid = pd.concat([matched_reid_table(summary, seed) for summary, seed in zip(summaries, common.SEEDS)])
    matched_reid.to_csv(os.path.join(target_dir, "matched_reid_per_seed.csv"), index=False)
    reid_grouped = matched_reid.drop(columns=["seed"]).groupby(["target_reid_top1", "curve"], sort=False)
    reid_grouped.mean().reset_index().to_csv(os.path.join(target_dir, "matched_reid_seed_mean.csv"), index=False)
    reid_grouped.std(ddof=0).reset_index().to_csv(os.path.join(target_dir, "matched_reid_seed_std.csv"), index=False)
    contrasts = contrast_table(matched)
    contrasts.to_csv(os.path.join(target_dir, "contrasts.csv"), index=False)
    errors = closed_form_error(summaries)
    errors.to_csv(os.path.join(target_dir, "closed_form_error.csv"), index=False)
    decomposition = classifier_weight_decomposition()
    decomposition.to_csv(os.path.join(target_dir, "classifier_weight_decomposition.csv"), index=False)
    common.save_json({"seeds": common.SEEDS, "targets": common.TARGET_COSINES, "curves": CURVES},
                     os.path.join(target_dir, "meta.json"))

    pd.set_option("display.width", 250)
    seed0 = matched[matched["seed"] == 0]
    columns = ["target_cos_v", "curve", "param_at_target", "reid_top1", "reid_top10", "gender_agree", "race_agree",
               "age_diff", "l2_gender_agree", "l2_race_agree", "l2_age_diff", "norm", "nn_cos", "diversity_cos"]
    print(seed0[columns].round(3).to_string(index=False))
    print(matched_reid[matched_reid["seed"] == 0].round(3).to_string(index=False))
    print(contrasts.round(4).to_string(index=False))
    print(errors.round(5).to_string(index=False))
    print(decomposition.round(4).to_string(index=False))


if __name__ == "__main__":
    run()
