"""
本実験の結果の集計．

1. 画像での再識別率をそろえた比較表: 方式ごとの設定を，画像での再識別率 top-1 について線形補間する．
2. 方式間の対応のある比較: ベクトル空間での再識別率をそろえた3つの水準で，M2・M3u と M0 の差を
   画像ごとの差のブートストラップで評価する．
3. ノルムの補正の効果: 同じソースの顔ベクトルに対する，補正の有無の差．
4. ベクトル空間の評価と画像での評価の対応．

    python programs/ex003_ghost_image_eval/summarize_main.py
"""

import os
from typing import Dict, List

import numpy as np
import pandas as pd

import calibrate
import settings

MAIN_DIR = os.path.join(settings.OUTPUT_DIR, "main", f"n{settings.NUM_IMAGES}", str(settings.SEED))
METHODS = ["M0", "M2", "M3u"]
NORM_MODES = ["raw", "fixed"]
INTERPOLATED = ["param", "cos_gen_v", "reid_top10", "dex_gender_match", "dex_age_abs_diff", "dex_age_rmse",
                "lin_gender_agree", "lin_race_agree", "vec_cos_v", "vec_reid_top1", "detection_rate"]
#: 目標の再識別率が掃引の範囲からこの値以内で外れる場合は，端の設定の値を用いる
EDGE_TOLERANCE = 0.03
NUM_BOOTSTRAP = 2000


def load_summary() -> pd.DataFrame:
    """
    本実験の設定ごとの要約表を，顔検出の成功率と結合して読み込む．

    引数:
        なし．
    戻り値:
        summary (pd.DataFrame): 設定ごとに1行の要約表．
    """
    summary = pd.read_csv(os.path.join(MAIN_DIR, "summary.csv"))
    detection_path = os.path.join(MAIN_DIR, "detection.csv")
    if os.path.exists(detection_path):
        summary = summary.merge(pd.read_csv(detection_path)[["key", "detection_rate"]], on="key", how="left")
    else:
        summary["detection_rate"] = np.nan
    return summary


def load_per_image() -> Dict[str, Dict[str, np.ndarray]]:
    """
    画像ごとの指標を，設定の名前をkeyとする辞書として読み込む．

    引数:
        なし．
    戻り値:
        per_image (dict): `{設定の名前: {指標名: 形状 (n,) の配列}}`．
    """
    archive = np.load(os.path.join(MAIN_DIR, "per_image.npz"))
    per_image: Dict[str, Dict[str, np.ndarray]] = {}
    for name in archive.files:
        if "::" not in name:
            continue
        key, metric = name.split("::")
        per_image.setdefault(key, {})[metric] = archive[name]
    return per_image


def curve_rows(summary: pd.DataFrame, method: str, norm_mode: str) -> pd.DataFrame:
    """
    1つの方式・1つのノルムの扱いについて，再識別率の昇順に並べた設定の表を返す．
    M0 の曲線には，alpha=0 に相当する Recon を含める．

    引数:
        summary (pd.DataFrame): 設定ごとの要約表．
        method (str): 方式の記号．
        norm_mode (str): "raw" または "fixed"．
    戻り値:
        rows (pd.DataFrame): 再識別率 top-1 の昇順に並べた表．
    """
    methods = [method, "Recon"] if method == "M0" else [method]
    rows = summary[summary["method"].isin(methods) & (summary["norm_mode"] == norm_mode)].copy()
    rows.loc[rows["method"] == "Recon", "param"] = 0.0
    return rows.sort_values("reid_top1")


def matched_reid_table(summary: pd.DataFrame) -> pd.DataFrame:
    """
    画像での再識別率 top-1 が目標値になる点での各指標を，方式ごとに線形補間で求める．

    引数:
        summary (pd.DataFrame): 設定ごとの要約表．
    戻り値:
        table (pd.DataFrame): (目標値, 方式, ノルムの扱い) ごとに1行の表．
    """
    rows = []
    for target in settings.TARGET_REID:
        for method in METHODS:
            for norm_mode in NORM_MODES:
                curve = curve_rows(summary, method, norm_mode)
                x = curve["reid_top1"].to_numpy()
                row = {"target_reid_top1": target, "method": method, "norm_mode": norm_mode,
                       "extrapolated_from_edge": bool(target < x[0] or target > x[-1])}
                in_range = x[0] - EDGE_TOLERANCE <= target <= x[-1] + EDGE_TOLERANCE
                for metric in INTERPOLATED:
                    row[metric] = float(np.interp(target, x, curve[metric].to_numpy())) if in_range else float("nan")
                rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_difference(first: np.ndarray, second: np.ndarray, rng: np.random.Generator) -> dict:
    """
    対応のある2つの設定の指標の差 (first - second) の平均と，ブートストラップによる95%信頼区間を求める．

    引数:
        first (np.ndarray): 形状 (n,) の1つ目の設定の画像ごとの値．
        second (np.ndarray): 形状 (n,) の2つ目の設定の画像ごとの値．
        rng (np.random.Generator): 乱数生成器．
    戻り値:
        result (dict): diff (差の平均)，ci_low，ci_high (95%信頼区間)．
    """
    diff = first - second
    samples = diff[rng.integers(0, len(diff), size=(NUM_BOOTSTRAP, len(diff)))].mean(axis=1)
    return {"diff": float(diff.mean()), "ci_low": float(np.quantile(samples, 0.025)), "ci_high": float(np.quantile(samples, 0.975))}


def level_keys(norm_mode: str) -> Dict[float, Dict[str, str]]:
    """
    ベクトル空間での再識別率の目標値ごとに，較正で選んだ各方式の設定の名前を返す．

    引数:
        norm_mode (str): "raw" または "fixed"．
    戻り値:
        keys (dict): `{目標値: {方式: 設定の名前}}`．
    """
    chosen = calibrate.load_chosen()
    return {
        target: {row.method: f"{row.method}|{row.param_name}={row.param:g}|{norm_mode}"
                 for row in chosen[chosen["target_reid_top1"] == target].itertuples()}
        for target in settings.TARGET_REID
    }


def paired_method_table(per_image: Dict[str, Dict[str, np.ndarray]]) -> pd.DataFrame:
    """
    ベクトル空間での再識別率をそろえた3つの水準で，M2・M3u と M0 を画像ごとの対応のある差で比べる．

    引数:
        per_image (dict): `load_per_image` の戻り値．
    戻り値:
        table (pd.DataFrame): (水準, ノルムの扱い, 比較) ごとに1行の表．再識別率，性別一致率，
            年齢の差の絶対値のそれぞれについて，差の平均と95%信頼区間を含む．
    """
    rng = np.random.default_rng(settings.SEED)
    rows = []
    for norm_mode in NORM_MODES:
        for target, keys in level_keys(norm_mode).items():
            for method in ("M2", "M3u"):
                first, second = per_image[keys[method]], per_image[keys["M0"]]
                row = {"vector_reid_level": target, "norm_mode": norm_mode, "contrast": f"{method} - M0",
                       "first": keys[method], "second": keys["M0"]}
                for metric, values in (("reid_top1", lambda m: m["reid_top1"]),
                                       ("dex_gender_match", lambda m: m["dex_gender_match"]),
                                       ("dex_age_abs_diff", lambda m: np.abs(m["dex_age_diff"]))):
                    result = bootstrap_difference(values(first), values(second), rng)
                    row.update({f"{metric}_{name}": value for name, value in result.items()})
                rows.append(row)
    return pd.DataFrame(rows)


def norm_effect_table(summary: pd.DataFrame, per_image: Dict[str, Dict[str, np.ndarray]]) -> pd.DataFrame:
    """
    同じソースの顔ベクトルに対する，ノルムの補正の有無の差 (fixed - raw) を設定ごとに求める．

    引数:
        summary (pd.DataFrame): 設定ごとの要約表．
        per_image (dict): `load_per_image` の戻り値．
    戻り値:
        table (pd.DataFrame): 設定ごとに1行の表．
    """
    rng = np.random.default_rng(settings.SEED)
    rows = []
    for setting in summary.loc[summary["norm_mode"] == "fixed", "setting"]:
        raw, fixed = per_image[f"{setting}|raw"], per_image[f"{setting}|fixed"]
        row = {"setting": setting, "raw_norm": float(raw["vec_norm"].mean()), "fixed_norm": float(fixed["vec_norm"].mean())}
        for metric, values in (("cos_gen_v", lambda m: m["cos_gen_v"]), ("reid_top1", lambda m: m["reid_top1"]),
                               ("dex_gender_match", lambda m: m["dex_gender_match"]),
                               ("dex_age_abs_diff", lambda m: np.abs(m["dex_age_diff"]))):
            result = bootstrap_difference(values(fixed), values(raw), rng)
            row.update({f"{metric}_{name}": value for name, value in result.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def correspondence_table(summary: pd.DataFrame, per_image: Dict[str, Dict[str, np.ndarray]]) -> pd.DataFrame:
    """
    ベクトル空間での評価と画像での評価の対応を，設定ごとにまとめる．

    引数:
        summary (pd.DataFrame): 設定ごとの要約表．
        per_image (dict): `load_per_image` の戻り値．
    戻り値:
        table (pd.DataFrame): 設定ごとに1行の表．類似度と再識別率のベクトル空間・画像での値，画像ごとの
            類似度の相関，再識別の成否の一致率，線形分類器とDEXの性別の予測の一致率を含む．
    """
    rows = []
    for row in summary.itertuples():
        metrics = per_image[row.key]
        rows.append({
            "key": row.key, "vec_cos_v": row.vec_cos_v, "cos_gen_v": row.cos_gen_v,
            "cos_ratio": row.cos_gen_v / row.vec_cos_v if row.vec_cos_v > 0.05 else float("nan"),
            "corr_per_image": row.corr_vec_cos_vs_gen_cos,
            "vec_reid_top1": row.vec_reid_top1, "reid_top1": row.reid_top1,
            "reid_agreement_per_image": float((metrics["vec_reid_top1"] == metrics["reid_top1"]).mean()),
            "lin_on_vec_vs_dex_gender": row.lin_vec_vs_dex_gender_agree,
            "lin_on_gen_vs_dex_gender": row.lin_gen_vs_dex_gender_agree,
            "lin_gender_agree": row.lin_gender_agree, "dex_gender_match": row.dex_gender_match,
        })
    return pd.DataFrame(rows)


def run() -> None:
    """
    集計を実行し，結果をCSVとして保存する．

    引数:
        なし．
    戻り値:
        なし．
    """
    target_dir = settings.condition_dir("main_analysis", f"n{settings.NUM_IMAGES}", settings.SEED)
    summary, per_image = load_summary(), load_per_image()
    tables = {
        "matched_image_reid": matched_reid_table(summary),
        "paired_methods": paired_method_table(per_image),
        "norm_effect": norm_effect_table(summary, per_image),
        "correspondence": correspondence_table(summary, per_image),
    }
    for name, table in tables.items():
        table.to_csv(os.path.join(target_dir, f"{name}.csv"), index=False)
    pooled_vec = np.concatenate([m["vec_cos_v"] for k, m in per_image.items() if not k.startswith("Self")])
    pooled_gen = np.concatenate([m["cos_gen_v"] for k, m in per_image.items() if not k.startswith("Self")])
    slope, intercept = np.polyfit(pooled_vec, pooled_gen, 1)
    settings.save_json({
        "pooled_corr_vec_cos_vs_gen_cos": float(np.corrcoef(pooled_vec, pooled_gen)[0, 1]),
        "pooled_fit_slope": float(slope), "pooled_fit_intercept": float(intercept),
        "setting_level_corr_reid": float(np.corrcoef(summary["vec_reid_top1"], summary["reid_top1"])[0, 1]),
    }, os.path.join(target_dir, "meta.json"))

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    for name, table in tables.items():
        print(f"\n== {name} ==\n" + table.round(3).to_string(index=False))
    print(settings.load_json(os.path.join(target_dir, "meta.json")))


if __name__ == "__main__":
    run()
