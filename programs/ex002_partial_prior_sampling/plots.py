"""
ex002_partial_prior_sampling の結果を可視化する関数をまとめたモジュール．

`outputs/ex002_partial_prior_sampling/` 以下に保存された結果を読み込み，複数のseedがある場合は
seed間の平均と標準偏差を計算して描画する．図は `outputs/ex002_partial_prior_sampling/figures/` に
保存する．ルートディレクトリの `visualize_result.ipynb` から呼び出して用いる．
"""

import os
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common

INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
SURFACE = "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"

#: 曲線ごとの描画スタイル．色は方式の系統を，破線と白抜きのマーカーは対照・比較対象を表す
CURVE_STYLES: Dict[str, dict] = {
    "M0": {"label": "M0  current (alpha scaling)", "color": BLUE, "linestyle": "-", "marker": "o", "filled": True},
    "M2": {"label": "M2  keep top-k dims, prior elsewhere", "color": ORANGE, "linestyle": "-", "marker": "s", "filled": True},
    "M2r": {"label": "M2r control: random k dims", "color": ORANGE, "linestyle": "--", "marker": "s", "filled": False},
    "M3_k16": {"label": "M3  k=16 (soft)", "color": YELLOW, "linestyle": "-", "marker": "v", "filled": True},
    "M3_k32": {"label": "M3  k=32 (soft)", "color": AQUA, "linestyle": "-", "marker": "D", "filled": True},
    "M3_k64": {"label": "M3  k=64 (soft)", "color": MAGENTA, "linestyle": "-", "marker": "^", "filled": True},
    "M3u": {"label": "M3u control: all dims soft", "color": AQUA, "linestyle": "--", "marker": "D", "filled": False},
    "M1": {"label": "M1  isotropic noise on v", "color": MUTED, "linestyle": "-", "marker": "x", "filled": True},
    "M1b": {"label": "M1b isotropic noise on v0", "color": MUTED, "linestyle": "--", "marker": "+", "filled": True},
}
MAIN_CURVES = ["M0", "M2", "M2r", "M3_k32", "M3u", "M1", "M1b"]
M3_CURVES = ["M2", "M3_k16", "M3_k32", "M3_k64", "M3u"]

FIGURE_DIR = os.path.join(common.OUTPUT_DIR, "figures")


def apply_style() -> None:
    """
    全ての図に共通する描画スタイルを設定する．

    引数:
        なし．
    戻り値:
        なし．
    """
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": SECONDARY_INK, "ytick.color": SECONDARY_INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
        "lines.linewidth": 2.0, "lines.markersize": 6, "font.size": 10,
        "axes.titlesize": 11, "legend.frameon": False, "legend.fontsize": 8.5,
    })


def save_figure(fig: plt.Figure, name: str) -> str:
    """
    図をPNGとして `FIGURE_DIR` に保存して閉じる．

    引数:
        fig (plt.Figure): 保存する図．
        name (str): 拡張子を除いたファイル名．
    戻り値:
        path (str): 保存したPNGのパス．
    """
    os.makedirs(FIGURE_DIR, exist_ok=True)
    path = os.path.join(FIGURE_DIR, f"{name}.png")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def load_expb(seeds: List[int], hyperparameter: str) -> Dict[str, pd.DataFrame]:
    """
    実験Bの要約表を複数のseedについて読み込み，設定ごとにseed間の平均と標準偏差を計算する．

    引数:
        seeds (list[int]): 集計するseedのリスト．
        hyperparameter (str): サンプル数を表す文字列 (例: "K10")．
    戻り値:
        result (dict): "mean" と "std" をkeyとし，設定ごとに1行の表をvalueとする辞書．
    """
    frames = [
        pd.read_csv(os.path.join(common.OUTPUT_DIR, "expB_sampling", hyperparameter, str(seed), "summary.csv"))
        for seed in seeds
    ]
    keys = ["setting", "method", "curve", "param_name", "param"]
    grouped = pd.concat(frames).groupby(keys, sort=False)
    return {"mean": grouped.mean(numeric_only=True).reset_index(), "std": grouped.std(numeric_only=True, ddof=0).reset_index()}


def draw_curves(ax: plt.Axes, data: Dict[str, pd.DataFrame], curves: List[str], x: str, y: str) -> None:
    """
    方式ごとの掃引点を1本の曲線として描画する．誤差棒はseed間の標準偏差である．

    引数:
        ax (plt.Axes): 描画先の軸．
        data (dict): `load_expb` の戻り値．
        curves (list[str]): 描画する曲線の名前のリスト．
        x (str): 横軸にとる指標の名前 (`{x}_mean` の列を用いる)．
        y (str): 縦軸にとる指標の名前 (`{y}_mean` の列を用いる)．
    戻り値:
        なし．
    """
    for curve in curves:
        style = CURVE_STYLES[curve]
        order = data["mean"][data["mean"]["curve"] == curve].sort_values("cos_v_mean").index
        mean, std = data["mean"].loc[order], data["std"].loc[order]
        ax.errorbar(
            mean[f"{x}_mean"], mean[f"{y}_mean"], yerr=std[f"{y}_mean"], label=style["label"], color=style["color"],
            linestyle=style["linestyle"], marker=style["marker"], linewidth=1.8, markersize=6,
            markerfacecolor=style["color"] if style["filled"] else SURFACE, markeredgewidth=1.5,
            ecolor=style["color"], elinewidth=1, capsize=2,
        )


def reference_line(ax: plt.Axes, value: float, label: str, below: bool = False) -> None:
    """
    基準値を表す水平線を，左端のラベルとともに描画する．

    引数:
        ax (plt.Axes): 描画先の軸．
        value (float): 水平線の縦軸の値．
        label (str): 線に付けるラベル．
        below (bool): True の場合，ラベルを線の下側に配置する．
    戻り値:
        なし．
    """
    ax.axhline(value, color=SECONDARY_INK, linestyle=":", linewidth=1.1)
    ax.text(0.01, value, f" {label}", color=SECONDARY_INK, ha="left", va="top" if below else "bottom",
            fontsize=8, transform=ax.get_yaxis_transform())


def tradeoff_figure(data: Dict[str, pd.DataFrame], curves: List[str], x: str, x_label: str, prefix: str,
                    title: str, name: str, age_limit: Optional[float] = 20.0) -> str:
    """
    トレードオフ曲線 (横軸: 匿名性の指標，縦軸: 性別の一致率・人種の一致率・年齢の予測差) を描画する．

    引数:
        data (dict): `load_expb` の戻り値．
        curves (list[str]): 描画する曲線の名前のリスト．
        x (str): 横軸にとる指標の名前．
        x_label (str): 横軸のラベル．
        prefix (str): 属性の指標の接頭辞 (正規化なしの分類器は ""，L2正規化ありは "l2_")．
        title (str): 図全体のタイトル．
        name (str): 保存するファイル名．
        age_limit (float または None): 年齢の予測差の縦軸の上限．None の場合は自動．
    戻り値:
        path (str): 保存した図のパス．
    """
    reference = data["mean"].set_index("setting")
    ceiling, floor = reference.loc["M0|alpha=0"], reference.loc["REF_random_real|none=0"]
    panels = [
        (f"{prefix}gender_agree", "gender agreement with prediction on v", (0.45, 0.82)),
        (f"{prefix}race_agree", "race agreement with prediction on v", (0.2, 0.75)),
        (f"{prefix}age_diff", "|predicted age difference| [years]  (lower is better)", (6, age_limit) if age_limit else None),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5))
    for ax, (metric, y_label, limits) in zip(axes, panels):
        draw_curves(ax, data, curves, x, metric)
        if limits:
            ax.set_ylim(*limits)
        reference_line(ax, ceiling[f"{metric}_mean"], "reconstruction only (alpha = 0)", below=metric.endswith("age_diff"))
        reference_line(ax, floor[f"{metric}_mean"], "random other real face", below=not metric.endswith("age_diff"))
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.invert_xaxis()
    axes[0].legend(loc="upper right")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return save_figure(fig, name)


def plot_expa() -> List[str]:
    """
    実験Aの図 (属性関連度の棒グラフと累積シェア，上位 k 次元での分類精度) を描画する．

    引数:
        なし．
    戻り値:
        paths (list[str]): 保存した図のパスのリスト．
    """
    apply_style()
    target = os.path.join(common.OUTPUT_DIR, "expA_attribute_relevance", "mu", "0")
    table = pd.read_csv(os.path.join(target, "relevance_all_images.csv")).sort_values("rank")
    probes = pd.read_csv(os.path.join(target, "probe_accuracy.csv"))
    summary = common.load_json(os.path.join(target, "summary.json"))
    positions = np.arange(1, len(table) + 1)
    paths = []

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].bar(positions, table["r_max"], width=0.8, color=BLUE)
    axes[0].set_xlabel("latent dimension (sorted by relevance, descending)")
    axes[0].set_ylabel("relevance r_j = max(eta^2 gender, race, age bin)")
    axes[0].set_title("Attribute relevance per latent dimension")
    cumulative = np.cumsum(table["r_max"]) / table["r_max"].sum()
    axes[1].plot(positions, cumulative, color=BLUE, label="cumulative share of r_j")
    axes[1].plot(positions, positions / len(table), color=MUTED, linestyle="--", linewidth=1.2, label="uniform (no concentration)")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_xlabel("number of top dimensions k")
    axes[1].set_ylabel("cumulative share")
    axes[1].set_title("Concentration of relevance")
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    paths.append(save_figure(fig, "fig_expA_relevance"))

    panels = [("gender_acc", "gender accuracy"), ("race_acc", "race accuracy"), ("age_mae", "age MAE [years]  (lower is better)")]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5))
    for ax, (metric, y_label) in zip(axes, panels):
        top = probes[probes["selection"] == "top"]
        random = probes[probes["selection"] == "random"]
        ax.plot(top["k"], top[metric], color=ORANGE, marker="s", label="top-k dims by relevance")
        ax.plot(random["k"], random[metric], color=ORANGE, marker="s", linestyle="--", markerfacecolor=SURFACE,
                markeredgewidth=1.5, label="random k dims (mean of 5 draws)")
        reference_line(ax, summary["majority_baseline"][metric], "majority / median baseline", below=(metric != "age_mae"))
        ax.set_xscale("log", base=2)
        ax.set_xticks(common.TOP_K_FOR_PROBE)
        ax.set_xticklabels([str(k) for k in common.TOP_K_FOR_PROBE])
        ax.set_xlabel("number of latent dimensions k used by the linear probe")
        ax.set_ylabel(y_label)
    axes[0].legend(loc="upper left")
    fig.suptitle("Linear probes on mu: are attributes aligned with a few axes?", fontsize=12)
    fig.tight_layout()
    paths.append(save_figure(fig, "fig_expA_probe_accuracy"))
    return paths


def plot_expb(seeds: List[int], hyperparameter: str = None) -> List[str]:
    """
    実験Bの図 (トレードオフ曲線，再識別率と類似度の関係，出力の自然さと多様性，閉形式の検証) を描画する．

    引数:
        seeds (list[int]): 平均と標準偏差の計算に用いるseedのリスト．
        hyperparameter (str): サンプル数を表す文字列．None の場合は既定のサンプル数．
    戻り値:
        paths (list[str]): 保存した図のパスのリスト．
    """
    apply_style()
    data = load_expb(seeds, hyperparameter or f"K{common.NUM_SAMPLES}")
    cos_label = "mean cos(v~, v)   (lower = more anonymous, to the right)"
    reid_label = "top-1 re-identification rate   (lower = more anonymous, to the right)"
    paths = [
        tradeoff_figure(data, MAIN_CURVES, "cos_v", cos_label, "", "Attribute retention vs. similarity to the original",
                        "fig_expB_tradeoff_cos"),
        tradeoff_figure(data, MAIN_CURVES, "reid_top1", reid_label, "", "Attribute retention vs. re-identification rate",
                        "fig_expB_tradeoff_reid"),
        tradeoff_figure(data, MAIN_CURVES, "cos_v", cos_label, "l2_",
                        "Attribute retention vs. similarity (classifiers on L2-normalized vectors)", "fig_expB_tradeoff_cos_l2"),
        tradeoff_figure(data, M3_CURVES, "cos_v", cos_label, "", "Soft variants (M3) vs. hard (M2) and control (M3u)",
                        "fig_expB_tradeoff_m3_family"),
    ]

    fig, ax = plt.subplots(figsize=(6.4, 5))
    draw_curves(ax, data, MAIN_CURVES, "cos_v", "reid_top1")
    ax.set_xlabel("mean cos(v~, v)")
    ax.set_ylabel("top-1 re-identification rate")
    ax.set_title("Re-identification stays near 1 until cos(v~, v) < 0.5")
    ax.invert_xaxis()
    ax.legend(loc="lower left")
    fig.tight_layout()
    paths.append(save_figure(fig, "fig_expB_reid_vs_cos"))

    real_norm = data["mean"].set_index("setting").loc["M1|tau=0", "norm_mean"]
    panels = [("norm", "mean ||v~||", (0, 60)), ("nn_cos", "cos to the nearest real vector", (0, 1.02)),
              ("diversity_cos", "cos between two samples of the same image", (0, 1.02))]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5))
    for ax, (metric, y_label, limits) in zip(axes, panels):
        draw_curves(ax, data, MAIN_CURVES, "cos_v", metric)
        ax.set_ylim(*limits)
        ax.set_xlabel("mean cos(v~, v)")
        ax.set_ylabel(y_label)
        ax.invert_xaxis()
    reference_line(axes[0], real_norm, "real ArcFace vectors (mean)")
    axes[0].legend(loc="upper left")
    fig.suptitle("Naturalness and diversity of the generated vectors", fontsize=12)
    fig.tight_layout()
    paths.append(save_figure(fig, "fig_expB_naturalness_diversity"))

    fig, ax = plt.subplots(figsize=(5.6, 5))
    table = data["mean"][data["mean"]["method"] != "REF"]
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1.2, linestyle="--")
    for curve in ["M0", "M2", "M2r", "M3_k16", "M3_k32", "M3_k64", "M3u", "M1", "M1b"]:
        style, rows = CURVE_STYLES[curve], table[table["curve"] == curve]
        ax.scatter(rows["pred_cos_v"], rows["cos_v_mean"], s=42, marker=style["marker"], color=style["color"],
                   facecolor=style["color"] if style["filled"] else SURFACE, linewidth=1.5, label=curve, zorder=3)
    ax.set_xlabel("closed-form prediction of cos(v~, v)")
    ax.set_ylabel("measured mean cos(v~, v)")
    ax.set_title("Closed form vs. measurement (all settings)")
    ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    paths.append(save_figure(fig, "fig_expB_closed_form"))
    return paths
