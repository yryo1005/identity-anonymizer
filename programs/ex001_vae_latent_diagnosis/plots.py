"""
ex001_vae_latent_diagnosis の結果を可視化する関数をまとめたモジュール．

`outputs/ex001_vae_latent_diagnosis/` 以下に保存された結果を読み込み，複数のseedがある場合は
seed間の平均と標準偏差を計算して描画する．図は `outputs/ex001_vae_latent_diagnosis/figures/` に保存する．
ルートディレクトリの `visualize_result.ipynb` から呼び出して用いる．
"""

import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common

# 配色: 系列の識別には固定順のカテゴリ色，alpha の大小には単一色相の濃淡を用いる
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SEQUENTIAL_BLUES = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
SURFACE = "#fcfcfb"

FIGURE_DIR = os.path.join(common.OUTPUT_DIR, "figures")


def apply_style() -> None:
    """
    全ての図に共通する描画スタイル (背景色，控えめな軸・グリッド，文字色) を設定する．

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
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def seed_dir(method: str, hyperparameter: str, seed: int) -> str:
    """
    条件ごとの結果の保存先ディレクトリのパスを返す (作成はしない)．

    引数:
        method (str): 実験の種類．
        hyperparameter (str): ハイパーパラメータを表す文字列．
        seed (int): 乱数シード．
    戻り値:
        path (str): 保存先ディレクトリのパス．
    """
    return os.path.join(common.OUTPUT_DIR, method, hyperparameter, str(seed))


def load_across_seeds(method: str, hyperparameter: str, filename: str, seeds: List[int], keys: List[str]) -> Dict[str, pd.DataFrame]:
    """
    複数のseedのCSVを読み込み，`keys` でグループ化して数値列のseed間の平均と標準偏差を計算する．

    引数:
        method (str): 実験の種類．
        hyperparameter (str): ハイパーパラメータを表す文字列．
        filename (str): 各seedのディレクトリ内のCSVファイル名．
        seeds (list[int]): 集計するseedのリスト．
        keys (list[str]): 条件を識別する列名のリスト (例: ["subset", "alpha"])．
    戻り値:
        result (dict): "mean" と "std" をkeyとし，`keys` を列に持つ表をvalueとする辞書．
    """
    frames = [pd.read_csv(os.path.join(seed_dir(method, hyperparameter, seed), filename)) for seed in seeds]
    grouped = pd.concat(frames).groupby(keys, sort=False)
    return {
        "mean": grouped.mean(numeric_only=True).reset_index(),
        "std": grouped.std(numeric_only=True, ddof=0).reset_index(),
    }


# ---------------------------------------------------------------------------
# 実験1
# ---------------------------------------------------------------------------

def plot_ex1(hyperparameter: str = "all_images") -> List[str]:
    """
    実験1の図 (標準偏差のヒストグラム，降順に並べたKL，mu の分散と標準偏差の平均の散布図，
    デコーダの特異値) を描画する．

    引数:
        hyperparameter (str): 実験1のハイパーパラメータを表す文字列．
    戻り値:
        paths (list[str]): 保存した図のパスのリスト．
    """
    apply_style()
    target = seed_dir("ex1_latent_stats", hyperparameter, 0)
    table = pd.read_csv(os.path.join(target, "per_dim_stats.csv"))
    summary = common.load_json(os.path.join(target, "summary.json"))
    std_sample = np.load(os.path.join(target, "std_sample.npy"))
    paths = []

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.hist(std_sample, bins=80, color=SERIES_COLORS[0], edgecolor=SURFACE, linewidth=0.3)
    ax.axvline(1.0, color=MUTED, linestyle="--", linewidth=1.2)
    ax.text(1.0, ax.get_ylim()[1] * 0.97, " prior std = 1", color=SECONDARY_INK, ha="right", va="top", fontsize=8.5)
    ax.set_xlabel("posterior std  exp(0.5 logvar)  (all images x all dims)")
    ax.set_ylabel("count (subsampled elements)")
    ax.set_title(f"Posterior std: mean {summary['std_all_elements_mean']:.3f}")
    paths.append(save_figure(fig, "fig_ex1_std_hist"))

    fig, ax = plt.subplots(figsize=(5, 5))
    ordered = table.sort_values("kl_mean", ascending=False)["kl_mean"].to_numpy()
    ax.bar(np.arange(len(ordered)), ordered, width=0.8, color=SERIES_COLORS[0])
    ax.set_xlabel("latent dimension (sorted by KL, descending)")
    ax.set_ylabel("mean KL per dimension [nat]")
    ax.set_title(f"KL per dimension: total {summary['kl_total_mean_nat']:.1f} nat")
    paths.append(save_figure(fig, "fig_ex1_kl_sorted"))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(table["mu_var"], table["std_mean"], s=28, color=SERIES_COLORS[0], edgecolor=SURFACE, linewidth=0.8)
    ax.axvline(common.ACTIVE_UNIT_THRESHOLD, color=MUTED, linestyle="--", linewidth=1.2)
    ax.set_xscale("log")
    ax.set_xlim(left=min(common.ACTIVE_UNIT_THRESHOLD * 0.5, table["mu_var"].min() * 0.7))
    ax.text(common.ACTIVE_UNIT_THRESHOLD, ax.get_ylim()[0], " active-unit threshold 0.01", color=SECONDARY_INK, va="bottom", fontsize=8.5)
    ax.set_xlabel("variance of mu over the dataset")
    ax.set_ylabel("mean posterior std")
    ax.set_title(f"Active units: {summary['num_active_units']} / {summary['latent_dim']}")
    paths.append(save_figure(fig, "fig_ex1_muvar_vs_std"))

    fig, ax = plt.subplots(figsize=(5, 5))
    singular_values = np.array(summary["decoder_singular_values"])
    ax.plot(np.arange(1, len(singular_values) + 1), singular_values, color=SERIES_COLORS[0])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("index of singular value")
    ax.set_ylabel("singular value of decoder weight (512 x d)")
    ax.set_title("Decoder gain spectrum (linear decoder)")
    paths.append(save_figure(fig, "fig_ex1_decoder_singular_values"))
    return paths


# ---------------------------------------------------------------------------
# 実験2
# ---------------------------------------------------------------------------

def _band(ax: plt.Axes, x: np.ndarray, table: pd.DataFrame, prefix: str, color: str, label: str) -> None:
    """
    指標の平均を線で，5%〜95%分位の範囲を帯で描画する．

    引数:
        ax (plt.Axes): 描画先の軸．
        x (np.ndarray): 形状 (alphaの数,) の横軸の値．
        table (pd.DataFrame): `{prefix}_mean`，`{prefix}_q05`，`{prefix}_q95` の列を持つ表．
        prefix (str): 指標の名前．
        color (str): 線と帯の色．
        label (str): 凡例に表示する名前．
    戻り値:
        なし．
    """
    ax.fill_between(x, table[f"{prefix}_q05"], table[f"{prefix}_q95"], color=color, alpha=0.16, linewidth=0)
    ax.plot(x, table[f"{prefix}_mean"], color=color, marker="o", label=label)


def _reference_line(ax: plt.Axes, value: float, label: str, linestyle: str, below: bool = False) -> None:
    """
    比較基準を表す水平線を，右端のラベルとともに描画する．

    引数:
        ax (plt.Axes): 描画先の軸．
        value (float): 水平線の縦軸の値．
        label (str): 線に付けるラベル．
        linestyle (str): 線種．
        below (bool): True の場合，ラベルを線の下側に配置する (近接する線のラベルの重なりを避けるため)．
    戻り値:
        なし．
    """
    ax.axhline(value, color=SECONDARY_INK, linestyle=linestyle, linewidth=1.2)
    ax.text(ax.get_xlim()[1], value, f"{label} ", color=SECONDARY_INK, ha="right",
            va="top" if below else "bottom", fontsize=8)


def plot_ex2(seeds: List[int], hyperparameter: str = None, subset: str = "all", scatter_seed: int = None) -> List[str]:
    """
    実験2の図 (alpha に対する移動量と最近傍距離，alpha に対する cos，移動量と cos の散布図，
    潜在空間と出力空間の移動量の散布図) を描画する．

    引数:
        seeds (list[int]): 平均と標準偏差の計算に用いるseedのリスト．
        hyperparameter (str): 実験2のハイパーパラメータを表す文字列．None の場合は既定のサンプル数．
        subset (str): 集計対象の部分集合 ("all"，"train"，"test")．
        scatter_seed (int): 散布図に用いるseed．None の場合は `seeds` の先頭．
    戻り値:
        paths (list[str]): 保存した図のパスのリスト．
    """
    apply_style()
    hyperparameter = hyperparameter or f"K{common.NUM_SAMPLES}"
    scatter_seed = seeds[0] if scatter_seed is None else scatter_seed
    loaded = load_across_seeds("ex2_displacement", hyperparameter, "summary_by_alpha.csv", seeds, ["subset", "alpha"])
    table = loaded["mean"][loaded["mean"]["subset"] == subset].reset_index(drop=True)
    table_std = loaded["std"][loaded["std"]["subset"] == subset].reset_index(drop=True)
    baselines = load_across_seeds("ex2_displacement", hyperparameter, "baselines_summary.csv", seeds, ["quantity"])["mean"].set_index("quantity")
    alphas = table["alpha"].to_numpy()
    paths = []

    fig, ax = plt.subplots(figsize=(5.6, 5))
    _band(ax, alphas, table, "d_latent", SERIES_COLORS[0], "d_latent (mean, 5-95% band)")
    ax.errorbar(alphas, table["d_latent_mean"], yerr=table_std["d_latent_mean"], fmt="none", ecolor=INK, capsize=2, linewidth=1)
    ax.set_xlim(0, alphas.max() * 1.02)
    _reference_line(ax, baselines.loc["mu_nn_dist", "value_q50"], "nearest other mu (median)", "--", below=True)
    _reference_line(ax, baselines.loc["mu_rand_dist", "value_q50"], "random other mu (median)", ":")
    _reference_line(ax, baselines.loc["mu_norm", "value_q50"], "||mu|| (median)", "-.")
    ax.set_xlabel("alpha (noise scale)")
    ax.set_ylabel("distance in latent space")
    ax.set_title("Latent displacement vs. distance to other images")
    ax.legend(loc="upper left")
    paths.append(save_figure(fig, f"fig_ex2_dlatent_vs_alpha_{subset}"))

    fig, ax = plt.subplots(figsize=(5.6, 5))
    _band(ax, alphas, table, "cos_v", SERIES_COLORS[0], "cos(v~, v)")
    _band(ax, alphas, table, "cos_dec", SERIES_COLORS[1], "cos(v~, decoder(mu))")
    ax.plot(alphas, table["pred_cos_v_mean"], color=INK, linestyle="--", linewidth=1.2, label="closed-form prediction of cos(v~, v)")
    ax.set_xlim(0, alphas.max() * 1.02)
    ax.set_ylim(-0.1, 1.02)
    _reference_line(ax, baselines.loc["arc_nn_cos", "value_q50"], "nearest other face (median)", "--")
    _reference_line(ax, baselines.loc["arc_rand_cos", "value_q95"], "random other face (95th pct.)", ":")
    ax.set_xlabel("alpha (noise scale)")
    ax.set_ylabel("cosine similarity")
    ax.set_title("Output similarity vs. alpha (mean, 5-95% band)")
    ax.legend(loc="upper right")
    paths.append(save_figure(fig, f"fig_ex2_cos_vs_alpha_{subset}"))

    target = seed_dir("ex2_displacement", hyperparameter, scatter_seed)
    samples = np.load(os.path.join(target, "per_sample_metrics.npz"))
    rng = np.random.default_rng(0)
    nonzero = [index for index, alpha in enumerate(samples["alphas"]) if alpha > 0]

    for y_key, y_label, title, name in [
        ("cos_v", "cos(v~, v)", "Decoder sensitivity: similarity vs. latent displacement", "fig_ex2_scatter_dlatent_cos"),
        ("d_out", "output displacement ||v~ - decoder(mu)||", "Decoder gain: output vs. latent displacement", "fig_ex2_scatter_dlatent_dout"),
    ]:
        fig, ax = plt.subplots(figsize=(5.6, 5))
        for index in nonzero:
            x, y = samples["d_latent"][index].ravel(), samples[y_key][index].ravel()
            picked = rng.choice(x.size, size=min(3000, x.size), replace=False)
            ax.scatter(x[picked], y[picked], s=4, alpha=0.35, linewidth=0, color=SEQUENTIAL_BLUES[index],
                       label=f"alpha = {samples['alphas'][index]:g}", rasterized=True)
        ax.set_xlabel("latent displacement d_latent = ||alpha * std * z||")
        ax.set_ylabel(y_label)
        ax.set_title(title)
        legend = ax.legend(loc="best", markerscale=3.5)
        for handle in legend.legend_handles:
            handle.set_alpha(1.0)
        paths.append(save_figure(fig, name))
    return paths


# ---------------------------------------------------------------------------
# 実験3
# ---------------------------------------------------------------------------

def plot_ex3(seeds: List[int], hyperparameter: str = None, hist_seed: int = None) -> List[str]:
    """
    実験3の図 (ガウスと半径固定の cos の比較，その差，実効的な alpha に対する cos，||z|| の分布) を描画する．

    引数:
        seeds (list[int]): 平均と標準偏差の計算に用いるseedのリスト．
        hyperparameter (str): 実験3のハイパーパラメータを表す文字列．None の場合は既定のサンプル数．
        hist_seed (int): ||z|| のヒストグラムに用いるseed．None の場合は `seeds` の先頭．
    戻り値:
        paths (list[str]): 保存した図のパスのリスト．
    """
    apply_style()
    hyperparameter = hyperparameter or f"K{common.NUM_SAMPLES}"
    hist_seed = seeds[0] if hist_seed is None else hist_seed
    tests = load_across_seeds("ex3_radius_sampling", hyperparameter, "gauss_vs_shell_tests.csv", seeds, ["alpha"])
    variants = load_across_seeds("ex3_radius_sampling", hyperparameter, "summary_by_variant.csv", seeds, ["variant", "alpha"])["mean"]
    mean, std = tests["mean"], tests["std"]
    alphas = mean["alpha"].to_numpy()
    paths = []

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(alphas, mean["gauss_cos_v_mean"], color=SERIES_COLORS[0], marker="o", markersize=9, label="Gaussian  z = u")
    ax.plot(alphas, mean["shell_cos_v_mean"], color=SERIES_COLORS[1], marker="o", markersize=4, linestyle="--",
            label="fixed radius  z = sqrt(d) u / ||u||")
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("alpha (noise scale)")
    ax.set_ylabel("mean cos(v~, v)")
    ax.set_title("Gaussian vs. fixed-radius sampling")
    ax.legend(loc="upper right")
    paths.append(save_figure(fig, "fig_ex3_gauss_vs_shell"))

    fig, ax = plt.subplots(figsize=(5, 5))
    positions = np.arange(len(alphas))
    ax.bar(positions, mean["image_diff_mean"], width=0.6, color=SERIES_COLORS[0],
           yerr=std["image_diff_mean"], ecolor=INK, capsize=3)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{alpha:g}" for alpha in alphas])
    ax.set_xlabel("alpha (noise scale)")
    ax.set_ylabel("mean difference of cos(v~, v):  Gaussian - fixed radius")
    ax.set_title("Paired difference (error bar: std over seeds)")
    paths.append(save_figure(fig, "fig_ex3_paired_difference"))

    fig, ax = plt.subplots(figsize=(5.6, 5))
    markers = {"gauss": "o", "shell_r0.5": "v", "shell_r1.0": "s", "shell_r1.5": "D", "shell_r2.0": "^"}
    for color, (name, marker) in zip(SERIES_COLORS, markers.items()):
        rows = variants[variants["variant"] == name].sort_values("effective_alpha")
        ax.plot(rows["effective_alpha"], rows["cos_v_mean"], color=color, marker=marker, linewidth=1.2,
                markersize=7, markerfacecolor="none", markeredgewidth=1.6, label=name)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("effective alpha = alpha * r   (radius = r * sqrt(d))")
    ax.set_ylabel("mean cos(v~, v)")
    ax.set_title("Fixing the radius is equivalent to rescaling alpha")
    ax.legend(loc="upper right")
    paths.append(save_figure(fig, "fig_ex3_effective_alpha"))

    target = seed_dir("ex3_radius_sampling", hyperparameter, hist_seed)
    meta = common.load_json(os.path.join(target, "meta.json"))
    u_norm = np.load(os.path.join(target, "cos_v_per_sample.npz"))["u_norm"].ravel()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.hist(u_norm, bins=80, color=SERIES_COLORS[0], edgecolor=SURFACE, linewidth=0.3)
    ax.axvline(meta["sqrt_d"], color=INK, linestyle="--", linewidth=1.2)
    ax.text(meta["sqrt_d"], ax.get_ylim()[1] * 0.97, f" sqrt(d) = {meta['sqrt_d']:.2f}", color=SECONDARY_INK, va="top", fontsize=8.5)
    ax.set_xlim(0, u_norm.max() * 1.05)
    ax.set_xlabel("||z||  for z ~ N(0, I)")
    ax.set_ylabel("count")
    ax.set_title(f"Norm of Gaussian samples: mean {meta['z_norm_mean']:.2f}, std {meta['z_norm_std']:.2f}")
    paths.append(save_figure(fig, "fig_ex3_znorm_hist"))
    return paths
