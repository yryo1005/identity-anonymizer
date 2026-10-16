"""
ex003_ghost_image_eval の作業前の確認の結果を可視化する関数をまとめたモジュール．

`outputs/ex003_ghost_image_eval/prechecks/` 以下に保存された結果を読み込み，図を
`outputs/ex003_ghost_image_eval/figures/` に保存する．ルートディレクトリの `visualize_result.ipynb` から呼び出す．
"""

import os
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import settings

INK, SECONDARY_INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
FIGURE_DIR = os.path.join(settings.OUTPUT_DIR, "figures")
PRECHECK_DIR = os.path.join(settings.OUTPUT_DIR, "prechecks", "default", str(settings.SEED))


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


def plot_reproduction() -> str:
    """
    予稿の値，リポジトリのnotebookに記録された値，本実験の処理による実測値を，alpha ごとに比べる図を描く．

    引数:
        なし．
    戻り値:
        path (str): 保存した図のパス．
    """
    apply_style()
    repo = pd.read_csv(os.path.join(PRECHECK_DIR, "repo_notebook_reproduction.csv"))
    ours = pd.read_csv(os.path.join(PRECHECK_DIR, "paper_reproduction.csv"))
    breakdown = pd.read_csv(os.path.join(PRECHECK_DIR, "reproduction_breakdown_by_age.csv"))
    self_row = breakdown[(breakdown["source"] == "Self") & (breakdown["age_group"] == "all")].iloc[0]
    panels = [
        ("cos", "mean cos(v_gen, v)", "cos_gen_v"),
        ("gender_match", "DEX gender agreement (original vs. generated)", "dex_gender_match"),
        ("age_rmse", "DEX age RMSE [years]  (lower is better)", "dex_age_rmse"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5))
    for ax, (metric, y_label, self_key) in zip(axes, panels):
        ax.plot(repo["alpha"], repo[f"{metric}_paper"], color=INK, marker="o", label="paper (23,088 images)")
        ax.plot(repo["alpha"], repo[f"{metric}_notebook"], color=ORANGE, marker="s", linestyle="--",
                markerfacecolor=SURFACE, markeredgewidth=1.5, label="repo notebook, recorded (467 sample images)")
        ax.plot(repo["alpha"], repo[f"{metric}_measured"], color=ORANGE, marker="s",
                label="this pipeline, same 467 sample images")
        ax.plot(ours["alpha"], ours[f"{metric}_measured"], color=BLUE, marker="D",
                label="this pipeline, 200 random evaluation images")
        ax.axhline(self_row[self_key], color=SECONDARY_INK, linestyle=":", linewidth=1.1)
        ax.text(0.99, self_row[self_key], "Self swap (200 images) ", color=SECONDARY_INK, ha="right", va="bottom",
                fontsize=8, transform=ax.get_yaxis_transform())
        ax.set_xlabel("alpha (noise scale of the current method M0)")
        ax.set_ylabel(y_label)
        ax.set_xticks(repo["alpha"])
    axes[0].legend(loc="lower left")
    fig.suptitle("Reproduction of the paper's evaluation", fontsize=12)
    fig.tight_layout()
    return save_figure(fig, "fig_precheck_reproduction")


def plot_age_breakdown() -> str:
    """
    Self と Recon について，DEXが推定した年齢の変化と性別一致率を，ラベルの年代ごとに比べる図を描く．

    引数:
        なし．
    戻り値:
        path (str): 保存した図のパス．
    """
    apply_style()
    breakdown = pd.read_csv(os.path.join(PRECHECK_DIR, "reproduction_breakdown_by_age.csv"))
    breakdown = breakdown[breakdown["age_group"] != "all"]
    groups = breakdown["age_group"].unique()
    positions = np.arange(len(groups))
    panels = [("dex_age_bias", "DEX age: generated - original [years]"),
              ("dex_gender_match", "DEX gender agreement"), ("cos_gen_v", "mean cos(v_gen, v)")]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5))
    for ax, (metric, y_label) in zip(axes, panels):
        for offset, (source, color, label) in zip((-0.2, 0.2), (("Self", BLUE, "Self: original v as source"),
                                                              ("Recon", ORANGE, "Recon: decoder(mu) as source"))):
            rows = breakdown[breakdown["source"] == source].set_index("age_group").loc[groups]
            ax.bar(positions + offset, rows[metric], width=0.36, color=color, label=label)
        ax.axhline(0, color=MUTED, linewidth=1)
        ax.set_xticks(positions)
        ax.set_xticklabels(groups)
        ax.set_xlabel("age group (UTKFace label)")
        ax.set_ylabel(y_label)
    axes[0].legend(loc="lower left")
    fig.suptitle("Where the degradation comes from: by age group (200 random evaluation images)", fontsize=12)
    fig.tight_layout()
    return save_figure(fig, "fig_precheck_age_breakdown")


def plot_prechecks() -> List[str]:
    """
    作業前の確認に関する全ての図を描く．

    引数:
        なし．
    戻り値:
        paths (list[str]): 保存した図のパスのリスト．
    """
    return [plot_reproduction(), plot_age_breakdown()]


# ---------------------------------------------------------------------------
# 本実験
# ---------------------------------------------------------------------------

METHOD_STYLES = {
    "M0": {"label": "M0  current (alpha scaling)", "color": BLUE, "marker": "o"},
    "M2": {"label": "M2  keep top-k dims, prior elsewhere", "color": ORANGE, "marker": "s"},
    "M3u": {"label": "M3u shrink all dims toward prior", "color": AQUA, "marker": "D"},
}


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


def plot_tradeoff() -> str:
    """
    画像での再識別率に対する属性保持のトレードオフ曲線を描く．実線はノルムの補正なし，破線は補正ありである．

    引数:
        なし．
    戻り値:
        path (str): 保存した図のパス．
    """
    import summarize_main

    apply_style()
    summary = summarize_main.load_summary()
    references = summary.set_index("key")
    panels = [("dex_gender_match", "DEX gender agreement (original vs. generated)", False),
              ("dex_age_abs_diff", "DEX |age difference| [years]  (lower is better)", True),
              ("lin_race_agree", "race agreement, linear classifier on v_gen (reference)", False)]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
    for ax, (metric, y_label, lower_is_better) in zip(axes, panels):
        for method, style in METHOD_STYLES.items():
            for norm_mode, linestyle, filled in (("raw", "-", True), ("fixed", "--", False)):
                rows = summarize_main.curve_rows(summary, method, norm_mode)
                ax.plot(rows["reid_top1"], rows[metric], color=style["color"], linestyle=linestyle, marker=style["marker"],
                        linewidth=1.8, markerfacecolor=style["color"] if filled else SURFACE, markeredgewidth=1.5,
                        label=f"{style['label']}" + ("" if filled else "  [norm fixed to 22.4]"))
        reference_line(ax, references.loc["Self|raw", metric], "Self (original v)", below=lower_is_better)
        reference_line(ax, references.loc["Recon|raw", metric], "Recon (decoder(mu))", below=not lower_is_better)
        reference_line(ax, references.loc["Other|raw", metric], "Other (another person's v)", below=not lower_is_better)
        ax.set_xlabel("top-1 re-identification rate from generated images\n(lower = more anonymous, to the right)")
        ax.set_ylabel(y_label)
        ax.invert_xaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, -0.09))
    fig.suptitle("Attribute retention vs. re-identification rate, measured on GHOST-generated images (1,000 images)", fontsize=12)
    fig.tight_layout()
    return save_figure(fig, "fig_main_tradeoff")


def plot_correspondence() -> str:
    """
    ベクトル空間での評価と，生成画像での評価の対応を描く (類似度の散布図，設定ごとの類似度と再識別率)．

    引数:
        なし．
    戻り値:
        path (str): 保存した図のパス．
    """
    import summarize_main

    apply_style()
    summary = summarize_main.load_summary()
    per_image = summarize_main.load_per_image()
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))

    keys = [key for key in per_image if key.endswith("|raw") and not key.startswith(("Self", "Other"))]
    x = np.concatenate([per_image[key]["vec_cos_v"] for key in keys])
    y = np.concatenate([per_image[key]["cos_gen_v"] for key in keys])
    picked = rng.choice(len(x), size=6000, replace=False)
    axes[0].scatter(x[picked], y[picked], s=4, alpha=0.3, linewidth=0, color=BLUE, rasterized=True)
    axes[0].plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1.2)
    axes[0].set_xlabel("cos(v~, v) in vector space (source embedding)")
    axes[0].set_ylabel("cos(v_gen, v) re-extracted from the generated image")
    axes[0].set_title(f"Per image, all settings pooled (r = {np.corrcoef(x, y)[0, 1]:.2f})")

    for ax, (x_key, y_key, x_label, y_label, title) in zip(axes[1:], [
        ("vec_cos_v", "cos_gen_v", "mean cos(v~, v) in vector space", "mean cos(v_gen, v) from images", "Similarity per setting"),
        ("vec_reid_top1", "reid_top1", "top-1 re-identification rate in vector space",
         "top-1 re-identification rate from images", "Re-identification rate per setting"),
    ]):
        ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1.2)
        for method, style in METHOD_STYLES.items():
            rows = summarize_main.curve_rows(summary, method, "raw")
            ax.plot(rows[x_key], rows[y_key], color=style["color"], marker=style["marker"], linewidth=1.5, label=style["label"])
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
    axes[1].legend(loc="upper left")
    fig.suptitle("Vector-space evaluation vs. evaluation on generated images", fontsize=12)
    fig.tight_layout()
    return save_figure(fig, "fig_main_correspondence")


def plot_image_grid(num_images: int = 12) -> str:
    """
    元画像と，各設定で生成した顔画像を並べた比較画像を作る．列が画像，行が設定である．

    引数:
        num_images (int): 並べる画像の枚数．
    戻り値:
        path (str): 保存した図のパス．
    """
    import calibrate
    import cv2
    import prepare
    import summarize_main

    apply_style()
    eval_data = prepare.load()
    summary = summarize_main.load_summary().set_index("key")
    main_dir = summarize_main.MAIN_DIR
    columns = np.linspace(0, settings.NUM_IMAGES - 1, num_images).astype(int)
    chosen = calibrate.load_chosen()
    rows = [("Original", None), ("Self", "Self|raw"), ("Recon", "Recon|raw")]
    for method in ("M0", "M2", "M3u"):
        for row in chosen[chosen["method"] == method].itertuples():
            rows.append((f"{method} {row.param_name}={row.param:g}", f"{method}|{row.param_name}={row.param:g}|raw"))
    rows.append(("Other", "Other|raw"))

    tile = 112
    fig, axes = plt.subplots(len(rows), 1, figsize=(num_images * 1.1 + 2.4, len(rows) * 1.12))
    for ax, (label, key) in zip(axes, rows):
        if key is None:
            images = np.asarray(eval_data.resized256[eval_data.main_index[columns]])
            note = ""
        else:
            images = np.load(os.path.join(main_dir, "generated", key.replace("|", "__") + ".npy"), mmap_mode="r")[columns]
            note = f"\nre-ID {summary.loc[key, 'reid_top1']:.2f}"
        strip = np.concatenate([cv2.resize(np.asarray(image), (tile, tile)) for image in images], axis=1)
        ax.imshow(strip[:, :, ::-1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_ylabel(label + note, rotation=0, ha="right", va="center", fontsize=8.5)
    fig.subplots_adjust(hspace=0.03, left=0.14, right=0.995, top=0.995, bottom=0.005)
    return save_figure(fig, "fig_main_image_grid")


def plot_main() -> List[str]:
    """
    本実験に関する全ての図を描く．

    引数:
        なし．
    戻り値:
        paths (list[str]): 保存した図のパスのリスト．
    """
    return [plot_tradeoff(), plot_correspondence(), plot_image_grid()]


# ---------------------------------------------------------------------------
# 貼り戻した画像での評価
# ---------------------------------------------------------------------------

def plot_pasteback() -> str:
    """
    DEXとArcFaceへの入力を変えたときの評価値を，予稿の値と比べる図を描く．

    引数:
        なし．
    戻り値:
        path (str): 保存した図のパス．
    """
    apply_style()
    path = os.path.join(settings.OUTPUT_DIR, "pasteback_reproduction", f"n{settings.NUM_REPRODUCTION_IMAGES}",
                        str(settings.SEED), "summary.csv")
    table = pd.read_csv(path)
    self_row = table[table["condition"] == "Self"].iloc[0]
    table = table[table["condition"] != "Self"].copy()
    table["alpha"] = table["condition"].str.split("=").str[1].astype(float)
    alphas = table["alpha"].to_numpy()
    paper = np.array([settings.PAPER_VALUES[alpha] for alpha in alphas])
    panels = [
        ("mean cos(v_gen, v)", 0, [("cos_crop", "generator output (crop)", BLUE, "o"),
                                   ("cos_recrop", "face re-cropped from the pasted image", ORANGE, "s")]),
        ("DEX gender agreement", 1, [("crop_gender_match", "crop vs. crop", BLUE, "o"),
                                     ("recrop_gender_match", "original crop vs. re-crop of pasted image", AQUA, "D"),
                                     ("full_gender_match", "full image vs. pasted full image", ORANGE, "s")]),
        ("DEX age RMSE [years]  (lower is better)", 2, [("crop_age_rmse", "crop vs. crop", BLUE, "o"),
                                                        ("recrop_age_rmse", "original crop vs. re-crop of pasted image", AQUA, "D"),
                                                        ("full_age_rmse", "full image vs. pasted full image", ORANGE, "s")]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5))
    for ax, (y_label, paper_column, series) in zip(axes, panels):
        ax.plot(alphas, paper[:, paper_column], color=INK, marker="o", linewidth=2.4, label="paper")
        for column, label, color, marker in series:
            ax.plot(alphas, table[column], color=color, marker=marker, label=label)
            ax.scatter([alphas.max() + 0.25], [self_row[column]], color=color, marker=marker, s=46,
                       facecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.set_xticks(list(alphas) + [alphas.max() + 0.25])
        ax.set_xticklabels([f"{alpha:g}" for alpha in alphas] + ["Self"])
        ax.set_xlabel("alpha of the current method M0   (open markers: Self swap)")
        ax.set_ylabel(y_label)
        ax.legend(loc="best")
    fig.suptitle("Which evaluation input reproduces the paper? (200 random evaluation images)", fontsize=12)
    fig.tight_layout()
    return save_figure(fig, "fig_pasteback_reproduction")
