"""
本実験: 各設定でソースの顔ベクトルを作り，GHOSTで顔画像を生成して評価する．

設定は，参照用の Self (元の v)，Other (他人の実際の v)，Recon (v0 = decoder(mu)) と，
較正で選んだパラメータの M0，M2，M3u である．Self と Other 以外は，ソースの顔ベクトルをそのまま
渡す場合 ("raw") と，学習用データの平均ノルムにスケーリングしてから渡す場合 ("fixed") の2通りで生成する．

    python programs/ex003_ghost_image_eval/generate.py            # 本実験 (1,000枚)
    python programs/ex003_ghost_image_eval/generate.py --trial    # 試行 (50枚)
"""

import argparse
import os
import time
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

import calibrate
import pipeline
import prepare
import samplers
import settings
from attributes import load_ranking
from evaluate import reidentification
from machine_learning_utils import set_seed

#: 較正で選ぶ設定に加えて，必ず生成する設定 (方式, パラメータ)．
#: 指示にある M0 の alpha=5，M2 の k=32・64 に加え，画像での再識別率が高い側の曲線を補うための
#: 掃引点 (M0 の alpha=1, 2, 3，M2 の k=80, 96，M3u の gamma=0.65, 0.75) を含める．
EXTRA_SETTINGS = [("M0", 5.0), ("M2", 32), ("M2", 64),
                  ("M0", 1.0), ("M0", 2.0), ("M0", 3.0), ("M2", 80), ("M2", 96), ("M3u", 0.65), ("M3u", 0.75)]
NORM_MODES = ["raw", "fixed"]


def make_setting(method: str, param: float) -> samplers.Setting:
    """
    方式とパラメータから，ex002 のサンプリング設定を作る．

    引数:
        method (str): 方式の記号 ("M0"，"M2"，"M3u")．
        param (float): パラメータ (M0 は alpha，M2 は k，M3u は gamma)．
    戻り値:
        setting (samplers.Setting): サンプリング設定．
    """
    if method == "M0":
        return samplers.Setting("M0", "M0", "alpha", float(param), alpha=float(param))
    if method == "M2":
        return samplers.Setting("M2", "M2", "k", int(param), k=int(param))
    return samplers.Setting("M3u", "M3u", "gamma", float(param), k=0, gamma=float(param))


def list_settings() -> List[Tuple[str, Optional[samplers.Setting], str]]:
    """
    生成する設定を列挙する．

    引数:
        なし．
    戻り値:
        entries (list[tuple]): (設定の名前, サンプリング設定または None, 目標の再識別率の説明) のリスト．
            Self と Other はサンプリング設定を持たない．
    """
    entries = [("Self", None, "reference"), ("Other", None, "reference"),
               ("Recon", make_setting("M0", 0.0), "reference")]
    chosen = calibrate.load_chosen()
    seen = set()
    for row in chosen.itertuples():
        key = (row.method, round(float(row.param), 4))
        if key in seen:
            continue
        seen.add(key)
        entries.append((f"{row.method}|{row.param_name}={row.param:g}", make_setting(row.method, row.param),
                        f"vector reid {row.target_reid_top1:g}"))
    for method, param in EXTRA_SETTINGS:
        if (method, round(float(param), 4)) in seen:
            continue
        seen.add((method, round(float(param), 4)))
        setting = make_setting(method, param)
        entries.append((f"{method}|{setting.param_name}={param:g}", setting, "extra"))
    return entries


def other_index(group: np.ndarray, pool_group: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    各画像に対し，重複グループが異なる画像を，候補の中から無作為に1枚ずつ選ぶ．

    引数:
        group (np.ndarray): 形状 (n,) の各画像の重複グループの番号．
        pool_group (np.ndarray): 形状 (E,) の候補の画像の重複グループの番号．
        rng (np.random.Generator): 乱数生成器．
    戻り値:
        index (np.ndarray): 形状 (n,) の，候補の中での画像の番号．
    """
    index = rng.integers(0, len(pool_group), size=len(group))
    while True:
        same = pool_group[index] == group
        if not same.any():
            return index
        index[same] = rng.integers(0, len(pool_group), size=int(same.sum()))


@torch.no_grad()
def vector_metrics(context: pipeline.Context, sources: torch.Tensor) -> dict:
    """
    ソースの顔ベクトルそのものに対する，ベクトル空間での指標を計算する．

    引数:
        context (pipeline.Context): モデルとデータ．
        sources (torch.Tensor): 形状 (n, 512) のソースの顔ベクトル．
    戻り値:
        metrics (dict): いずれも形状 (n,) の配列．vec_cos_v (元の v とのコサイン類似度)，
            vec_reid_top1 / vec_reid_top10 (再識別の成否)，vec_norm (ノルム)，
            vec_lin_gender (線形分類器による性別の予測)．
    """
    reid = reidentification(sources.unsqueeze(1), context.gallery, context.gallery_group, context.group)
    return {
        "vec_cos_v": F.cosine_similarity(sources, context.v, dim=1).cpu().numpy(),
        "vec_reid_top1": reid["reid_top1"][:, 0].cpu().numpy(),
        "vec_reid_top10": reid["reid_top10"][:, 0].cpu().numpy(),
        "vec_norm": sources.norm(dim=1).cpu().numpy(),
        "vec_lin_gender": context.predictor.predict(sources)["gender"].cpu().numpy().astype(np.float64),
    }


def run(trial: bool = False) -> None:
    """
    全ての設定について生成と評価を行い，設定ごとの要約表，画像ごとの指標，生成画像を保存する．

    引数:
        trial (bool): True の場合，最初の50枚だけで試行し，処理時間を測る．
    戻り値:
        なし．
    """
    import common as ex002_common

    ex002_common.NUM_SAMPLES = 1
    eval_data = prepare.load()
    num_images = settings.NUM_TRIAL_IMAGES if trial else settings.NUM_IMAGES
    target_dir = settings.condition_dir("trial" if trial else "main", f"n{num_images}", settings.SEED)
    if settings.is_finished(target_dir):
        print(f"実行済みのためスキップ: {target_dir}")
        return

    index = eval_data.main_index[:num_images]
    context = pipeline.build_context(eval_data, index)
    ranking = load_ranking()
    set_seed(settings.SEED)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(settings.SEED)
    others = other_index(eval_data.group[index], eval_data.group, np.random.default_rng(settings.SEED))
    image_dir = os.path.join(target_dir, "generated")
    os.makedirs(image_dir, exist_ok=True)

    rows, per_image = [], {}
    for name, setting, purpose in tqdm(list_settings(), desc="settings", leave=False):
        if name == "Self":
            base = context.v
        elif name == "Other":
            base = torch.from_numpy(eval_data.v[others]).cuda()
        else:
            mask = samplers.kept_mask(setting, ranking, ranking)
            base = samplers.sample(setting, context.vae, context.v, context.v0, context.mu, context.std, mask, generator, None)[:, 0]

        for norm_mode in (["raw"] if setting is None else NORM_MODES):
            sources = base if norm_mode == "raw" else base * (context.real_mean_norm / base.norm(dim=1, keepdim=True))
            key = f"{name}|{norm_mode}"
            start_time = time.time()
            metrics, generated, v_gen = pipeline.generate_and_evaluate(context, sources, key)
            elapsed = time.time() - start_time
            metrics.update(vector_metrics(context, sources))
            metrics["v_gen"] = v_gen
            per_image[key] = metrics
            np.save(os.path.join(image_dir, key.replace("|", "__") + ".npy"), generated)

            row = {"key": key, "setting": name, "method": name.split("|")[0], "norm_mode": norm_mode, "purpose": purpose,
                   "param": float("nan") if setting is None else setting.param, "elapsed_sec": elapsed}
            row.update(pipeline.summarize(metrics))
            row.update({k: float(v.mean()) for k, v in metrics.items() if k.startswith("vec_")})
            row["lin_vec_vs_dex_gender_agree"] = float((metrics["vec_lin_gender"] == metrics["dex_is_female"]).mean())
            row["lin_gen_vs_dex_gender_agree"] = float((metrics["lin_gender"] == metrics["dex_is_female"]).mean())
            row["corr_vec_cos_vs_gen_cos"] = float(np.corrcoef(metrics["vec_cos_v"], metrics["cos_gen_v"])[0, 1]) if name != "Self" else float("nan")
            rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(target_dir, "summary.csv"), index=False)
    np.savez_compressed(os.path.join(target_dir, "per_image.npz"), index=index,
                        **{f"{key}::{name}": value for key, metrics in per_image.items() for name, value in metrics.items()})
    np.savez_compressed(os.path.join(target_dir, "originals.npz"), index=index, other_index=others,
                        dex_age=context.original_dex["age"], dex_is_female=context.original_dex["is_female"],
                        lin_gender=context.original_linear["gender"], lin_race=context.original_linear["race"],
                        lin_age=context.original_linear["age"])
    total = float(summary["elapsed_sec"].sum())
    settings.save_json({
        "seed": settings.SEED, "num_images": int(num_images), "num_generations": int(len(summary)),
        "real_mean_norm": context.real_mean_norm, "total_generation_sec": total,
        "sec_per_setting": total / len(summary), "sec_per_image": total / len(summary) / num_images,
    }, os.path.join(target_dir, "meta.json"))

    pd.set_option("display.width", 250)
    columns = ["key", "vec_cos_v", "cos_gen_v", "vec_reid_top1", "reid_top1", "reid_top10", "dex_gender_match",
               "dex_age_abs_diff", "dex_age_rmse", "lin_race_agree", "vec_norm", "elapsed_sec"]
    print(summary[columns].round(3).to_string(index=False))
    print(f"生成と評価の合計: {total:.0f} 秒 ({len(summary)} 設定，1設定あたり {total / len(summary):.1f} 秒)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GHOSTによる生成と評価を実行する")
    parser.add_argument("--trial", action="store_true")
    run(trial=parser.parse_args().trial)
