"""
予稿の再現で生じたずれの切り分け．

1. 年代別の内訳: 再現用の200枚について，Self (元の v をソースにした自分自身との顔交換) と
   Recon (v0 = decoder(mu)) の性別一致率・年齢の差を，ラベルの年代ごとに集計する．
2. 画像の集合による違い: リポジトリ同梱のサンプル画像と，評価用データの年齢の分布を比べる．
3. DEXの入力による違い: 元画像の全体をDEXに入力した場合と，切り出した顔画像を入力した場合の
   予測の違いを調べる (予稿の評価が，切り出す前の画像を用いていた可能性の検討材料)．
4. 処理時間: 1設定あたりの生成と評価の時間を測る．

    python programs/ex003_ghost_image_eval/diagnose.py
"""

import os
import time

import cv2
import numpy as np
import pandas as pd

import ghost
import pipeline
import prepare
import settings
from data import age_bin, parse_label

AGE_BIN_NAMES = ["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]


def breakdown_by_age(context: pipeline.Context, eval_data: prepare.EvalData) -> pd.DataFrame:
    """
    Self と Recon について，性別一致率と年齢の差をラベルの年代ごとに集計する．

    引数:
        context (pipeline.Context): 再現用の200枚について準備したモデルとデータ．
        eval_data (prepare.EvalData): 評価用データ一式．
    戻り値:
        table (pd.DataFrame): (ソース, 年代) ごとに1行の表．
    """
    bins = age_bin(eval_data.age[context.index])
    rows = []
    for name, sources in (("Self", context.v), ("Recon", context.v0)):
        start_time = time.time()
        metrics, _, _ = pipeline.generate_and_evaluate(context, sources, name)
        elapsed = time.time() - start_time
        groups = [("all", np.ones(len(bins), dtype=bool))] + [(AGE_BIN_NAMES[b], bins == b) for b in range(8)]
        for group_name, mask in groups:
            if mask.sum() == 0:
                continue
            diff = metrics["dex_age_diff"][mask]
            rows.append({
                "source": name, "age_group": group_name, "num_images": int(mask.sum()),
                "cos_gen_v": float(metrics["cos_gen_v"][mask].mean()),
                "reid_top1": float(metrics["reid_top1"][mask].mean()),
                "dex_gender_match": float(metrics["dex_gender_match"][mask].mean()),
                "dex_age_original": float(context.original_dex["age"][mask].mean()),
                "dex_age_bias": float(diff.mean()), "dex_age_rmse": float(np.sqrt((diff ** 2).mean())),
                "elapsed_sec_for_all": elapsed,
            })
    return pd.DataFrame(rows)


def age_distributions(eval_data: prepare.EvalData) -> pd.DataFrame:
    """
    画像の集合ごとに，ラベルの年代の構成比を求める．

    引数:
        eval_data (prepare.EvalData): 評価用データ一式．
    戻り値:
        table (pd.DataFrame): 年代ごとに1行，画像の集合ごとに1列の構成比の表．
    """
    sample_dir = os.path.join(settings.REPO_ROOT, "data", "utkface_sample")
    sample_ages = np.array([label[0] for label in map(parse_label, os.listdir(sample_dir)) if label is not None])
    sets = {
        "repo_sample_500": sample_ages, "all_eval_4614": eval_data.age,
        "reproduction_200": eval_data.age[eval_data.reproduction_index], "main_1000": eval_data.age[eval_data.main_index],
    }
    table = pd.DataFrame({"age_group": AGE_BIN_NAMES})
    for name, ages in sets.items():
        table[name] = np.bincount(age_bin(ages), minlength=8) / len(ages)
    return table


def full_image_vs_crop(context: pipeline.Context, eval_data: prepare.EvalData) -> dict:
    """
    元画像の全体をDEXに入力した場合と，切り出した顔画像を入力した場合の予測を比べる．

    引数:
        context (pipeline.Context): 再現用の200枚について準備したモデルとデータ．
        eval_data (prepare.EvalData): 評価用データ一式．
    戻り値:
        result (dict): 性別の正解率 (全体・切り出し)，両者の予測の一致率，年齢のMAE，年齢の差の統計．
    """
    index = context.index
    full_images = [cv2.imread(os.path.join(settings.REPO_ROOT, eval_data.paths[i])) for i in index]
    full = {key: np.concatenate([ghost.dex_predict(context.models, image[None])[key] for image in full_images])
            for key in ("age", "is_female")}
    crop = context.original_dex
    is_female = eval_data.gender[index] == 1
    return {
        "num_images": int(len(index)),
        "gender_acc_full_image": float((full["is_female"] == is_female).mean()),
        "gender_acc_crop": float((crop["is_female"] == is_female).mean()),
        "gender_agreement_full_vs_crop": float((full["is_female"] == crop["is_female"]).mean()),
        "age_mae_full_image": float(np.abs(full["age"] - eval_data.age[index]).mean()),
        "age_mae_crop": float(np.abs(crop["age"] - eval_data.age[index]).mean()),
        "age_rmse_full_vs_crop": float(np.sqrt(((full["age"] - crop["age"]) ** 2).mean())),
        "age_bias_crop_minus_full": float((crop["age"] - full["age"]).mean()),
    }


def run() -> None:
    """
    切り分けを実行し，結果を保存する．

    引数:
        なし．
    戻り値:
        なし．
    """
    target_dir = settings.condition_dir("prechecks", "default", settings.SEED)
    eval_data = prepare.load()
    context = pipeline.build_context(eval_data, eval_data.reproduction_index)
    breakdown = breakdown_by_age(context, eval_data)
    distributions = age_distributions(eval_data)
    framing = full_image_vs_crop(context, eval_data)
    breakdown.to_csv(os.path.join(target_dir, "reproduction_breakdown_by_age.csv"), index=False)
    distributions.to_csv(os.path.join(target_dir, "age_distributions.csv"), index=False)
    settings.save_json(framing, os.path.join(target_dir, "dex_full_image_vs_crop.json"))
    pd.set_option("display.width", 250)
    print(breakdown.round(3).to_string(index=False))
    print(distributions.round(3).to_string(index=False))
    print(pd.Series(framing).round(4).to_string())


if __name__ == "__main__":
    run()


def channel_order_check() -> dict:
    """
    既存の `FaceAnonymizerPipeline.anonymize_image` が，匿名化モデルへ渡すターゲットの顔ベクトルを
    BGRの順序の顔画像から計算していること (VAEの学習と評価はRGBの順序) の影響を調べる．
    再現用の200枚について，2通りの顔ベクトルの類似度と，それぞれを Self と Recon のソースにした場合の
    生成画像の類似度を求めて保存する．

    引数:
        なし．
    戻り値:
        result (dict): 2通りの顔ベクトルのコサイン類似度とノルム，Self と Recon の類似度．
    """
    import torch
    import torch.nn.functional as F
    from model import load_model
    from utils.inference.image_processing import normalize_and_torch_batch

    eval_data = prepare.load()
    models = ghost.load_models(with_dex=False)
    vae = load_model()
    index = eval_data.reproduction_index
    target224 = np.asarray(eval_data.target224[index])
    resized256 = np.asarray(eval_data.resized256[index])
    batches = range(0, len(index), settings.BATCH_SIZE)

    def generated_cosine(sources: torch.Tensor, reference: torch.Tensor) -> float:
        """ソースの顔ベクトルから生成した画像の，基準の顔ベクトルに対する類似度の平均を返す．"""
        cosines = []
        for start in batches:
            stop = start + settings.BATCH_SIZE
            generated = ghost.generate(models, resized256[start:stop], sources[start:stop])
            v_gen = ghost.embed(models, ghost.resize_to_crop(models, generated))
            cosines.append(F.cosine_similarity(v_gen, reference[start:stop], dim=1))
        return float(torch.cat(cosines).mean())

    with torch.no_grad():
        v_rgb = torch.cat([ghost.embed(models, target224[s:s + settings.BATCH_SIZE]) for s in batches])
        v_bgr = torch.cat([
            models.arcface(F.interpolate(normalize_and_torch_batch(target224[s:s + settings.BATCH_SIZE]),
                                         scale_factor=0.5, mode="bilinear", align_corners=True))
            for s in batches
        ])
        recon_rgb, recon_bgr = vae.decode(vae.encode(v_rgb)[0]), vae.decode(vae.encode(v_bgr)[0])
        result = {
            "num_images": int(len(index)),
            "cos_bgr_vs_rgb_mean": float(F.cosine_similarity(v_bgr, v_rgb, dim=1).mean()),
            "cos_bgr_vs_rgb_min": float(F.cosine_similarity(v_bgr, v_rgb, dim=1).min()),
            "norm_rgb": float(v_rgb.norm(dim=1).mean()), "norm_bgr": float(v_bgr.norm(dim=1).mean()),
            "self_cos_gen_source_rgb": generated_cosine(v_rgb, v_rgb),
            "self_cos_gen_source_bgr": generated_cosine(v_bgr, v_rgb),
            "recon_vector_cos_input_rgb": float(F.cosine_similarity(recon_rgb, v_rgb, dim=1).mean()),
            "recon_vector_cos_input_bgr": float(F.cosine_similarity(recon_bgr, v_rgb, dim=1).mean()),
            "recon_cos_gen_input_rgb": generated_cosine(recon_rgb, v_rgb),
            "recon_cos_gen_input_bgr": generated_cosine(recon_bgr, v_rgb),
        }
    target_dir = settings.condition_dir("prechecks", "default", settings.SEED)
    settings.save_json(result, os.path.join(target_dir, "channel_order_check.json"))
    return result
