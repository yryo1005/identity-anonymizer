"""
パラメータの較正．

本実験に用いる1,000枚について，ベクトル空間で各方式 (M0，M2，M3u) のパラメータを細かく掃引し，
再識別率 top-1 が目標値 (0.9，0.5，0.1) に最も近くなるパラメータを選ぶ．
サンプリングと再識別率の計算は ex002 の実装をそのまま用いる．

    python programs/ex003_ghost_image_eval/calibrate.py
"""

import os
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

import prepare
import samplers
import settings
from attributes import load_ranking
from evaluate import reidentification
from machine_learning_utils import set_seed
from model import encode_all, load_model

METHOD = "calibration"
HYPERPARAMETER = f"n{settings.NUM_IMAGES}_K{settings.CALIBRATION_NUM_SAMPLES}"


def candidate_settings() -> List[samplers.Setting]:
    """
    較正のために掃引する設定を列挙する．

    引数:
        なし．
    戻り値:
        candidates (list[samplers.Setting]): M0 (alpha: 2.0〜7.0，0.1刻み)，M2 (k: 0〜128の全整数)，
            M3u (gamma: 0〜0.9，0.01刻み) の設定のリスト．
    """
    candidates = [samplers.Setting("M0", "M0", "alpha", float(a), alpha=float(a)) for a in np.round(np.arange(2.0, 7.01, 0.1), 2)]
    candidates += [samplers.Setting("M2", "M2", "k", k, k=k) for k in range(0, 129)]
    candidates += [samplers.Setting("M3u", "M3u", "gamma", float(g), k=0, gamma=float(g)) for g in np.round(np.arange(0.0, 0.901, 0.01), 2)]
    return candidates


@torch.no_grad()
def sweep(eval_data: prepare.EvalData) -> pd.DataFrame:
    """
    全ての候補について，ベクトル空間での類似度と再識別率を計算する．

    引数:
        eval_data (prepare.EvalData): 評価用データ一式．
    戻り値:
        table (pd.DataFrame): 候補ごとに1行の表 (method，param_name，param，cos_v，reid_top1，reid_top10，norm)．
    """
    import common as ex002_common

    ex002_common.NUM_SAMPLES = settings.CALIBRATION_NUM_SAMPLES
    index = eval_data.main_index
    vae = load_model()
    v = torch.from_numpy(eval_data.v[index]).cuda()
    mu, std = encode_all(vae, eval_data.v[index])
    v0 = vae.decode(mu)
    group = torch.from_numpy(eval_data.group[index]).cuda()
    gallery = F.normalize(torch.from_numpy(eval_data.gallery_v).cuda(), dim=1)
    gallery_group = torch.from_numpy(eval_data.gallery_group).cuda()
    ranking = load_ranking()

    set_seed(settings.SEED)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(settings.SEED)
    rows = []
    for setting in tqdm(candidate_settings(), desc="calibrate", leave=False):
        mask = samplers.kept_mask(setting, ranking, ranking)
        v_tilde = samplers.sample(setting, vae, v, v0, mu, std, mask, generator, None)
        reid = reidentification(v_tilde, gallery, gallery_group, group)
        rows.append({
            "method": setting.method, "param_name": setting.param_name, "param": setting.param,
            "cos_v": float(F.cosine_similarity(v_tilde, v.unsqueeze(1), dim=2).mean()),
            "reid_top1": float(reid["reid_top1"].mean()), "reid_top10": float(reid["reid_top10"].mean()),
            "norm": float(v_tilde.norm(dim=2).mean()),
        })
    return pd.DataFrame(rows)


def choose(table: pd.DataFrame) -> pd.DataFrame:
    """
    方式ごと・目標値ごとに，再識別率 top-1 が目標値に最も近いパラメータを選ぶ．

    引数:
        table (pd.DataFrame): `sweep` の戻り値．
    戻り値:
        chosen (pd.DataFrame): (方式, 目標値) ごとに1行の表．選んだパラメータと，そのときのベクトル空間での
            類似度・再識別率，および掃引の範囲で到達できた再識別率の最小値を含む．
    """
    rows = []
    for method, rows_of_method in table.groupby("method", sort=False):
        for target in settings.TARGET_REID:
            best = rows_of_method.iloc[(rows_of_method["reid_top1"] - target).abs().argmin()]
            rows.append({
                "method": method, "target_reid_top1": target, "param_name": best["param_name"], "param": best["param"],
                "vector_cos_v": best["cos_v"], "vector_reid_top1": best["reid_top1"],
                "vector_reid_top10": best["reid_top10"],
                "reachable": bool(abs(best["reid_top1"] - target) < 0.05),
                "min_reid_top1_in_sweep": float(rows_of_method["reid_top1"].min()),
            })
    return pd.DataFrame(rows)


def run() -> None:
    """
    較正を実行し，掃引の結果と選んだパラメータを保存する．

    引数:
        なし．
    戻り値:
        なし．
    """
    target_dir = settings.condition_dir(METHOD, HYPERPARAMETER, settings.SEED)
    if settings.is_finished(target_dir):
        print(f"実行済みのためスキップ: {target_dir}")
    else:
        table = sweep(prepare.load())
        table.to_csv(os.path.join(target_dir, "sweep.csv"), index=False)
        choose(table).to_csv(os.path.join(target_dir, "chosen.csv"), index=False)
        settings.save_json({"seed": settings.SEED, "num_candidates": int(len(table))}, os.path.join(target_dir, "meta.json"))
    print(load_chosen().round(4).to_string(index=False))


def load_chosen() -> pd.DataFrame:
    """
    選んだパラメータの表を読み込む．

    引数:
        なし．
    戻り値:
        chosen (pd.DataFrame): `choose` の戻り値．
    """
    return pd.read_csv(os.path.join(settings.OUTPUT_DIR, METHOD, HYPERPARAMETER, str(settings.SEED), "chosen.csv"))


if __name__ == "__main__":
    run()
