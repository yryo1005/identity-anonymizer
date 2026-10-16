"""
評価に用いる画像の準備．

1. ex002 の評価用データ (4,614枚) の全画像について，匿名化の対象となる顔を切り出してキャッシュする．
2. 本実験に用いる1,000枚を，性別・人種・年代で層化抽出する．
3. 予稿の再現に用いる200枚を，単純無作為抽出する．

    python programs/ex003_ghost_image_eval/prepare.py --num_workers 4
"""

import argparse
import multiprocessing as mp
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

import settings


@dataclass
class EvalData:
    """
    評価用データ一式．E は評価用データの画像数 (4,614)，G は探索対象の全画像数 (23,125) である．

    属性:
        paths (list[str]): 要素数 E の画像の相対パス．
        v (np.ndarray): 形状 (E, 512) の元の顔ベクトル (ex001 のキャッシュ)．
        age, gender, race (np.ndarray): 形状 (E,) のラベル．
        group (np.ndarray): 形状 (E,) の重複グループの番号．
        target224 (np.ndarray): 形状 (E, 224, 224, 3) の切り出した顔画像 (BGR，uint8)．
        resized256 (np.ndarray): 形状 (E, 256, 256, 3) の生成器のターゲット (BGR，uint8)．
        main_index (np.ndarray): 本実験に用いる画像の番号 (E の中での位置，要素数 1,000)．
        reproduction_index (np.ndarray): 予稿の再現に用いる画像の番号 (要素数 200)．
        train_v (np.ndarray): 形状 (N_train, 512) の学習用データの顔ベクトル．
        gallery_v (np.ndarray): 形状 (G, 512) の探索対象の顔ベクトル．
        gallery_group (np.ndarray): 形状 (G,) の探索対象の重複グループの番号．
    """

    paths: List[str]
    v: np.ndarray
    age: np.ndarray
    gender: np.ndarray
    race: np.ndarray
    group: np.ndarray
    target224: np.ndarray
    resized256: np.ndarray
    main_index: np.ndarray
    reproduction_index: np.ndarray
    train_v: np.ndarray
    gallery_v: np.ndarray
    gallery_group: np.ndarray


def labeled_paths() -> List[str]:
    """
    ラベルを取得できた画像の相対パスを，ex002 の `Dataset` と同じ順序で返す．

    引数:
        なし．
    戻り値:
        paths (list[str]): 画像の相対パスのリスト．
    """
    import common as ex002_common

    with open(os.path.join(settings.ARCFACE_CACHE_DIR, "paths.txt"), "r", encoding="utf-8") as f:
        paths = f.read().split("\n")
    has_label = np.load(os.path.join(ex002_common.DATASET_DIR, "has_label.npy"))
    return [path for path, keep in zip(paths, has_label) if keep]


def crop_worker(args: Tuple[List[str], int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    画像のパスの一覧から，顔を切り出す (1プロセス分の処理)．

    引数:
        args (tuple): 以下の3要素のタプル．
            paths (list[str]): 画像の相対パスのリスト (要素数 M)．
            worker_index (int): プロセスの番号 (tqdmの表示位置に用いる)．
            num_threads (int): このプロセスが使用するCPUスレッド数．
    戻り値:
        target224 (np.ndarray): 形状 (M, 224, 224, 3) の切り出した顔画像．検出に失敗した行は0で埋める．
        resized256 (np.ndarray): 形状 (M, 256, 256, 3) の生成器のターゲット．
        detected (np.ndarray): 形状 (M,) のbool配列．顔を検出できた行が True．
    """
    import cv2
    from tqdm import tqdm

    import ghost

    paths, worker_index, num_threads = args
    models = ghost.load_models(with_generator=False, with_dex=False, num_threads=num_threads)
    target224 = np.zeros((len(paths), 224, 224, 3), dtype=np.uint8)
    resized256 = np.zeros((len(paths), 256, 256, 3), dtype=np.uint8)
    detected = np.zeros(len(paths), dtype=bool)
    for i, path in enumerate(tqdm(paths, desc=f"crop[{worker_index}]", position=worker_index, leave=False)):
        crops = ghost.crop_target(models, cv2.imread(os.path.join(settings.REPO_ROOT, path)))
        if crops is None:
            continue
        target224[i], resized256[i], detected[i] = crops[0], crops[1], True
    return target224, resized256, detected


def stratified_sample(strata: np.ndarray, num_samples: int, rng: np.random.Generator) -> np.ndarray:
    """
    層ごとの枚数がなるべく均等になるように層化抽出する．枚数が足りない層は全て採用し，
    不足分を残りの層に均等に再配分する．

    引数:
        strata (np.ndarray): 形状 (E,) の層の番号．
        num_samples (int): 抽出する枚数．
        rng (np.random.Generator): 乱数生成器．
    戻り値:
        index (np.ndarray): 形状 (num_samples,) の抽出した画像の番号 (昇順)．
    """
    labels, counts = np.unique(strata, return_counts=True)
    quota = np.zeros(len(labels), dtype=int)
    open_strata = np.ones(len(labels), dtype=bool)
    remaining = num_samples
    while remaining > 0 and open_strata.any():
        share = max(remaining // open_strata.sum(), 1)
        for i in rng.permutation(np.nonzero(open_strata)[0]):
            take = min(share, counts[i] - quota[i], remaining)
            quota[i] += take
            remaining -= take
            open_strata[i] = quota[i] < counts[i]
            if remaining == 0:
                break
    picked = [rng.choice(np.nonzero(strata == label)[0], size=q, replace=False) for label, q in zip(labels, quota)]
    return np.sort(np.concatenate(picked))


def build(num_workers: int = 4) -> None:
    """
    顔の切り出しのキャッシュと，画像の抽出結果を作成して保存する．

    引数:
        num_workers (int): 顔の切り出しを並列に実行するプロセス数．
    戻り値:
        なし．
    """
    os.makedirs(settings.CROP_DIR, exist_ok=True)
    if settings.is_finished(settings.CROP_DIR):
        print(f"作成済みのためスキップ: {settings.CROP_DIR}")
        return

    from data import age_bin, load_dataset

    dataset = load_dataset()
    eval_position = np.nonzero(dataset.is_eval)[0]
    paths = [labeled_paths()[i] for i in eval_position]

    num_threads = max(1, (os.cpu_count() or 1) // num_workers)
    shards = [list(shard) for shard in np.array_split(np.array(paths, dtype=object), num_workers)]
    with mp.get_context("spawn").Pool(num_workers) as pool:
        results = pool.map(crop_worker, [(shard, i, num_threads) for i, shard in enumerate(shards)])
    target224 = np.concatenate([result[0] for result in results])
    resized256 = np.concatenate([result[1] for result in results])
    detected = np.concatenate([result[2] for result in results])

    rng = np.random.default_rng(settings.SEED)
    age, gender, race = (getattr(dataset, name)[eval_position] for name in ("age", "gender", "race"))
    strata = gender * 100 + race * 10 + age_bin(age)
    candidates = np.nonzero(detected)[0]
    main_index = candidates[stratified_sample(strata[candidates], settings.NUM_IMAGES, rng)]
    reproduction_index = np.sort(rng.choice(candidates, size=settings.NUM_REPRODUCTION_IMAGES, replace=False))

    np.save(os.path.join(settings.CROP_DIR, "target224.npy"), target224)
    np.save(os.path.join(settings.CROP_DIR, "resized256.npy"), resized256)
    np.save(os.path.join(settings.CROP_DIR, "eval_position.npy"), eval_position)
    np.save(os.path.join(settings.CROP_DIR, "main_index.npy"), main_index)
    np.save(os.path.join(settings.CROP_DIR, "reproduction_index.npy"), reproduction_index)
    meta = {
        "num_eval_images": int(len(paths)), "num_detected": int(detected.sum()),
        "num_main": int(len(main_index)), "num_reproduction": int(len(reproduction_index)),
        "num_strata_total": int(len(np.unique(strata))),
        "main_stratum_size_min_max": [int(np.bincount(np.unique(strata[main_index], return_inverse=True)[1]).min()),
                                      int(np.bincount(np.unique(strata[main_index], return_inverse=True)[1]).max())],
        "main_gender_counts": np.bincount(gender[main_index]).tolist(),
        "main_race_counts": np.bincount(race[main_index]).tolist(),
        "main_age_bin_counts": np.bincount(age_bin(age[main_index]), minlength=8).tolist(),
        "eval_gender_counts": np.bincount(gender).tolist(),
        "eval_race_counts": np.bincount(race).tolist(),
        "eval_age_bin_counts": np.bincount(age_bin(age), minlength=8).tolist(),
        "seed": settings.SEED,
    }
    settings.save_json(meta, os.path.join(settings.CROP_DIR, "meta.json"))
    print("\n" + "\n".join(f"{key}: {value}" for key, value in meta.items()))


def load() -> EvalData:
    """
    評価用データ一式を読み込む．

    引数:
        なし．
    戻り値:
        eval_data (EvalData): 評価用データ一式．
    """
    from data import load_dataset

    dataset = load_dataset()
    eval_position = np.load(os.path.join(settings.CROP_DIR, "eval_position.npy"))
    all_paths = labeled_paths()
    return EvalData(
        paths=[all_paths[i] for i in eval_position],
        v=dataset.v[eval_position], age=dataset.age[eval_position], gender=dataset.gender[eval_position],
        race=dataset.race[eval_position], group=dataset.group[eval_position],
        target224=np.load(os.path.join(settings.CROP_DIR, "target224.npy"), mmap_mode="r"),
        resized256=np.load(os.path.join(settings.CROP_DIR, "resized256.npy"), mmap_mode="r"),
        main_index=np.load(os.path.join(settings.CROP_DIR, "main_index.npy")),
        reproduction_index=np.load(os.path.join(settings.CROP_DIR, "reproduction_index.npy")),
        train_v=dataset.v[~dataset.is_eval], gallery_v=dataset.gallery_v, gallery_group=dataset.gallery_group,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="評価用画像の顔の切り出しと抽出を行う")
    parser.add_argument("--num_workers", type=int, default=4)
    build(num_workers=parser.parse_args().num_workers)
