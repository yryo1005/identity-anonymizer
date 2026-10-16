"""
ex001で抽出したArcFaceの顔ベクトルに，UTKFaceのラベル，重複グループ，学習用・評価用の分割を
対応付けて保存・読み込みするモジュール．

単体で実行すると `datasets/ex002_partial_prior_sampling/labels_groups_split/` を作成する．

    python programs/ex002_partial_prior_sampling/data.py
"""

import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import GroupShuffleSplit

import common

#: UTKFaceのファイル名の正規の形式 `[age]_[gender]_[race]_[date].jpg`
FILENAME_PATTERN = re.compile(r"^(\d{1,3})_([01])_([0-4])_(\d+)\.jpg$", re.IGNORECASE)


@dataclass
class Dataset:
    """
    実験で用いるデータ一式．N はラベルを取得できた画像数，G は探索対象の全画像数．

    属性:
        v (np.ndarray): 形状 (N, 512) のArcFaceの顔ベクトル (正規化なし)．
        age (np.ndarray): 形状 (N,) の年齢．
        gender (np.ndarray): 形状 (N,) の性別 (0: 男性，1: 女性)．
        race (np.ndarray): 形状 (N,) の人種 (0〜4)．
        group (np.ndarray): 形状 (N,) の重複グループの番号．
        is_eval (np.ndarray): 形状 (N,) のbool配列．評価用データが True．
        gallery_v (np.ndarray): 形状 (G, 512) の，再識別の探索対象となる全画像の顔ベクトル．
        gallery_group (np.ndarray): 形状 (G,) の，探索対象の各画像の重複グループの番号．
    """

    v: np.ndarray
    age: np.ndarray
    gender: np.ndarray
    race: np.ndarray
    group: np.ndarray
    is_eval: np.ndarray
    gallery_v: np.ndarray
    gallery_group: np.ndarray


def parse_label(path: str) -> Optional[Tuple[int, int, int]]:
    """
    UTKFaceのファイル名から年齢・性別・人種を取得する．形式が崩れている場合は None を返す．

    引数:
        path (str): 画像のパスまたはファイル名．
    戻り値:
        label (tuple[int, int, int] または None): (age, gender, race)．
    """
    match = FILENAME_PATTERN.match(os.path.basename(path))
    if match is None:
        return None
    age, gender, race = (int(match.group(i)) for i in (1, 2, 3))
    return age, gender, race


def age_bin(age: np.ndarray) -> np.ndarray:
    """
    年齢を年代ビン (0-9, 10-19, ..., 70+) の番号に変換する．

    引数:
        age (np.ndarray): 形状 (N,) の年齢．
    戻り値:
        bins (np.ndarray): 形状 (N,) のビン番号 (0〜7)．
    """
    return np.digitize(age, common.AGE_BIN_EDGES[1:])


@torch.no_grad()
def duplicate_groups(v: np.ndarray, threshold: float, chunk_size: int = 2048) -> np.ndarray:
    """
    コサイン類似度が閾値以上の画像の組を辺とするグラフの連結成分を，重複グループとして求める．

    引数:
        v (np.ndarray): 形状 (G, 512) の顔ベクトル．
        threshold (float): 重複とみなすコサイン類似度の閾値．
        chunk_size (int): 1回に類似度行列を計算する行数．
    戻り値:
        group (np.ndarray): 形状 (G,) の重複グループの番号．
    """
    normalized = torch.nn.functional.normalize(torch.from_numpy(v).cuda(), dim=1)
    rows, cols = [], []
    for start in range(0, len(v), chunk_size):
        sim = normalized[start:start + chunk_size] @ normalized.T
        index = torch.nonzero(sim >= threshold)
        rows.append((index[:, 0] + start).cpu().numpy())
        cols.append(index[:, 1].cpu().numpy())
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    graph = coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(len(v), len(v)))
    return connected_components(graph, directed=False)[1]


def build_dataset() -> None:
    """
    ラベルのパース，重複グループ化，重複グループを単位とする8:2の分割を行い，結果を保存する．

    引数:
        なし．
    戻り値:
        なし (`labels.npy`，`has_label.npy`，`group.npy`，`is_eval.npy`，`meta.json` を保存する)．
    """
    os.makedirs(common.DATASET_DIR, exist_ok=True)
    if common.is_finished(common.DATASET_DIR):
        print(f"作成済みのためスキップ: {common.DATASET_DIR}")
        return

    v = np.load(os.path.join(common.ARCFACE_CACHE_DIR, "embeddings.npy"))
    with open(os.path.join(common.ARCFACE_CACHE_DIR, "paths.txt"), "r", encoding="utf-8") as f:
        paths = f.read().split("\n")
    all_names = [name for name in os.listdir(common.UTKFACE_IMAGE_DIR) if name.lower().endswith(".jpg")]

    parsed = [parse_label(path) for path in paths]
    has_label = np.array([label is not None for label in parsed])
    labels = np.array([label if label is not None else (-1, -1, -1) for label in parsed], dtype=np.int64)
    failed_names = [os.path.basename(path) for path, label in zip(paths, parsed) if label is None]

    group = duplicate_groups(v, common.DUPLICATE_COSINE)
    splitter = GroupShuffleSplit(n_splits=1, test_size=common.EVAL_FRACTION, random_state=common.SPLIT_SEED)
    _, eval_index = next(splitter.split(v, groups=group))
    is_eval = np.zeros(len(v), dtype=bool)
    is_eval[eval_index] = True

    group_sizes = np.bincount(group)
    np.save(os.path.join(common.DATASET_DIR, "labels.npy"), labels)
    np.save(os.path.join(common.DATASET_DIR, "has_label.npy"), has_label)
    np.save(os.path.join(common.DATASET_DIR, "group.npy"), group)
    np.save(os.path.join(common.DATASET_DIR, "is_eval.npy"), is_eval)
    meta = {
        "num_jpg_files": len(all_names),
        "num_jpg_files_parse_failed": int(sum(parse_label(name) is None for name in all_names)),
        "num_embeddings": int(len(v)),
        "num_embeddings_parse_failed": int((~has_label).sum()),
        "parse_failed_names_in_embeddings": failed_names,
        "duplicate_cosine_threshold": common.DUPLICATE_COSINE,
        "num_groups": int(group.max() + 1),
        "num_groups_with_duplicates": int((group_sizes > 1).sum()),
        "num_images_in_duplicate_groups": int(group_sizes[group_sizes > 1].sum()),
        "max_group_size": int(group_sizes.max()),
        "num_train": int((~is_eval & has_label).sum()),
        "num_eval": int((is_eval & has_label).sum()),
        "num_groups_in_both_sides": int(len(set(group[is_eval]) & set(group[~is_eval]))),
        "split_seed": common.SPLIT_SEED,
    }
    common.save_json(meta, os.path.join(common.DATASET_DIR, "meta.json"))
    print(json_summary(meta))


def json_summary(meta: dict) -> str:
    """
    メタデータを，1行に1項目の文字列に整形する．

    引数:
        meta (dict): `build_dataset` が保存するメタデータ．
    戻り値:
        text (str): 整形した文字列．
    """
    return "\n".join(f"{key}: {value}" for key, value in meta.items())


def load_dataset() -> Dataset:
    """
    顔ベクトル，ラベル，重複グループ，分割を読み込む．ラベルを取得できなかった画像は，
    属性の評価対象 (v, age など) からは除くが，再識別の探索対象 (gallery) には残す．

    引数:
        なし．
    戻り値:
        dataset (Dataset): 実験で用いるデータ一式．
    """
    gallery_v = np.load(os.path.join(common.ARCFACE_CACHE_DIR, "embeddings.npy"))
    labels = np.load(os.path.join(common.DATASET_DIR, "labels.npy"))
    has_label = np.load(os.path.join(common.DATASET_DIR, "has_label.npy"))
    group = np.load(os.path.join(common.DATASET_DIR, "group.npy"))
    is_eval = np.load(os.path.join(common.DATASET_DIR, "is_eval.npy"))
    return Dataset(
        v=gallery_v[has_label], age=labels[has_label, 0], gender=labels[has_label, 1], race=labels[has_label, 2],
        group=group[has_label], is_eval=is_eval[has_label], gallery_v=gallery_v, gallery_group=group,
    )


def class_counts(values: np.ndarray) -> List[int]:
    """
    クラスごとの件数を数える．

    引数:
        values (np.ndarray): 形状 (N,) の非負整数のクラス番号．
    戻り値:
        counts (list[int]): クラス番号順の件数．
    """
    return np.bincount(values).tolist()


if __name__ == "__main__":
    build_dataset()
    dataset = load_dataset()
    print("gender:", class_counts(dataset.gender), "race:", class_counts(dataset.race),
          "age bins:", class_counts(age_bin(dataset.age)))
