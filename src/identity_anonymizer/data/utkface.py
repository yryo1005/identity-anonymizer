import os
from typing import List, Tuple

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from identity_anonymizer.faceswap.models import GhostModels
from utils.inference.image_processing import crop_face, normalize_and_torch_batch


def list_utkface_image_paths(image_dir: str) -> List[str]:
    """
    UTKFaceのディレクトリから，顔画像のパス一覧を取得する．

    引数:
        image_dir (str): UTKFaceの画像が格納されたディレクトリのパス．
    戻り値:
        image_paths (list[str]): 画像ファイルのパスのリスト．
    """
    return sorted(
        os.path.join(image_dir, name)
        for name in os.listdir(image_dir)
        if name.lower().endswith(".jpg")
    )


def extract_arcface_embeddings(
    image_paths: List[str],
    models: GhostModels,
    batch_size: int = 64,
) -> np.ndarray:
    """
    顔画像のパス一覧から，バッチ処理でArcFaceの顔ベクトルを抽出する．

    引数:
        image_paths (list[str]): 画像ファイルのパスのリスト．
        models (GhostModels): `load_ghost_models` で読み込んだモデル一式(顔検出・ArcFaceを使用)．
        batch_size (int): 1回のGPU処理でまとめて扱う画像数．
    戻り値:
        embeddings (np.ndarray): 形状 (N, 512) のArcFace顔ベクトル(顔検出に失敗した画像は除外
            されるため N <= len(image_paths))．
    """
    all_embeddings = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="extract_arcface_embeddings", leave=False):
        batch_paths = image_paths[i:i + batch_size]
        faces = []
        for path in batch_paths:
            image_full = cv2.imread(path)
            try:
                face = crop_face(image_full, models.app, models.crop_size)[0]
                faces.append(face[:, :, ::-1])
            except TypeError:
                continue

        if len(faces) == 0:
            continue

        faces_norm = normalize_and_torch_batch(np.array(faces))
        with torch.no_grad():
            embeds = models.arcface(F.interpolate(faces_norm, scale_factor=0.5, mode="bilinear", align_corners=True))
        all_embeddings.extend(embeds.cpu().numpy())

    return np.array(all_embeddings)


def build_face_embedding_dataset(
    image_dir: str,
    models: GhostModels,
    output_dir: str,
    test_size: float = 0.1,
    seed: int = 0,
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    UTKFaceの顔画像からArcFace顔ベクトルを抽出し，学習用・検証用に分割して保存する．
    (`01_make_dataset.ipynb` から利用される)

    引数:
        image_dir (str): UTKFaceの画像が格納されたディレクトリのパス．
        models (GhostModels): `load_ghost_models` で読み込んだモデル一式．
        output_dir (str): 抽出した顔ベクトルの `.npy` ファイルを保存するディレクトリ．
        test_size (float): 検証用データの割合．
        seed (int): 学習用・検証用データの分割に用いるseed値．
        batch_size (int): ArcFaceへ入力する際のバッチサイズ．
    戻り値:
        train_embeddings (np.ndarray): 形状 (N_train, 512) の学習用顔ベクトル．
        test_embeddings (np.ndarray): 形状 (N_test, 512) の検証用顔ベクトル．
    """
    image_paths = list_utkface_image_paths(image_dir)
    embeddings = extract_arcface_embeddings(image_paths, models, batch_size=batch_size)

    train_embeddings, test_embeddings = train_test_split(embeddings, test_size=test_size, random_state=seed)

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "source_embeds.npy"), embeddings)
    np.save(os.path.join(output_dir, "train_source_embeds.npy"), train_embeddings)
    np.save(os.path.join(output_dir, "test_source_embeds.npy"), test_embeddings)

    return train_embeddings, test_embeddings
