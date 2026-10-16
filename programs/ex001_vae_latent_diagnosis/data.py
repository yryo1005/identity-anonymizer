"""
UTKFaceの全画像からArcFaceの顔ベクトルを抽出してキャッシュし，実験から読み込むためのモジュール．

前処理は既存の `identity_anonymizer.data.utkface.extract_arcface_embeddings_with_paths` と同一
(顔検出 → `crop_face` で 224x224 に切り出し → 1/2 に縮小 → ArcFace(iresnet100)) である．
既存の `load_ghost_models` はランドマーク検出器のために mxnet を import するが，本実験では
顔検出器とArcFaceしか使わないため，この2つだけを本モジュール内で読み込む．

単体で実行するとキャッシュを作成する．

    python programs/ex001_vae_latent_diagnosis/data.py --num_workers 4
"""

import argparse
import multiprocessing as mp
import os
import time
from typing import List, Tuple

import numpy as np

import common


def list_image_paths(image_dir: str) -> List[str]:
    """
    UTKFaceのディレクトリから，拡張子が `.jpg` の画像パスを辞書順で取得する．

    引数:
        image_dir (str): UTKFaceの画像が格納されたディレクトリのパス．
    戻り値:
        image_paths (list[str]): 画像ファイルのパスのリスト (要素数は画像の総数)．
    """
    return sorted(
        os.path.join(image_dir, name)
        for name in os.listdir(image_dir)
        if name.lower().endswith(".jpg")
    )


def _patch_onnxruntime_providers(num_threads: int = 0) -> None:
    """
    insightface==0.2.1 は `onnxruntime.InferenceSession` を providers 無しで生成するが，
    onnxruntime-gpu>=1.9 ではこれが例外になる．このプロセス内に限り，providers の既定値を
    補うラッパーへ差し替える (site-packages や既存コードは書き換えない)．
    あわせて，複数プロセスで並列に実行した際にCPUスレッドが過剰に競合しないよう，
    1セッションあたりのスレッド数を制限する．

    引数:
        num_threads (int): 1セッションが使用するCPUスレッド数．0 の場合は制限しない．
    戻り値:
        なし．
    """
    import onnxruntime

    original_session = onnxruntime.InferenceSession
    if getattr(original_session, "_ex001_patched", False):
        return

    class _SessionWithDefaultProviders(original_session):
        _ex001_patched = True

        def __init__(self, path, sess_options=None, providers=None, **kwargs):
            providers = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if sess_options is None and num_threads > 0:
                sess_options = onnxruntime.SessionOptions()
                sess_options.intra_op_num_threads = num_threads
            super().__init__(path, sess_options, providers=providers, **kwargs)

    onnxruntime.InferenceSession = _SessionWithDefaultProviders


def load_face_models(det_thresh: float = 0.6, det_size: Tuple[int, int] = (640, 640), num_threads: int = 0):
    """
    顔検出器(SCRFD)とArcFace(iresnet100)を読み込む．設定値は既存の `load_ghost_models` と同じ．

    引数:
        det_thresh (float): 顔検出の信頼度の閾値．
        det_size (tuple[int, int]): 顔検出時の入力解像度．
        num_threads (int): 顔検出器が使用するCPUスレッド数．0 の場合は制限しない．
    戻り値:
        app (Face_detect_crop): 顔検出・切り出しモデル．
        arcface (torch.nn.Module): GPU上に配置した評価モードのArcFace．
    """
    import cv2
    import torch

    if num_threads > 0:
        cv2.setNumThreads(num_threads)
        torch.set_num_threads(num_threads)
    _patch_onnxruntime_providers(num_threads)
    from identity_anonymizer import config
    from arcface_model.iresnet import iresnet100
    from insightface_func.face_detect_crop_multi import Face_detect_crop

    app = Face_detect_crop(name="antelope", root=config.GHOST_WEIGHTS_DIR)
    app.prepare(ctx_id=0, det_thresh=det_thresh, det_size=det_size)

    arcface = iresnet100(fp16=False)
    weight = torch.load(os.path.join(config.GHOST_WEIGHTS_DIR, "backbone.pth"), map_location="cpu")
    arcface.load_state_dict(weight)
    return app, arcface.cuda().eval()


def _crop_faces(batch_paths: List[str], app, crop_size: int) -> Tuple[list, List[str]]:
    """
    1バッチ分の画像から顔を切り出す．顔検出に失敗した画像は除外する．

    引数:
        batch_paths (list[str]): 画像パスのリスト (要素数 B)．
        app (Face_detect_crop): 顔検出・切り出しモデル．
        crop_size (int): 切り出す顔画像の一辺のサイズ．
    戻り値:
        faces (list[np.ndarray]): 形状 (crop_size, crop_size, 3) のRGB顔画像のリスト (要素数 B' <= B)．
        kept_paths (list[str]): `faces` の各要素に対応する画像パス (要素数 B')．
    """
    import cv2
    from utils.inference.image_processing import crop_face

    faces, kept_paths = [], []
    for path in batch_paths:
        image_full = cv2.imread(path)
        try:
            face = crop_face(image_full, app, crop_size)[0]
        except TypeError:
            # 顔が検出されなかった場合 (app.get が None を返す)
            continue
        faces.append(face[:, :, ::-1])
        kept_paths.append(path)
    return faces, kept_paths


def extract_embeddings(args: Tuple[List[str], int, int, int, int]) -> Tuple[np.ndarray, List[str], dict]:
    """
    画像パスの一覧からArcFaceの顔ベクトルを抽出する (1プロセス分の処理)．

    引数:
        args (tuple): 以下の5要素のタプル．
            image_paths (list[str]): 画像パスのリスト (要素数 M)．
            batch_size (int): ArcFaceへ入力するバッチサイズ．
            crop_size (int): 切り出す顔画像の一辺のサイズ．
            worker_index (int): プロセスの番号 (tqdmの表示位置に用いる)．
            num_threads (int): このプロセスが使用するCPUスレッド数．
    戻り値:
        embeddings (np.ndarray): 形状 (M', 512) の顔ベクトル (float32，M' <= M)．
        kept_paths (list[str]): `embeddings` の各行に対応する画像パス (要素数 M')．
        usage (dict): このプロセスのDRAM・VRAMの最大使用量 [MiB]．
    """
    import psutil
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm

    image_paths, batch_size, crop_size, worker_index, num_threads = args
    app, arcface = load_face_models(num_threads=num_threads)
    from utils.inference.image_processing import normalize_and_torch_batch

    all_embeddings, all_kept_paths = [], []
    starts = range(0, len(image_paths), batch_size)
    for start in tqdm(starts, desc=f"arcface[{worker_index}]", position=worker_index, leave=False):
        faces, kept_paths = _crop_faces(image_paths[start:start + batch_size], app, crop_size)
        if len(faces) == 0:
            continue
        faces_norm = normalize_and_torch_batch(np.array(faces))
        with torch.no_grad():
            faces_small = F.interpolate(faces_norm, scale_factor=0.5, mode="bilinear", align_corners=True)
            embeds = arcface(faces_small)
        all_embeddings.append(embeds.cpu().numpy())
        all_kept_paths.extend(kept_paths)

    usage = {
        "dram_mib": psutil.Process().memory_info().rss / 2 ** 20,
        "vram_mib": torch.cuda.max_memory_allocated() / 2 ** 20,
    }
    return np.concatenate(all_embeddings).astype(np.float32), all_kept_paths, usage


def match_test_split(embeddings: np.ndarray, reference_test_path: str, threshold: float = 0.9999) -> np.ndarray:
    """
    各顔ベクトルが，VAEの学習時に検証用として分割されたデータに含まれるかを判定する．
    `data/processed/test_source_embeds.npy` の各行とのコサイン類似度の最大値が閾値を超えるものを
    検証用データとみなす．

    引数:
        embeddings (np.ndarray): 形状 (N, 512) の顔ベクトル．
        reference_test_path (str): 学習時の検証用顔ベクトル (形状 (N_test, 512)) の `.npy` のパス．
        threshold (float): 同一ベクトルとみなすコサイン類似度の閾値．
    戻り値:
        is_test (np.ndarray): 形状 (N,) のbool配列．検証用データに一致した行が True．
    """
    import torch

    reference = torch.from_numpy(np.load(reference_test_path)).cuda()
    reference = torch.nn.functional.normalize(reference, dim=1)
    target = torch.nn.functional.normalize(torch.from_numpy(embeddings).cuda(), dim=1)
    best = torch.cat([(chunk @ reference.T).max(dim=1).values for chunk in target.split(4096)])
    return (best > threshold).cpu().numpy()


def build_arcface_cache(num_workers: int = 4, batch_size: int = 64, crop_size: int = 224) -> None:
    """
    UTKFaceの全画像からArcFaceの顔ベクトルを抽出し，`datasets/` 以下にキャッシュする．
    画像を `num_workers` 個に等分割し，プロセスごとにモデルを読み込んで並列に抽出する．

    引数:
        num_workers (int): 並列に実行するプロセス数．
        batch_size (int): ArcFaceへ入力するバッチサイズ．
        crop_size (int): 切り出す顔画像の一辺のサイズ (GHOSTの前提である224)．
    戻り値:
        なし (`embeddings.npy`，`paths.txt`，`is_test.npy`，`meta.json` を保存する)．
    """
    os.makedirs(common.DATASET_DIR, exist_ok=True)
    if common.is_finished(common.DATASET_DIR):
        print(f"キャッシュ済みのためスキップ: {common.DATASET_DIR}")
        return

    image_paths = list_image_paths(common.UTKFACE_IMAGE_DIR)
    shards = [list(shard) for shard in np.array_split(np.array(image_paths, dtype=object), num_workers)]
    num_threads = max(1, (os.cpu_count() or 1) // num_workers)
    jobs = [(shard, batch_size, crop_size, i, num_threads) for i, shard in enumerate(shards)]

    start_time = time.time()
    with mp.get_context("spawn").Pool(num_workers) as pool:
        results = pool.map(extract_embeddings, jobs)
    elapsed = time.time() - start_time

    embeddings = np.concatenate([result[0] for result in results])
    kept_paths = [path for result in results for path in result[1]]
    is_test = match_test_split(embeddings, common.REFERENCE_TEST_EMBEDS_PATH)

    np.save(os.path.join(common.DATASET_DIR, "embeddings.npy"), embeddings)
    np.save(os.path.join(common.DATASET_DIR, "is_test.npy"), is_test)
    with open(os.path.join(common.DATASET_DIR, "paths.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(os.path.relpath(path, common.REPO_ROOT) for path in kept_paths))
    common.save_json({
        "num_images": len(image_paths),
        "num_embeddings": int(len(embeddings)),
        "num_detection_failed": int(len(image_paths) - len(embeddings)),
        "num_matched_test_split": int(is_test.sum()),
        "num_workers": num_workers,
        "batch_size": batch_size,
        "elapsed_sec": elapsed,
        "max_dram_mib_per_worker": max(result[2]["dram_mib"] for result in results),
        "max_vram_mib_per_worker": max(result[2]["vram_mib"] for result in results),
    }, os.path.join(common.DATASET_DIR, "meta.json"))
    print(f"\n抽出完了: {len(embeddings)} / {len(image_paths)} 枚 ({elapsed:.0f} 秒)")


def load_arcface_cache() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    キャッシュ済みのArcFaceの顔ベクトルを読み込む．

    引数:
        なし．
    戻り値:
        embeddings (np.ndarray): 形状 (N, 512) の顔ベクトル (float32，正規化はしていない)．
        is_test (np.ndarray): 形状 (N,) のbool配列．VAEの学習時に検証用だった画像が True．
        paths (list[str]): 各行に対応する画像の相対パス (要素数 N)．
    """
    embeddings = np.load(os.path.join(common.DATASET_DIR, "embeddings.npy"))
    is_test = np.load(os.path.join(common.DATASET_DIR, "is_test.npy"))
    with open(os.path.join(common.DATASET_DIR, "paths.txt"), "r", encoding="utf-8") as f:
        paths = f.read().split("\n")
    return embeddings, is_test, paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UTKFaceのArcFace顔ベクトルをキャッシュする")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)
    build_arcface_cache(num_workers=parser.parse_args().num_workers, batch_size=parser.parse_args().batch_size)
