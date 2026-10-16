"""
生成画像での顔検出の成功率．

本実験で保存した生成画像 (256x256) に，生成時と同じ顔検出器を適用し，顔を検出できた割合を求める．
生成画像は顔が画面いっぱいに写っており，そのままでは検出器が想定する顔の大きさを外れて，元の切り出し
画像でも全く検出できない．そのため，周囲に幅 PADDING の黒い余白を付けてから検出する
(この条件で，元の切り出し画像の検出率が 1.0 になることを確認している)．

    python programs/ex003_ghost_image_eval/detect.py --num_workers 4
"""

import argparse
import multiprocessing as mp
import os
from typing import List, Tuple

import numpy as np
import pandas as pd

import settings


#: 検出の前に画像の周囲に付ける黒い余白の幅 [画素]
PADDING = 64
#: 設定ごとに検出を行う画像数 (先頭から)
NUM_DETECTION_IMAGES = 300


def detect_worker(args: Tuple[List[str], int, int]) -> List[dict]:
    """
    生成画像のファイルの一覧について，顔検出の成功率を計算する (1プロセス分の処理)．

    引数:
        args (tuple): 以下の3要素のタプル．
            files (list[str]): 生成画像を保存した `.npy` ファイルのパスのリスト．
            worker_index (int): プロセスの番号 (tqdmの表示位置に用いる)．
            num_threads (int): このプロセスが使用するCPUスレッド数．
    戻り値:
        rows (list[dict]): ファイルごとの，設定の名前，枚数，検出できた枚数，成功率の辞書のリスト．
    """
    import cv2
    from tqdm import tqdm

    import ghost

    files, worker_index, num_threads = args
    models = ghost.load_models(with_generator=False, with_dex=False, num_threads=num_threads)
    rows = []
    for path in tqdm(files, desc=f"detect[{worker_index}]", position=worker_index, leave=False):
        images = np.load(path, mmap_mode="r")[:NUM_DETECTION_IMAGES]
        padded = (cv2.copyMakeBorder(np.asarray(image), PADDING, PADDING, PADDING, PADDING, cv2.BORDER_CONSTANT)
                  for image in images)
        detected = sum(models.app.get(image, models.crop_size) is not None for image in padded)
        rows.append({"key": os.path.basename(path)[:-4].replace("__", "|"), "num_images": len(images),
                     "num_detected": int(detected), "detection_rate": detected / len(images)})
    return rows


def run(num_workers: int = 4) -> None:
    """
    本実験の全ての設定について顔検出の成功率を計算し，保存する．

    引数:
        num_workers (int): 並列に実行するプロセス数．
    戻り値:
        なし．
    """
    main_dir = os.path.join(settings.OUTPUT_DIR, "main", f"n{settings.NUM_IMAGES}", str(settings.SEED))
    output_path = os.path.join(main_dir, "detection.csv")
    if os.path.exists(output_path):
        print(pd.read_csv(output_path).round(4).to_string(index=False))
        return
    image_dir = os.path.join(main_dir, "generated")
    files = sorted(os.path.join(image_dir, name) for name in os.listdir(image_dir) if name.endswith(".npy"))
    num_threads = max(1, (os.cpu_count() or 1) // num_workers)
    shards = [list(shard) for shard in np.array_split(np.array(files, dtype=object), num_workers)]
    with mp.get_context("spawn").Pool(num_workers) as pool:
        results = pool.map(detect_worker, [(shard, i, num_threads) for i, shard in enumerate(shards)])
    table = pd.DataFrame([row for rows in results for row in rows])
    table.to_csv(output_path, index=False)
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成画像での顔検出の成功率を計算する")
    parser.add_argument("--num_workers", type=int, default=4)
    run(num_workers=parser.parse_args().num_workers)
