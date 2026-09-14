import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from typing import List, Optional

import psutil
import torch

from identity_anonymizer.evaluation.metrics import AnonymizationEvalResult

# 各workerプロセス内でのみ参照するグローバル状態(プロセスごとに1回だけモデルをロードして使い回す)
_worker_state = {}


def _init_worker(
    ghost_weights_dir: str,
    dex_weights_dir: str,
    anonymizer_name: str,
    anonymizer_weight_path: Optional[str],
) -> None:
    """
    `ProcessPoolExecutor` の各workerプロセスの起動時に1回だけ呼ばれ，GHOST・DEX・Anonymizerの
    各モデルをロードしてプロセスローカルな状態として保持する．

    引数:
        ghost_weights_dir (str): GHOST由来の重みを格納したディレクトリ．
        dex_weights_dir (str): DEXの重みを格納したディレクトリ．
        anonymizer_name (str): 使用する匿名化モデルの識別名(`registry` 参照)．
        anonymizer_weight_path (str または None): 匿名化モデルの学習済み重みのパス．
    戻り値:
        なし．
    """
    from identity_anonymizer.anonymizers.registry import get_anonymizer
    from identity_anonymizer.evaluation.age_gender import load_dex_models
    from identity_anonymizer.faceswap.models import load_ghost_models
    from identity_anonymizer.faceswap.pipeline import FaceAnonymizerPipeline

    models = load_ghost_models(weights_dir=ghost_weights_dir)
    anonymizer = get_anonymizer(anonymizer_name, weight_path=anonymizer_weight_path)

    _worker_state["pipeline"] = FaceAnonymizerPipeline(models, anonymizer)
    _worker_state["dex_models"] = load_dex_models(weights_dir=dex_weights_dir, device="cuda")


def _evaluate_shard(image_paths: List[str], noise_level: float) -> List[AnonymizationEvalResult]:
    """
    1プロセス(1shard)分の画像リストを評価する．`_init_worker` が事前に呼ばれている前提．

    引数:
        image_paths (list[str]): このプロセスが担当する画像パスのリスト．
        noise_level (float): `Anonymizer.anonymize` に渡すスケール係数．
    戻り値:
        results (list[AnonymizationEvalResult]): 顔検出に成功した画像分の評価結果．
    """
    from identity_anonymizer.evaluation.metrics import evaluate_single_image

    pipeline = _worker_state["pipeline"]
    dex_models = _worker_state["dex_models"]

    results = []
    for path in image_paths:
        result = evaluate_single_image(path, pipeline, dex_models, noise_level=noise_level)
        if result is not None:
            results.append(result)
    return results


def _measure_single_worker_resource_usage(
    ghost_weights_dir: str,
    dex_weights_dir: str,
    anonymizer_name: str,
    anonymizer_weight_path: Optional[str],
    sample_image_paths: List[str],
    noise_level: float,
    result_queue: "mp.Queue",
) -> None:
    """
    1プロセス分のパイプラインを実際にロード・実行し，そのプロセスのピークDRAM/VRAM使用量を
    計測して `result_queue` へ書き込む(`determine_worker_count` から専用プロセスとして起動される)．

    親プロセス側の `result_queue.get()` が失敗時に無限に待機し続けることを防ぐため，
    途中で例外が発生した場合も必ず `result_queue` へ結果(またはエラー)を書き込む．
    """
    try:
        _init_worker(ghost_weights_dir, dex_weights_dir, anonymizer_name, anonymizer_weight_path)
        torch.cuda.reset_peak_memory_stats()

        _evaluate_shard(sample_image_paths, noise_level)

        peak_vram_bytes = torch.cuda.max_memory_allocated()
        peak_dram_bytes = psutil.Process(os.getpid()).memory_info().rss
        result_queue.put((peak_dram_bytes, peak_vram_bytes, None))
    except Exception as exc:  # noqa: BLE001  (計測専用プロセスの例外を親プロセスへ伝搬させるため)
        import traceback

        result_queue.put((0, 0, traceback.format_exc()))
        raise exc


def determine_worker_count(
    ghost_weights_dir: str,
    dex_weights_dir: str,
    anonymizer_name: str,
    anonymizer_weight_path: Optional[str],
    sample_image_paths: List[str],
    noise_level: float = 1.0,
    max_workers: Optional[int] = None,
    safety_margin: float = 0.8,
    measurement_timeout: float = 300.0,
) -> int:
    """
    並列数1でパイプラインを1回実行した際のDRAM/VRAM使用量を計測し，使用可能なメモリの
    範囲に収まるプロセス数を決定する．

    GPU上で複数プロセスが同一のGPUを共有するため，計算資源の競合により並列化してもGPU計算部分の
    スループットが線形には向上しない場合がある．本関数はメモリ超過によるクラッシュを防ぐための
    上限を与えるものであり，実際に速度が向上するプロセス数は別途計測することが望ましい．

    引数:
        ghost_weights_dir (str): GHOST由来の重みを格納したディレクトリ．
        dex_weights_dir (str): DEXの重みを格納したディレクトリ．
        anonymizer_name (str): 使用する匿名化モデルの識別名．
        anonymizer_weight_path (str または None): 匿名化モデルの学習済み重みのパス．
        sample_image_paths (list[str]): 計測用に1プロセスで処理する少数の画像パス．
        noise_level (float): `Anonymizer.anonymize` に渡すスケール係数．
        max_workers (int または None): プロセス数の上限．None の場合は `os.cpu_count()` を用いる．
        safety_margin (float): 空きメモリのうち実際に使用してよい割合(既定0.8)．
        measurement_timeout (float): 計測用プロセスからの応答を待つ最大秒数．
            モデルの読み込みに時間がかかる場合は大きめの値を指定する．
    戻り値:
        num_workers (int): 決定されたプロセス数(1以上)．
    """
    if not torch.cuda.is_available():
        return 1

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    process = ctx.Process(
        target=_measure_single_worker_resource_usage,
        args=(
            ghost_weights_dir, dex_weights_dir, anonymizer_name, anonymizer_weight_path,
            sample_image_paths, noise_level, result_queue,
        ),
    )
    process.start()
    try:
        peak_dram_bytes, peak_vram_bytes, error_traceback = result_queue.get(timeout=measurement_timeout)
    except Exception as exc:
        process.terminate()
        process.join()
        raise RuntimeError(
            "並列数決定のためのリソース計測プロセスから既定時間内に応答がありませんでした．"
            "ghost_weights_dir/dex_weights_dir/anonymizer_weight_path が正しいか確認してください．"
        ) from exc
    process.join()

    if error_traceback is not None:
        raise RuntimeError(f"リソース計測プロセスでエラーが発生しました:\n{error_traceback}")

    free_vram_bytes, _total_vram_bytes = torch.cuda.mem_get_info()
    free_dram_bytes = psutil.virtual_memory().available
    cpu_count = max_workers if max_workers is not None else (os.cpu_count() or 1)

    return _compute_worker_count(
        peak_dram_bytes=peak_dram_bytes,
        peak_vram_bytes=peak_vram_bytes,
        free_dram_bytes=free_dram_bytes,
        free_vram_bytes=free_vram_bytes,
        cpu_count=cpu_count,
        safety_margin=safety_margin,
    )


def _compute_worker_count(
    peak_dram_bytes: int,
    peak_vram_bytes: int,
    free_dram_bytes: int,
    free_vram_bytes: int,
    cpu_count: int,
    safety_margin: float,
) -> int:
    """
    1プロセスあたりのDRAM/VRAM使用量の実測値から，安全に並列実行できるプロセス数を計算する
    (`determine_worker_count` から分離した純粋関数．プロセス起動やGPUを必要としないため，
    単体テストが容易である)．

    引数:
        peak_dram_bytes (int): 1プロセスあたりのピークDRAM使用量(バイト)．
        peak_vram_bytes (int): 1プロセスあたりのピークVRAM使用量(バイト)．
        free_dram_bytes (int): システム全体の空きDRAM(バイト)．
        free_vram_bytes (int): GPU全体の空きVRAM(バイト)．
        cpu_count (int): プロセス数の上限(CPUコア数，または明示的な `max_workers`)．
        safety_margin (float): 空きメモリのうち実際に使用してよい割合．
    戻り値:
        num_workers (int): 決定されたプロセス数(1以上)．
    """
    n_by_vram = int((free_vram_bytes * safety_margin) // peak_vram_bytes) if peak_vram_bytes > 0 else cpu_count
    n_by_dram = int((free_dram_bytes * safety_margin) // peak_dram_bytes) if peak_dram_bytes > 0 else cpu_count

    return max(1, min(n_by_vram, n_by_dram, cpu_count))


def run_evaluation_parallel(
    image_paths: List[str],
    ghost_weights_dir: str,
    dex_weights_dir: str,
    anonymizer_name: str = "vae",
    anonymizer_weight_path: Optional[str] = None,
    noise_level: float = 1.0,
    num_workers: Optional[int] = None,
) -> List[AnonymizationEvalResult]:
    """
    画像リストに対する匿名化前後の評価(年齢・性別・コサイン類似度)を複数プロセスで並列に実行する．

    `num_workers` が指定されない場合は，`determine_worker_count` で計測したDRAM/VRAM使用量に
    基づいてプロセス数を自動決定する．各workerプロセスはGHOST・DEX・Anonymizerの各モデルを
    1回だけロードし，割り当てられた画像を順に処理する．

    引数:
        image_paths (list[str]): 評価対象の画像パスのリスト．
        ghost_weights_dir (str): GHOST由来の重みを格納したディレクトリ．
        dex_weights_dir (str): DEXの重みを格納したディレクトリ．
        anonymizer_name (str): 使用する匿名化モデルの識別名．
        anonymizer_weight_path (str または None): 匿名化モデルの学習済み重みのパス．
        noise_level (float): `Anonymizer.anonymize` に渡すスケール係数．
        num_workers (int または None): 並列実行するプロセス数．None の場合は自動決定する．
    戻り値:
        results (list[AnonymizationEvalResult]): 顔検出に成功した画像分の評価結果．
    """
    if num_workers is None:
        sample_size = min(5, len(image_paths))
        num_workers = determine_worker_count(
            ghost_weights_dir, dex_weights_dir, anonymizer_name, anonymizer_weight_path,
            sample_image_paths=image_paths[:sample_size], noise_level=noise_level,
        )

    shards = [image_paths[i::num_workers] for i in range(num_workers)]

    ctx = mp.get_context("spawn")
    results: List[AnonymizationEvalResult] = []
    with ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(ghost_weights_dir, dex_weights_dir, anonymizer_name, anonymizer_weight_path),
    ) as executor:
        futures = [executor.submit(_evaluate_shard, shard, noise_level) for shard in shards]
        for future in futures:
            results.extend(future.result())

    return results
