import torch

from identity_anonymizer.evaluation import parallel


def test_determine_worker_count_returns_one_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    num_workers = parallel.determine_worker_count(
        ghost_weights_dir="unused",
        dex_weights_dir="unused",
        anonymizer_name="vae",
        anonymizer_weight_path=None,
        sample_image_paths=[],
    )

    assert num_workers == 1


def test_compute_worker_count_bounded_by_vram():
    # 1プロセスあたり1GiB使うが空きVRAMが3.5GiBしかない場合，3プロセスまでに制限される
    num_workers = parallel._compute_worker_count(
        peak_dram_bytes=1,
        peak_vram_bytes=1 * 1024 ** 3,
        free_dram_bytes=1024 ** 4,
        free_vram_bytes=int(3.5 * 1024 ** 3),
        cpu_count=32,
        safety_margin=0.8,
    )
    assert num_workers == 2  # floor(3.5 * 0.8 / 1) = 2


def test_compute_worker_count_bounded_by_cpu_count():
    # メモリに十分な余裕があっても cpu_count(max_workers) を超えない
    num_workers = parallel._compute_worker_count(
        peak_dram_bytes=1,
        peak_vram_bytes=1,
        free_dram_bytes=1024 ** 4,
        free_vram_bytes=1024 ** 4,
        cpu_count=4,
        safety_margin=0.8,
    )
    assert num_workers == 4


def test_compute_worker_count_at_least_one():
    # 1プロセス分のメモリ使用量が空きメモリを上回っていても，最低1プロセスは確保する
    num_workers = parallel._compute_worker_count(
        peak_dram_bytes=1,
        peak_vram_bytes=10 * 1024 ** 3,
        free_dram_bytes=1024 ** 4,
        free_vram_bytes=1 * 1024 ** 3,
        cpu_count=8,
        safety_margin=0.8,
    )
    assert num_workers == 1
