"""
匿名化デモ (programs/demo) の人物の追跡と方式の定義のテスト．
GPUやモデルを必要としない部分のみを対象とする．
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "programs", "demo"))

import tracking  # noqa: E402


def make_face(direction: np.ndarray, noise: float, rng: np.random.Generator) -> np.ndarray:
    """
    ある人物の顔ベクトルに，ノイズを加えたものを作る．

    引数:
        direction (np.ndarray): 形状 (512,) の人物の顔ベクトル．
        noise (float): ノイズの大きさ．
        rng (np.random.Generator): 乱数生成器．
    戻り値:
        face (np.ndarray): 形状 (512,) の顔ベクトル．
    """
    return direction + noise * rng.standard_normal(512)


def make_kps(x: float, y: float, eye_distance: float = 40.0) -> np.ndarray:
    """
    鼻の位置と両目の間隔から，5点のランドマークを作る．

    引数:
        x (float): 鼻のx座標．
        y (float): 鼻のy座標．
        eye_distance (float): 両目の間隔．
    戻り値:
        kps (np.ndarray): 形状 (5, 2) のランドマーク．
    """
    half = eye_distance / 2
    return np.array([[x - half, y - half], [x + half, y - half], [x, y], [x - half, y + half], [x + half, y + half]])


def test_two_people_keep_their_ids():
    """2人が同時に映る場合，それぞれが動画全体で同じ番号を保つ．"""
    rng = np.random.default_rng(0)
    people = rng.standard_normal((2, 512))
    frames = [np.stack([make_face(people[0], 0.3, rng), make_face(people[1], 0.3, rng)]) for _ in range(20)]
    ids = tracking.assign_tracks(frames, threshold=0.35)
    assert all(list(frame_ids) == [0, 1] for frame_ids in ids)


def test_order_change_within_frame():
    """フレームごとに検出の順序が入れ替わっても，人物の番号は顔ベクトルに従う．"""
    rng = np.random.default_rng(1)
    people = rng.standard_normal((2, 512))
    frames = [np.stack([make_face(people[i % 2], 0.3, rng), make_face(people[1 - i % 2], 0.3, rng)]) for i in range(10)]
    ids = tracking.assign_tracks(frames, threshold=0.35)
    for i, frame_ids in enumerate(ids):
        assert list(frame_ids) == ([0, 1] if i % 2 == 0 else [1, 0])


def test_low_similarity_face_is_kept_by_position():
    """横を向くなどして見た目の類似度が下がっても，位置が近ければ同じ人物とみなす．"""
    rng = np.random.default_rng(2)
    person = rng.standard_normal(512)
    frames, kps = [], []
    for i in range(10):
        face = make_face(person, 0.3, rng) if i != 5 else rng.standard_normal(512)  # 5フレーム目は別物に見える
        frames.append(face[None])
        kps.append([make_kps(100 + i, 100)])
    ids = tracking.assign_tracks(frames, threshold=0.35, kps_per_frame=kps)
    assert all(frame_ids[0] == 0 for frame_ids in ids)


def test_new_person_far_away_gets_new_id():
    """見た目も位置も異なる顔は，新しい人物として登録する．"""
    rng = np.random.default_rng(3)
    people = rng.standard_normal((2, 512))
    frames = [make_face(people[0], 0.3, rng)[None], make_face(people[1], 0.3, rng)[None]]
    kps = [[make_kps(100, 100)], [make_kps(400, 100)]]
    ids = tracking.assign_tracks(frames, threshold=0.35, kps_per_frame=kps)
    assert [int(frame_ids[0]) for frame_ids in ids] == [0, 1]


def test_empty_frames_and_track_embeddings():
    """顔が無いフレームを挟んでも処理でき，人物ごとの顔ベクトルのノルムは個々の顔ベクトルのノルムの平均になる．"""
    rng = np.random.default_rng(4)
    person = rng.standard_normal(512) * 3
    frames = [make_face(person, 0.1, rng)[None], np.zeros((0, 512)), make_face(person, 0.1, rng)[None]]
    ids = tracking.assign_tracks(frames, threshold=0.35)
    assert [len(frame_ids) for frame_ids in ids] == [1, 0, 1]
    embeddings = tracking.track_embeddings(frames, ids)
    assert embeddings.shape == (1, 512)
    expected = np.mean([np.linalg.norm(frames[0][0]), np.linalg.norm(frames[2][0])])
    assert np.linalg.norm(embeddings[0]) == pytest.approx(expected, rel=1e-5)
