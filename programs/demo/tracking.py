"""
動画中の複数の人物を，フレームをまたいで追跡するためのモジュール．

各フレームで検出した顔を，それまでに現れた人物 (トラック) に次の2段階で対応付ける．
    1. 見た目: ArcFaceの顔ベクトルと各人物の平均の顔ベクトルのコサイン類似度に基づき，ハンガリアン法で
       対応付ける．類似度が閾値以上の組だけを採用する．
    2. 位置: 1で対応付かなかった顔は，直近のフレームに映っていた人物のうち，顔の位置 (鼻の位置) が十分に
       近いものに対応付ける．横を向いた顔やぼやけた顔のように，一時的に見た目の類似度が下がる場合に，
       同じ人物が別人として分かれるのを防ぐ．
どちらでも対応付かなかった顔は，新しい人物として登録する．同じフレーム内の2つの顔が同じ人物に
割り当てられることはない．
"""

from typing import List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

#: 5点のランドマークのうち，左目・右目・鼻の番号
LEFT_EYE, RIGHT_EYE, NOSE = 0, 1, 2


def assign_tracks(embeddings_per_frame: List[np.ndarray], threshold: float,
                  kps_per_frame: Optional[List[List[np.ndarray]]] = None,
                  max_gap: int = 15, max_distance_ratio: float = 1.0) -> List[np.ndarray]:
    """
    フレームごとの顔の顔ベクトルと位置から，各顔に人物の番号を割り当てる．

    引数:
        embeddings_per_frame (list[np.ndarray]): 要素数 F のリスト．各要素は形状 (n_f, 512) の，
            そのフレームで検出した顔の顔ベクトル (顔が無いフレームは形状 (0, 512))．
        threshold (float): 見た目で同一人物とみなすコサイン類似度の閾値．
        kps_per_frame (list または None): 要素数 F のリスト．各要素は，そのフレームの各顔の5点の
            ランドマーク (形状 (5, 2)) のリスト．None の場合は位置による対応付けを行わない．
        max_gap (int): 位置による対応付けの対象とする，人物が最後に映っていたフレームからの最大の間隔．
        max_distance_ratio (float): 位置による対応付けで許す鼻の移動量の上限 (両目の間隔に対する比)．
    戻り値:
        track_ids (list[np.ndarray]): 要素数 F のリスト．各要素は形状 (n_f,) の人物の番号 (0始まり)．
    """
    track_sums: List[np.ndarray] = []
    last_frame: List[int] = []
    last_nose: List[np.ndarray] = []
    track_ids: List[np.ndarray] = []
    for frame_index, embeddings in enumerate(embeddings_per_frame):
        ids = np.full(len(embeddings), -1, dtype=int)
        kps_list = kps_per_frame[frame_index] if kps_per_frame is not None else None
        if len(embeddings) > 0 and track_sums:
            faces = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            tracks = np.stack(track_sums)
            tracks = tracks / np.linalg.norm(tracks, axis=1, keepdims=True)
            similarity = faces @ tracks.T
            rows, cols = linear_sum_assignment(-similarity)
            for row, col in zip(rows, cols):
                if similarity[row, col] >= threshold:
                    ids[row] = col
            if kps_list is not None:
                assign_by_position(ids, kps_list, frame_index, last_frame, last_nose, max_gap, max_distance_ratio)
        for i in np.nonzero(ids < 0)[0]:
            ids[i] = len(track_sums)
            track_sums.append(np.zeros(embeddings.shape[1]))
            last_frame.append(frame_index)
            last_nose.append(np.zeros(2))
        for i, track in enumerate(ids):
            track_sums[track] = track_sums[track] + embeddings[i]
            last_frame[track] = frame_index
            if kps_list is not None:
                last_nose[track] = np.asarray(kps_list[i])[NOSE]
        track_ids.append(ids)
    return track_ids


def assign_by_position(ids: np.ndarray, kps_list: List[np.ndarray], frame_index: int, last_frame: List[int],
                       last_nose: List[np.ndarray], max_gap: int, max_distance_ratio: float) -> None:
    """
    見た目で対応付かなかった顔を，直近に映っていた人物のうち顔の位置が近いものに対応付ける (ids を直接更新する)．

    引数:
        ids (np.ndarray): 形状 (n,) の，このフレームの各顔の人物の番号 (未割り当ては -1)．
        kps_list (list[np.ndarray]): このフレームの各顔の5点のランドマーク (形状 (5, 2))．
        frame_index (int): このフレームの番号．
        last_frame (list[int]): 各人物が最後に映っていたフレームの番号．
        last_nose (list[np.ndarray]): 各人物が最後に映っていたときの鼻の位置 (形状 (2,))．
        max_gap (int): 対象とする，最後に映っていたフレームからの最大の間隔．
        max_distance_ratio (float): 許す鼻の移動量の上限 (両目の間隔に対する比)．
    戻り値:
        なし．
    """
    used = set(int(track) for track in ids if track >= 0)
    for i in np.nonzero(ids < 0)[0]:
        kps = np.asarray(kps_list[i])
        eye_distance = max(float(np.linalg.norm(kps[LEFT_EYE] - kps[RIGHT_EYE])), 1.0)
        best, best_distance = -1, np.inf
        for track in range(len(last_frame)):
            if track in used or frame_index - last_frame[track] > max_gap:
                continue
            distance = float(np.linalg.norm(kps[NOSE] - last_nose[track]))
            if distance <= max_distance_ratio * eye_distance and distance < best_distance:
                best, best_distance = track, distance
        if best >= 0:
            ids[i] = best
            used.add(best)


def track_embeddings(embeddings_per_frame: List[np.ndarray], track_ids: List[np.ndarray]) -> np.ndarray:
    """
    人物ごとに，その人物に割り当てられた全ての顔の顔ベクトルの平均を求める．平均によってノルムが
    小さくなるのを避けるため，向きは平均ベクトルの向き，ノルムは個々の顔ベクトルのノルムの平均とする．

    引数:
        embeddings_per_frame (list[np.ndarray]): `assign_tracks` と同じ．
        track_ids (list[np.ndarray]): `assign_tracks` の戻り値．
    戻り値:
        embeddings (np.ndarray): 形状 (P, 512) の人物ごとの顔ベクトル (P は人物の数)．
    """
    stacked = np.concatenate([e for e in embeddings_per_frame if len(e) > 0])
    ids = np.concatenate([i for i in track_ids if len(i) > 0])
    num_tracks = int(ids.max()) + 1
    result = np.zeros((num_tracks, stacked.shape[1]))
    for track in range(num_tracks):
        members = stacked[ids == track]
        direction = members.mean(axis=0)
        result[track] = direction / np.linalg.norm(direction) * np.linalg.norm(members, axis=1).mean()
    return result.astype(np.float32)
