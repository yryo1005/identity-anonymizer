"""
匿名化デモの設定と，既存のモジュールを読み込むためのパスの設定．

サンプリング方式は `programs/ex002_partial_prior_sampling/samplers.py`，GHOSTの生成とモデルの読み込みは
`programs/ex003_ghost_image_eval/ghost.py` を，変更せずに読み込んで再利用する．
"""

import os
import sys

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(DEMO_DIR))
EX002_DIR = os.path.join(REPO_ROOT, "programs", "ex002_partial_prior_sampling")
EX003_DIR = os.path.join(REPO_ROOT, "programs", "ex003_ghost_image_eval")

# ex003 の settings.py が src/ と .ai/ai-dev-kit/src/ を sys.path へ追加する
for _path in (EX003_DIR, EX002_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

STATIC_DIR = os.path.join(DEMO_DIR, "static")
RANKING_PATH = os.path.join(DEMO_DIR, "assets", "attribute_ranking.json")
JOBS_DIR = os.path.join(REPO_ROOT, "outputs", "demo", "jobs")

# --- 入力の制限 ---
#: 処理する動画の長さの上限 [秒]．これを超える部分は切り捨てる
MAX_DURATION_SEC = 60
#: 処理する動画の高さの上限 [画素]．これを超える動画は縮小する
MAX_HEIGHT = 720
#: 処理する動画のフレームレートの上限
MAX_FPS = 30
#: アップロードを受け付けるファイルサイズの上限 [バイト]
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
#: 保持するジョブの数．これを超えると古いジョブのファイルから削除する
MAX_KEPT_JOBS = 20

# --- 処理の設定 ---
#: 顔検出を並列に実行するスレッド数
DETECTION_THREADS = 4
#: 同一人物とみなすArcFaceのコサイン類似度の閾値 (人物の追跡に用いる)
TRACK_SIMILARITY_THRESHOLD = 0.35
#: GHOSTの生成器に一度に入力する顔の数
GENERATION_BATCH_SIZE = 64
#: ノルムの補正に用いる，実データの顔ベクトルの平均ノルム (ex003 で学習用データから計算した値)
REAL_MEAN_NORM = 22.40
