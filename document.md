# document.md

本ドキュメントは，`identity_anonymizer.anonymizers.Anonymizer` を継承して新しい匿名化モデルを
実装する方法をまとめたものである．セットアップ・実行方法は [README.md](README.md) を参照すること．

## 1. `Anonymizer` インターフェースの設計方針

GHOSTは，ソース顔として常にArcFaceの512次元埋め込みを要求する．そのため `Anonymizer` は，
**出力**を形状 `(B, 512)` の顔ベクトルに固定する(`output_dim = 512`)．

一方，**入力**の形式・形状はモデルごとに異なってよい．`references/` の予稿・発表資料では，
以下のように入力の異なる複数の手法が紹介されている．

| 手法 | 入力 | 出力 |
| :--- | :--- | :--- |
| 提案手法(`VAEAnonymizer`) | ターゲットのArcFace顔ベクトル $` \mathbf{v} `$ (512次元) | 匿名化後の顔ベクトル $` \hat{\mathbf{v}} `$ (512次元) |
| 従来手法2(`AttributeNNAnonymizer`，本ドキュメントの例) | ターゲットの属性 $` I `$ (性別・年齢・人種) | 顔ベクトル $` \hat{\mathbf{v}} `$ (512次元) |

このように入力の型・次元が手法ごとに異なるため，`Anonymizer` は入力の型を `Any` とし，
各サブクラスが自身の想定する入力を検証する `validate_input` を実装する設計とした．

## 2. `Anonymizer` のAPI(テンプレートメソッドパターン)

`identity_anonymizer/anonymizers/base.py` の `Anonymizer` は，以下の3つのメソッドで構成される．

```text
anonymize(x, **kwargs)              # 公開API．基底クラスが実装する(サブクラスは上書きしない)
  ├─ validate_input(x)              # サブクラスが実装: 入力の型・形状を検証する
  ├─ _anonymize(x, **kwargs)        # サブクラスが実装: 実際の変換処理
  └─ 出力形状 (B, output_dim) の検証  # 基底クラスが実装
```

`anonymize` は，入力検証(`validate_input`) → 実際の変換(`_anonymize`) → 出力形状の検証，を
行うテンプレートメソッドである．この設計により，

- 呼び出し側(`FaceAnonymizerPipeline` 等)は，常に同じ `anonymize(x, **kwargs)` という
  シグネチャでモデルを差し替えられる．
- 各サブクラスは，入力検証を実装し忘れることがない(`anonymize` が必ず `validate_input` を
  呼び出すため)．
- 出力形状が誤っている実装は，`anonymize` の時点で早期に検出される．

## 3. 新しいAnonymizerを実装する手順

1. **入力の形式を決める．** テンソル(顔ベクトル等)である必要はない．dict，dictのlist(バッチ)，
   その他任意の型でよい．
2. **`Anonymizer` を継承したクラスを作成し，`validate_input(self, x)` を実装する．**
   入力 `x` の型・形状・値域が想定と一致するかを確認し，一致しない場合は `TypeError` または
   `ValueError` を送出する．
3. **`_anonymize(self, x, **kwargs) -> torch.Tensor` を実装する．** `validate_input` を
   通過した入力から，形状 `(B, self.output_dim)` の顔ベクトルを生成する．
4. **学習方法を実装する．** 学習手順(教師なし再構成，回帰，敵対的学習等)はモデルごとに
   大きく異なるため，`Anonymizer` では規定しない．`VAEAnonymizer` の学習は
   `notebooks/02_train_vae_anonymizer.ipynb` に，`AttributeNNAnonymizer` の学習は
   `identity_anonymizer.anonymizers.attribute_nn.train_attribute_nn_anonymizer` と
   `notebooks/06_train_attribute_nn_anonymizer.ipynb` に実装している．新しいモデルも同様に，
   `src/identity_anonymizer/anonymizers/<model_name>.py` にモデルとその学習用の関数を実装し，
   `notebooks/0N_train_<model_name>.ipynb` から呼び出す形にすることを推奨する．
5. **`identity_anonymizer.anonymizers.registry.register_anonymizer` へ登録する．**
   登録後は `get_anonymizer("<識別名>")` でインスタンス化できるようになる．
6. **入力がArcFace顔ベクトルではない場合，パイプライン呼び出し時に `anonymizer_input` を
   指定する．** `FaceAnonymizerPipeline.anonymize_image` / `anonymize_video` /
   `get_unified_identity_embedding` は，ターゲット画像・動画から計算したArcFace顔ベクトルを
   常に内部で保持しているが(これは複数人物が映る場合の対象人物の追跡に使うため)，
   `Anonymizer.anonymize` へ渡す入力は `anonymizer_input` 引数で明示的に上書きできる．
   `anonymizer_input=None`(既定値)の場合は，このArcFace顔ベクトルがそのまま
   `Anonymizer.anonymize` の入力として使われる(`VAEAnonymizer` はこの既定動作を利用する)．

## 4. 実装例: `AttributeNNAnonymizer`(従来手法2)

`references/` の予稿・発表資料で紹介されている「従来手法2」(性別・年齢・人種の属性を入力とし，
ArcFaceの顔ベクトルを近似的に出力するNN)を，入力がArcFace顔ベクトルではないAnonymizerの
実装例として `src/identity_anonymizer/anonymizers/attribute_nn.py` に実装した．

### 4.1 入力の形式

年齢(`age`)・性別(`gender`: 0=男性，1=女性)・人種(`race`: 0〜4，UTKFaceの定義に従う)を
キーに持つ `dict`，またはそのバッチ分の `list[dict]` を入力とする．

```python
{"age": 20, "gender": 0, "race": 0}
```

### 4.2 `validate_input` の実装

```python
def validate_input(self, x):
    attributes_list = x if isinstance(x, list) else [x]
    if len(attributes_list) == 0:
        raise ValueError("AttributeNNAnonymizerの入力(属性のリスト)が空です．")
    for attributes in attributes_list:
        if not isinstance(attributes, dict):
            raise TypeError(...)
        for key in ("age", "gender", "race"):
            if key not in attributes:
                raise ValueError(...)
        if attributes["gender"] not in (0, 1):
            raise ValueError(...)
        if not (0 <= attributes["race"] < self.NUM_RACE_CATEGORIES):
            raise ValueError(...)
```

単体(`dict`)とバッチ(`list[dict]`)の両方を受け付けられるよう，`list` でなければ長さ1の
`list` に揃えたうえで，各要素を検証している．

### 4.3 `_anonymize` の実装

属性のdictを，年齢(正規化した1次元)・性別のone-hot(2次元)・人種のone-hot(5次元)からなる
8次元の特徴量へ変換し，3層のMLP(`nn.Linear` → `ReLU` を2回，最後に `nn.Linear` で
`output_dim=512` へ写像)に入力する．

```python
def _anonymize(self, x, **kwargs):
    attributes_list = x if isinstance(x, list) else [x]
    features = self._encode_attributes(attributes_list).to(device)
    return self.net(features)
```

### 4.4 学習

UTKFaceの各顔画像 $` X_n `$ とその属性 $` I_n `$(ファイル名 `{age}_{gender}_{race}_{日時}.jpg`
から取得)の組を用いて，以下の目的関数を最小化する(`train_attribute_nn_anonymizer`)．

$$
\arg\min_{NN} \frac{1}{N} \sum_{n=1}^{N} \| \mathrm{ArcFace}(X_n) - NN(I_n) \|_2
$$

属性と顔ベクトルのペアの抽出には `identity_anonymizer.data.build_attribute_embedding_dataset`
を用いる．`notebooks/06_train_attribute_nn_anonymizer.ipynb` で，同梱のUTKFaceサンプル
(500枚)を用いた学習と，学習済みモデルをパイプラインへ差し替えて属性のみから顔を匿名化する
デモを実行している．

### 4.5 登録と利用

```python
# identity_anonymizer/anonymizers/registry.py
_ANONYMIZER_REGISTRY = {
    "vae": VAEAnonymizer,
    "attribute_nn": AttributeNNAnonymizer,
}
```

```python
from identity_anonymizer.anonymizers import get_anonymizer
from identity_anonymizer.faceswap import load_ghost_models, FaceAnonymizerPipeline

models = load_ghost_models()
anonymizer = get_anonymizer("attribute_nn", weight_path="weights/anonymizers/attribute_nn/attribute_nn.pt")
pipeline = FaceAnonymizerPipeline(models, anonymizer)

# 入力がArcFace顔ベクトルではないため，anonymizer_input で明示的に属性を渡す
face_image, full_image = pipeline.anonymize_image(
    "sample_images/beckham.jpg",
    anonymizer_input={"age": 20, "gender": 0, "race": 0},
)
```

### 4.6 `VAEAnonymizer` との違い・トレードオフ

`AttributeNNAnonymizer` はターゲットの顔画像を必要とせず，属性のみから顔ベクトルを生成できる
反面，**ターゲットの属性を何らかの方法で事前に取得しておく必要がある**．これは `references/`
の予稿でも「従来手法の課題」として指摘されている点であり，`VAEAnonymizer`(ターゲットの顔画像
から直接ArcFace顔ベクトルを計算するため，属性の事前取得が不要)が解決しようとしている課題
そのものである．`AttributeNNAnonymizer` は，この設計上のトレードオフと，`Anonymizer` が
入力の異なるモデルを問題なく扱えることの両方を示す実装例として実装している．

## 5. 関連ファイル

| ファイル | 役割 |
| :--- | :--- |
| `src/identity_anonymizer/anonymizers/base.py` | `Anonymizer` 抽象基底クラス |
| `src/identity_anonymizer/anonymizers/vae.py` | `VAEAnonymizer`(提案手法) |
| `src/identity_anonymizer/anonymizers/attribute_nn.py` | `AttributeNNAnonymizer`(従来手法2，本ドキュメントの例)と学習関数 |
| `src/identity_anonymizer/anonymizers/registry.py` | モデルの登録・取得(`register_anonymizer`, `get_anonymizer`) |
| `src/identity_anonymizer/faceswap/pipeline.py` | `FaceAnonymizerPipeline`(`anonymizer_input` 引数) |
| `src/identity_anonymizer/data/utkface.py` | UTKFaceの属性パース(`parse_utkface_attributes`)とデータセット構築 |
| `notebooks/06_train_attribute_nn_anonymizer.ipynb` | `AttributeNNAnonymizer` の学習・デモの実行例 |
| `tests/test_attribute_nn_anonymizer.py` | `validate_input` の検証を含む単体テスト |
