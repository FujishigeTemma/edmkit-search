# CODING.md

設計方針・コーディングスタイルガイド。

---

## 1. モジュール設計と依存性注入

各モジュールは単一の責務を持ち、モジュール間の依存は最小限にする。上位モジュールは下位の具象を直接呼ばず、コールバック (関数ポインタ) 経由で利用する。

**原則:**
- ユーティリティ層はドメインロジックに依存しない
- 振る舞いの差し替え点はパラメータとして設計する
- パラメータの部分適用には `functools.partial` を使い、高階関数で戦略を差し替え可能にする
- コールバック型には型エイリアスで名前を付け、インターフェースを明文化する

### NG

```python
def run(data, method="fast"):
    if method == "fast":
        result = fast_predict(data)
    elif method == "accurate":
        result = accurate_predict(data)
```

### OK

```python
PredictFunc = Callable[[np.ndarray], np.ndarray]

def run(data, predict: PredictFunc):
    result = predict(data)

# 呼び出し側で戦略を具体化
run(data, predict=fast_predict)
run(data, predict=partial(accurate_predict, regularization=1e-3))
```

**テスタビリティの利点:**
- コールバックを差し替えればロジック単体をテストできる
- 乱数を含む処理もサンプラーを注入すれば決定論的テストが書ける
- 集約関数を差し替えれば平均・中央値・信頼区間など柔軟に切り替えられる

---

## 2. イミュータビリティ

入力を変更せず、常に新しいオブジェクトを返す。

### NG

```python
def normalize(x: np.ndarray) -> np.ndarray:
    x -= x.mean()   # 呼び出し元の配列を破壊
    x /= x.std()
    return x
```

### OK

```python
def normalize(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / x.std()  # 新しい配列を返す
```

in-place 演算 (`-=`, `/=`, `np.add(out=...)`) は使わず、式で新しい配列を生成する。

---

## 3. Fail-Fast バリデーション

関数の先頭でガード節により入力を検証し、不正な入力は即座に弾く。

**原則:**
- 計算に入る前にすべての前提条件をチェック
- エラーメッセージには **期待値と実際の値** の両方を含める
- 未実装は `NotImplementedError` で明示する

### NG

```python
def predict(X, Y, theta):
    # バリデーションなし → 計算途中で cryptic なエラーが出る
    D = cdist(X, Y)
    weights = np.exp(-theta * D / D.mean(axis=1, keepdims=True))
```

### OK

```python
def predict(X, Y, theta):
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X and Y must have the same length, got {X.shape[0]} and {Y.shape[0]}")
    if theta < 0:
        raise ValueError(f"theta must be non-negative, got {theta}")
    # ここから計算ロジック
```

---

## 4. 型と契約

すべての関数に引数・戻り値の型を付ける。docstring で形状・制約・振る舞いを明記する。

**型アノテーション:**
- Union 型は `|` 構文 (PEP 604) を使用
- 複雑なコールバック型は型エイリアスで名前を付ける
- `# type: ignore` はやむを得ない場合に限定
- keyword-only 引数 (`*` 以降) で意図を明確にする

**docstring:**
- NumPy スタイル
- 必須セクション: Summary, Parameters (型 + 形状), Returns (型 + 形状), Raises
- 主要な公開関数には Examples を付ける

```python
TransformFunc = Callable[[np.ndarray], np.ndarray]

def fit(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    transform: TransformFunc,
    alpha: float | None = None,
) -> np.ndarray:
    """Fit the model.

    Parameters
    ----------
    X : np.ndarray of shape (N, D)
        Input features.
    Y : np.ndarray of shape (N,)
        Target values.
    transform : TransformFunc
        Feature transformation applied before fitting.
    alpha : float | None
        Regularization strength. None disables regularization.

    Returns
    -------
    np.ndarray of shape (D,)

    Raises
    ------
    ValueError
        If X and Y have different lengths.
    """
```

---

## 5. 命名規約

**基本:**
- 関数・変数・引数: `snake_case`
- 型エイリアス: `PascalCase` (`TransformFunc`, `Sampler`)
- private: `_` プレフィックス (`_impl`, `_validate`)
- モジュール: `snake_case`。1ファイル = 1概念

**ドメイン標準記法の尊重:**
論文や分野で確立された記法はそのまま使う。冗長な命名に展開しない。

| OK | NG |
|----|----|
| `tau` | `time_delay` |
| `alpha` | `regularization_strength` |
| `theta` | `locality_parameter` |
| `dt` | `time_step_size` |
| `E` | `embedding_dimension` |

ただし略記が分野外の読者に不明な場合は、docstring の Parameters で意味を説明する。コード中の変数名を冗長にするのではなく、ドキュメントで補う。

**配列の形状変数:**
数学的慣習に従い大文字1字を使い、インラインコメントで意味を明記する。

```python
B, N, E = X.shape    # batch, points, embedding dimension
M = query.shape[1]   # number of query points
k = E + 1            # number of neighbors
```

**関数名は「何を」であり「どうやって」ではない:**

| OK | NG |
|----|----|
| `pairwise_distance` | `compute_distance_with_matmul` |
| `autocorrelation` | `fft_based_autocorrelation` |
| `fit` | `fit_using_ridge_regression` |

実装手法は変わりうるが、関数が返すものは変わらない。名前は結果を表す。

---

## 6. 公開 API の設計

**公開面の最小化:**
- `__init__.py` は re-export のみ。ロジックを持たない
- ユーザーに見せる関数だけを公開し、内部ヘルパーは `_` プレフィックスで隠蔽する

**入力の正規化:**
- ユーザーには柔軟な入力を許容し (1D でも 2D でも渡せる等)、関数内部で統一された形に正規化してから処理する。出力は入力に合わせた形に戻す

**バックエンド分離:**
- 複数の実装 (バックエンド) がある場合、公開関数はディスパッチのみを行い、各バックエンドは独立した private 関数として実装する。戻り値の型は統一する

---

## 7. 計算の設計原則

### 等価変換

「同じ結果を返す、別の計算に書き換える」ことが最も根本的な思考様式である。コードを書く前に「この計算は別の形で表現できないか」を問う。

- **代数的変換:** 恒等式を利用して演算の構造を変える。要素ごとのループが行列積に帰着することがある
- **定理による計算クラスの変換:** 計算量のオーダーを変える
- **依存構造の分析:** 逐次的に見える計算でも、データ依存グラフを分析すれば独立に処理できる部分が見つかる
- **数値的に等価な手法の選択:** 逆行列の明示的計算と連立方程式の直接求解は数学的に同じだが、安定性と計算量が異なる

変換先の候補を引き出しとして持つこと (ループ → 行列積、畳み込み → FFT、逆行列 → solve 等) と、適用可能性を判断する目を持つことが重要。

### 必要十分性

結果の生成に厳密に必要な計算だけを行う。問題の定義に照らして不要な仕事を認識して排除する。

- 上位 k 個の集合が必要なだけなら全体のソートは不要。部分ソートで足りる
- 中間データ構造 (対角行列など) が必要でないなら実体化しない。ブロードキャストや einsum で同じ結果を得る
- 共通する構造は一度だけ構築し再利用する

判断基準は「この中間データ構造・中間計算は、最終結果を得るために本当に必要か」である。

### ホットパスの純化

ループの内側から判断・準備・冗長な計算をすべて追い出し、ループ本体を最小かつ分岐のない形にする。

- 入力の正規化はループの前に済ませ、ループ内の条件分岐を排除する
- ループ変数に依存しない部分式は事前計算し、ループ内ではスライス (ビュー) で参照する

ひとつの設計判断が複数の原則を同時に満たすとき、それは良い設計である。「入力の正規化」は API の利便性 (セクション 6) であると同時に、ホットパスの最適化でもある。

### 制約下での表現力

フレームワークの API が必要な操作をサポートしていない場合、Python ループにフォールバックするのではなく、利用可能なプリミティブの組み合わせで等価な操作を構築する。reshape と算術的なインデックス変換の組み合わせで、ループなしのバッチ処理を実現できることが多い。

---

## 8. 数値安定性

浮動小数点演算は実数演算ではない。理論上の不変条件が計算上は破れうることを前提にコードを設計する。

1. **理論的不変条件の防衛的保証:** 二乗ノルムは理論上非負だが、丸め誤差で微小な負値が生じうる。後段で平方根を取る前にクランプすれば NaN を防げる
2. **正則化の構造的設計:** 正則化から除外すべきパラメータ (切片項など) を識別する。正則化の強さをデータ行列のスケールに適応させることで、ハイパーパラメータの意味を移植可能にする
3. **計算手法の選択:** 数学的に等価な手法の中から数値的に安定なものを選ぶ。これはセクション 7 (等価変換) の判断基準のひとつ

---

## 9. 数式と実装の構造的対応

コードの構造が数学的定義の構造を反映するようにする。論文や教科書との照合による正しさの検証が容易になる。

ODE 系を行列・ベクトル積の形で記述すれば、系の構造が行列の配置として視覚化される。einsum の添字表記はアインシュタイン縮約記法と直接対応するため、数式との整合性を添字レベルで検証できる。

数式と実装の間の「翻訳層」が薄いほど、バグの入り込む余地が少ない。

---

## まとめ

| # | 原則 | 要約 |
|---|------|------|
| 1 | **モジュール設計と DI** | 高凝集・低結合。コールバック注入と `partial` で戦略を差し替え |
| 2 | **イミュータビリティ** | 入力を変更しない。常に新しいオブジェクトを返す |
| 3 | **Fail-Fast** | ガード節で早期検証。期待値と実際値をエラーに含める |
| 4 | **型と契約** | 型エイリアス、NumPy スタイル docstring、keyword-only 引数 |
| 5 | **命名規約** | snake_case 基本。ドメイン標準記法を尊重。関数名は結果を表す |
| 6 | **公開 API** | 公開面の最小化。入力の正規化。バックエンド分離 |
| 7 | **計算の設計** | 等価変換、必要十分性、ホットパスの純化、制約下での表現力 |
| 8 | **数値安定性** | 浮動小数点の限界を設計段階で組み込む |
| 9 | **数式との対応** | 実装の構造が数学的定義を反映し、検証を容易にする |
