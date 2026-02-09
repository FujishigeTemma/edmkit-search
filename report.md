# MDE コードベース可読性分析レポート

## 概要

| 項目 | 値 |
|------|-----|
| 分析対象 | `src/mde/` (14ファイル), `scripts/` (4ファイル) |
| 総行数 | 約1,500行 |
| 全体評価 | コアモジュールは良好、スクリプト層に改善の余地あり |

---

## 1. 高優先度の問題

### 1.1 コード重複 (scripts/)

**問題箇所:**
- `scripts/fly_mde.py:38-73` - `run_experiment()`
- `scripts/lorenz96_mde.py:68-107` - `run_experiment()`

**具体例:**

```python
# fly_mde.py:38-73
def run_experiment(
    candidates: np.ndarray,
    target: np.ndarray,
    split: TemporalSplit,
    target_label: str,
    cand_var_names: list[str],
    metric_name: str,
    mode: str,
) -> dict:
    cfg = METRIC_CONFIGS[metric_name]
    result = mde(
        candidates=candidates,
        target=target,
        split=split,
        predict=simplex_projection,
        metric=cfg.fn,
        threshold=cfg.threshold,
        max_dim=MAX_DIM,
    )
    selected_vars = [cand_var_names[i] for i in result.selected_indices]
    # ... 以下同様のパターン
```

```python
# lorenz96_mde.py:68-107 - ほぼ同一の構造
def run_experiment(
    candidates: np.ndarray,
    target: np.ndarray,
    split: TemporalSplit,
    target_label: str,
    candidate_indices: list[int],  # 追加引数
    cand_var_names: list[str],
    metric_name: str,
    mode: str,
) -> dict:
    cfg = METRIC_CONFIGS[metric_name]
    result = mde(...)  # 同じパターン
    selected_vars = [cand_var_names[i] for i in result.selected_indices]
    # ...
```

**原因:**
- 実験スクリプト間で共通ロジックの抽出が不十分
- `experiment_util.py` に共通化の一部は存在するが、`run_experiment()` 自体は各スクリプトに重複

**影響:**
- 変更時に複数箇所の修正が必要
- バグ修正の漏れリスク

---

### 1.2 脆弱な文字列解析 (lorenz96_mde.py:148)

**問題箇所:**
- `scripts/lorenz96_mde.py:147-149`

**具体例:**

```python
# lorenz96_mde.py:147-149
if mode.startswith("single_"):
    target_idx = int(mode.split("_x")[1])  # "single_x5" -> "5"
    target_indices = [target_idx]
```

**原因:**
- 文字列フォーマットに依存したインデックス抽出
- `"_x"` が複数含まれる場合や、数値以外が来る場合の考慮なし

**リスク:**
- `"single_x10"` は正常だが、`"single_experiment_x5"` では壊れる
- 例外処理がないため、不正入力でクラッシュ

**改善案:**
```python
import re
match = re.match(r"single_x(\d+)", mode)
if match:
    target_idx = int(match.group(1))
```

---

### 1.3 非効率なシリアライズ (lorenz96_mde.py:210-212)

**問題箇所:**
- `scripts/lorenz96_mde.py:160-168` (書き込み側)
- `scripts/lorenz96_mde.py:209-218` (読み込み側)

**具体例:**

```python
# 書き込み側 (lorenz96_mde.py:160-168)
rows.append({
    ...
    "tp": str(v["tp"]),      # [1, 2, 3] -> "[1, 2, 3]"
    "fp": str(v["fp"]),
    "fn": str(v["fn"]),
    ...
})

# 読み込み側 (lorenz96_mde.py:209-212)
for row_idx, row in enumerate(verification_df.iter_rows(named=True)):
    tp = ast.literal_eval(row["tp"])    # "[1, 2, 3]" -> [1, 2, 3]
    fn = ast.literal_eval(row["fn"])
    fp = ast.literal_eval(row["fp"])
```

**原因:**
- CSV形式ではリストをネイティブに保存できないため、文字列化
- 読み込み時に `ast.literal_eval()` で再パース

**問題点:**
- セキュリティリスク (`ast.literal_eval` は `eval` より安全だが、複雑な式も評価可能)
- パフォーマンスオーバーヘッド
- フォーマット依存 (余分な空白等でパース失敗の可能性)

**改善案:**
- JSON/Parquet形式の使用
- または `build_verification_csv()` の戻り値を直接使用し、CSV保存を廃止

---

## 2. 中優先度の問題

### 2.1 docstring 欠如 (scripts/)

**問題箇所:**
- `scripts/fly_mde.py:38` - `run_experiment()`
- `scripts/fly_mde.py:76` - `main()`
- `scripts/lorenz96_mde.py:68` - `run_experiment()`
- `scripts/lorenz96_mde.py:251` - `main()`
- `scripts/data_analysis.py:29` - `main()`

**具体例:**

```python
# fly_mde.py:76-77
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # docstring なし - 何をするスクリプトか不明

# lorenz96_mde.py:68-77
def run_experiment(
    candidates: np.ndarray,
    target: np.ndarray,
    ...
) -> dict:
    # docstring なし - 引数の意味、戻り値のキーが不明
    cfg = METRIC_CONFIGS[metric_name]
```

**原因:**
- スクリプトは「動けばよい」という意識
- コアモジュールと異なり、ドキュメント基準が緩い

**影響:**
- 新規参加者が処理内容を理解しにくい
- 戻り値の辞書キーを把握するにはコードを読む必要がある

---

### 2.2 型ヒントの曖昧さ (scripts/)

**問題箇所:**
- `scripts/fly_mde.py:46` - `-> dict`
- `scripts/lorenz96_mde.py:77` - `-> dict`
- `scripts/experiment_util.py:16` - `exp: dict`

**具体例:**

```python
# fly_mde.py:38-46
def run_experiment(...) -> dict:
    # 戻り値のキーが型から判別不能
    return {
        "metric_name": metric_name,
        "mode": mode,
        "target_label": target_label,
        "result": result,
        "selected_vars": selected_vars,
        "candidates": candidates,
        "target_values": target,
    }

# experiment_util.py:16
def build_plot_data(exp: dict, *, split_name: str) -> MDEPlotData:
    # exp に必要なキーが不明
    result = exp["result"]
    ...
```

**原因:**
- `TypedDict` を使用していない
- 辞書スキーマの明示的な定義がない

**改善案:**

```python
from typing import TypedDict

class ExperimentResult(TypedDict):
    metric_name: str
    mode: str
    target_label: str
    result: MDEResult
    selected_vars: list[str]
    candidates: np.ndarray
    target_values: np.ndarray
```

---

### 2.3 関数の責任過多 (lorenz96_mde.py:251-320)

**問題箇所:**
- `scripts/lorenz96_mde.py:251-320` - `main()` (70行)

**具体例:**

```python
# lorenz96_mde.py:251-320
def main():
    # フェーズ1: データ生成 (254-259)
    rng = np.random.default_rng(L96_SEED)
    t, X = lorenz96(...)

    # フェーズ2: 軌跡プロット (261-262)
    fig = plot_trajectories(X, t)
    save_figure(fig, ...)

    # フェーズ3: モード設定構築 (265-285)
    modes = []
    for idx in SINGLE_TARGET_INDICES:
        ...

    # フェーズ4: 実験実行ループ (290-303)
    experiments = []
    for metric_name in METRIC_CONFIGS:
        ...

    # フェーズ5: 結果CSV出力 (306-307)
    results_df = build_results_csv(...)

    # フェーズ6: 可視化 + 検証 (310-316)
    generate_plots(...)
    verification_df = build_verification_csv(...)
```

**原因:**
- 実験スクリプトの「一気通貫」設計
- 各フェーズを独立関数に分割していない

**影響:**
- テストしにくい
- 部分的な再実行が困難
- コードの見通しが悪い

---

## 3. 低優先度の問題

### 3.1 バリデーション不足 (src/mde/data/)

**問題箇所:**
- `src/mde/data/lorenz96.py:6-69` - `lorenz96()`
- `src/mde/data/fly.py:43-60` - `get_fly_columns()`

**具体例:**

```python
# lorenz96.py:6-15
def lorenz96(
    N: int,
    F: float,      # バリデーションなし (F <= 0 でも動作)
    dt: float,     # バリデーションなし (dt <= 0 でも動作)
    t_max: float,  # バリデーションなし (t_max <= 0 でも動作)
    ...
):
    if N < 4:      # N のみチェック
        raise ValueError(...)

# fly.py:43-60
def get_fly_columns(df: pl.DataFrame) -> tuple[list[str], list[str]]:
    ts_cols = [c for c in df.columns if c.startswith("TS")]
    target_cols = ["Left_Right", "FWD"]  # 存在確認なし
    return ts_cols, target_cols
```

**原因:**
- 内部利用を前提とした設計
- 外部入力のバリデーション意識が低い

**影響:**
- 不正な入力で予期せぬ挙動 (無限ループ、ゼロ除算等)

---

### 3.2 Raises セクション欠如 (src/mde/viz/)

**問題箇所:**
- `src/mde/viz/diagnostics.py` - 大部分の関数
- `src/mde/viz/results.py` - 全関数

**具体例:**

```python
# diagnostics.py:8-30 - Raises あり (良い例)
def plot_targets_timeseries(...) -> Figure:
    """...
    Raises
    ------
    ValueError
        If targets does not have shape (T, 2)...
    """

# diagnostics.py:66-97 - Raises なし
def plot_ts_overlay(data: np.ndarray, *, column_names: list[str]) -> Figure:
    """...
    Returns
    -------
    Figure
    """
    # column_names と data の長さが一致しない場合の動作が不明
```

**原因:**
- 一部の関数にのみ Raises セクションを記載
- 一貫性のないドキュメントスタイル

---

## 4. 良好な点 (参考)

コードベースには以下の優れた設計が見られる:

### 4.1 コアモジュールの型ヒント充実

```python
# selection.py - 型エイリアスの適切な使用
CandidateFilter: TypeAlias = Callable[
    [int, np.ndarray, np.ndarray, list[int]],
    bool,
]

# metrics.py - 関数型の明示
MetricFn: TypeAlias = Callable[[np.ndarray, np.ndarray], tuple[float, np.ndarray]]
```

### 4.2 dataclass による構造化

```python
# selection.py
@dataclass
class MDEResult:
    split: TemporalSplit
    selected_indices: list[int]
    manifold: np.ndarray
    val_scores: list[float]
    ...
```

### 4.3 命名規則の一貫性

- Public 関数: `snake_case` (mde, greedy_select, temporal_split)
- Private 関数: `_leading_underscore` (_ensure_2d, _eval_candidate)
- クラス/TypeAlias: `PascalCase` (MDEResult, MetricFn)

### 4.4 充実した docstring (コアモジュール)

```python
# selection.py:274-346 - mde() の docstring
def mde(...) -> MDEResult:
    """Perform Manifold Dimension Expansion to discover causal relationships.

    Parameters
    ----------
    candidates : np.ndarray of shape (T, N)
        ...

    Examples
    --------
    >>> from mde import mde, mean_rho, temporal_split
    ...
    """
```

---

## 改善提案サマリー

| 優先度 | カテゴリ | 対象 | 改善内容 |
|--------|----------|------|----------|
| 高 | DRY | scripts/*.py | `run_experiment()` を `experiment_util.py` に共通化 |
| 高 | 堅牢性 | lorenz96_mde.py:148 | 正規表現または辞書ベースの設計に変更 |
| 高 | データ形式 | lorenz96_mde.py | `ast.literal_eval` 廃止、JSON/直接利用へ |
| 中 | ドキュメント | scripts/*.py | `run_experiment()`, `main()` に docstring 追加 |
| 中 | 型安全性 | scripts/*.py | `TypedDict` による戻り値型の明示 |
| 中 | 責任分離 | lorenz96_mde.py | `main()` を複数の独立関数に分割 |
| 低 | 堅牢性 | data/lorenz96.py | F, dt, t_max のバリデーション追加 |
| 低 | 堅牢性 | data/fly.py | 列存在確認の追加 |
| 低 | ドキュメント | viz/*.py | Raises セクションの統一追加 |
