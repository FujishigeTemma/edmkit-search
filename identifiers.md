# MDE Identifiers Catalog

全パッケージ・モジュールの識別子一覧と命名規則の分析。

---

## 1. パッケージ / モジュール名

| パッケージ / モジュール | 場所 | 命名パターン |
|---|---|---|
| `mde` | `src/mde/` | 略語 (小文字) |
| `mde.data` | `src/mde/data/` | snake_case |
| `mde.viz` | `src/mde/viz/` | 略語 (小文字) |
| `mde.ccm` | `src/mde/ccm.py` | 略語 (小文字) |
| `mde.selection` | `src/mde/selection.py` | snake_case |
| `mde.metrics` | `src/mde/metrics.py` | snake_case |
| `mde.skill` | `src/mde/skill.py` | snake_case |
| `mde.splits` | `src/mde/splits.py` | snake_case |
| `mde.data.fly` | `src/mde/data/fly.py` | snake_case |
| `mde.data.lorenz96` | `src/mde/data/lorenz96.py` | snake_case + 数字 |
| `mde.viz.results` | `src/mde/viz/results.py` | snake_case |
| `mde.viz.diagnostics` | `src/mde/viz/diagnostics.py` | snake_case |
| `mde.viz.types` | `src/mde/viz/types.py` | snake_case |
| `mde.viz.io` | `src/mde/viz/io.py` | 略語 (小文字) |

## 2. クラス (Public NamedTuple)

| クラス名 | 定義元 | フィールド |
|---|---|---|
| `Diagnostics` | `ccm.py` | `lib_sizes`, `score_mean`, `score_var`, `aicc_saturation`, `aicc_linear`, `delta_aicc`, `saturation_params`, `linear_params` |
| `Config` | `metrics.py` | `metric`, `threshold` |
| `Step` | `selection.py` | `dim`, `var_idx`, `score`, `scores_per_target` |
| `Selection` | `selection.py` | `selected_indices`, `val_scores`, `val_scores_per_target` |
| `Evaluation` | `selection.py` | `scores`, `scores_per_target`, `predictions` |
| `Result` | `selection.py` | `split`, `selected_indices`, `manifold`, `val`, `test` |
| `Split` | `splits.py` | `train`, `val`, `test` |
| `PlotData` | `viz/types.py` | `target_label`, `selected_var_names`, `val_scores`, `test_scores`, `val_scores_per_target`, `test_scores_per_target`, `query_indices`, `observations`, `predictions_per_dim` |

## 3. クラス (Internal NamedTuple)

| クラス名 | 定義元 | フィールド |
|---|---|---|
| `_SampleStats` | `ccm.py` | `lib_sizes`, `score_mean`, `score_var` |
| `_ModelFit` | `ccm.py` | `aicc_saturation`, `aicc_linear`, `delta_aicc`, `saturation_params`, `linear_params` |
| `_EvalArgs` | `selection.py` | `var_idx`, `selected_indices`, `X_train`, `X_val`, `Y_train`, `Y_val`, `threshold`, `metric`, `predict` |
| `_EvalResult` | `selection.py` | `var_idx`, `score`, `scores_per_target` |

## 4. 型エイリアス (TypeAlias)

| 名前 | 定義元 | 型 |
|---|---|---|
| `MetricFn` | `metrics.py` | `Callable[[ndarray, ndarray], tuple[float, ndarray]]` |
| `PredictFn` | `skill.py` | `Callable[[ndarray, ndarray, ndarray], ndarray]` |
| `Filter` | `selection.py` | `Callable[[int, ndarray, ndarray, list[int]], bool]` |

## 5. Public 関数

### コアアルゴリズム (`selection.py`)

| 関数名 | パラメータ |
|---|---|
| `mde()` | `candidates`, `target`, `split`, `predict`, `metric`, `threshold`, `max_dim`, `max_workers`, `candidate_filter`, `store_predictions` |
| `select()` | `candidates`, `target`, `train_indices`, `val_indices`, `predict`, `metric`, `threshold`, `max_dim`, `max_workers`, `candidate_filter` |
| `select_iter()` | `candidates`, `target`, `train_indices`, `val_indices`, `predict`, `metric`, `threshold`, `max_dim`, `max_workers`, `candidate_filter` |
| `get_predictions()` | `candidates`, `target`, `selected_indices`, `train_indices`, `query_indices`, `predict`, `dim` |
| `evaluate_manifold()` | `candidates`, `target`, `selected_indices`, `train_indices`, `query_indices`, `predict`, `metric`, `store_predictions` |

### 予測スキル (`skill.py`)

| 関数名 | パラメータ |
|---|---|
| `prediction_skill()` | `manifold`, `target`, `train_indices`, `query_indices`, `predict`, `metric` |

### CCM (`ccm.py`)

| 関数名 | パラメータ |
|---|---|
| `converged()` | `cause`, `effect`, `lib_sizes`, `E`, `tau`, `num_samples`, `predict`, `metric`, `aicc_threshold`, `rng` |
| `diagnostics()` | `cause`, `effect`, `lib_sizes`, `E`, `tau`, `num_samples`, `predict`, `metric`, `rng` |
| `make_filter()` | `target`, `train_indices`, `predict`, `metric`, `E`, `tau`, `num_samples`, `aicc_threshold`, `rng` |

### メトリクス (`metrics.py`)

| 関数名 | パラメータ |
|---|---|
| `mean_rho()` | `predictions`, `observations` |
| `rmse()` | `predictions`, `observations` |
| `mae()` | `predictions`, `observations` |
| `negate()` | `metric` |

### データ分割 (`splits.py`)

| 関数名 | パラメータ |
|---|---|
| `temporal_split()` | `length`, `train_ratio`, `val_ratio`, `gap` |

### データ (`data/`)

| 関数名 | パラメータ |
|---|---|
| `load()` | `path` |
| `columns()` | `df` |
| `lorenz96()` | `N`, `F`, `dt`, `t_max`, `X0`, `t_spinup`, `rng` |
| `true_neighbors()` | `target_idx`, `N` |

### 可視化 (`viz/`)

| 関数名 | パラメータ |
|---|---|
| `plot_mde_results()` | `mde_data` |
| `plot_mde_results_multi()` | `mde_data`, `target_names` |
| `plot_predictions()` | `mde_data` |
| `plot_predictions_multi()` | `mde_data`, `target_names` |
| `plot_targets_timeseries()` | `targets`, `target_names` |
| `plot_ts_overlay()` | `data`, `column_names` |
| `plot_distribution_stats()` | `data`, `column_names` |
| `plot_autocorrelation()` | `data`, `column_names`, `target_data`, `target_names`, `lags` |
| `plot_correlation_heatmap()` | `candidates`, `targets`, `candidate_names`, `target_names` |
| `save_figure()` | `fig`, `path`, `dpi` |

## 6. Private / Internal 関数

| 関数名 | 定義元 |
|---|---|
| `_validate_metric_inputs()` | `metrics.py` |
| `_ensure_2d()` | `skill.py` |
| `_prepare_ccm_embedding()` | `ccm.py` |
| `_sample_ccm_scores()` | `ccm.py` |
| `_compute_sample_statistics()` | `ccm.py` |
| `_fit_linear()` | `ccm.py` |
| `_fit_saturation()` | `ccm.py` |
| `_aicc()` | `ccm.py` |
| `_compute_ccm_statistics()` | `ccm.py` |
| `_fit_convergence_models()` | `ccm.py` |
| `_eval_candidate()` | `selection.py` |
| `_rhs()` | `data/lorenz96.py` |
| `_rk4_step()` | `data/lorenz96.py` |

## 7. モジュールレベル定数・変数

### パッケージ内

| 名前 | 定義元 | パターン |
|---|---|---|
| `_DEFAULT_DATA_PATH` | `data/fly.py` | `_SCREAMING_SNAKE` |

### スクリプト定数

| 名前 | 定義元 | パターン |
|---|---|---|
| `OUTPUT_DIR` | 全スクリプト | `SCREAMING_SNAKE` |
| `DATA_PATH` | `fly_mde.py`, `fly_mde_ccm.py`, `data_analysis.py` | `SCREAMING_SNAKE` |
| `MAX_DIM` | `fly_mde.py`, `lorenz96_mde.py`, `fly_mde_ccm.py` | `SCREAMING_SNAKE` |
| `GAP` | 同上 | `SCREAMING_SNAKE` |
| `TRAIN_RATIO` | 同上 | `SCREAMING_SNAKE` |
| `VAL_RATIO` | 同上 | `SCREAMING_SNAKE` |
| `TARGET_NAMES` | 同上 | `SCREAMING_SNAKE` |
| `METRIC_CONFIGS` | 同上 | `SCREAMING_SNAKE` |
| `MODES` | `fly_mde.py`, `fly_mde_ccm.py` | `SCREAMING_SNAKE` |
| `PREFIX` | `data_analysis.py` | `SCREAMING_SNAKE` |
| `L96_N`, `L96_F`, `L96_DT`, `L96_T_MAX`, `L96_T_SPINUP`, `L96_SEED` | `lorenz96_mde.py` | `SCREAMING_SNAKE` |
| `SINGLE_TARGET_INDICES`, `MULTI_TARGET_INDICES` | `lorenz96_mde.py` | `SCREAMING_SNAKE` |

## 8. スクリプト関数

| 関数名 | 定義元 |
|---|---|
| `build_plot_data()` | `experiment_util.py` |
| `build_results_csv()` | `experiment_util.py` |
| `generate_plots()` | `experiment_util.py` |
| `run_experiment()` | `fly_mde.py`, `lorenz96_mde.py`, `fly_mde_ccm.py` |
| `main()` | 全スクリプト |
| `exclude_targets()` | `lorenz96_mde.py` |
| `verify_neighbor_selection()` | `lorenz96_mde.py` |
| `build_verification_csv()` | `lorenz96_mde.py` |
| `plot_trajectories()` | `lorenz96_mde.py` |
| `plot_neighbor_selection()` | `lorenz96_mde.py` |

## 9. 主要なローカル変数パターン

### 数学的慣習 (大文字1文字)

科学計算コードでの標準的パターン:

| 変数名 | 意味 | 使用箇所 |
|---|---|---|
| `T` | 時系列長 | `selection.py`, `skill.py`, 他 |
| `N` | 変数数 | `selection.py`, `lorenz96.py`, 他 |
| `M` | ターゲット数 | `ccm.py`, `viz/results.py` |
| `E` | 埋め込み次元 | `ccm.py` |
| `F` | 強制定数 | `lorenz96.py` |
| `L` | ライブラリサイズ配列 | `ccm.py` |
| `X` | 状態配列 | `lorenz96.py`, `lorenz96_mde.py` |
| `X0` | 初期条件 | `lorenz96.py` |
| `A` | デザイン行列 | `ccm.py:_fit_linear` |

### 接頭辞付き変数 (PascalCase接頭辞 + snake_case)

| 変数名 | 使用箇所 |
|---|---|
| `X_train`, `X_val`, `X_lib`, `X_query` | `selection.py`, `skill.py` |
| `Y_train`, `Y_val`, `Y_lib` | `selection.py`, `skill.py` |
| `X_all`, `X_all[i]` | `lorenz96.py` |
| `Y_train_full` | `ccm.py:make_filter` |

### 一般的な snake_case ローカル変数

`selected_indices`, `available_indices`, `train_indices`, `val_indices`, `query_indices`,
`effect_embedded`, `cause_aligned`, `all_samples`, `all_preds`, `all_obs`,
`pred_matrix`, `obs_matrix`, `valid_results`, `best_result`,
`eval_args`, `candidate_indices`, `manifold_train`, `manifold_val`,
`per_target`, `scores_per_target`, `predictions_list`, `ground_truth`,
`score_mean`, `score_var`, `lib_sizes`, `lib_indices`, `test_indices`, `test_mask`,
`sqrt_w`, `residual_fn`, `pred_c`, `obs_c`, `pred_std`, `obs_std`,
`nan_mask`, `has_nan`, `denom`, `cov`,
`train_end`, `val_end`, `train_ratio`, `val_ratio`,
`ts_cols`, `target_cols`, `var_names`, `cand_var_names`, `cand_names`,
`results_df`, `verification_df`, `experiments`, `rows`, `row`,
`fig`, `ax`, `axes`, `im`, `colors`, `cmap`, `labels`, `matrix`,
`rng`, `rss`, `acf`, `preds`, `pred`, `obs`

---

## 10. カテゴリ別命名分析

### A. データ配列を表す命名

データそのものを保持する numpy 配列の命名パターン。

| 名前 | 意味 | 使用箇所 | 型 |
|---|---|---|---|
| `candidates` | 候補変数の時系列行列 | `selection.py`, scripts | `(T, N)` |
| `target` | 目的変数 | `selection.py`, scripts | `(T,)` or `(T, M)` |
| `manifold` | 構築された多様体 | `selection.py`, `skill.py` | `(T, D)` |
| `cause` | CCMの原因変数 | `ccm.py` | `(T,)` or `(T, M)` |
| `effect` | CCMの結果変数 | `ccm.py` | `(T,)` or `(T, M)` |
| `observations` | 観測値 (正解値) | `ccm.py`, `experiment_util.py`, `viz/types.py` | `(N,)` or `(N, M)` |
| `predictions` | 予測値 | `selection.py`, `skill.py`, `ccm.py` | `(N,)` or `(N, M)` |
| `data` | 汎用データ配列 | `viz/diagnostics.py` | `(T, N)` |

### B. インデックスを表す命名

規則: **`_indices` (配列) vs `_idx` (スカラー)** の使い分けは全体的に一貫。

| 名前 | 意味 | 型 | 使用箇所 |
|---|---|---|---|
| `train_indices` | 訓練用インデックス配列 | `ndarray` | `selection.py`, `skill.py`, `ccm.py` |
| `val_indices` | 検証用インデックス配列 | `ndarray` | `selection.py` |
| `test_indices` | テスト用インデックス配列 | `ndarray` (ローカル) | `ccm.py` |
| `query_indices` | 予測クエリ用インデックス配列 | `ndarray` | `skill.py`, `selection.py`, `viz/types.py` |
| `selected_indices` | 選択済み変数インデックス | `list[int]` | `selection.py`, `ccm.py` |
| `candidate_indices` | 候補変数インデックス | `list[int]` | `selection.py`, `lorenz96_mde.py` |
| `lib_indices` | CCMライブラリインデックス | `ndarray` | `ccm.py` |
| `target_indices` | ターゲット列インデックス | `list[int]` | scripts |
| `var_idx` | 単一変数インデックス | `int` | `selection.py`, `ccm.py` |
| `cause_idx` | 単一原因変数インデックス | `int` | `ccm.py` |
| `target_idx` | 単一ターゲットインデックス | `int` | `lorenz96_mde.py` |
| `dim_idx` | 次元インデックス | `int` | `viz/results.py`, scripts |
| `row_idx` | 行インデックス | `int` | `lorenz96_mde.py` |

### C. Train / Validation / Test に関連する命名

#### C-1. 2つの分割パラダイムの共存

| パラダイム | 用語 | 使用箇所 | 文脈 |
|---|---|---|---|
| 時系列分割 | `train` / `val` / `test` | `splits.py`, `selection.py` | MDE の貪欲選択 |
| EDM/CCM | `lib` / `query` | `ccm.py`, `skill.py` | ライブラリ構築・予測 |

対応関係:
```
train_indices → X_train / Y_train   (selection.py)
train_indices → X_lib / Y_lib       (skill.py, get_predictions)
val_indices   → X_val / Y_val       (selection.py)
query_indices → X_query             (skill.py, get_predictions)
lib_indices   → X_lib / Y_lib       (ccm.py)
```

`train` と `lib` は同じ概念を指すが、文脈ごとに用語が切り替わる。
`PredictFn` のドキュメントは `(X_lib, Y_lib, X_query)` だが、`select_iter()` 内では `X_train` / `Y_train` から抽出したデータを predict に渡す。ドメイン文脈に応じた使い分けであり許容範囲。

#### C-2. `val` 略称の統一

| 場所 | フィールド名 | 形式 |
|---|---|---|
| `Split.val` | `val` | 略称 |
| `Selection.val_scores` | `val_scores` | 略称 |
| `Selection.val_scores_per_target` | `val_scores_per_target` | 略称 |
| `PlotData.val_scores` | `val_scores` | 略称 |
| `Result.val` | `val` | 略称 |

**解決済み:** `val` に統一。

### D. 候補変数・目的変数に関連する命名

#### D-1. 分解後のローカル変数パターン

| ローカル変数 | 命名パターン | 使用箇所 |
|---|---|---|
| `X_train`, `X_val`, `Y_train`, `Y_val` | 大文字接頭辞 + `_split` | `selection.py` |
| `X_lib`, `Y_lib`, `X_query` | 大文字接頭辞 + `_role` | `skill.py`, `get_predictions` |
| `manifold_train`, `manifold_val` | snake_case + `_split` | `selection.py:_eval_candidate` |
| `target_2d`, `cause_2d`, `effect_2d` | snake_case + `_2d` | `selection.py:mde`, `ccm.py` |
| `cause_aligned`, `effect_embedded` | snake_case + `_状態` | `ccm.py` |
| `Y_train_full` | 大文字接頭辞 + `_train_full` | `ccm.py:make_filter` |

#### D-2. スクリプトでの略称不統一

| 変数 | 使用箇所 |
|---|---|
| `cand_var_names` | `fly_mde.py`, `fly_mde_ccm.py` |
| `cand_names` | `lorenz96_mde.py` |

同じ意味だがスクリプト間で名前が異なる。

### E. 各種結果を表す命名

#### E-1. Result コンテナの階層

```
Result (トップレベル)
├── split: Split
├── selected_indices: list[int]
├── manifold: ndarray
├── val: Evaluation
│   ├── scores: list[float]
│   ├── scores_per_target: list[ndarray]
│   └── predictions: list[ndarray] | None
└── test: Evaluation
    ├── scores
    ├── scores_per_target
    └── predictions

Selection (中間結果)
├── selected_indices: list[int]
├── val_scores: list[float]              ← "val" (略称)
└── val_scores_per_target: list[ndarray]

Step (1ステップ)
├── dim: int
├── var_idx: int
├── score: float
└── scores_per_target: ndarray

_EvalResult (内部)
├── var_idx: int
├── score: float
└── scores_per_target: ndarray

PlotData (可視化)
├── val_scores, test_scores
├── val_scores_per_target, test_scores_per_target
├── observations: ndarray
└── predictions_per_dim: list[ndarray]
```

#### E-2. per-target スコアの語順

**解決済み:** `scores_per_target` (suffix 形) に統一。全箇所が suffix 形。

| 場所 | フィールド名 |
|---|---|
| `Step` | `scores_per_target` |
| `_EvalResult` | `scores_per_target` |
| `Evaluation` | `scores_per_target` |
| `Selection` | `val_scores_per_target` |
| `PlotData` | `val_scores_per_target` |
| `PlotData` | `test_scores_per_target` |

#### E-3. predictions の命名バリエーション

| 変数名 | 型 | 使用箇所 |
|---|---|---|
| `predictions` | `ndarray` | `skill.py`, `_eval_candidate` (1回の予測) |
| `predictions` | `list[ndarray] \| None` | `Evaluation` フィールド (次元ごと) |
| `preds` | `ndarray` | `get_predictions`, `evaluate_manifold` (ローカル略称) |
| `all_preds` | `list` | `ccm.py` (CCM内の蓄積) |
| `pred_matrix` | `ndarray` | `ccm.py` (列結合した行列) |
| `predictions_list` | `list[ndarray]` | `viz/results.py` (可視化用) |
| `predictions_per_dim` | `list[ndarray]` | `PlotData` フィールド |

---

## 11. 設計上の問題点まとめ (重要度順)

### ~~1. `per_target_scores` vs `scores_per_target` 語順の不統一~~ [解決済み]

`scores_per_target` (suffix 形) に統一。

### ~~2. `target` vs `targets` の不統一~~ [解決済み]

`target` に統一。

### ~~3. `val` vs `validation` の不統一~~ [解決済み]

`val` に統一。

### 4. `train`/`lib` と `val`/`query` の二重用語 [低 - 許容]

EDM ドメインの `lib`/`query` と時系列分割の `train`/`val` が共存。
各関数のスコープ内では一貫しており、ドメイン文脈に応じた使い分け。
**対応:** 変更なし。

### 5. `Evaluation.predictions` vs `PlotData.predictions_per_dim` [低 - 許容]

異なる抽象レベルでの命名差異。`PlotData` は可視化特化で `_per_dim` が内容を明示。
**対応:** 変更なし。
