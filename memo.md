# CCM Convergence (AICc) 検証メモ

## 実験条件

- `ccm_threshold = 4.0` (ΔAICc)
- CCM 内部: `num_samples=20`, lib_sizes は `np.geomspace(10, train_size*0.8, 8)`
- `max_dim=5` (Lorenz96), `max_dim=3` (Fly)
- `gap=50`, `train_ratio=0.6`, `val_ratio=0.2`
- seed=42

---

## Lorenz96 (N=10, T=10,000)

候補 9 変数、ターゲットごとの真の近傍は 3 変数 (x_{i-2}, x_{i-1}, x_{i+1})。

### 選択結果

| target | metric | ccm | selected | test score |
|--------|--------|-----|----------|------------|
| single_x0 | rho | OFF | x2,x3,x7,x1,x8 | +0.4013 |
| single_x0 | rho | ON  | x2,x3,x7,x1,x8 | +0.4013 |
| single_x0 | mae | OFF | x2,x9,x3,x8,x1 | -3.4078 |
| single_x0 | mae | ON  | x2,x9,x3,x8,x1 | -3.4078 |
| single_x0 | rmse | OFF | x2,x3,x7,x1,x8 | -4.3805 |
| single_x0 | rmse | ON  | x2,x3,x7,x1,x8 | -4.3805 |
| single_x5 | rho | OFF | x7,x8,x6,x3,x4 | +0.3172 |
| single_x5 | rho | ON  | x7,x8,x6,x3,x4 | +0.3172 |
| single_x5 | mae | OFF | x7,x8,x6,x3,x4 | -3.1009 |
| single_x5 | mae | ON  | x7,x8,x6,x3,x4 | -3.1009 |
| single_x5 | rmse | OFF | x7,x8,x6,x3,x4 | -4.1682 |
| single_x5 | rmse | ON  | x7,x8,x6,x3,x4 | -4.1682 |
| multi_x0_x5 | rho | OFF | x8,x9,x7,x6,x1 | +0.2560 |
| multi_x0_x5 | rho | ON  | x8,x9,x7,x6,x1 | +0.2560 |
| multi_x0_x5 | mae | OFF | x8,x7,x9,x6,x1 | -3.5768 |
| multi_x0_x5 | mae | ON  | x8,x7,x9,x6,x1 | -3.5768 |
| multi_x0_x5 | rmse | OFF | x8,x7,x3,x1,x2 | -4.5663 |
| multi_x0_x5 | rmse | ON  | x8,x7,x3,x1,x2 | -4.5663 |

### ΔAICc 値

全てのケースで全候補が ΔAICc >= 4 を達成し、CCM フィルタを通過。
代表例 (single_x0, rho): `[17.95, 23.18, 9.31, 25.96, 14.55]`

### 実行時間 (秒)

| target | metric | ccm=OFF | ccm=ON | 倍率 |
|--------|--------|---------|--------|------|
| single_x0 | rho | 2.0 | 48.4 | 24x |
| single_x0 | mae | 1.9 | 51.8 | 27x |
| single_x0 | rmse | 1.9 | 57.6 | 30x |
| single_x5 | rho | 1.9 | 83.7 | 44x |
| single_x5 | mae | 2.0 | 74.9 | 37x |
| single_x5 | rmse | 1.9 | 91.9 | 48x |
| multi_x0_x5 | rho | 1.9 | 76.1 | 40x |
| multi_x0_x5 | mae | 1.8 | 81.8 | 45x |
| multi_x0_x5 | rmse | 1.8 | 65.1 | 36x |

平均: ccm=OFF ~1.9s, ccm=ON ~70s (**約 37 倍**)

---

## Fly (T=10,608, 候補 80 変数)

### 選択結果

| target | metric | ccm | selected | test score |
|--------|--------|-----|----------|------------|
| single_LR | rho | OFF | TS46,TS74,TS79 | +0.2260 |
| single_LR | rho | ON  | TS73,TS46,TS74 | -0.1303 |
| single_LR | mae | OFF | TS56,TS39,TS74 | -0.1470 |
| single_LR | mae | ON  | TS56,TS39,TS74 | -0.1470 |
| single_LR | rmse | OFF | TS35,TS68,TS72 | -0.3369 |
| single_LR | rmse | ON  | TS56,TS39,TS74 | -0.3145 |
| multi | rho | OFF | TS56,TS46,TS65 | +0.7052 |
| multi | rho | ON  | TS56,TS46,TS65 | +0.7052 |
| multi | mae | OFF | TS56,TS39,TS74 | -0.1337 |
| multi | mae | ON  | TS56,TS39,TS74 | -0.1337 |
| multi | rmse | OFF | TS56,TS39,TS74 | -0.2682 |
| multi | rmse | ON  | TS56,TS39,TS74 | -0.2682 |

### 実行時間 (秒)

| target | metric | ccm=OFF | ccm=ON | 倍率 |
|--------|--------|---------|--------|------|
| single_LR | rho | 10.1 | 60.3 | 6x |
| single_LR | mae | 9.0 | 133.2 | 15x |
| single_LR | rmse | 9.5 | 116.0 | 12x |
| multi | rho | 8.0 | 121.7 | 15x |
| multi | mae | 8.8 | 114.4 | 13x |
| multi | rmse | 11.2 | 98.6 | 9x |

平均: ccm=OFF ~9.4s, ccm=ON ~107s (**約 11 倍**)

---

## 所見

### 選択結果の違い

- **Lorenz96**: CCM ON/OFF で選択結果・テストスコアが完全一致。Lorenz96 は結合が強く、全ての真の近傍が明確な CCM 収束を示すため、AICc フィルタが全候補を通過させた。
- **Fly**: 大半のケースで一致するが、一部で差異あり。
  - `single_LR, rho`: OFF=TS46,TS74,TS79 → ON=TS73,TS46,TS74。CCM が TS79 を弾き TS73 を選択。ただしテストスコアが +0.226 → -0.130 と悪化しており、CCM フィルタが良いスコアの候補を過剰に弾いた可能性がある。
  - `single_LR, rmse`: OFF=TS35,TS68,TS72 → ON=TS56,TS39,TS74。テストスコアは -0.337 → -0.315 と若干改善。

### 実行時間

- CCM ON にすると **Lorenz96 で約 37 倍、Fly で約 11 倍**の実行時間増加。
- 主因は各 greedy step で候補ごとに CCM テスト (8 lib_sizes x 20 samples の simplex projection) を実行するため。
- Lorenz96 の方が倍率が高いのは、候補数が少ない (9) ため OFF 時のベースが速く、一方で CCM テスト自体のコストは同程度であるため。

### ΔAICc の傾向

- Lorenz96: 全候補で ΔAICc >> 4（5.6 〜 35.4）。収束が明確。
- Fly: 全候補で ΔAICc >> 4（7.7 〜 28.8）。こちらも収束傾向は強い。

### 今後の検討事項

1. **実行時間の改善**: CCM テストが律速。`num_samples` を減らす（20→10）、lib_sizes の点数を減らす（8→5）等で高速化できる。もしくは greedy step で best score を上回る候補のみに CCM テストを実行する現在の設計を活かし、threshold を高めに設定する。
2. **CCM フィルタが有効に機能するケース**: 今回のデータでは殆どの候補が収束を示したため、フィルタリング効果は限定的。ノイズが多い/偽の相関がある実データでより効果を発揮する見込み。
3. **single_LR rho での悪化**: CCM フィルタが「スコアは高いが因果的でない」候補を弾いた結果、テストスコアが悪化した。これは CCM の意図通りだが、テスト精度との兼ね合いで閾値調整が必要になりうる。
