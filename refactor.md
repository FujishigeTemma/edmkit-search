# リファクタリング方針

## 大目的
- mdeのアルゴリズムを汎用的で拡張しやすくミニマルで可読性の高い形で実装する
- mdeを使って見つけた変数の組を使ってsimplex projectionしたときの精度が、LSTM+baselineなどmachine learning系の手法より優れていることを示すこと

## 前提

- mdeの実装がある
  - まだまだ綺麗な実装とは言い難い
  - 後方互換性を無視したAPIの変更を許容する
- ~~fly dataとlorenz96にmdeを使って変数の組を最適化するコードとその可視化がある~~ 削除済み
  - あくまでPoCなので、重要でない
- 時系列データを扱うため、validationではrolling validationが必要です。  

## ~~1. Dataset + DataLoder移行~~ 完了済み

現状のデータの受け渡し方から、Dataset/DataLoaderを利用した方法に移行する
dataset.py にベース実装が存在する。必要であれば変更して構わない。

## 2. 全体のリファクタリング

全体の設計を見直すためにPoC用だった「fly dataとlorenz96にmdeを使って変数の組を最適化するコードとその可視化」を削除した
「Empirical Dynamic ModelingにおけるSimplex ProjectionやS-Mapの精度をもとに、最適な変数を探索する」というのが本質的な処理

@CODING.md

TODO `# REFACTOR: `コメントを参考にリファクタリング方針を計画する（記載がない部分も含めて検討が必要）。

## 3. MLモデルをtinygradで実装

ドキュメントのexampleを参考に実装する

## 4. なるべく共通の流れで学習と検証を実装する

極力、mlの標準的なワークフローに合わせて実装する

1. dataの準備
2. train
  - mde: mdeを使って変数の組を見つける -> 保存
  - ml: モデルの訓練 -> 保存
3. validation
  - mde: 見つかった変数の組を使ってsimplex_projection
  - ml: 訓練済みモデルを使った推論
