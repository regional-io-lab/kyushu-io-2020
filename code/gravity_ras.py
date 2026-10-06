"""
gravity_ras.py — Gravity-RAS法による県間交易額の推計：参考実装

【重要な注記】
このスクリプトは、本リポジトリのdata/に収録した推計値を実際に生成した
本番コードそのものではありません。method/methodology.mdに記載した手法
（山田光男 2013「グラビティ-RAS法による地域間交易の推計」と同一の手法体系）
を、第三者が理解・再現しやすい形で書き直した参考実装です。

目的：
  - 手法の計算ロジックを明示し、再現性・検証可能性を確保すること
  - 自分のデータ（他地域の県別IO表など）に同じ手法を適用したい人の出発点
    となること

このコードをそのまま実行しても、本リポジトリのCSV/Excelデータと完全に
一致する数値が出るとは限りません（実際の推計では、部門ごとの個別調整や
長崎県のコード不一致バグ修正など、手作業による補正が入っています。詳細は
README.mdの「データ品質に関する既知の経緯」を参照してください）。

必要パッケージ: numpy, pandas
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def gravity_initial_allocation(
    origin_output: pd.Series,
    dest_demand: pd.Series,
    distance: pd.DataFrame,
    beta: float = 1.5,
) -> pd.DataFrame:
    """重力モデルによる県間取引額の初期配分を行う。

    Parameters
    ----------
    origin_output : 各県（発）の当該部門の移出可能額（index=県名）
    dest_demand : 各県（着）の当該部門の移入需要額（index=県名）
    distance : 県間の道路移動時間（分）等を格納したDataFrame（index=発, columns=着）
        ※ 直線距離ではなく実際の道路移動時間を使うことを推奨
        （長崎県の有明海迂回ルートのような、直線距離では捉えられない
        移動コストを反映するため）
    beta : 距離減衰係数（既定値1.5）

    Returns
    -------
    県間取引額の初期配分マトリクス（index=発, columns=着）
    """
    origins = origin_output.index
    dests = dest_demand.index

    allocation = pd.DataFrame(0.0, index=origins, columns=dests)
    for o in origins:
        for d in dests:
            if o == d:
                continue  # 県内取引は対象外（移出入のみを推計）
            dist = distance.loc[o, d]
            if dist <= 0:
                continue
            weight = origin_output[o] * dest_demand[d] / (dist**beta)
            allocation.loc[o, d] = weight

    # 各県の移出可能額の合計に合わせてスケーリング（初期値）
    row_sums = allocation.sum(axis=1)
    for o in origins:
        if row_sums[o] > 0:
            allocation.loc[o] *= origin_output[o] / row_sums[o]

    return allocation


def ras_balance(
    initial_matrix: pd.DataFrame,
    row_targets: pd.Series,
    col_targets: pd.Series,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> pd.DataFrame:
    """RAS法により、行和・列和が指定値に収束するまで反復スケーリングする。

    Parameters
    ----------
    initial_matrix : 重力モデルによる初期配分マトリクス
    row_targets : 各県（発）の移出合計の目標値
    col_targets : 各県（着）の移入合計の目標値
    max_iter : 最大反復回数
    tol : 収束判定の許容誤差

    Returns
    -------
    行和・列和が目標値に収束した（または最大反復回数に達した）マトリクス
    """
    matrix = initial_matrix.copy()

    for i in range(max_iter):
        # 行方向のスケーリング
        row_sums = matrix.sum(axis=1)
        row_factors = row_targets / row_sums.replace(0, np.nan)
        matrix = matrix.mul(row_factors, axis=0).fillna(0)

        # 列方向のスケーリング
        col_sums = matrix.sum(axis=0)
        col_factors = col_targets / col_sums.replace(0, np.nan)
        matrix = matrix.mul(col_factors, axis=1).fillna(0)

        row_err = (matrix.sum(axis=1) - row_targets).abs().max()
        col_err = (matrix.sum(axis=0) - col_targets).abs().max()
        if max(row_err, col_err) < tol:
            break

    return matrix


def estimate_interregional_trade(
    origin_output: pd.Series,
    dest_demand: pd.Series,
    distance: pd.DataFrame,
    beta: float = 1.5,
) -> pd.DataFrame:
    """重力モデル初期配分 → RAS収束、までを一括実行する。"""
    initial = gravity_initial_allocation(origin_output, dest_demand, distance, beta=beta)
    balanced = ras_balance(initial, row_targets=origin_output, col_targets=dest_demand)
    return balanced


if __name__ == "__main__":
    # 使い方の最小例（ダミーデータ）
    prefectures = ["福岡", "佐賀", "長崎", "熊本", "大分", "宮崎", "鹿児島", "山口", "沖縄"]

    rng = np.random.default_rng(0)
    output = pd.Series(rng.uniform(1000, 10000, len(prefectures)), index=prefectures)
    demand = pd.Series(rng.uniform(1000, 10000, len(prefectures)), index=prefectures)

    # 実際には道路移動時間（分）のテーブルを使用する
    dist_values = rng.uniform(30, 300, (len(prefectures), len(prefectures)))
    np.fill_diagonal(dist_values, 0)
    dist = pd.DataFrame(dist_values, index=prefectures, columns=prefectures)

    result = estimate_interregional_trade(output, demand, dist)
    print(result.round(1))
