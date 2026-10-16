# core/calc.py
import numpy as np
import pandas as pd


def _find_col(columns, *contains):
    for col in columns:
        if all(k in col for k in contains):
            return col
    return None


def _num(df, col):
    if col is None:
        return pd.Series(float('nan'), index=df.index)
    return pd.to_numeric(df[col], errors='coerce')


def calculate_safety_stock(df):
    """计算安全库存及衍生指标，原地新增列并返回 df。"""
    quality_col = _find_col(df.columns, '质量')
    strategy_col = _find_col(df.columns, '策略')
    cycle_col = _find_col(df.columns, '采购周期')
    future_avg_col = _find_col(df.columns, '未来', '月均')
    actual_col = _find_col(df.columns, '实际库存')

    quality = _num(df, quality_col)
    strategy = _num(df, strategy_col)
    cycle = _num(df, cycle_col)
    future_avg = _num(df, future_avg_col)
    actual = _num(df, actual_col)

    # 风险加权 = 质量系数 * 0.4 + 策略系数 * 0.6（两个口径共用）
    risk = quality * 0.4 + strategy * 0.6
    df['风险加权'] = risk.round(2)

    def _run(demand_col, suffix, safety_name):
        demand = _num(df, demand_col)
        base = (future_avg + demand) / 2
        safety = base * risk * cycle

        gap = (safety - actual).clip(lower=0)

        coverage = actual / safety
        coverage = coverage.where(safety != 0, 1.0)

        status = np.where(
            coverage.isna(), '未知',
            np.where(coverage < 0.5, '严重',
                     np.where(coverage < 1.0, '预警', '安全'))
        )

        df[f'基础月均需求{suffix}'] = base.round(2)
        df[safety_name] = safety.round(2)
        df[f'缺口{suffix}'] = gap.round(2)
        df[f'覆盖率{suffix}'] = coverage.round(2)
        df[f'状态{suffix}'] = status

    # 主口径：O_原始
    o_orig = 'O_原始' if 'O_原始' in df.columns else _find_col(df.columns, '月均量')
    _run(o_orig, '', '安全库存')

    # 对照口径：O_修正
    if 'O_修正' in df.columns:
        _run('O_修正', '_修正', '安全库存_修正版')

    return df