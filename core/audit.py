# core/audit.py
import re
import numpy as np
import pandas as pd


def _month_key(col):
    m = re.search(r'(\d{4})\D*M(\d{2})', col)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    m = re.search(r'M(\d{2})', col)
    if m:
        return (0, int(m.group(1)))
    return None


def _split_months(month_cols, meta):
    keys = {c: _month_key(c) for c in month_cols}
    valid = [c for c in month_cols if keys[c] is not None]
    snap = None
    if meta.get('snapshot_date'):
        snap = (int(meta['snapshot_date'][:4]), int(meta['snapshot_date'][5:7]))
    if snap:
        future_cols = [c for c in valid if keys[c] > snap]
        past_cols = [c for c in valid if keys[c] <= snap]
    else:
        ordered = sorted(valid, key=lambda c: keys[c])
        past_cols = ordered[:6]
        future_cols = ordered[6:]
    return future_cols, past_cols


def audit_data(df, meta):
    """纯计算：统计空值/0值/重复编码，生成 O 列对照与标记列，返回统计字典。

    不包含任何网页渲染逻辑，展示在 app.py 中完成。
    """
    result = {}

    month_cols = meta.get('month_cols', [])
    sap_col = meta.get('sap_col')
    mat_col = meta.get('mat_col')

    future_cols, past_cols = _split_months(month_cols, meta)

    # 总行数
    result['total_rows'] = int(len(df))

    # 各月用量列空值 / 0值统计
    result['month_null_counts'] = {c: int(df[c].isna().sum()) for c in month_cols}
    result['month_zero_counts'] = {c: int((df[c] == 0).sum()) for c in month_cols}

    # SAP 缺失列表与重复组
    if sap_col:
        missing_mask = df[sap_col].isna()
        if mat_col:
            result['sap_missing'] = df.loc[missing_mask, mat_col].astype(str).tolist()
        else:
            result['sap_missing'] = df.index[missing_mask].astype(str).tolist()

        dup_groups = {}
        valid = df[df[sap_col].notna()]
        if mat_col and len(valid):
            for sap, grp in valid.groupby(sap_col, dropna=True):
                codes = grp[mat_col].astype(str).tolist()
                if len(codes) > 1:
                    dup_groups[str(int(sap))] = codes
        result['sap_dup_groups'] = dup_groups
    else:
        result['sap_missing'] = []
        result['sap_dup_groups'] = {}

    # O 列对照：O_原始 / O_修正 = SUM(过去6月) / COUNT(过去6月)
    o_col = next((c for c in df.columns if '月均量' in c), None)
    if o_col is not None:
        df['O_原始'] = pd.to_numeric(df[o_col], errors='coerce')
    else:
        df['O_原始'] = pd.NA

    if past_cols:
        total = df[past_cols].sum(axis=1)
        cnt = df[past_cols].notna().sum(axis=1).astype('float64')
        df['O_修正'] = total / cnt.replace(0, np.nan)
    else:
        df['O_修正'] = pd.NA

    # 标记列
    if future_cols:
        df['未来3月_零值数'] = (df[future_cols] == 0).sum(axis=1)
        df['未来3月_全为0'] = (df[future_cols] == 0).all(axis=1)
    else:
        df['未来3月_零值数'] = 0
        df['未来3月_全为0'] = False

    if 'SAP缺失' not in df.columns:
        df['SAP缺失'] = df[sap_col].isna() if sap_col else True

    result['o_col'] = o_col
    result['future_cols'] = future_cols
    result['past_cols'] = past_cols
    return result