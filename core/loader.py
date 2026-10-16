# core/loader.py
import re
import calendar
import pandas as pd


def _clean_header(value):
    return str(value).replace('\n', '').replace('\r', '').strip()


def _to_str(value):
    if pd.isna(value):
        return ''
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value).strip()


def _find_col(columns, *contains):
    for col in columns:
        if all(k in col for k in contains):
            return col
    return None


def _find_month_columns(columns):
    pattern = re.compile(r'M\d{2}')
    return [c for c in columns if pattern.search(c)]


def _parse_sheet_month(sheet_name):
    m = re.search(r'(\d{4})年(\d{1,2})月', sheet_name)
    if m:
        year, month = int(m.group(1)), int(m.group(2))
    else:
        m = re.search(r'(\d{4})(\d{2})月', sheet_name)
        if m:
            year, month = int(m.group(1)), int(m.group(2))
        else:
            return None, None
    return f'{year:04d}{month:02d}', f'{year:04d}-{month:02d}'


def load_safety_stock(path):
    """读取安全库存 Excel，返回 (df, meta)。path 可为文件路径或 file-like 对象。"""
    xls = pd.ExcelFile(path)

    sheet_name = None
    for name in xls.sheet_names:
        if '安全库存' in str(name):
            sheet_name = name
            break
    if sheet_name is None:
        sheet_name = xls.sheet_names[0]

    df = xls.parse(sheet_name=sheet_name, header=0)
    df.columns = [_clean_header(c) for c in df.columns]

    mat_col = '物料编码' if '物料编码' in df.columns else _find_col(df.columns, '物料编码')
    sap_col = 'SAP编码' if 'SAP编码' in df.columns else _find_col(df.columns, 'SAP')
    actual_stock_col = _find_col(df.columns, '月末实际库存')
    safety_stock_col = _find_col(df.columns, '安全库存')

    month_str, month_dash = _parse_sheet_month(str(sheet_name))

    # snapshot_date：从实际库存列名提取月份，如 "6月末实际库存" -> "2025-06-30"
    snapshot_date = None
    if actual_stock_col and month_dash:
        m = re.search(r'(\d{1,2})月末', actual_stock_col)
        if m:
            month = int(m.group(1))
            year = int(month_dash[:4])
            last_day = calendar.monthrange(year, month)[1]
            snapshot_date = f'{year:04d}-{month:02d}-{last_day:02d}'

    # 月份用量列（未来3个月 + 过去6个月）与月均列转数值，保留 NaN
    month_cols = _find_month_columns(df.columns)
    avg_cols = [c for c in df.columns if '月均' in c]
    numeric_cols = list(dict.fromkeys(month_cols + avg_cols))
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # SAP 编码：to_numeric + Int64
    if sap_col:
        sap_raw = (
            df[sap_col]
            .astype(str)
            .str.strip()
            .str.replace(',', '', regex=False)
            .replace({'#N/A': '', 'N/A': '', 'nan': '', 'None': '', 'NaN': '', '<NA>': ''})
        )
        df[sap_col] = pd.to_numeric(sap_raw, errors='coerce').astype('Int64')
        df['SAP缺失'] = df[sap_col].isna()

        def _join_key(row):
            if pd.notna(row[sap_col]):
                return f'SAP:{int(row[sap_col])}'
            return f'MAT:{_to_str(row[mat_col] if mat_col else "")}'

        df['join_key'] = df.apply(_join_key, axis=1)
    else:
        df['SAP缺失'] = True
        if mat_col:
            df['join_key'] = 'MAT:' + df[mat_col].map(_to_str)
        else:
            df['join_key'] = 'MAT:' + df.index.astype(str)

    meta = {
        'sheet_name': sheet_name,
        'month_str': month_str,
        'month_dash': month_dash,
        'actual_stock_col': actual_stock_col,
        'safety_stock_col': safety_stock_col,
        'snapshot_date': snapshot_date,
        'mat_col': mat_col,
        'sap_col': sap_col,
        'month_cols': month_cols,
        'avg_cols': avg_cols,
    }
    return df, meta