# app.py —— 安全库存审计与计算主入口
# 整合 读取清洗(loader) → 审计(audit) → 计算(calc) → KPI/图表/明细 → 导出(exporter)。
# 本文件含页面视觉美化 CSS（浅灰蓝背景、卡片、图表容器、表格样式、按钮等）。

import io
import os
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go

from core.loader import load_safety_stock
from core.audit import audit_data
from core.calc import calculate_safety_stock
from core.exporter import export_to_excel


COLOR_MAP = {'安全': '#2E86DE', '预警': '#F5A623', '严重': '#E74C3C'}


def fmt_num(n):
    """数字格式化：亿 / 万 / 原样（千分位）。"""
    n = float(n) if n is not None else 0.0
    if n >= 1e8:
        return f'{n / 1e8:.2f} 亿'
    if n >= 1e4:
        return f'{n / 1e4:.2f} 万'
    return f'{n:,.0f}'


def kpi_card(label, value, color):
    """自定义 HTML 指标卡片（样式由 CSS 的 .kpi-card 控制）。"""
    st.markdown(
        f"""<div class="kpi-card" style="--accent:{color};">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def _detect_biz_cols(df):
    """自动检测业务列（名称/规格/单位/供应商），找不到返回 None。"""
    biz = {'name': None, 'spec': None, 'unit': None, 'supplier': None}
    for c in df.columns:
        if biz['name'] is None and ('名称' in c or '品名' in c):
            biz['name'] = c
        if biz['spec'] is None and ('规格' in c or '型号' in c):
            biz['spec'] = c
        if biz['unit'] is None and ('单位' in c and '用量' not in c):
            biz['unit'] = c
        if biz['supplier'] is None and ('供应商' in c or '厂商' in c):
            biz['supplier'] = c
    return biz


def _first_match(columns, keywords, exclude=None):
    """按关键词（AND 关系）查找第一个未使用的匹配列。"""
    exclude = exclude or set()
    for c in columns:
        if c in exclude:
            continue
        if all(k in c for k in keywords):
            return c
    return None


AUX_KEY = '物料编码'


def _pick_col(df, names):
    for n in names:
        if n in df.columns:
            return n
    for c in df.columns:
        s = str(c).replace('\n', '').replace(' ', '')
        for n in names:
            if n.replace('\n', '').replace(' ', '') in s:
                return c
    return None


@st.cache_data(show_spinner=False)
def load_aux_usage(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_excel(io.BytesIO(file_bytes))
    df.columns = [str(c).strip() for c in df.columns]
    rename_map = {
        '物料': '物料编码', '移动类型': '移动类型', '过账日期': '过账日期',
        '以录入单位表示的数量': '数量', '录入单位': '录入单位',
        '物料凭证': '物料凭证', '工厂': '工厂', '存储地点': '存储地点',
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    if '物料编码' not in df.columns:
        raise ValueError('使用量表缺少「物料」列')
    if '数量' not in df.columns:
        raise ValueError('使用量表缺少「以录入单位表示的数量」列')
    if '移动类型' in df.columns:
        df = df[df['移动类型'].astype(str).str.strip() == '261'].copy()
    df['数量'] = pd.to_numeric(df['数量'], errors='coerce').abs()
    if '过账日期' in df.columns:
        df['过账日期'] = pd.to_datetime(df['过账日期'], errors='coerce')
        df['日期'] = df['过账日期'].dt.date
    else:
        df['日期'] = pd.NaT
    df['物料编码'] = df['物料编码'].astype(str).str.strip()
    g = df.groupby('物料编码', dropna=False)
    out = pd.DataFrame(index=g.size().index)
    out['aux_消耗总量'] = g['数量'].sum(min_count=1)
    if '过账日期' in df.columns and df['过账日期'].notna().any():
        df['年月'] = df['过账日期'].dt.to_period('M')
        month_cnt = df.groupby('物料编码')['年月'].nunique(dropna=True)
        out['aux_月均用量'] = out['aux_消耗总量'] / month_cnt.replace(0, np.nan)
        out['aux_月数'] = month_cnt
    else:
        out['aux_月均用量'] = out['aux_消耗总量']
        out['aux_月数'] = 1
    out['aux_月消耗量'] = out['aux_消耗总量']
    if '过账日期' in df.columns and df['过账日期'].notna().any():
        date_range = df.groupby('物料编码')['过账日期'].agg(lambda s: (s.max() - s.min()).days + 1)
        out['aux_日均用量'] = out['aux_消耗总量'] / date_range.replace(0, np.nan)
    else:
        out['aux_日均用量'] = out['aux_消耗总量'] / 30.0
    out['aux_使用天数'] = g['日期'].nunique(dropna=True)
    out['aux_最近消耗日期'] = g['过账日期'].max() if '过账日期' in df.columns else pd.NaT
    out['aux_消耗凭证数'] = g['物料凭证'].nunique(dropna=False) if '物料凭证' in df.columns else g.size()
    if '日期' in df.columns and df['日期'].notna().any():
        daily = df.groupby(['物料编码', '日期'], dropna=False)['数量'].sum().reset_index(name='日消耗')
        out['aux_单日最大消耗'] = daily.groupby('物料编码')['日消耗'].max()
        out['aux_消耗波动率'] = daily.groupby('物料编码')['日消耗'].agg(lambda s: s.std() / s.mean() if s.mean() else np.nan)
    else:
        out['aux_单日最大消耗'] = np.nan
        out['aux_消耗波动率'] = np.nan
    if '录入单位' in df.columns:
        out['aux_使用量单位'] = g['录入单位'].agg(lambda s: ','.join(sorted(set(s.dropna().astype(str)))))
    return out.reset_index()


@st.cache_data(show_spinner=False)
def load_aux_forecast(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_excel(io.BytesIO(file_bytes))
    df.columns = [str(c).strip() for c in df.columns]
    rename_map = {'物料代码': '物料编码', '物料理论数量（该订单）': '预测数量', '跟踪号': '跟踪号'}
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    if '物料编码' not in df.columns:
        raise ValueError('预测表缺少「物料代码」列')
    if '预测数量' not in df.columns:
        raise ValueError('预测表缺少「物料理论数量（该订单）」列')
    df['物料编码'] = df['物料编码'].astype(str).str.strip()
    df['预测数量'] = pd.to_numeric(df['预测数量'], errors='coerce')
    g = df.groupby('物料编码', dropna=False)
    out = pd.DataFrame(index=g.size().index)
    out['aux_预测需求总量'] = g['预测数量'].sum(min_count=1)
    out['aux_预测订单数'] = g['跟踪号'].nunique(dropna=False) if '跟踪号' in df.columns else g.size()
    out['aux_预测平均单订单量'] = g['预测数量'].mean()
    out['aux_预测最大单订单量'] = g['预测数量'].max()
    total = out['aux_预测需求总量'].replace(0, np.nan)
    out['aux_预测需求集中度'] = out['aux_预测最大单订单量'] / total
    return out.reset_index()


@st.cache_data(show_spinner=False)
def load_aux_inventory(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_excel(io.BytesIO(file_bytes))
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={'物料': '物料编码'})
    if '物料编码' not in df.columns:
        raise ValueError('库存表缺少「物料」列')
    df['物料编码'] = df['物料编码'].astype(str).str.strip()
    num_cols = ['非限制使用的库存', '质量检验', '已冻结的库存', '限制使用的库存', '退货', 'OCS可用库存']
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    if '到期日期' in df.columns:
        df['到期日期'] = pd.to_datetime(df['到期日期'], errors='coerce')
    g = df.groupby('物料编码', dropna=False)
    out = pd.DataFrame(index=g.size().index)
    for c in num_cols:
        out[f'aux_{c}'] = g[c].sum(min_count=1) if c in df.columns else np.nan
    base_cols = [c for c in ['非限制使用的库存', '质量检验', '已冻结的库存', '限制使用的库存', '退货'] if c in df.columns]
    out['aux_总库存'] = g[base_cols].sum().sum(axis=1) if base_cols else np.nan
    out['aux_库存批次数'] = g['批次'].nunique(dropna=False) if '批次' in df.columns else np.nan
    out['aux_库存地点数'] = g['存储地点'].nunique(dropna=False) if '存储地点' in df.columns else np.nan
    if '到期日期' in df.columns:
        out['aux_最早到期日期'] = g['到期日期'].min()
    return out.reset_index()


def attach_aux_tables(df, meta, usage_bytes=None, forecast_bytes=None, inventory_bytes=None,
                      override_usage=False, override_forecast=False, join_key=None):
    mat_col = meta.get('mat_col') or AUX_KEY
    join_col = join_key if (join_key and join_key in df.columns) else mat_col
    if join_col not in df.columns:
        return df
    aux_list = []
    if usage_bytes:
        aux_list.append(load_aux_usage(usage_bytes))
    if forecast_bytes:
        aux_list.append(load_aux_forecast(forecast_bytes))
    if inventory_bytes:
        aux_list.append(load_aux_inventory(inventory_bytes))
    for aux in aux_list:
        if aux is None or aux.empty:
            continue
        aux = aux.copy()
        aux['_aux_key'] = aux['物料编码'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
        aux = aux.drop(columns=['物料编码'], errors='ignore')
        df['_aux_key'] = df[join_col].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
        df = df.merge(aux, on='_aux_key', how='left')
        df = df.drop(columns=['_aux_key'], errors='ignore')
    if override_usage and 'aux_月均用量' in df.columns:
        usage_col = _pick_col(df, ['月均量(半年）', '月均量（半年）', '月均量(半年)', '月均量（半年)', '月均量\n(半年）'])
        if usage_col:
            df[usage_col] = df['aux_月均用量'].fillna(df[usage_col])
    if override_forecast and 'aux_预测需求总量' in df.columns:
        forecast_col = _pick_col(df, ['未来3个月月均用量', '未来3个月月均'])
        if forecast_col:
            df[forecast_col] = (df['aux_预测需求总量'] / 3.0).fillna(df[forecast_col])
    return df


def render_realtime_clock():
    components.html(
        """
        <div id="rt-clock" style="font-family:'Microsoft YaHei',sans-serif;background:#ffffff;border:1px solid #e3e8f0;border-left:4px solid #2E86DE;border-radius:12px;padding:10px 14px;box-shadow:0 2px 8px rgba(31,45,61,0.06);color:#1f2d3d;">
            <div style="font-size:12px;color:#6b7a90;margin-bottom:4px;">当前时间</div>
            <div id="rt-date" style="font-size:13px;color:#1f2d3d;margin-bottom:2px;"></div>
            <div id="rt-time" style="font-size:20px;font-weight:600;color:#2E86DE;letter-spacing:1px;"></div>
        </div>
        <script>
        (function(){
            const weekdays=['星期日','星期一','星期二','星期三','星期四','星期五','星期六'];
            function pad(n){return n<10?'0'+n:''+n;}
            function tick(){
                const now=new Date();
                document.getElementById('rt-date').textContent=now.getFullYear()+'年'+(now.getMonth()+1)+'月'+now.getDate()+'日 '+weekdays[now.getDay()];
                document.getElementById('rt-time').textContent=pad(now.getHours())+':'+pad(now.getMinutes())+':'+pad(now.getSeconds());
            }
            tick();setInterval(tick,1000);
        })();
        </script>
        """,
        height=95,
    )


@st.cache_data(show_spinner='正在读取与计算…')
def _load_and_calc(file_bytes, usage_bytes=None, forecast_bytes=None, inventory_bytes=None,
                   override_usage=False, override_forecast=False, join_key=None):
    import tempfile, os
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
    tmp.write(file_bytes)
    tmp.close()
    df, meta = load_safety_stock(tmp.name)
    audit_data(df, meta)

    df_orig = df.copy()
    df_orig = calculate_safety_stock(df_orig)
    orig_safety = float(df_orig['安全库存'].sum())
    orig_gap = float(df_orig['缺口'].sum())

    df = attach_aux_tables(df, meta, usage_bytes, forecast_bytes, inventory_bytes,
                           override_usage, override_forecast, join_key)
    df = calculate_safety_stock(df)
    actual_col = meta.get('actual_stock_col')
    if actual_col:
        df[actual_col] = pd.to_numeric(df[actual_col], errors='coerce')

    new_safety = float(df['安全库存'].sum())
    new_gap = float(df['缺口'].sum())
    compare = {'orig_safety': orig_safety, 'orig_gap': orig_gap,
               'new_safety': new_safety, 'new_gap': new_gap}
    return df, meta, tmp.name, compare


st.set_page_config(page_title='安全库存审计与计算', layout='wide')

# ===== 自定义 CSS：页面视觉美化 =====
st.markdown(
    """
    <style>
    /* 全局字体 */
    html, body, .stApp {
        font-family: "Microsoft YaHei", "PingFang SC", "Helvetica Neue", Arial, sans-serif;
        color: #1f2d3d;
    }

    /* 主内容区：浅灰蓝背景，顶部留白 1.2rem */
    [data-testid="stAppViewContainer"] {
        background: #e9edf5;
    }
    [data-testid="stAppViewContainer"] .block-container {
        padding-top: 1.2rem !important;
    }

    /* 顶部 header 透明 */
    [data-testid="stHeader"] {
        background: transparent;
    }

    /* 侧边栏：稍深背景 + 白底输入控件 */
    [data-testid="stSidebar"] {
        background: #eef1f7;
        border-right: 1px solid #e3e8f0;
    }
    [data-testid="stSidebar"] [data-baseweb="input"],
    [data-testid="stSidebar"] [data-baseweb="select"],
    [data-testid="stSidebar"] [data-baseweb="textarea"] {
        background-color: #ffffff;
    }

    /* 上传区域：1.5px 虚线边框 + 圆角 + hover 变蓝 */
    [data-testid="stFileUploader"] {
        border: 1.5px dashed #2E86DE;
        border-radius: 12px;
        background: #ffffff;
        padding: 8px;
        transition: border-color 0.2s ease;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #4A6CF7;
    }

    /* KPI 卡片：等高、白底、圆角、彩条、阴影、hover 上浮 */
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e3e8f0;
        border-left: 4px solid var(--accent, #2E86DE);
        border-radius: 12px;
        padding: 12px 14px;
        height: 100%;
        min-height: 100px;
        box-shadow: 0 2px 8px rgba(31,45,61,0.06);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(31,45,61,0.12);
    }
    .kpi-label {
        font-size: 13px;
        color: #6b7a90;
        margin-bottom: 8px;
    }
    .kpi-value {
        font-size: 20px;
        font-weight: 600;
        color: #1f2d3d;
        word-break: break-all;
        line-height: 1.25;
    }

    /* 标题左侧蓝色竖条 */
    [data-testid="stMarkdownContainer"] h1::before,
    [data-testid="stMarkdownContainer"] h2::before,
    [data-testid="stMarkdownContainer"] h3::before {
        content: "";
        display: inline-block;
        width: 4px;
        height: 18px;
        background: #2E86DE;
        border-radius: 2px;
        margin-right: 8px;
        vertical-align: middle;
    }

    /* 图表容器卡片：圆角裁剪，避免 SVG 直角交叠 */
    [data-testid="stPlotlyChart"] {
        background: #ffffff;
        border: 1px solid #e3e8f0;
        border-radius: 12px;
        padding: 12px;
        box-shadow: 0 2px 8px rgba(31,45,61,0.06);
        overflow: hidden;
        isolation: isolate;
        will-change: transform;
        transform: translateZ(0);
    }
    [data-testid="stPlotlyChart"] > div,
    [data-testid="stPlotlyChart"] .js-plotly-plot,
    [data-testid="stPlotlyChart"] .plot-container,
    [data-testid="stPlotlyChart"] .svg-container {
        border-radius: 12px !important;
        overflow: hidden !important;
        background: transparent !important;
    }

    /* 明细表外层容器 + 深色表头白字（单元格背景交给 Styler） */
    .stDataFrame {
        border: 1px solid #e3e8f0;
        border-radius: 12px;
        overflow: hidden;
        isolation: isolate;
        box-shadow: 0 2px 8px rgba(31,45,61,0.06);
    }
    .stDataFrame > div,
    .stDataFrame [data-testid="stDataFrameResizable"] {
        border-radius: 12px;
        overflow: hidden;
    }
    .stDataFrame thead tr th {
        background-color: #1f2d3d !important;
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    /* 下载按钮：渐变 + hover 上浮 */
    [data-testid="stDownloadButton"] button {
        background: linear-gradient(90deg, #2E86DE, #4A6CF7) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    [data-testid="stDownloadButton"] button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(46,134,222,0.35);
    }

    /* caption 文字加深 */
    [data-testid="stCaptionContainer"], [data-testid="stCaption"] {
        color: #6b7a90 !important;
        font-size: 13px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ===== 列菜单中文化（JavaScript 注入） =====
st.components.v1.html(
    """
    <script>
    (function() {
        const dict = {
            'Sort ascending':  '升序排列',
            'Sort descending': '降序排列',
            'Statistics':      '统计信息',
            'Autosize':        '自动调整列宽',
            'Autosize all columns': '自动调整所有列宽',
            'Pin column':      '固定此列',
            'Unpin column':    '取消固定',
            'Hide column':     '隐藏此列',
            'Format':          '格式化',
            'Search':          '搜索',
            'Reset':           '重置',
        };
        function translate(root) {
            const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, null);
            let node;
            while ((node = walker.nextNode())) {
                const t = node.nodeValue.trim();
                if (dict[t]) node.nodeValue = node.nodeValue.replace(t, dict[t]);
            }
        }
        const obs = new MutationObserver(muts => {
            muts.forEach(m => {
                m.addedNodes.forEach(n => {
                    if (n.nodeType === 1) setTimeout(() => translate(n), 30);
                });
            });
        });
        obs.observe(document.body, {childList: true, subtree: true});
    })();
    </script>
    """,
    height=0,
)

st.title('安全库存审计与计算')

# 侧边栏：上传 Excel
with st.sidebar:
    render_realtime_clock()
uploaded = st.sidebar.file_uploader('上传 Excel 文件', type=['xlsx', 'xls'])

with st.sidebar.expander('可选辅助表（不传不影响原功能）', expanded=False):
    usage_file = st.file_uploader('使用量表 Excel', type=['xlsx', 'xls'], key='aux_usage')
    forecast_file = st.file_uploader('预测表 Excel', type=['xlsx', 'xls'], key='aux_forecast')
    inventory_file = st.file_uploader('库存表 Excel', type=['xlsx', 'xls'], key='aux_inventory')
    st.caption(
        f"使用量：{'已上传' if usage_file else '未上传'} | "
        f"预测：{'已上传' if forecast_file else '未上传'} | "
        f"库存：{'已上传' if inventory_file else '未上传'}"
    )
    st.markdown('---')
    mode = st.radio('辅助表使用模式', ['增强模式（推荐）', '覆盖模式'], index=0,
                    help='增强模式：辅助表只做参考，不影响安全库存计算；覆盖模式：用辅助表数据改写主表用量字段。')
    if mode == '增强模式（推荐）':
        st.caption('**增强模式（推荐）**\n\n辅助表只作为额外参考列展示，不改动主表原有口径。安全库存、缺口、覆盖率、状态、图表结果完全不变。\n\n适合：第一次使用、或不确定辅助表数据质量时。')
        override_usage = False
        override_forecast = False
    else:
        st.caption('**覆盖模式**\n\n会用辅助表口径改写主表用量字段，进而影响安全库存、缺口、覆盖率、KPI、图表和导出结果。\n\n- 使用量表 → 覆盖「月均量(半年）」\n- 预测表 → 覆盖「未来3个月月均用量」\n\n适合：辅助表数据完整、单位与时间口径对齐时。')
        override_usage = st.checkbox('用使用量表覆盖「月均量(半年）」', value=False)
        override_forecast = st.checkbox('用预测表覆盖「未来3个月月均用量」', value=False)

    join_key = st.selectbox('辅助表关联主键', ['SAP编码', '物料编码'], index=0,
                            help='选择主表中与辅助表对应的编码列进行关联。')

if uploaded is None:
    st.info('请在左侧上传 data/raw/安全库存.xlsx 开始审计与计算。')
    st.stop()

try:
    df, meta, src_path, compare = _load_and_calc(
        uploaded.getvalue(),
        usage_file.getvalue() if usage_file else None,
        forecast_file.getvalue() if forecast_file else None,
        inventory_file.getvalue() if inventory_file else None,
        override_usage,
        override_forecast,
        join_key,
    )
except Exception as e:
    st.error(f'处理失败：{e}')
    st.stop()

if any([usage_file, forecast_file, inventory_file]):
    if mode == '增强模式（推荐）':
        st.info('当前为 **增强模式**：辅助表仅作参考展示，不参与安全库存计算，下方所有 KPI、图表、明细表仍基于主表原始口径。')
    else:
        st.warning('当前为 **覆盖模式**：已用辅助表口径改写主表用量字段，下方 KPI、图表、明细表均为覆盖后的结果。如需对比，请切换回增强模式。')

if mode == '覆盖模式' and any([override_usage, override_forecast]):
    with st.expander('覆盖前后对比', expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            kpi_card('原始 总安全库存', fmt_num(compare['orig_safety']), '#4A6CF7')
            kpi_card('原始 总缺口', fmt_num(compare['orig_gap']), '#4A6CF7')
        with c2:
            kpi_card('覆盖后 总安全库存', fmt_num(compare['new_safety']), '#2E86DE')
            kpi_card('覆盖后 总缺口', fmt_num(compare['new_gap']), '#E74C3C')

actual_col = meta.get('actual_stock_col')
mat_col = meta.get('mat_col') or '物料编码'

if actual_col is None:
    st.error('未能识别「月末实际库存」列，请检查表头。')
    st.stop()

# 侧边栏：选择主安全库存口径
basis_options = ['O_原始']
if '安全库存_修正版' in df.columns:
    basis_options.append('O_修正')
basis = st.sidebar.selectbox('选择主安全库存口径', basis_options)
st.sidebar.caption('O_原始：按源表原始系数直接计算；O_修正：按修正后的系数重新计算。')

if basis == 'O_原始':
    safety_col = '安全库存'
    gap_col = '缺口'
    cov_col = '覆盖率'
    status_col = '状态'
else:
    safety_col = '安全库存_修正版'
    gap_col = '缺口_修正'
    cov_col = '覆盖率_修正'
    status_col = '状态_修正'

# ===== 关键指标 =====
total = int(len(df))
low = int(df[status_col].isin(['严重', '预警']).sum())
severe = int((df[status_col] == '严重').sum())
total_safety = float(df[safety_col].sum())
total_actual = float(df[actual_col].sum())
total_gap = float(df[gap_col].sum())

overall_coverage = total_actual / total_safety if total_safety else float('nan')
if pd.isna(overall_coverage):
    cov_display = '—'
    cov_color = '#4A6CF7'
else:
    cov_display = f'{overall_coverage:.2%}'
    if overall_coverage > 1.0:
        cov_color = '#27AE60'
    elif overall_coverage >= 0.7:
        cov_color = '#F5A623'
    else:
        cov_color = '#E74C3C'

zero_stock = int((df[actual_col] <= 0).sum())
zero_color = '#E74C3C' if zero_stock > 0 else '#27AE60'

gap_sum = float(df[gap_col].sum())
if gap_sum > 0:
    concentration = float(df[gap_col].nlargest(10).sum()) / gap_sum
    conc_display = f'{concentration:.1%}'
else:
    conc_display = '—'

# 第二行补齐指标
avg_coverage = float(df[cov_col].mean())
avg_cov_display = f'{avg_coverage:.1%}' if pd.notna(avg_coverage) else '—'

if gap_sum > 0:
    top1_display = f'{float(df[gap_col].max()) / gap_sum:.1%}'
else:
    top1_display = '—'

net_diff = total_safety - total_actual
if net_diff > 0:
    net_display = '+' + fmt_num(net_diff)
    net_color = '#E74C3C'
elif net_diff < 0:
    net_display = '-' + fmt_num(abs(net_diff))
    net_color = '#27AE60'
else:
    net_display = '0'
    net_color = '#27AE60'

st.subheader('关键指标')
k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1:
    kpi_card('物料总数', f'{total:,}', '#5B8DB8')
with k2:
    kpi_card('低库存物料数', f'{low:,}', '#F5A623')
with k3:
    kpi_card('严重物料数', f'{severe:,}', '#E74C3C')
with k4:
    kpi_card('总安全库存', fmt_num(total_safety), '#2E86DE')
with k5:
    kpi_card('总实际库存', fmt_num(total_actual), '#17A2B8')
with k6:
    kpi_card('总缺口', fmt_num(total_gap), '#E74C3C')

r1, r2, r3, r4, r5, r6 = st.columns(6)
with r1:
    kpi_card('整体覆盖率', cov_display, cov_color)
with r2:
    kpi_card('零库存物料数', f'{zero_stock:,}', zero_color)
with r3:
    kpi_card('缺口集中度', conc_display, '#4A6CF7')
with r4:
    kpi_card('平均覆盖率', avg_cov_display, '#4A6CF7')
with r5:
    kpi_card('Top1 缺口占比', top1_display, '#B23A48')
with r6:
    kpi_card('净差值', net_display, net_color)

# 辅助表增强指标
aux_cols = [c for c in df.columns if c.startswith('aux_')]
if aux_cols:
    with st.expander('辅助表增强指标（可选上传后出现）', expanded=False):
        search_kw = st.text_input('按物料编码搜索', '', key='aux_search')
        show_df = df[[c for c in [mat_col] + aux_cols if c in df.columns]].copy()
        if search_kw:
            show_df = show_df[show_df[mat_col].astype(str).str.contains(search_kw, case=False, na=False)]
        st.dataframe(show_df, use_container_width=True, height=320)

        total_mats = len(df)
        if usage_file:
            matched_u = df['aux_消耗总量'].notna().sum() if 'aux_消耗总量' in df.columns else 0
            st.caption(f'使用量表匹配率：{matched_u}/{total_mats} 物料，未匹配 {total_mats - matched_u}')
        if forecast_file:
            matched_f = df['aux_预测需求总量'].notna().sum() if 'aux_预测需求总量' in df.columns else 0
            st.caption(f'预测表匹配率：{matched_f}/{total_mats} 物料，未匹配 {total_mats - matched_f}')
        if inventory_file:
            matched_i = df['aux_总库存'].notna().sum() if 'aux_总库存' in df.columns else 0
            st.caption(f'库存表匹配率：{matched_i}/{total_mats} 物料，未匹配 {total_mats - matched_i}')

        if usage_file and 'aux_最近消耗日期' in df.columns:
            max_date = df['aux_最近消耗日期'].max()
            st.caption(f'使用量表最近过账日期：{max_date}')
        if inventory_file and 'aux_最早到期日期' in df.columns:
            min_date = df['aux_最早到期日期'].min()
            st.caption(f'库存表最早到期日期：{min_date}')

        if mode == '覆盖模式' and override_usage and 'aux_使用量单位' in df.columns:
            units = df['aux_使用量单位'].dropna().unique()
            bad_units = [u for u in units if '公斤' not in str(u)]
            if bad_units:
                st.warning(f'检测到使用量表单位包含非“公斤”：{", ".join(bad_units)}，覆盖模式下可能造成数量级不一致，请谨慎使用。')

# 辅助表未匹配物料
if any([usage_file, forecast_file, inventory_file]):
    with st.expander('辅助表未匹配物料', expanded=False):
        main_ids = set(df[mat_col].astype(str).str.strip())
        if usage_file:
            aux_u_ids = set(load_aux_usage(usage_file.getvalue())['物料编码'].astype(str).str.strip())
            unmatched_u = sorted(aux_u_ids - main_ids)
            st.write(f'使用量表未匹配物料数：{len(unmatched_u)}')
            if unmatched_u:
                st.dataframe(pd.DataFrame({mat_col: unmatched_u}), use_container_width=True)
        if forecast_file:
            aux_f_ids = set(load_aux_forecast(forecast_file.getvalue())['物料编码'].astype(str).str.strip())
            unmatched_f = sorted(aux_f_ids - main_ids)
            st.write(f'预测表未匹配物料数：{len(unmatched_f)}')
            if unmatched_f:
                st.dataframe(pd.DataFrame({mat_col: unmatched_f}), use_container_width=True)
        if inventory_file:
            aux_i_ids = set(load_aux_inventory(inventory_file.getvalue())['物料编码'].astype(str).str.strip())
            unmatched_i = sorted(aux_i_ids - main_ids)
            st.write(f'库存表未匹配物料数：{len(unmatched_i)}')
            if unmatched_i:
                st.dataframe(pd.DataFrame({mat_col: unmatched_i}), use_container_width=True)

# ===== 可视化 =====
st.subheader('可视化')

sort_key = st.radio(
    '安全库存 vs 实际库存图 — 排序方式',
    ['按缺口降序', '按覆盖率升序', '按安全库存降序'],
    horizontal=True,
    index=0,
    label_visibility='collapsed',
)

c1, c2 = st.columns(2)

with c1:
    if sort_key == '按缺口降序':
        vs = df.nlargest(20, gap_col)
    elif sort_key == '按覆盖率升序':
        vs = df[df[safety_col] > 0].nsmallest(20, cov_col)
    else:
        vs = df.nlargest(20, safety_col)

    vs = vs[[mat_col, safety_col, actual_col]].copy()
    vs = vs.rename(columns={safety_col: '安全库存', actual_col: '实际库存'})
    vs = vs.dropna(subset=['安全库存', '实际库存'])
    vs['显示名'] = vs[mat_col].astype(str).str[-8:]
    vs = vs.melt(id_vars=[mat_col, '显示名'], value_vars=['安全库存', '实际库存'],
                 var_name='指标', value_name='数量')
    fig_vs = px.bar(vs, x='显示名', y='数量', color='指标', barmode='group',
                    hover_data={mat_col: True, '显示名': False},
                    title='安全库存 vs 实际库存（Top 20）')
    fig_vs.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        legend_title_text='',
        margin=dict(l=40, r=40, t=60, b=120),
        height=500
    )
    fig_vs.update_xaxes(tickangle=-30)
    st.plotly_chart(fig_vs, use_container_width=True, config={'locale': 'zh-CN'})

with c2:
    status_counts = df[status_col].value_counts().reset_index()
    status_counts.columns = ['状态', '数量']
    status_counts = status_counts.set_index('状态').reindex(['安全', '预警', '严重']).dropna().reset_index()
    fig_pie = px.pie(status_counts, names='状态', values='数量', title='状态分布')
    fig_pie.update_traces(marker=dict(colors=[COLOR_MAP.get(s, '#999999') for s in status_counts['状态']]))
    fig_pie.update_layout(
        legend=dict(orientation="h", yanchor="top", y=-0.05, xanchor="center", x=0.5),
        legend_title_text='',
        margin=dict(l=40, r=40, t=60, b=80)
    )
    st.plotly_chart(fig_pie, use_container_width=True, config={'locale': 'zh-CN'})

# Top 20 缺口物料（按覆盖率着色）
gap_top = df[[mat_col, gap_col, cov_col]].dropna(subset=[gap_col]).nlargest(20, gap_col)
gap_top = gap_top.sort_values(gap_col, ascending=True)
gap_top['显示名'] = gap_top[mat_col].astype(str).str[-8:]
gap_top['覆盖率截断'] = gap_top[cov_col].clip(0, 1)
fig_gap = px.bar(
    gap_top, x=gap_col, y='显示名', orientation='h',
    color='覆盖率截断',
    color_continuous_scale=['#E74C3C', '#F5A623', '#F1C40F', '#27AE60'],
    hover_data={mat_col: True, '显示名': False, cov_col: ':.2%'},
    title='Top 20 缺口物料（按覆盖率着色）'
)
fig_gap.update_layout(
    coloraxis_showscale=False,
    margin=dict(l=40, r=40, t=60, b=60),
    height=560,
)
st.plotly_chart(fig_gap, use_container_width=True, config={'locale': 'zh-CN'})
st.caption('颜色含义：红色=覆盖率接近 0（极度缺货），橙色=接近预警，绿色=接近安全')

# 帕累托分析（X 轴范围压缩）
gap_sorted = df[df[gap_col] > 0][[mat_col, gap_col]].dropna().sort_values(gap_col, ascending=False).reset_index(drop=True)
if not gap_sorted.empty:
    gap_sorted['序号'] = gap_sorted.index + 1
    total_gap_val = gap_sorted[gap_col].sum()
    gap_sorted['累积占比'] = gap_sorted[gap_col].cumsum() / total_gap_val
    n_total = len(gap_sorted)
    n_80 = int((gap_sorted['累积占比'] < 0.8).sum()) + 1
    # 只保留累积占比首次 >= 0.95 之前的物料
    cutoff = int((gap_sorted['累积占比'] < 0.95).sum()) + 1
    gap_sorted = gap_sorted.iloc[:cutoff].copy()

    fig_pareto = go.Figure()
    fig_pareto.add_trace(go.Bar(
        x=gap_sorted['序号'], y=gap_sorted[gap_col],
        name='缺口', marker_color='#2E86DE',
        hovertemplate='序号 %{x}<br>缺口 %{y:,.2f}<extra></extra>',
    ))
    fig_pareto.add_trace(go.Scatter(
        x=gap_sorted['序号'], y=gap_sorted['累积占比'],
        name='累积占比', yaxis='y2',
        line=dict(color='#E74C3C', width=2),
        hovertemplate='序号 %{x}<br>累积 %{y:.1%}<extra></extra>',
    ))
    fig_pareto.add_hline(y=0.8, line=dict(color='#999999', dash='dash'),
                         yref='y2', annotation_text='80%', annotation_position='right')
    fig_pareto.update_layout(
        title='缺口帕累托分析（Top N 贡献 80% 缺口）',
        xaxis=dict(title='物料序号（按缺口降序）'),
        yaxis=dict(title='缺口'),
        yaxis2=dict(title='累积占比', overlaying='y', side='right', tickformat='.0%', range=[0, 1.05]),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=40, r=60, t=60, b=60),
        height=480,
    )
    fig_pareto.update_xaxes(dtick=5 if len(gap_sorted) > 20 else 2)
    st.plotly_chart(fig_pareto, use_container_width=True, config={'locale': 'zh-CN'})
    st.caption(f'共 {n_total} 个有缺口物料，前 {n_80} 个物料覆盖了 80% 的总缺口。图表仅展示累积达 95% 的前 {len(gap_sorted)} 个物料。')

# ===== 明细表 =====
st.subheader('安全库存明细表')

biz = _detect_biz_cols(df)

biz_cols = []
if biz.get('name'):
    biz_cols.append((biz['name'], '物料名称'))
if biz.get('spec'):
    biz_cols.append((biz['spec'], '规格'))
if biz.get('unit'):
    biz_cols.append((biz['unit'], '单位'))
if biz.get('supplier'):
    biz_cols.append((biz['supplier'], '供应商'))

detail_src = [mat_col] + [c for c, _ in biz_cols] + [safety_col, actual_col, gap_col, cov_col, status_col]
detail_names = ['物料编码'] + [d for _, d in biz_cols] + ['安全库存', '实际库存', '缺口', '覆盖率', '状态']

detail = df[detail_src].copy()
detail.columns = detail_names
detail['状态'] = detail['状态'].astype(str).str.strip()

fc1, fc2, fc3 = st.columns([3, 1, 1])
with fc1:
    kw = st.text_input('按物料编码搜索', '')
with fc2:
    status_filter = st.multiselect('状态筛选', ['安全', '预警', '严重'], default=['安全', '预警', '严重'])
with fc3:
    sort_opt = st.selectbox('排序', ['缺口降序', '覆盖率升序', '安全库存降序'], index=0)

view = detail.copy()
if kw:
    view = view[view['物料编码'].astype(str).str.contains(kw, case=False, na=False)]
if status_filter:
    view = view[view['状态'].isin(status_filter)]

if sort_opt == '缺口降序':
    view = view.sort_values('缺口', ascending=False)
elif sort_opt == '覆盖率升序':
    view = view.sort_values('覆盖率', ascending=True)
else:
    view = view.sort_values('安全库存', ascending=False)


def _row_style(row):
    s = str(row['状态']).strip()
    n = len(row)
    cols = list(row.index)
    # 先按状态色铺整行
    if s == '严重':
        styles = ['background-color: #ffe4e4'] * n
    elif s == '预警':
        styles = ['background-color: #fff7d6'] * n
    else:
        styles = ['' for _ in range(n)]
    # 再给「安全库存」列单独覆盖
    if '安全库存' in cols:
        idx = cols.index('安全库存')
        styles[idx] = 'background-color: #E3EFFC; color: #1E5FAF; font-weight: 700'
    return styles


styled = (
    view.style
    .apply(_row_style, axis=1)
    .format({
        '安全库存': '{:,.2f}',
        '实际库存': '{:,.2f}',
        '缺口': '{:,.2f}',
        '覆盖率': '{:.2f}',
    })
)
st.dataframe(styled, use_container_width=True, height=420)

# 计算过程与中间量（内联展示）
with st.expander('查看计算过程与中间量', expanded=False):
    st.caption('安全库存 = 基础月均需求 × 风险加权 × 采购周期系数；风险加权 = 质量系数 × 0.4 + 策略系数 × 0.6；基础月均需求 = (未来3个月月均用量 + 过去6个月月均用量) / 2')

    used = set()
    proc_specs = [
        ['质量'],
        ['策略'],
        ['采购周期'],
        ['未来', '月均'],
        ['过去', '月均'],
        ['风险'],
        ['基础', '月均'],
    ]
    proc_cols = []
    for kws in proc_specs:
        c = _first_match(df.columns, kws, used)
        if c:
            used.add(c)
            proc_cols.append(c)

    proc_df = df[[mat_col] + proc_cols + [safety_col, gap_col, cov_col, status_col]].copy()
    proc_df = proc_df.rename(columns={mat_col: '物料编码'})
    st.dataframe(proc_df.style.format(precision=4, na_rep='—'), use_container_width=True, height=300)

# ===== 冗余库存告警 =====
with st.expander('冗余库存告警（覆盖率 > 5）', expanded=False):
    redundant = df[df[cov_col] > 5].copy()
    if redundant.empty:
        st.success('当前没有覆盖率超过 5 的冗余物料。')
    else:
        total_redundant = len(redundant)
        redundant_top = redundant.nlargest(20, cov_col).copy()
        redundant_top['冗余量'] = redundant_top[actual_col] - redundant_top[safety_col]

        biz = _detect_biz_cols(df)
        red_cols = [mat_col]
        if biz.get('name'):
            red_cols.append(biz['name'])
        red_cols += [actual_col, safety_col, cov_col, '冗余量']

        red_out = redundant_top[red_cols].copy()
        red_rename = {mat_col: '物料编码', actual_col: '实际库存', safety_col: '安全库存', cov_col: '覆盖率', '冗余量': '冗余量'}
        if biz.get('name'):
            red_rename[biz['name']] = '物料名称'
        red_out = red_out.rename(columns=red_rename)
        red_out['覆盖率'] = red_out['覆盖率'].apply(lambda x: f'{x:.2%}')

        st.caption(f'共 {total_redundant} 项覆盖率 > 5 的物料，可能存在资金占用过高的问题。')
        st.dataframe(red_out.style.format({'实际库存': '{:,.2f}', '安全库存': '{:,.2f}', '冗余量': '{:,.2f}'}), use_container_width=True, hide_index=True)

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as w:
            red_out.to_excel(w, index=False, sheet_name='冗余清单')
        st.download_button('下载冗余清单 (Excel)', data=buf.getvalue(),
                           file_name='冗余清单.xlsx',
                           mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

# ===== 重点关注清单 =====
st.subheader('重点关注清单')
focus = df[(df[cov_col] < 0.5) & (df[gap_col] > 0)].copy()
if focus.empty:
    st.success('当前没有覆盖率低于 50% 的物料。')
else:
    focus = focus.nlargest(20, gap_col)
    biz = _detect_biz_cols(df)
    cols = [mat_col]
    for k in ['name', 'spec', 'unit', 'supplier']:
        if biz.get(k):
            cols.append(biz[k])
    cols += [safety_col, actual_col, gap_col, cov_col]
    out = focus[cols].copy()
    out['建议补货量'] = out[gap_col].apply(lambda x: int(x * 1.1) + 1)
    rename_map = {mat_col: '物料编码', safety_col: '安全库存', actual_col: '当前库存', gap_col: '缺口', cov_col: '覆盖率'}
    out = out.rename(columns=rename_map)
    out['覆盖率'] = out['覆盖率'].apply(lambda x: f'{x:.1%}')
    st.caption(f'共 {len(focus)} 项，覆盖率 < 50% 且按缺口降序取前 20。建议补货量 = 缺口 × 1.1 向上取整。')

    def _hl_safety_col(s):
        return ['background-color: #E3EFFC; color: #1E5FAF; font-weight: 700'] * len(s)

    styled_focus = (
        out.style
        .apply(_hl_safety_col, axis=0, subset=['安全库存'])
        .format({'安全库存': '{:,.0f}', '当前库存': '{:,.0f}', '缺口': '{:,.0f}'})
    )
    st.dataframe(styled_focus, use_container_width=True, hide_index=True)

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as w:
        out.to_excel(w, index=False, sheet_name='重点关注清单')
    st.download_button('下载重点关注清单 (Excel)', data=buf.getvalue(),
                       file_name='重点关注清单.xlsx',
                       mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

# ===== 双口径对比 =====
if '安全库存_修正版' in df.columns:
    with st.expander('O_原始 vs O_修正 差异对比', expanded=False):
        st.caption('差异 = 修正版 - 原始。正数表示修正后安全库存提高（更保守），负数表示降低（更激进）。')

        d1, d2, d3, d4 = st.columns(4)
        with d1:
            kpi_card('原始 总安全库存', fmt_num(float(df['安全库存'].sum())), '#4A6CF7')
        with d2:
            kpi_card('修正 总安全库存', fmt_num(float(df['安全库存_修正版'].sum())), '#4A6CF7')
        with d3:
            kpi_card('原始 总缺口', fmt_num(float(df['缺口'].sum())), '#4A6CF7')
        with d4:
            kpi_card('修正 总缺口', fmt_num(float(df['缺口_修正'].sum())), '#4A6CF7')

        diff = df.copy()
        diff['安全库存差异'] = diff['安全库存_修正版'] - diff['安全库存']
        diff['绝对差异'] = diff['安全库存差异'].abs()
        top_diff = diff.nlargest(10, '绝对差异')[[mat_col, '安全库存', '安全库存_修正版', '安全库存差异', '缺口', '缺口_修正']].copy()
        top_diff = top_diff.rename(columns={
            mat_col: '物料编码',
            '安全库存': '原始安全库存',
            '安全库存_修正版': '修正安全库存',
            '安全库存差异': '差异',
            '缺口': '原始缺口',
            '缺口_修正': '修正缺口',
        })
        st.dataframe(top_diff.style.format({
            '原始安全库存': '{:,.2f}',
            '修正安全库存': '{:,.2f}',
            '差异': '{:,.2f}',
            '原始缺口': '{:,.2f}',
            '修正缺口': '{:,.2f}',
        }), use_container_width=True, hide_index=True)

# ===== 下载 =====
st.subheader('下载')
try:
    month_str = meta.get('month_str') or '000000'
    out_name = f'安全库存_计算结果_{month_str}.xlsx'
    out_path = os.path.join(tempfile.gettempdir(), out_name)

    export_df = df.copy()
    if safety_col != '安全库存':
        export_df['安全库存'] = export_df[safety_col]
    export_to_excel(src_path, export_df, out_path)

    # 用 openpyxl 追加「分析摘要」sheet
    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill, Alignment

        wb = openpyxl.load_workbook(out_path)
        summary_name = '分析摘要'
        if summary_name in wb.sheetnames:
            del wb[summary_name]
        ws = wb.create_sheet(summary_name, 0)

        title_font = Font(name='微软雅黑', size=14, bold=True, color='FFFFFF')
        title_fill = PatternFill('solid', fgColor='2E86DE')
        label_font = Font(name='微软雅黑', size=11, bold=True)
        value_font = Font(name='微软雅黑', size=11)

        ws['A1'] = '安全库存分析摘要'
        ws['A1'].font = title_font
        ws['A1'].fill = title_fill
        ws.merge_cells('A1:B1')
        ws['A1'].alignment = Alignment(horizontal='center', vertical='center')
        ws.row_dimensions[1].height = 28

        rows = [
            ('生成时间', pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')),
            ('主口径', basis),
            ('总物料数', len(df)),
            ('低库存物料数', low),
            ('严重物料数', severe),
            ('零库存物料数', zero_stock),
            ('总安全库存', f'{total_safety:,.2f}'),
            ('总实际库存', f'{total_actual:,.2f}'),
            ('总缺口', f'{total_gap:,.2f}'),
            ('整体覆盖率', cov_display),
            ('缺口集中度', conc_display),
            ('冗余物料数（覆盖率>5）', int((df[cov_col] > 5).sum())),
            ('辅助表模式', mode),
            ('覆盖字段', '、'.join([k for k, v in {'使用量覆盖': override_usage, '预测覆盖': override_forecast}.items() if v]) or '无'),
        ]
        for i, (k, v) in enumerate(rows, start=2):
            ws.cell(row=i, column=1, value=k).font = label_font
            ws.cell(row=i, column=2, value=v).font = value_font
        ws.column_dimensions['A'].width = 26
        ws.column_dimensions['B'].width = 24

        aux_cols = [c for c in df.columns if c.startswith('aux_')]
        if aux_cols:
            from openpyxl.utils.dataframe import dataframe_to_rows
            aux_sheet_name = '辅助指标'
            if aux_sheet_name in wb.sheetnames:
                del wb[aux_sheet_name]
            ws_aux = wb.create_sheet(aux_sheet_name)
            export_aux = df[[c for c in [mat_col] + aux_cols if c in df.columns]].copy()
            for r in dataframe_to_rows(export_aux, index=False, header=True):
                ws_aux.append(r)

        wb.save(out_path)
    except Exception as e:
        st.warning(f'分析摘要 sheet 生成失败（不影响主文件）：{e}')

    with open(out_path, 'rb') as f:
        xlsx_bytes = f.read()

    st.caption(f'导出文件将基于当前选择的「{basis}」口径回填安全库存列，共 {len(export_df)} 行。')
    st.download_button('下载 Excel（回填安全库存列）', data=xlsx_bytes,
                       file_name=out_name,
                       mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
except Exception as e:
    st.error(f'导出失败：{e}')