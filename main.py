import os
import json
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

# ====================== 板塊定義 ======================
CORE_SECTORS = {
    'XLK': '科技',
    'XLF': '金融',
    'XLV': '醫療保健',
    'XLY': '非必需消費',
    'XLP': '必需消費',
    'XLE': '能源',
    'XLI': '工業',
    'XLU': '公用事業',
    'XLB': '原材料',
    'XLRE': '房地產',
    'XLC': '通訊服務',
    'SMH': '半導體',
    'XBI': '生物科技',
    'IGV': '軟體應用',
    'KRE': '地區銀行',
    'XHB': '房屋建築',
    'XRT': '零售',
    'IYT': '運輸',
    'ITA': '國防航天',
    'XME': '金屬礦業'
}

THEMATIC_SECTORS = {
    'CIBR': '網絡安全',
    'DRAM': '記憶體',
    'UFO': '太空探索',
    'LYTE': '光通信',
    'WGMI': '加密挖礦',
    'ICLN': '清潔能源',
    'TAN': '太陽能',
    'BOTZ': '機械人AI',
    'SOCL': '社交媒體',
    'FINX': '金融科技',
    'COPX': '銅礦金屬',
    'IHI': '醫療設備',
    'SKYY': '雲端運算',
    'PAVE': '基礎設施',
    'JETS': '航空旅遊'
}

SECTORS = {**CORE_SECTORS, **THEMATIC_SECTORS}
BENCHMARK = 'SPY'
HISTORY_FILE = 'history.json'
MIN_BARS = 60          # 至少要有多少根 K 線才納入計算
MAX_RANK_GAP_DAYS = 4  # 歷史日期差距超過此天數就不算排名變化

# ====================== 歷史讀取 / 儲存（含日期校驗） ======================
def load_previous_history():
    if not os.path.exists(HISTORY_FILE):
        return {}

    try:
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not data:
            return {}

        first_val = next(iter(data.values()))

        # 舊格式：純 int
        if isinstance(first_val, int):
            new_data = {}
            for k, v in data.items():
                new_data[k] = {
                    "ranks": [v],
                    "quadrants": [None],
                    "rank": v,
                    "quadrant": None,
                    "last_date": None
                }
            return new_data

        return data

    except Exception as e:
        print(f"讀取歷史紀錄失敗：{e}")
        return {}

def save_current_history(df_result, trade_date: str):
    prev_data = load_previous_history()
    history_data = {}

    for idx, row in df_result.iterrows():
        ticker = row['ticker']
        curr_rank = int(idx + 1)
        curr_quad = row['quadrant']

        item_hist = prev_data.get(ticker, {})
        ranks = item_hist.get('ranks', [])
        quadrants = item_hist.get('quadrants', [])

        ranks.append(curr_rank)
        quadrants.append(curr_quad)
        ranks = ranks[-30:]
        quadrants = quadrants[-30:]

        history_data[ticker] = {
            "ranks": ranks,
            "quadrants": quadrants,
            "rank": curr_rank,
            "quadrant": curr_quad,
            "last_date": trade_date
        }

    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history_data, f, ensure_ascii=False, indent=2)

def is_history_fresh(item_hist: dict, today: datetime) -> bool:
    """判斷歷史是否足夠新，可拿來算排名變化"""
    last_date_str = item_hist.get('last_date')
    if not last_date_str:
        return False
    try:
        last_date = datetime.strptime(last_date_str, "%Y-%m-%d")
        return (today - last_date).days <= MAX_RANK_GAP_DAYS
    except Exception:
        return False

# ====================== 資料下載與計算 ======================
def calculate_rrg_and_rs():
    print(f"正在下載 {len(SECTORS)} 個板塊數據（個別失敗會跳過）...")

    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })
    retries = Retry(total=5, backoff_factor=5, status_forcelist=[429, 500, 502, 503, 504], raise_on_status=False)
    adapter = HTTPAdapter(max_retries=retries)
    session.mount('https://', adapter)
    session.mount('http://', adapter)

    original_request = session.request
    def request_with_timeout(*args, **kwargs):
        kwargs.setdefault('timeout', 30)
        return original_request(*args, **kwargs)
    session.request = request_with_timeout

    all_tickers = list(SECTORS.keys()) + [BENCHMARK]

    try:
        raw = yf.download(all_tickers, period='1y', session=session, progress=False, threads=True)
        if raw is None or raw.empty:
            raise ValueError("Yahoo Finance 返回空資料")
        data = raw['Close'] if 'Close' in raw.columns else raw
    except Exception as e:
        raise RuntimeError(f"Yahoo Finance 下載失敗: {e}")

    # 個別 ticker 有效性檢查
    valid_tickers = []
    skipped = []
    for t in SECTORS.keys():
        if t not in data.columns:
            skipped.append(f"{t}(無欄位)")
            continue
        series = data[t].dropna()
        if len(series) < MIN_BARS:
            skipped.append(f"{t}(僅{len(series)}根)")
            continue
        valid_tickers.append(t)

    if BENCHMARK not in data.columns or data[BENCHMARK].dropna().shape[0] < 200:
        raise ValueError(f"基準 {BENCHMARK} 資料不足，無法繼續")

    if skipped:
        print(f"⚠️ 已跳過 {len(skipped)} 檔：{', '.join(skipped)}")
    print(f"✅ 有效標的：{len(valid_tickers)} 檔")

    # 大市環境
    spy_series = data[BENCHMARK].dropna()
    spy_sma200 = spy_series.rolling(200).mean().iloc[-1]
    spy_latest = spy_series.iloc[-1]
    market_regime = (
        "🟢 大市環境：多頭牛市 (SPY 企穩 SMA200)"
        if spy_latest >= spy_sma200
        else "🔴 大市環境：空頭警戒 (SPY 跌破 SMA200)"
    )

    trade_date = spy_series.index[-1].strftime("%Y-%m-%d")

    # 相對強度線
    ratios = pd.DataFrame({t: data[t] / data[BENCHMARK] for t in valid_tickers})

    # 自訂動能象限（非經典 RRG）
    rs_ratio = pd.DataFrame()
    rs_momentum = pd.DataFrame()
    for t in valid_tickers:
        series = ratios[t].dropna()
        sma_short = series.rolling(10).mean()
        sma_long = series.rolling(50).mean()
        rs_ratio[t] = 100 * (sma_short / sma_long)

        mom_short = rs_ratio[t].rolling(5).mean()
        mom_long = rs_ratio[t].rolling(20).mean()
        rs_momentum[t] = 100 * (mom_short / mom_long)

    latest_ratio = rs_ratio.iloc[-1]
    latest_mom = rs_momentum.iloc[-1]

    # 多週期相對報酬
    ret_1d = data.pct_change(1).iloc[-1]
    ret_5d = data.pct_change(5).iloc[-1]
    ret_2w = data.pct_change(10).iloc[-1]
    ret_1m = data.pct_change(21).iloc[-1]

    spy_1d, spy_5d, spy_2w, spy_1m = (
        ret_1d[BENCHMARK], ret_5d[BENCHMARK],
        ret_2w[BENCHMARK], ret_1m[BENCHMARK]
    )

    # 近 20 日波動（年化）
    vol_20 = data.pct_change().rolling(20).std().iloc[-1] * np.sqrt(252)

    results = []
    for t in valid_tickers:
        e1d = (ret_1d[t] - spy_1d) * 100
        e5d = (ret_5d[t] - spy_5d) * 100
        e2w = (ret_2w[t] - spy_2w) * 100
        e1m = (ret_1m[t] - spy_1m) * 100

        # 原始加權得分
        raw_score = (e1d * 0.10) + (e5d * 0.30) + (e2w * 0.30) + (e1m * 0.30)

        # 簡單波動調整（避免高波動主題壟斷）
        vol = vol_20.get(t, np.nan)
        if pd.isna(vol) or vol <= 0:
            adj_score = raw_score
        else:
            # 以 20% 年化波動為基準做正規化
            adj_score = raw_score / (vol / 0.20)

        # 多週期一致性（幾個週期同時 > 0）
        consistency = sum([
            1 if e1d > 0 else 0,
            1 if e5d > 0 else 0,
            1 if e2w > 0 else 0,
            1 if e1m > 0 else 0
        ])

        r_ratio = float(latest_ratio.get(t, 100.0))
        r_mom = float(latest_mom.get(t, 100.0))

        if r_ratio >= 100 and r_mom >= 100:
            quadrant, quad_emoji = "Leading", "🟢"
        elif r_ratio < 100 and r_mom >= 100:
            quadrant, quad_emoji = "Improving", "🔵"
        elif r_ratio >= 100 and r_mom < 100:
            quadrant, quad_emoji = "Weakening", "🟡"
        else:
            quadrant, quad_emoji = "Lagging", "🔴"

        results.append({
            'ticker': t,
            'name': SECTORS[t],
            'group': 'core' if t in CORE_SECTORS else 'thematic',
            'rel_1d': e1d,
            'rel_5d': e5d,
            'rel_2w': e2w,
            'rel_1m': e1m,
            'raw_score': raw_score,
            'adj_score': adj_score,
            'consistency': consistency,
            'vol_20': vol * 100 if not pd.isna(vol) else np.nan,  # 百分比
            'r_ratio': r_ratio,
            'r_mom': r_mom,
            'quadrant': quadrant,
            'quad_emoji': quad_emoji
        })

    df_rs = pd.DataFrame(results)
    # 主排序改用波動調整後得分
    df_rs = df_rs.sort_values(by='adj_score', ascending=False).reset_index(drop=True)
    return df_rs, market_regime, trade_date

# ====================== 排名變化與表格 ======================
def format_rank_change(current_rank, prev_rank, is_fresh: bool):
    if not is_fresh or prev_rank is None:
        return "NEW"
    diff = prev_rank - current_rank
    if diff > 0:
        return f"▲{diff}"
    elif diff < 0:
        return f"▼{abs(diff)}"
    return "─"

def get_quadrant_emoji(quad):
    return {"Leading": "🟢", "Improving": "🔵", "Weakening": "🟡", "Lagging": "🔴"}.get(quad, "⚪")

def generate_table_lines(sub_df, prev_history, today: datetime):
    header = (
        f"{'名':<3}{'變':<4}{'ETF':<6}{'名稱':<8}{'象限':<10}"
        f"{'1D':<6}{'5D':<6}{'2W':<6}{'1M':<6}"
        f"{'一致':<4}{'波動':<5}{'品質':<8}"
    )
    lines = [header, "-" * 78]
    rotations = []

    for _, row in sub_df.iterrows():
        rank = int(row['global_rank'])
        ticker = row['ticker']
        item_hist = prev_history.get(ticker, {})
        past_ranks = item_hist.get('ranks', [])
        past_quads = item_hist.get('quadrants', [])
        prev_rank = past_ranks[-1] if past_ranks else None
        prev_quad = past_quads[-1] if past_quads else None
        fresh = is_history_fresh(item_hist, today)

        change_str = format_rank_change(rank, prev_rank, fresh)
        name = row['name']
        emoji = row['quad_emoji']
        quad = row['quadrant']

        full_ranks = past_ranks + [rank]
        full_quads = past_quads + [quad]
        ranks_5d = full_ranks[-5:]
        avg_rank_5d = sum(ranks_5d) / len(ranks_5d) if ranks_5d else rank
        rank_std = np.std(ranks_5d) if len(ranks_5d) > 1 else 0.0

        leading_days = 0
        for q in reversed(full_quads):
            if q == "Leading":
                leading_days += 1
            else:
                break

        if quad == "Leading":
            quality_str = "剛轉強⭐" if leading_days <= 2 else f"長期強({leading_days}d)"
        elif rank_std > 3.5:
            quality_str = "忽上忽下"
        elif avg_rank_5d < rank - 1:
            quality_str = "穩步上升"
        elif avg_rank_5d > rank + 1:
            quality_str = "回落中"
        else:
            quality_str = "橫盤"

        if prev_quad and prev_quad != quad and fresh:
            rotations.append(
                f"• **{ticker} ({name})**: {get_quadrant_emoji(prev_quad)} `{prev_quad}` ➔ {emoji} `{quad}`"
            )

        vol_str = f"{row['vol_20']:.0f}%" if not pd.isna(row['vol_20']) else "N/A"
        line = (
            f"{rank:<3}{change_str:<4}{ticker:<6}{name:<8}{emoji}{quad[:7]:<9}"
            f"{row['rel_1d']:+.1f}% {row['rel_5d']:+.1f}% {row['rel_2w']:+.1f}% {row['rel_1m']:+.1f}% "
            f"{row['consistency']}/4  {vol_str:<5}{quality_str:<8}"
        )
        lines.append(line)

    return lines, rotations

def split_into_fields(title_prefix: str, lines: list, max_len: int = 1000):
    fields = []
    current_chunk, current_len, part = [], 0, 1

    for line in lines:
        add_len = len(line) + 1
        if current_chunk and current_len + add_len > max_len:
            name = title_prefix if part == 1 else f"{title_prefix}（續{part}）"
            fields.append({
                "name": name,
                "value": "```text\n" + "\n".join(current_chunk) + "\n```",
                "inline": False
            })
            part += 1
            current_chunk = [line]
            current_len = add_len
        else:
            current_chunk.append(line)
            current_len += add_len

    if current_chunk:
        name = title_prefix if part == 1 else f"{title_prefix}（續{part}）"
        fields.append({
            "name": name,
            "value": "```text\n" + "\n".join(current_chunk) + "\n```",
            "inline": False
        })
    return fields

# ====================== Discord ======================
def send_to_discord(df, prev_history, market_regime, trade_date: str):
    webhook_url = os.environ.get('DISCORD_WEBHOOK_URL')
    if not webhook_url:
        print("錯誤：未設定 DISCORD_WEBHOOK_URL")
        return

    today = datetime.strptime(trade_date, "%Y-%m-%d")
    df = df.copy()
    df['global_rank'] = np.arange(1, len(df) + 1)

    df_core = df[df['group'] == 'core']
    df_thematic = df[df['group'] == 'thematic']

    lines_core, rot_core = generate_table_lines(df_core, prev_history, today)
    lines_thematic, rot_thematic = generate_table_lines(df_thematic, prev_history, today)
    all_rotations = rot_core + rot_thematic

    rotation_text = "\n".join(all_rotations) if all_rotations else "*今日暫無跨象限轉倉*"
    quad_counts = df['quadrant'].value_counts()
    breadth = (
        f"🟢 Leading `{quad_counts.get('Leading', 0)}` | "
        f"🔵 Improving `{quad_counts.get('Improving', 0)}` | "
        f"🟡 Weakening `{quad_counts.get('Weakening', 0)}` | "
        f"🔴 Lagging `{quad_counts.get('Lagging', 0)}`"
    )

    # Embed 1 核心
    fields_core = [
        {"name": "大市環境", "value": market_regime, "inline": False},
        {"name": "🔄 自訂動能象限轉倉", "value": rotation_text[:1000], "inline": False},
    ]
    fields_core.extend(split_into_fields("核心板塊排名（波動調整後）", lines_core))

    embed_core = {
        "title": f"📊 美股核心板塊監控 ({trade_date})",
        "color": 3066993,
        "fields": fields_core
    }

    # Embed 2 主題
    fields_thematic = split_into_fields("前沿主題排名（波動調整後）", lines_thematic)
    fields_thematic.append({
        "name": "📈 象限分佈 & 說明",
        "value": (
            f"{breadth}\n\n"
            f"**排序**：波動調整得分（原始加權 ÷ 相對波動）\n"
            f"**一致**：1D/5D/2W/1M 同時跑贏 SPY 的週期數\n"
            f"**象限**：自訂動能象限（非經典 RRG）"
        ),
        "inline": False
    })

    embed_thematic = {
        "title": f"🚀 前沿主題監控 ({trade_date})",
        "color": 3447003,
        "fields": fields_thematic,
        "footer": {
            "text": "短線模型 1D10%+5D30%+2W30%+1M30% | 已做波動調整 | 自訂動能象限"
        }
    }

    resp = requests.post(webhook_url, json={"embeds": [embed_core, embed_thematic]})
    if resp.status_code == 204:
        print("✅ 成功發送到 Discord")
    else:
        print(f"❌ 發送失敗 {resp.status_code}: {resp.text}")

# ====================== 主程式 ======================
if __name__ == "__main__":
    prev_history = load_previous_history()
    df_result, market_regime, trade_date = calculate_rrg_and_rs()
    send_to_discord(df_result, prev_history, market_regime, trade_date)
    save_current_history(df_result, trade_date)
    print(f"完成。交易日：{trade_date}，有效標的：{len(df_result)}")
