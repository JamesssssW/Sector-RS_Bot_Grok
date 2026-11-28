import io
import os
import json
import time
from datetime import datetime, date, timezone
import requests
import yfinance as yf
import pandas as pd
import numpy as np

# ==========================================
# 1. 核心選股、買點型態、止跌濾網、真龍頭與風控參數
# ==========================================
BENCHMARK = 'SPY'
EQUAL_WEIGHT_BENCHMARK = 'RSP'
VIX_TICKER = '^VIX'
VIX3M_TICKER = '^VIX3M'
HISTORY_FILE = 'stock_history.json'

# 基礎篩選門檻
MIN_PRICE = 10.0                  # 最低股價（過濾仙股）
MIN_AVG_DOLLAR_VOL = 30_000_000   # 20 日均成交額至少 3,000 萬美元（兼顧中型熱門主題股 VCP 縮量期）
MIN_RS_RATING = 80                # 只推薦全市場 RS 80 分以上強勢股
MAX_DIST_52W_HIGH_PCT = 20.0      # 距離 52 週高點唔超過 20%
MIN_ABOVE_52W_LOW_PCT = 25.0      # 較 52 週低點已反彈至少 25%

# 防追高買點區間、VCP 縮量與 EMA21 止跌濾網參數
BUY_ZONE_MIN_EXT = -1.0           # 收緊至只容許輕微跌穿 EMA21 1% 以內
BUY_ZONE_MAX_EXT = 8.0            # 距離 EMA21 8% 以內屬於安全可買區；超過 8% 撥入「乖離偏高觀察」
EMA21_PULLBACK_MAX_EXT = 2.5      # 回踩 EMA21 判定上限（-1.0% 至 +2.5%）
PULLBACK_MAX_DOWN_VOL = 1.0       # 下跌日量比上限（≤ 1.0x 代表縮量回調，唔放量下跌）
PULLBACK_MIN_CLOSE_POS = 0.55     # 下影線支撐門檻（收市價位於全日波幅 55% 以上）
VCP_RANGE_RATIO_MAX = 0.78        # 價窄：近 5 日平均振幅 ÷ 近 20 日平均振幅 上限
VCP_VOL_RATIO_MAX = 0.85          # 真縮量：近 5 日均量 ÷ 近 20 日均量 上限
VCP_MAX_DIST_HIGH_PCT = 10.0      # 高位蓄勢：距離 52 週高點 10% 以內

# 真龍頭強化參數（RS Line 52 週新高 + 個股跑贏所屬板塊 ETF）
RS_LINE_HIGH_TOLERANCE = 0.995    # RS Line 處於 52 週高點 99.5% 以上即視為創年高
RS_BLUE_DOT_MIN_DIST = 2.0        # 股價仍距年高 ≥ 2% 但 RS 線已創高（先行創高藍點）
RS_BLUE_DOT_BONUS = 5.0           # 🔵RS先行創高 加分
RS_NEW_HIGH_BONUS = 3.0           # ⚡RS創年高 加分
SECTOR_LEADER_BONUS = 3.0         # 👑近 1M & 3M 雙雙跑贏所屬板塊 ETF 加分
SECTOR_LAGGARD_PENALTY = -2.0     # 🐢近 1M & 3M 雙雙跑輸所屬板塊 ETF 扣分

# 單筆風險倉位與止損參數
RISK_PER_TRADE_PCT = 1.0          # 每筆交易最大承受總資金虧損 1.0% (1R)
MAX_SINGLE_POS_PCT = 15.0         # 單隻個股最大倉位上限 15%
ATR_PERIOD = 14                   # 14 日 Wilder ATR
ATR_MULTIPLIER = 2.0              # 2 倍 ATR 止損
EARNINGS_WARN_DAYS = 7            # 財報倒數 7 天內亮紅燈避雷
TOP_BUY_COUNT = 10                # 優先推薦買入前 10 檔
TOP_WATCH_COUNT = 10              # 強勢但乖離偏高觀察前 10 檔

# 11 大一級板塊 ETF 與中文名稱對照（Top-Down 聯動基準）
SECTOR_ETFS = {
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
    'XLC': '通訊服務'
}

# 板塊 RRG 象限對個股買點評分嘅 Top-Down 聯動加減分
SECTOR_QUAD_BONUS = {
    'Leading': 6.0,
    'Improving': 3.0,
    'Weakening': 0.0,
    'Lagging': -3.0
}

QUADRANT_META = {
    'Leading': ('🟢', '領先'),
    'Improving': ('🔵', '改善'),
    'Weakening': ('🟡', '轉弱'),
    'Lagging': ('🔴', '落後')
}

# ==========================================
# 2. 自訂 10 大熱門主題補充股池：ticker -> (中文主題標籤, 對應板塊ETF)
# ==========================================
CUSTOM_THEMATIC_STOCKS = {
    # --- 熱門 AI 算力與晶片 ---
    'NVDA': ('熱門AI', 'XLK'),
    'AVGO': ('熱門AI', 'XLK'),
    'TSM':  ('熱門AI', 'XLK'),
    'ASML': ('熱門AI', 'XLK'),
    'ARM':  ('熱門AI', 'XLK'),
    'AMD':  ('熱門AI', 'XLK'),
    'MU':   ('熱門AI', 'XLK'),
    'MRVL': ('熱門AI', 'XLK'),
    'SMCI': ('熱門AI', 'XLK'),
    'PLTR': ('熱門AI', 'XLK'),
    'APP':  ('熱門AI', 'XLK'),
    'ALAB': ('熱門AI', 'XLK'),
    'CRDO': ('熱門AI', 'XLK'),
    'NBIS': ('熱門AI', 'XLK'),
    'CRWV': ('熱門AI', 'XLK'),
    'IONQ': ('量子AI', 'XLK'),

    # --- AI 電力與核電基建 ---
    'CEG':  ('AI核電', 'XLU'),
    'VST':  ('AI電力', 'XLU'),
    'TLN':  ('AI電力', 'XLU'),
    'OKLO': ('AI核電', 'XLU'),
    'SMR':  ('AI核電', 'XLU'),
    'GEV':  ('AI電力', 'XLI'),

    # --- 光通信與高速互連 ---
    'COHR': ('光通信', 'XLK'),
    'LITE': ('光通信', 'XLK'),
    'AAOI': ('光通信', 'XLK'),
    'FN':   ('光通信', 'XLK'),
    'CIEN': ('光通信', 'XLK'),
    'GLW':  ('光通信', 'XLK'),
    'ANET': ('光通信', 'XLK'),
    'CLS':  ('光通信', 'XLK'),

    # --- 數據中心、液冷與電氣基建 ---
    'VRT':  ('數據中心', 'XLI'),
    'MOD':  ('數據中心', 'XLI'),
    'EQIX': ('數據中心', 'XLRE'),
    'DLR':  ('數據中心', 'XLRE'),
    'IRM':  ('數據中心', 'XLRE'),
    'APLD': ('數據中心', 'XLK'),
    'ETN':  ('數據中心', 'XLI'),
    'PWR':  ('數據中心', 'XLI'),
    'FIX':  ('數據中心', 'XLI'),

    # --- 軟件、雲端與網絡安全 ---
    'SNOW': ('軟件雲端', 'XLK'),
    'NET':  ('軟件雲端', 'XLK'),
    'DDOG': ('軟件雲端', 'XLK'),
    'MDB':  ('軟件雲端', 'XLK'),
    'GTLB': ('軟件雲端', 'XLK'),
    'IOT':  ('軟件雲端', 'XLK'),
    'MNDY': ('軟件雲端', 'XLK'),
    'CFLT': ('軟件雲端', 'XLK'),
    'CRWD': ('軟件網安', 'XLK'),
    'ZS':   ('軟件網安', 'XLK'),
    'PANW': ('軟件網安', 'XLK'),
    'S':    ('軟件網安', 'XLK'),
    'OKTA': ('軟件網安', 'XLK'),
    'NOW':  ('軟件雲端', 'XLK'),
    'CRM':  ('軟件雲端', 'XLK'),
    'ORCL': ('軟件雲端', 'XLK'),
    'MSFT': ('軟件雲端', 'XLK'),
    'SHOP': ('軟件電商', 'XLK'),
    'RDDT': ('軟件社群', 'XLC'),
    'SPOT': ('軟件串流', 'XLC'),
    'DUOL': ('軟件AI',   'XLY'),

    # --- 醫療設備、AI醫療與健康服務 ---
    'ISRG': ('醫療設備', 'XLV'),
    'BSX':  ('醫療設備', 'XLV'),
    'SYK':  ('醫療設備', 'XLV'),
    'DXCM': ('醫療設備', 'XLV'),
    'PODD': ('醫療設備', 'XLV'),
    'TMDX': ('醫療設備', 'XLV'),
    'HIMS': ('數碼醫療', 'XLV'),
    'TEM':  ('AI醫療',   'XLV'),
    'UNH':  ('醫療保險', 'XLV'),
    'ELV':  ('醫療保險', 'XLV'),
    'MCK':  ('醫療分銷', 'XLV'),
    'COR':  ('醫療分銷', 'XLV'),
    'CAH':  ('醫療分銷', 'XLV'),
    'IDXX': ('醫療診斷', 'XLV'),
    'EW':   ('醫療設備', 'XLV'),
    'MDT':  ('醫療設備', 'XLV'),

    # --- 製藥與生技 ---
    'LLY':  ('製藥減肥', 'XLV'),
    'NVO':  ('製藥減肥', 'XLV'),
    'VKTX': ('生技減肥', 'XLV'),
    'VRTX': ('生技製藥', 'XLV'),
    'REGN': ('生技製藥', 'XLV'),
    'ABBV': ('大型製藥', 'XLV'),
    'AMGN': ('大型製藥', 'XLV'),
    'GILD': ('大型製藥', 'XLV'),
    'BMY':  ('大型製藥', 'XLV'),
    'MRK':  ('大型製藥', 'XLV'),
    'PFE':  ('大型製藥', 'XLV'),
    'ARGX': ('生技製藥', 'XLV'),
    'ALNY': ('生技製藥', 'XLV'),
    'INSM': ('生技製藥', 'XLV'),
    'PCVX': ('生技製藥', 'XLV'),
    'AZN':  ('大型製藥', 'XLV'),

    # --- 加密礦股與數位資產 ---
    'IREN': ('加密礦股', 'XLK'),
    'CLSK': ('加密礦股', 'XLK'),
    'MARA': ('加密礦股', 'XLK'),
    'RIOT': ('加密礦股', 'XLK'),
    'CORZ': ('加密礦股', 'XLK'),
    'HUT':  ('加密礦股', 'XLK'),
    'WULF': ('加密礦股', 'XLK'),
    'CIFR': ('加密礦股', 'XLK'),
    'BTDR': ('加密礦股', 'XLK'),
    'MSTR': ('加密資產', 'XLK'),
    'COIN': ('加密交易', 'XLF'),
    'HOOD': ('加密券商', 'XLF'),

    # --- 太空、衛星與無人機國防 ---
    'RKLB': ('太空火箭', 'XLI'),
    'ASTS': ('太空衛星', 'XLC'),
    'LUNR': ('太空探索', 'XLI'),
    'PL':   ('太空衛星', 'XLI'),
    'RDW':  ('太空基建', 'XLI'),
    'KTOS': ('無人機國防', 'XLI'),
    'AVAV': ('無人機國防', 'XLI'),
    'LHX':  ('航天國防', 'XLI'),
    'NOC':  ('航天國防', 'XLI'),
    'LMT':  ('航天國防', 'XLI'),
    'RTX':  ('航天國防', 'XLI'),
    'HEI':  ('航天零件', 'XLI'),
    'TDG':  ('航天零件', 'XLI'),

    # --- 防守必消、公用與穩健現金牛 ---
    'COST':  ('防守零售', 'XLP'),
    'WMT':   ('防守零售', 'XLP'),
    'PG':    ('防守必消', 'XLP'),
    'KO':    ('防守必消', 'XLP'),
    'PEP':   ('防守必消', 'XLP'),
    'PM':    ('防守高息', 'XLP'),
    'MO':    ('防守高息', 'XLP'),
    'CL':    ('防守必消', 'XLP'),
    'KR':    ('防守零售', 'XLP'),
    'NEE':   ('防守公用', 'XLU'),
    'SO':    ('防守公用', 'XLU'),
    'DUK':   ('防守公用', 'XLU'),
    'AEP':   ('防守公用', 'XLU'),
    'WM':    ('防守環保', 'XLI'),
    'RSG':   ('防守環保', 'XLI'),
    'BRK-B': ('防守金融', 'XLF'),
    'PGR':   ('防守保險', 'XLF'),
    'CB':    ('防守保險', 'XLF'),
}


def map_sector_text_to_etf(raw_text):
    """將 Wikipedia 嘅 GICS / ICB 行業名稱統一對應至 11 檔板塊 ETF"""
    s = str(raw_text).strip().lower()
    if not s or s == 'nan':
        return None
    if 'technology' in s:
        return 'XLK'
    if 'financial' in s:
        return 'XLF'
    if 'health' in s:
        return 'XLV'
    if 'discretionary' in s or 'cyclical' in s:
        return 'XLY'
    if 'staples' in s or 'defensive' in s:
        return 'XLP'
    if 'energy' in s:
        return 'XLE'
    if 'industrial' in s:
        return 'XLI'
    if 'utilit' in s:
        return 'XLU'
    if 'material' in s:
        return 'XLB'
    if 'real estate' in s:
        return 'XLRE'
    if 'communication' in s or 'telecom' in s:
        return 'XLC'
    return None


# ==========================================
# 3. 動態獲取美股股票池 (S&P 500 + Nasdaq 100 + 自訂熱門主題池)
# ==========================================
def fetch_us_stock_universe():
    """從 Wikipedia 抓取 S&P 500 + Nasdaq 100 及行業，並無縫合併自訂熱門主題股池"""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; StockRSBot/1.0)"}
    tickers_set = set()
    ticker_sector_map = {}

    sources = [
        ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "Symbol"),
        ("https://en.wikipedia.org/wiki/Nasdaq-100", "Ticker"),
    ]

    for url, col_hint in sources:
        try:
            resp = requests.get(url, headers=headers, timeout=20)
            resp.raise_for_status()
            tables = pd.read_html(io.StringIO(resp.text))
            for tbl in tables:
                cols = [str(c) for c in tbl.columns]
                sym_col = next((c for c in cols if col_hint.lower() in c.lower() or "symbol" in c.lower()), None)
                if not sym_col or len(tbl) < 80:
                    continue

                sec_col = next((c for c in cols if "gics sector" in c.lower()), None)
                if not sec_col:
                    sec_col = next(
                        (c for c in cols if ("sector" in c.lower() or "icb" in c.lower() or "industry" in c.lower())
                         and "sub" not in c.lower()),
                        None
                    )

                for _, r in tbl.iterrows():
                    raw_sym = r.get(sym_col)
                    if pd.isna(raw_sym):
                        continue
                    clean = str(raw_sym).strip().upper().replace('.', '-')
                    if not (clean and len(clean) <= 6 and clean.replace('-', '').isalpha()):
                        continue

                    tickers_set.add(clean)
                    if sec_col and clean not in ticker_sector_map:
                        etf_code = map_sector_text_to_etf(r.get(sec_col))
                        if etf_code:
                            ticker_sector_map[clean] = etf_code
                break
        except Exception as e:
            print(f"⚠️ 抓取成分股來源失敗 ({url}): {e}")

    for sym, (_, sec_etf) in CUSTOM_THEMATIC_STOCKS.items():
        tickers_set.add(sym)
        if sec_etf in SECTOR_ETFS:
            ticker_sector_map[sym] = sec_etf

    for dup in ('GOOG', 'FOXA', 'NWS'):
        tickers_set.discard(dup)

    universe = sorted(tickers_set)
    print(
        f"✅ 成功載入美股監控池：共 {len(universe)} 檔"
        f"（含 {len(CUSTOM_THEMATIC_STOCKS)} 檔熱門主題股｜已匹配行業 {len(ticker_sector_map)} 檔）"
    )
    return universe, ticker_sector_map


# ==========================================
# 4. 歷史上榜紀錄處理（按美股交易日去重，追蹤新上榜與排名變化）
# ==========================================
def load_previous_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"讀取歷史紀錄失敗：{e}")
    return {}


def save_current_history(buy_df, watch_df, trade_date):
    prev = load_previous_history()
    updated = {}

    combined = pd.concat([buy_df, watch_df], ignore_index=True)
    for _, row in combined.iterrows():
        t = row['ticker']
        rank = int(row['buy_rank']) if pd.notna(row.get('buy_rank')) else int(row['watch_rank'])
        bucket = row['bucket']

        records = list(prev.get(t, {}).get('records', []))
        entry = {"date": trade_date, "rank": rank, "bucket": bucket, "rs": int(row['rs_rating'])}

        if records and records[-1].get('date') == trade_date:
            records[-1] = entry
        else:
            records = [r for r in records if r.get('date') != trade_date]
            records.append(entry)

        updated[t] = {"records": records[-20:]}

    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(updated, f, ensure_ascii=False, indent=2)


# ==========================================
# 5. 分批安全下載行情（防 SQLite 鎖死＋自動單線程補抓）
# ==========================================
def download_ohlcv_in_batches(ticker_list, batch_size=50):
    """分批下載 1 年日線數據，限制並發防 SQLite database is locked，並自動補抓失敗標的"""
    close_parts, high_parts, low_parts, vol_parts = [], [], [], []

    def _fetch_chunk(chunk, threads_opt):
        raw = yf.download(
            chunk,
            period='1y',
            progress=False,
            auto_adjust=True,
            threads=threads_opt
        )
        if raw is None or raw.empty:
            return
        if isinstance(raw.columns, pd.MultiIndex):
            close_parts.append(raw['Close'])
            high_parts.append(raw['High'])
            low_parts.append(raw['Low'])
            vol_parts.append(raw['Volume'])
        else:
            sym = chunk[0]
            close_parts.append(raw[['Close']].rename(columns={'Close': sym}))
            high_parts.append(raw[['High']].rename(columns={'High': sym}))
            low_parts.append(raw[['Low']].rename(columns={'Low': sym}))
            vol_parts.append(raw[['Volume']].rename(columns={'Volume': sym}))

    def _merge(parts):
        cleaned = [p.dropna(axis=1, how='all') for p in parts if p is not None and not p.empty]
        if not cleaned:
            return pd.DataFrame()
        df = pd.concat(cleaned, axis=1)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df.loc[:, ~df.columns.duplicated(keep='last')]

    for i in range(0, len(ticker_list), batch_size):
        chunk = ticker_list[i:i + batch_size]
        print(f"正在下載第 {i + 1}–{min(i + batch_size, len(ticker_list))} 檔行情...")
        try:
            _fetch_chunk(chunk, threads_opt=4)
        except Exception as e:
            print(f"⚠️ 批次 {i // batch_size + 1} 下載異常：{e}")

        if i + batch_size < len(ticker_list):
            time.sleep(0.8)

    close_df = _merge(close_parts)
    high_df = _merge(high_parts)
    low_df = _merge(low_parts)
    vol_df = _merge(vol_parts)

    if close_df.empty:
        raise RuntimeError("❌ Yahoo Finance 全部批次下載失敗，無可用數據。")

    missing_syms = [
        t for t in ticker_list
        if t not in close_df.columns or close_df[t].dropna().empty
    ]
    if missing_syms and len(missing_syms) <= 40:
        print(f"🔄 正在單線程補抓 {len(missing_syms)} 檔下載失敗標的：{', '.join(missing_syms)}")
        time.sleep(1.0)
        try:
            _fetch_chunk(missing_syms, threads_opt=False)
            close_df = _merge(close_parts)
            high_df = _merge(high_parts)
            low_df = _merge(low_parts)
            vol_df = _merge(vol_parts)
        except Exception as e:
            print(f"⚠️ 補抓失敗（已略過）：{e}")

    return close_df, high_df, low_df, vol_df


def get_earnings_warning(ticker):
    """僅對最終入選 Top 20 查詢下次財報日期，提前 7 天亮燈避雷"""
    try:
        cal = yf.Ticker(ticker).calendar
        earn_dates = []
        if isinstance(cal, dict):
            raw_ed = cal.get('Earnings Date')
            if isinstance(raw_ed, (list, tuple)):
                earn_dates = list(raw_ed)
            elif raw_ed is not None:
                earn_dates = [raw_ed]
        elif isinstance(cal, pd.DataFrame) and not cal.empty:
            if 'Earnings Date' in cal.index:
                earn_dates = list(cal.loc['Earnings Date'].values)

        today = date.today()
        valid_days = []
        for ed in earn_dates:
            if isinstance(ed, datetime):
                d = ed.date()
            elif isinstance(ed, date):
                d = ed
            elif isinstance(ed, pd.Timestamp):
                d = ed.date()
            else:
                continue
            diff = (d - today).days
            if diff >= 0:
                valid_days.append((diff, d))

        if valid_days:
            valid_days.sort(key=lambda x: x[0])
            days_left, next_d = valid_days[0]
            if days_left <= EARNINGS_WARN_DAYS:
                return f"🚨財報剩{days_left}日({next_d.strftime('%m/%d')})", True
            elif days_left <= 14:
                return f"📅財報 {next_d.strftime('%m/%d')}", False
    except Exception:
        pass
    return "", False


def calc_sector_rrg_states(close_df):
    """用與板塊 Bot 一致嘅 EMA 100 軸標準化 RRG 判定 11 大板塊強弱象限"""
    spy_s = close_df[BENCHMARK].dropna()
    sector_states = {}

    for etf, zh_name in SECTOR_ETFS.items():
        if etf not in close_df.columns:
            continue
        aligned = pd.concat([close_df[etf].rename('a'), spy_s.rename('b')], axis=1, join='inner').dropna()
        aligned = aligned[(aligned['a'] > 0) & (aligned['b'] > 0)]
        if len(aligned) < 40:
            continue

        rel = aligned['a'] / aligned['b']
        raw_ratio = 100.0 * (rel.ewm(span=10, adjust=False).mean() / rel.ewm(span=40, adjust=False).mean())
        raw_mom = 100.0 * (raw_ratio.ewm(span=5, adjust=False).mean() / raw_ratio.ewm(span=15, adjust=False).mean())

        ratio_std = raw_ratio.rolling(60, min_periods=20).std().fillna(1.0).clip(lower=0.30)
        mom_std = raw_mom.rolling(60, min_periods=20).std().fillna(0.50).clip(lower=0.15)

        ratio_z = float(((raw_ratio - 100.0) / ratio_std).iloc[-1])
        mom_z = float(((raw_mom - 100.0) / mom_std).iloc[-1])
        if not (np.isfinite(ratio_z) and np.isfinite(mom_z)):
            continue

        if ratio_z >= 0 and mom_z >= 0:
            quad = 'Leading'
        elif ratio_z < 0 and mom_z >= 0:
            quad = 'Improving'
        elif ratio_z >= 0 and mom_z < 0:
            quad = 'Weakening'
        else:
            quad = 'Lagging'

        emoji, quad_zh = QUADRANT_META[quad]
        sector_states[etf] = {
            'etf': etf,
            'name': zh_name,
            'quadrant': quad,
            'emoji': emoji,
            'quad_zh': quad_zh,
            'ratio_z': ratio_z,
            'bonus': SECTOR_QUAD_BONUS[quad]
        }

    return sector_states


def build_macro_header(close_df, raw_vix, raw_vix3m, trade_date, sector_states):
    """頂部大市濾網（SPY / RSP / VIX / VIX3M + 今日領先板塊速覽）"""
    spy_s = close_df[BENCHMARK].dropna()
    spy_sma200 = float(spy_s.rolling(200).mean().iloc[-1])
    spy_sma50 = float(spy_s.rolling(50).mean().iloc[-1])
    spy_now = float(spy_s.iloc[-1])

    is_bull = spy_now >= spy_sma200
    if spy_now >= spy_sma50 and is_bull:
        spy_emoji, spy_state = "🟢", "強勢多頭 (企穩 50/200MA)"
    elif is_bull:
        spy_emoji, spy_state = "🟡", "震盪整理 (破 50MA 守 200MA)"
    else:
        spy_emoji, spy_state = "🔴", "空頭警戒 (跌破 200MA，嚴控倉位)"

    rsp_line = ""
    if EQUAL_WEIGHT_BENCHMARK in close_df.columns:
        rsp_s = close_df[EQUAL_WEIGHT_BENCHMARK].dropna()
        if len(rsp_s) >= 50:
            rsp_now = float(rsp_s.iloc[-1])
            rsp_sma50 = float(rsp_s.rolling(50).mean().iloc[-1])
            rsp_e = "🟢" if rsp_now >= rsp_sma50 else "🟠"
            rsp_t = "企穩 50MA" if rsp_now >= rsp_sma50 else "弱於 50MA"
            rsp_line = f"\n{rsp_e} **RSP 等權** · {rsp_t}"

    vix_line = ""
    if raw_vix is not None and not raw_vix.empty:
        v_now = float(raw_vix.iloc[-1])
        v_prev = float(raw_vix.iloc[-2]) if len(raw_vix) >= 2 else v_now
        v_chg = ((v_now - v_prev) / v_prev * 100.0) if v_prev > 0 else 0.0
        v_emoji = "🟢" if v_now < 15 else ("🟡" if v_now < 20 else ("🟠" if v_now < 30 else "🔴"))

        term_txt = ""
        if raw_vix3m is not None and not raw_vix3m.empty:
            aligned = pd.concat([raw_vix.rename('v'), raw_vix3m.rename('v3')], axis=1, join='inner').dropna()
            aligned = aligned[aligned['v3'] > 0]
            if not aligned.empty:
                ratio = float((aligned['v'] / aligned['v3']).iloc[-1])
                t_tag = "倒掛⚠️" if ratio >= 1.0 else "正價差"
                term_txt = f" · 期限比 `{ratio:.2f}` ({t_tag})"

        vix_line = f"\n{v_emoji} **VIX** `{v_now:.2f}` (`{v_chg:+.1f}%`){term_txt}"

    leading_secs = sorted(
        [v for v in sector_states.values() if v['quadrant'] == 'Leading'],
        key=lambda x: x['ratio_z'],
        reverse=True
    )
    if leading_secs:
        lead_tags = " · ".join(f"`{s['name']}({s['etf']})`" for s in leading_secs[:5])
        sector_line = f"\n🔥 **領先板塊**：{lead_tags}"
    else:
        sector_line = "\n🔥 **領先板塊**：暫無（資金高度防守）"

    header = (
        f"### 📡 美股環境濾網｜{trade_date}\n"
        f"{spy_emoji} **SPY** · {spy_state}"
        f"{rsp_line}"
        f"{vix_line}"
        f"{sector_line}"
    )
    return header, is_bull


# ==========================================
# 6. Minervini 趨勢篩選、止跌濾網、真縮量 VCP、RS Line 創高與板塊領跑評分
# ==========================================
def run_stock_screener():
    universe, ticker_sector_map = fetch_us_stock_universe()
    macro_tickers = [BENCHMARK, EQUAL_WEIGHT_BENCHMARK, VIX_TICKER, VIX3M_TICKER] + list(SECTOR_ETFS.keys())
    all_symbols = sorted(set(universe + macro_tickers))

    close_raw, high_raw, low_raw, vol_raw = download_ohlcv_in_batches(all_symbols, batch_size=50)

    if BENCHMARK not in close_raw.columns or close_raw[BENCHMARK].dropna().empty:
        raise ValueError(f"❌ 無法取得基準 {BENCHMARK} 數據。")

    raw_vix = close_raw[VIX_TICKER].dropna() if VIX_TICKER in close_raw.columns else pd.Series(dtype=float)
    raw_vix3m = close_raw[VIX3M_TICKER].dropna() if VIX3M_TICKER in close_raw.columns else pd.Series(dtype=float)

    spy_dates = close_raw[BENCHMARK].dropna().index
    close_df = close_raw.reindex(spy_dates).ffill(limit=2)
    high_df = high_raw.reindex(spy_dates).ffill(limit=2)
    low_df = low_raw.reindex(spy_dates).ffill(limit=2)
    vol_df = vol_raw.reindex(spy_dates).fillna(0.0)

    if len(close_df) < 210:
        raise ValueError(f"❌ SPY 數據不足 210 個交易日（目前僅 {len(close_df)} 筆）。")

    trade_date = spy_dates[-1].strftime('%Y-%m-%d')
    sector_states = calc_sector_rrg_states(close_df)
    macro_header, is_bull = build_macro_header(close_df, raw_vix, raw_vix3m, trade_date, sector_states)

    # 計算全體股票池多週期回報與 20 日波幅（IBD 風格權重：1M 25% + 3M 35% + 6M 25% + 9M 15%）
    daily_ret = close_df.pct_change(1, fill_method=None)
    vol_20d_daily = (daily_ret.rolling(20, min_periods=10).std().iloc[-1] * 100.0).clip(lower=0.5)

    r_1m = close_df.pct_change(21, fill_method=None).iloc[-1]
    r_3m = close_df.pct_change(63, fill_method=None).iloc[-1]
    r_6m = close_df.pct_change(126, fill_method=None).iloc[-1]
    r_9m = close_df.pct_change(189, fill_method=None).iloc[-1]

    spy_1m = float(r_1m[BENCHMARK]) if pd.notna(r_1m[BENCHMARK]) else 0.0
    spy_3m = float(r_3m[BENCHMARK]) if pd.notna(r_3m[BENCHMARK]) else 0.0
    spy_6m = float(r_6m[BENCHMARK]) if pd.notna(r_6m[BENCHMARK]) else spy_3m
    spy_9m = float(r_9m[BENCHMARK]) if pd.notna(r_9m[BENCHMARK]) else spy_6m

    # Step A：對全體有效股票計算 Risk-Adjusted RS，嚴格過濾 NaN / inf 後映射為 1–99 IBD 評分
    raw_rs_map = {}
    for t in universe:
        if t not in close_df.columns or pd.isna(close_df[t].iloc[-1]):
            continue
        if pd.isna(r_3m.get(t)):
            continue
        vol_val = vol_20d_daily.get(t)
        if pd.isna(vol_val) or not np.isfinite(vol_val) or float(vol_val) <= 0:
            continue

        e1m = float((r_1m[t] - spy_1m) * 100) if pd.notna(r_1m.get(t)) else 0.0
        e3m = float((r_3m[t] - spy_3m) * 100) if pd.notna(r_3m.get(t)) else 0.0
        e6m = float((r_6m[t] - spy_6m) * 100) if pd.notna(r_6m.get(t)) else e3m
        e9m = float((r_9m[t] - spy_9m) * 100) if pd.notna(r_9m.get(t)) else e6m

        weighted_excess = (e1m * 0.25) + (e3m * 0.35) + (e6m * 0.25) + (e9m * 0.15)
        score = weighted_excess / float(vol_val)
        if np.isfinite(score):
            raw_rs_map[t] = score

    rs_series = pd.Series(raw_rs_map, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if rs_series.empty:
        raise ValueError("❌ 錯誤：無法計算任何有效個股嘅 RS 分數，請檢查行情數據源。")

    n_valid = len(rs_series)
    asc_rank = rs_series.rank(method='average', ascending=True)
    rs_rating_map = (
        (1 + (asc_rank - 1) / max(n_valid - 1, 1) * 98)
        .round()
        .clip(1, 99)
        .astype(int)
        .to_dict()
    )

    # Step B：逐檔執行 Minervini 趨勢模板、止跌濾網、真縮量 VCP、RS Line 創高與板塊領跑打分
    passed_trend = []
    for t, raw_rs in rs_series.items():
        rs_rating = rs_rating_map.get(t, 1)
        if rs_rating < MIN_RS_RATING:
            continue

        c = close_df[t].dropna()
        if len(c) < 200:
            continue
        h = high_df[t].reindex(c.index).ffill()
        l = low_df[t].reindex(c.index).ffill()
        v = vol_df[t].reindex(c.index).fillna(0.0)

        price = float(c.iloc[-1])
        if not np.isfinite(price) or price < MIN_PRICE:
            continue

        avg_vol_5 = float(v.tail(5).mean())
        avg_vol_20 = float(v.tail(20).mean())
        avg_dollar_vol = price * avg_vol_20
        if avg_dollar_vol < MIN_AVG_DOLLAR_VOL:
            continue

        ema21 = float(c.ewm(span=21, adjust=False).mean().iloc[-1])
        sma50 = float(c.rolling(50).mean().iloc[-1])
        sma150 = float(c.rolling(150).mean().iloc[-1])
        sma200_series = c.rolling(200).mean()
        sma200 = float(sma200_series.iloc[-1])
        sma200_prev20 = float(sma200_series.iloc[-21]) if len(sma200_series) >= 220 else sma200

        if any(pd.isna(x) or not np.isfinite(x) for x in (ema21, sma50, sma150, sma200, sma200_prev20)):
            continue

        # Minervini 核心多頭趨勢模板
        if not (price > sma50 > sma150 > sma200):
            continue
        if sma200 < sma200_prev20:
            continue

        high_52w = float(h.tail(252).max())
        low_52w = float(l.tail(252).min())
        if pd.isna(high_52w) or pd.isna(low_52w) or high_52w <= 0 or low_52w <= 0:
            continue

        dist_high_pct = (high_52w - price) / high_52w * 100.0
        above_low_pct = (price - low_52w) / low_52w * 100.0

        if dist_high_pct > MAX_DIST_52W_HIGH_PCT or above_low_pct < MIN_ABOVE_52W_LOW_PCT:
            continue

        # 14 日 Wilder ATR 與 1% 單筆風險倉位計算
        prev_c = c.shift(1)
        tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
        atr_val = tr.ewm(alpha=1 / ATR_PERIOD, min_periods=ATR_PERIOD, adjust=False).mean().iloc[-1]
        if pd.isna(atr_val) or not np.isfinite(atr_val):
            continue

        atr_14 = float(atr_val)
        stop_price = max(price - ATR_MULTIPLIER * atr_14, 0.0)
        stop_pct = ((price - stop_price) / price * 100.0) if price > 0 else 8.0

        base_pos = min((RISK_PER_TRADE_PCT / max(stop_pct, 1.5)) * 100.0, MAX_SINGLE_POS_PCT)
        pos_size = base_pos if is_bull else (base_pos * 0.5)

        ext_ema21_pct = ((price - ema21) / ema21 * 100.0) if ema21 > 0 else 0.0
        rel_vol = float(v.iloc[-1] / avg_vol_20) if avg_vol_20 > 0 else 1.0
        vol_dry_ratio = float(avg_vol_5 / avg_vol_20) if avg_vol_20 > 0 else 1.0

        # VCP 波幅收斂比（近 5 日平均振幅 ÷ 近 20 日平均振幅）
        bar_range_pct = ((h - l) / c).replace([np.inf, -np.inf], np.nan)
        range_5d = bar_range_pct.tail(5).mean()
        range_20d = bar_range_pct.tail(20).mean()
        vcp_ratio = float(range_5d / range_20d) if pd.notna(range_5d) and pd.notna(range_20d) and range_20d > 0 else 1.0

        raw_chg = daily_ret[t].iloc[-1] if t in daily_ret.columns else np.nan
        chg_1d_pct = float(raw_chg * 100.0) if pd.notna(raw_chg) and np.isfinite(raw_chg) else 0.0

        # 當日收盤位置比率（衡量下影線支撐）與止跌濾網
        day_high = float(h.iloc[-1])
        day_low = float(l.iloc[-1])
        day_range = day_high - day_low
        close_pos = ((price - day_low) / day_range) if day_range > 0 else 0.5

        no_heavy_sell = (chg_1d_pct >= 0.0) or (rel_vol <= PULLBACK_MAX_DOWN_VOL)
        has_lower_wick_support = (day_range > 0) and (close_pos >= PULLBACK_MIN_CLOSE_POS)
        is_stabilized = no_heavy_sell or has_lower_wick_support

        # 主題標籤與 Top-Down 板塊聯動加減分
        theme_tag = CUSTOM_THEMATIC_STOCKS[t][0] if t in CUSTOM_THEMATIC_STOCKS else ""
        sec_etf = ticker_sector_map.get(t)
        sec_info = sector_states.get(sec_etf) if sec_etf else None
        if sec_info:
            sector_label = f"{sec_info['name']}({sec_info['etf']} {sec_info['emoji']}{sec_info['quad_zh']})"
            sector_bonus = float(sec_info['bonus'])
        else:
            sector_label = "其他行業"
            sector_bonus = 0.0

        # 真龍頭指標 1：RS Line 創 52 週新高偵測（個股收市價 ÷ SPY 收市價）
        rs_line_tag = ""
        rs_line_bonus = 0.0
        spy_aligned = close_df[BENCHMARK].reindex(c.index).ffill()
        rs_line = (c / spy_aligned).replace([np.inf, -np.inf], np.nan).dropna()
        if len(rs_line) >= 120:
            rs_window = rs_line.tail(252)
            rs_now = float(rs_window.iloc[-1])
            rs_52w_high = float(rs_window.max())
            if np.isfinite(rs_now) and np.isfinite(rs_52w_high) and rs_52w_high > 0:
                if rs_now >= rs_52w_high * RS_LINE_HIGH_TOLERANCE:
                    if dist_high_pct >= RS_BLUE_DOT_MIN_DIST:
                        rs_line_tag = "🔵RS先行創高"
                        rs_line_bonus = RS_BLUE_DOT_BONUS
                    else:
                        rs_line_tag = "⚡RS創年高"
                        rs_line_bonus = RS_NEW_HIGH_BONUS

        # 真龍頭指標 2：個股相對所屬板塊 ETF 領跑驗證（1M & 3M 雙週期）
        sector_alpha_tag = ""
        sector_alpha_bonus = 0.0
        if (
            sec_etf
            and sec_etf in r_1m.index
            and sec_etf in r_3m.index
            and pd.notna(r_1m.get(t))
            and pd.notna(r_3m.get(t))
            and pd.notna(r_1m.get(sec_etf))
            and pd.notna(r_3m.get(sec_etf))
        ):
            alpha_1m = float((r_1m[t] - r_1m[sec_etf]) * 100.0)
            alpha_3m = float((r_3m[t] - r_3m[sec_etf]) * 100.0)
            if np.isfinite(alpha_1m) and np.isfinite(alpha_3m):
                if alpha_1m > 0 and alpha_3m > 0:
                    sector_alpha_tag = f"👑跑贏板塊(3M {alpha_3m:+.0f}%)"
                    sector_alpha_bonus = SECTOR_LEADER_BONUS
                elif alpha_1m < 0 and alpha_3m < 0:
                    sector_alpha_tag = f"🐢落後板塊(3M {alpha_3m:+.0f}%)"
                    sector_alpha_bonus = SECTOR_LAGGARD_PENALTY

        # 買點型態精確判定（真縮量 VCP + EMA21 止跌共振優先，放量陰跌收低自動避開）
        in_buy_zone = (BUY_ZONE_MIN_EXT <= ext_ema21_pct <= BUY_ZONE_MAX_EXT)
        is_breakout = (dist_high_pct <= 4.0 and rel_vol >= 1.25 and chg_1d_pct > 0)
        is_vcp = (
            vcp_ratio <= VCP_RANGE_RATIO_MAX
            and vol_dry_ratio <= VCP_VOL_RATIO_MAX
            and dist_high_pct <= VCP_MAX_DIST_HIGH_PCT
        )
        in_ema21_band = (BUY_ZONE_MIN_EXT <= ext_ema21_pct <= EMA21_PULLBACK_MAX_EXT)
        is_ema21_pullback = in_ema21_band and is_stabilized

        if in_buy_zone and not (in_ema21_band and not is_stabilized):
            bucket = 'buy'
            if is_vcp and is_ema21_pullback:
                setup_tag = "💎VCP+EMA21共振"
                setup_bonus = 10.0
            elif is_breakout:
                setup_tag = "🚀放量突破"
                setup_bonus = 8.0
            elif is_vcp:
                setup_tag = "🎯VCP縮量緊湊"
                setup_bonus = 7.0
            elif is_ema21_pullback:
                if has_lower_wick_support and chg_1d_pct < 0:
                    setup_tag = "🛡️回踩EMA21(下影線)"
                else:
                    setup_tag = "🛡️回踩EMA21"
                setup_bonus = 6.0
            else:
                setup_tag = "✅健康趨勢區"
                setup_bonus = 3.0

            tightness_score = max(0.0, 10.0 - abs(ext_ema21_pct - 2.0) * 1.2)
            near_high_score = max(0.0, 10.0 - dist_high_pct * 0.5)
            composite_score = (
                (rs_rating * 0.65)
                + tightness_score
                + near_high_score
                + setup_bonus
                + sector_bonus
                + rs_line_bonus
                + sector_alpha_bonus
            )
        else:
            bucket = 'watch'
            if in_ema21_band and not is_stabilized:
                setup_tag = "⚠️放量下跌(勿接刀)"
            elif ext_ema21_pct > BUY_ZONE_MAX_EXT:
                setup_tag = "⚠️乖離偏高(等回踩)"
            else:
                setup_tag = "⏳破EMA21觀察"
            composite_score = (
                float(rs_rating)
                + (float(raw_rs) * 0.5)
                + (sector_bonus * 0.5)
                + rs_line_bonus
                + (sector_alpha_bonus * 0.5)
            )

        passed_trend.append({
            'ticker': t,
            'price': price,
            'chg_1d': chg_1d_pct,
            'rs_rating': rs_rating,
            'raw_rs': float(raw_rs),
            'ext_ema21': ext_ema21_pct,
            'dist_high': dist_high_pct,
            'rel_vol': rel_vol,
            'vol_dry_ratio': vol_dry_ratio,
            'vcp_ratio': vcp_ratio,
            'is_vcp_setup': bool(in_buy_zone and is_vcp and not (in_ema21_band and not is_stabilized)),
            'stop_price': stop_price,
            'stop_pct': stop_pct,
            'pos_size': pos_size,
            'theme_tag': theme_tag,
            'sector_label': sector_label,
            'sector_bonus': sector_bonus,
            'rs_line_tag': rs_line_tag,
            'sector_alpha_tag': sector_alpha_tag,
            'setup_tag': setup_tag,
            'bucket': bucket,
            'composite_score': composite_score
        })

    df_all = pd.DataFrame(passed_trend)
    if df_all.empty:
        buy_df = pd.DataFrame()
        watch_df = pd.DataFrame()
    else:
        buy_df = (
            df_all[df_all['bucket'] == 'buy']
            .sort_values(by=['composite_score', 'rs_rating'], ascending=False)
            .head(TOP_BUY_COUNT)
            .reset_index(drop=True)
        )
        buy_df['buy_rank'] = buy_df.index + 1

        watch_df = (
            df_all[df_all['bucket'] == 'watch']
            .sort_values(by=['composite_score', 'rs_rating'], ascending=False)
            .head(TOP_WATCH_COUNT)
            .reset_index(drop=True)
        )
        watch_df['watch_rank'] = watch_df.index + 1

    # Step C：只對最終上榜嘅最多 20 檔個股查詢財報日期（極速且防限流）
    print(f"正在檢查上榜 {len(buy_df) + len(watch_df)} 檔個股嘅近期財報避雷日期...")
    for target_df in (buy_df, watch_df):
        if not target_df.empty:
            notes, warns = [], []
            for t in target_df['ticker']:
                note, is_warn = get_earnings_warning(t)
                notes.append(note)
                warns.append(is_warn)
                time.sleep(0.15)
            target_df['earnings_note'] = notes
            target_df['earnings_warn'] = warns

    funnel_stats = {
        'universe_cnt': n_valid,
        'trend_cnt': len(df_all),
        'buy_cnt': int((df_all['bucket'] == 'buy').sum()) if not df_all.empty else 0,
        'watch_cnt': int((df_all['bucket'] == 'watch').sum()) if not df_all.empty else 0,
    }
    return buy_df, watch_df, macro_header, is_bull, trade_date, funnel_stats


# ==========================================
# 7. 手機精簡卡片排版（含 TradingView 超連結與真龍頭副行）與 Discord 推送
# ==========================================
def format_tv_link(ticker):
    """將美股代號轉成 TradingView 直開 K 線圖超連結（自動修正 BRK-B 類符號）"""
    tv_symbol = ticker.replace('-', '.')
    return f"[**{ticker}**](https://www.tradingview.com/chart/?symbol={tv_symbol})"


def format_stock_cards(df, prev_history, trade_date, is_buy_list=True):
    if df.empty:
        return "-# 今日暫無符合嚴格趨勢與買點條件嘅個股，建議保留現金耐心等待。"

    cards = []
    for idx, row in df.iterrows():
        rank = idx + 1
        t = row['ticker']

        past_recs = [r for r in prev_history.get(t, {}).get('records', []) if r.get('date') != trade_date]
        if not past_recs or past_recs[-1].get('rank') is None:
            badge_hist = "🆕"
        else:
            prev_r = int(past_recs[-1]['rank'])
            diff = prev_r - rank
            if diff > 0:
                badge_hist = f"▲{diff}"
            elif diff < 0:
                badge_hist = f"▼{abs(diff)}"
            else:
                badge_hist = "─"

        lead_emoji = "🚨" if row.get('earnings_warn') else ("🟢" if is_buy_list else "🟡")
        earn_str = f" · {row['earnings_note']}" if row.get('earnings_note') else ""

        bonus_val = float(row.get('sector_bonus', 0.0))
        bonus_txt = f" {bonus_val:+.0f}分" if bonus_val != 0 else ""
        sec_str = f"{row.get('sector_label', '其他行業')}{bonus_txt}"
        theme_tag = row.get('theme_tag', '')
        tag_prefix = f"🏷️{theme_tag}｜" if theme_tag else "板塊："

        line_1 = (
            f"{lead_emoji} **#{rank:02d}** {format_tv_link(t)} "
            f"`${row['price']:.2f}` ({row['chg_1d']:+.1f}%) `RS {int(row['rs_rating']):02d}` {badge_hist}"
        )
        line_2 = f"-#  {tag_prefix}{sec_str}"

        leader_parts = [
            tag for tag in (row.get('rs_line_tag', ''), row.get('sector_alpha_tag', ''))
            if tag
        ]
        line_leader = f"-#  {' · '.join(leader_parts)}" if leader_parts else ""

        if row.get('is_vcp_setup'):
            line_3 = (
                f"-#  {row['setup_tag']} · EMA21 {row['ext_ema21']:+.1f}% · "
                f"年高 -{row['dist_high']:.0f}% · 5D量 {row['vol_dry_ratio']:.2f}x"
            )
        else:
            line_3 = (
                f"-#  {row['setup_tag']} · EMA21 {row['ext_ema21']:+.1f}% · "
                f"年高 -{row['dist_high']:.1f}% · 量比 {row['rel_vol']:.1f}x"
            )

        line_4 = (
            f"-#  止損 ${row['stop_price']:.2f} (-{row['stop_pct']:.1f}%) · "
            f"1%風險倉 {row['pos_size']:.1f}%{earn_str}"
        )

        card_lines = [line_1, line_2]
        if line_leader:
            card_lines.append(line_leader)
        card_lines.extend([line_3, line_4])
        cards.append("\n".join(card_lines))

    return "\n".join(cards)


def send_single_embed(webhook_url, embed, label):
    payload = {"embeds": [embed]}
    for attempt in range(2):
        resp = requests.post(webhook_url, json=payload, timeout=30)
        if resp.status_code in (200, 204):
            print(f"✅ 成功發送：{label}")
            return
        elif resp.status_code == 429 and attempt == 0:
            wait = float(resp.json().get("retry_after", 2.0))
            time.sleep(wait + 0.5)
        else:
            raise RuntimeError(f"❌ [{label}] 發送失敗，狀態碼：{resp.status_code}, 回應：{resp.text}")


def send_to_discord(buy_df, watch_df, macro_header, is_bull, trade_date, funnel_stats, prev_history):
    webhook_url = os.environ.get('DISCORD_WEBHOOK_URL', '').strip()
    if not webhook_url:
        raise RuntimeError("❌ 錯誤：未設定 DISCORD_WEBHOOK_URL 環境變數。")

    funnel_block = (
        f"### 🔍 強勢個股漏斗統計\n"
        f"掃描 `{funnel_stats['universe_cnt']}` 檔 ➔ 多頭模板 `{funnel_stats['trend_cnt']}` 檔 ➔ "
        f"🎯買點區 `{funnel_stats['buy_cnt']}` 檔｜⚠️偏高／觀察 `{funnel_stats['watch_cnt']}` 檔"
    )
    if not is_bull:
        funnel_block += "\n-# ⚠️ SPY 跌破 200MA 空頭警戒：下方單股建議倉位已自動減半"

    buy_cards = format_stock_cards(buy_df, prev_history, trade_date, is_buy_list=True)
    watch_cards = format_stock_cards(watch_df, prev_history, trade_date, is_buy_list=False)
    now_iso = datetime.now(timezone.utc).isoformat()

    embeds = [
        (
            "Embed 1/2 優先買入推薦",
            {
                "title": f"🎯 [1/2] 美股強勢個股買入推薦 Top {len(buy_df)}",
                "description": (
                    f"{macro_header}\n\n"
                    f"{funnel_block}\n\n"
                    f"### 🟢 優先買點排名（RS創高 / 板塊龍頭 / VCP共振）\n"
                    f"{buy_cards}"
                ),
                "color": 3066993
            }
        ),
        (
            "Embed 2/2 觀察與防追高名單",
            {
                "title": f"⚠️ [2/2] 強勢龍頭觀察名單 Top {len(watch_df)}（乖離偏高 / 止跌觀察）",
                "description": (
                    f"-# 以下個股 RS 評分極高但已偏離 EMA21 超過 {BUY_ZONE_MAX_EXT:g}% 或處於放量下跌／破線觀察，建議等企穩先買\n\n"
                    f"{watch_cards}"
                ),
                "color": 15844367,
                "footer": {
                    "text": f"美股數據日：{trade_date} • RS線創高+跑贏板塊+真縮量VCP • 1%風險定倉({ATR_MULTIPLIER:g}xATR)"
                },
                "timestamp": now_iso
            }
        )
    ]

    for idx, (label, embed) in enumerate(embeds):
        send_single_embed(webhook_url, embed, label)
        if idx < len(embeds) - 1:
            time.sleep(0.8)


# ==========================================
# 8. 主程式入口
# ==========================================
if __name__ == "__main__":
    prev_history = load_previous_history()
    buy_df, watch_df, macro_header, is_bull, trade_date, funnel_stats = run_stock_screener()
    send_to_discord(buy_df, watch_df, macro_header, is_bull, trade_date, funnel_stats, prev_history)
    save_current_history(buy_df, watch_df, trade_date)
