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
# 1. 核心選股與風控參數（可按交易風格微調）
# ==========================================
BENCHMARK = 'SPY'
EQUAL_WEIGHT_BENCHMARK = 'RSP'
VIX_TICKER = '^VIX'
VIX3M_TICKER = '^VIX3M'
HISTORY_FILE = 'stock_history.json'

# 篩選門檻
MIN_PRICE = 10.0                  # 最低股價（過濾仙股）
MIN_AVG_DOLLAR_VOL = 50_000_000   # 20 日均成交額至少 5,000 萬美元（確保流動性）
MIN_RS_RATING = 80                # 只推薦全市場 RS 80 分以上強勢股
MAX_DIST_52W_HIGH_PCT = 20.0      # 距離 52 週高點唔超過 20%
MIN_ABOVE_52W_LOW_PCT = 25.0      # 較 52 週低點已反彈至少 25%

# 防追高買點區間（相對 EMA21 乖離率 %）
BUY_ZONE_MIN_EXT = -3.0           # 允許輕微回踩 EMA21 下方 3%
BUY_ZONE_MAX_EXT = 8.0            # 距離 EMA21 8% 以內屬於安全可買區；超過 8% 撥入「乖離偏高觀察」

# 單筆風險倉位與止損參數
RISK_PER_TRADE_PCT = 1.0          # 每筆交易最大承受總資金虧損 1.0% (1R)
MAX_SINGLE_POS_PCT = 15.0         # 單隻個股最大倉位上限 15%
ATR_PERIOD = 14                   # 14 日 Wilder ATR
ATR_MULTIPLIER = 2.0              # 2 倍 ATR 止損
EARNINGS_WARN_DAYS = 7            # 財報倒數 7 天內亮紅燈避雷
TOP_BUY_COUNT = 10                # 優先推薦買入前 10 檔
TOP_WATCH_COUNT = 10              # 強勢但乖離偏高觀察前 10 檔

# 備用高流動性龍頭池（僅當 Wikipedia 暫時無法連線時自動啟用）
FALLBACK_TICKERS = [
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'AVGO', 'TSLA', 'LLY', 'JPM',
    'V', 'MA', 'UNH', 'XOM', 'COST', 'HD', 'PG', 'NFLX', 'AMD', 'CRM',
    'ORCL', 'ADBE', 'CSCO', 'ACN', 'LIN', 'TMO', 'ABT', 'MRK', 'PEP', 'KO',
    'WMT', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'AXP', 'SPGI', 'CAT', 'GE',
    'RTX', 'LMT', 'HON', 'UNP', 'ETN', 'UBER', 'ISRG', 'SYK', 'VRTX', 'REGN',
    'AMAT', 'LRCX', 'KLAC', 'MU', 'ANET', 'PANW', 'CRWD', 'PLTR', 'APP', 'NOW',
    'INTU', 'SNPS', 'CDNS', 'MELI', 'CEG', 'VST', 'GEV', 'AXON', 'FICO', 'SPOT',
    'TSM', 'ASML', 'ARM', 'MRVL', 'APH', 'PWR', 'URI', 'PH', 'TT', 'FIX'
]


# ==========================================
# 2. 動態獲取美股主流股票池 (S&P 500 + Nasdaq 100)
# ==========================================
def fetch_us_stock_universe():
    """從 Wikipedia 動態抓取 S&P 500 + Nasdaq 100 成分股，失敗時無縫退回備用池"""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; StockRSBot/1.0)"}
    tickers_set = set()

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
                match_col = next((c for c in cols if col_hint.lower() in c.lower() or "symbol" in c.lower()), None)
                if match_col and len(tbl) >= 80:
                    raw_syms = tbl[match_col].dropna().astype(str).tolist()
                    for sym in raw_syms:
                        clean = sym.strip().upper().replace('.', '-')
                        if clean and len(clean) <= 6 and clean.replace('-', '').isalpha():
                            tickers_set.add(clean)
                    break
        except Exception as e:
            print(f"⚠️ 抓取成分股來源失敗 ({url}): {e}")

    if len(tickers_set) < 100:
        print("⚠️ 動態股票池不足，改用內置備用高流動性龍頭池。")
        tickers_set.update(FALLBACK_TICKERS)

    # 移除雙重類別重複股（保留流動性較佳者）
    for dup in ('GOOG', 'FOXA', 'NWS'):
        tickers_set.discard(dup)

    universe = sorted(tickers_set)
    print(f"✅ 成功載入美股監控池：共 {len(universe)} 檔個股")
    return universe


# ==========================================
# 3. 歷史上榜紀錄處理（按美股交易日去重，追蹤新上榜與排名變化）
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
# 4. 分批安全下載行情（防 SQLite 鎖死＋自動單線程補抓）
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
        # 先移除全空失敗欄位，合併後保留最新有效補抓結果
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
            # 限制最多 4 條並發線程，避開 yfinance 內部 SQLite 快取鎖死
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

    # 偵測漏抓或全空標的（例如曾報 database is locked 嘅個股），用單線程安全補抓一次
    missing_syms = [
        t for t in ticker_list
        if t not in close_df.columns or close_df[t].dropna().empty
    ]
    if missing_syms and len(missing_syms) <= 30:
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


def build_macro_header(close_df, raw_vix, raw_vix3m, trade_date):
    """精簡版頂部大市濾網（SPY / RSP / VIX / VIX3M）"""
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

    header = (
        f"### 📡 美股環境濾網｜{trade_date}\n"
        f"{spy_emoji} **SPY** · {spy_state}"
        f"{rsp_line}"
        f"{vix_line}"
    )
    return header, is_bull


# ==========================================
# 5. Minervini 趨勢篩選、IBD 1–99 RS 與買點綜合評分
# ==========================================
def run_stock_screener():
    universe = fetch_us_stock_universe()
    macro_tickers = [BENCHMARK, EQUAL_WEIGHT_BENCHMARK, VIX_TICKER, VIX3M_TICKER]
    all_symbols = sorted(set(universe + macro_tickers))

    close_raw, high_raw, low_raw, vol_raw = download_ohlcv_in_batches(all_symbols, batch_size=50)

    if BENCHMARK not in close_raw.columns or close_raw[BENCHMARK].dropna().empty:
        raise ValueError(f"❌ 無法取得基準 {BENCHMARK} 數據。")

    raw_vix = close_raw[VIX_TICKER].dropna() if VIX_TICKER in close_raw.columns else pd.Series(dtype=float)
    raw_vix3m = close_raw[VIX3M_TICKER].dropna() if VIX3M_TICKER in close_raw.columns else pd.Series(dtype=float)

    # 鎖定對齊 SPY 交易日
    spy_dates = close_raw[BENCHMARK].dropna().index
    close_df = close_raw.reindex(spy_dates).ffill(limit=2)
    high_df = high_raw.reindex(spy_dates).ffill(limit=2)
    low_df = low_raw.reindex(spy_dates).ffill(limit=2)
    vol_df = vol_raw.reindex(spy_dates).fillna(0.0)

    if len(close_df) < 210:
        raise ValueError(f"❌ SPY 數據不足 210 個交易日（目前僅 {len(close_df)} 筆）。")

    trade_date = spy_dates[-1].strftime('%Y-%m-%d')
    macro_header, is_bull = build_macro_header(close_df, raw_vix, raw_vix3m, trade_date)

    # 計算全體股票池多週期回報與 20 日波幅（IBD 風格中短期動能權重：1M 25% + 3M 35% + 6M 25% + 9M 15%）
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

    # Step B：逐檔執行 Minervini 趨勢模板、流動性、量價收斂與買點打分
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

        # 1% 風險定倉：倉位 = 1.0% / 止損幅度(%)，空頭市自動減半
        base_pos = min((RISK_PER_TRADE_PCT / max(stop_pct, 1.5)) * 100.0, MAX_SINGLE_POS_PCT)
        pos_size = base_pos if is_bull else (base_pos * 0.5)

        # 乖離率、量比、VCP 波幅收斂比（近 5 日平均振幅 ÷ 近 20 日平均振幅）
        ext_ema21_pct = ((price - ema21) / ema21 * 100.0) if ema21 > 0 else 0.0
        rel_vol = float(v.iloc[-1] / avg_vol_20) if avg_vol_20 > 0 else 1.0

        bar_range_pct = ((h - l) / c).replace([np.inf, -np.inf], np.nan)
        range_5d = bar_range_pct.tail(5).mean()
        range_20d = bar_range_pct.tail(20).mean()
        vcp_ratio = float(range_5d / range_20d) if pd.notna(range_5d) and pd.notna(range_20d) and range_20d > 0 else 1.0

        raw_chg = daily_ret[t].iloc[-1] if t in daily_ret.columns else np.nan
        chg_1d_pct = float(raw_chg * 100.0) if pd.notna(raw_chg) and np.isfinite(raw_chg) else 0.0

        # 買點型態分類與買點綜合評分（結合 RS + 緊湊度 + 貼近 52W 高點 + 量能確認）
        in_buy_zone = (BUY_ZONE_MIN_EXT <= ext_ema21_pct <= BUY_ZONE_MAX_EXT)
        if in_buy_zone:
            bucket = 'buy'
            if dist_high_pct <= 4.0 and rel_vol >= 1.25 and chg_1d_pct > 0:
                setup_tag = "🚀放量突破"
                setup_bonus = 8.0
            elif vcp_ratio <= 0.78 and dist_high_pct <= 10.0:
                setup_tag = "🎯VCP縮量緊湊"
                setup_bonus = 7.0
            elif ext_ema21_pct <= 2.5:
                setup_tag = "🛡️回踩EMA21"
                setup_bonus = 6.0
            else:
                setup_tag = "✅健康趨勢區"
                setup_bonus = 3.0

            tightness_score = max(0.0, 10.0 - abs(ext_ema21_pct - 2.0) * 1.2)
            near_high_score = max(0.0, 10.0 - dist_high_pct * 0.5)
            composite_score = (rs_rating * 0.65) + tightness_score + near_high_score + setup_bonus
        else:
            bucket = 'watch'
            setup_tag = "⚠️乖離偏高(等回踩)" if ext_ema21_pct > BUY_ZONE_MAX_EXT else "⏳短線回調觀察"
            composite_score = float(rs_rating) + (float(raw_rs) * 0.5)

        passed_trend.append({
            'ticker': t,
            'price': price,
            'chg_1d': chg_1d_pct,
            'rs_rating': rs_rating,
            'raw_rs': float(raw_rs),
            'ext_ema21': ext_ema21_pct,
            'dist_high': dist_high_pct,
            'rel_vol': rel_vol,
            'vcp_ratio': vcp_ratio,
            'stop_price': stop_price,
            'stop_pct': stop_pct,
            'pos_size': pos_size,
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
            .sort_values(by=['rs_rating', 'raw_rs'], ascending=False)
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
# 6. 手機三層清晰排版與 Discord 雙 Embed 推送
# ==========================================
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
        vol_str = f"量比 {row['rel_vol']:.1f}x"

        line_1 = f"{lead_emoji} **#{rank:02d} {t}** `${row['price']:.2f}` ({row['chg_1d']:+.1f}%) `RS {int(row['rs_rating']):02d}` {badge_hist}"
        line_2 = (
            f"-#  {row['setup_tag']} · 距EMA21 {row['ext_ema21']:+.1f}% · "
            f"距年高 -{row['dist_high']:.1f}% · {vol_str}"
        )
        line_3 = (
            f"-#  止損 ${row['stop_price']:.2f} (-{row['stop_pct']:.1f}%) · "
            f"1%風險建議倉 {row['pos_size']:.1f}%{earn_str}"
        )
        cards.append(f"{line_1}\n{line_2}\n{line_3}")

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
        f"🎯買點區 `{funnel_stats['buy_cnt']}` 檔｜⚠️偏高 `{funnel_stats['watch_cnt']}` 檔"
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
                    f"### 🟢 優先買點排名（貼近 EMA21 / VCP 縮量 / 放量突破）\n"
                    f"{buy_cards}"
                ),
                "color": 3066993
            }
        ),
        (
            "Embed 2/2 乖離偏高觀察名單",
            {
                "title": f"⚠️ [2/2] 強勢龍頭觀察名單 Top {len(watch_df)}（乖離偏高，等回踩）",
                "description": (
                    f"-# 以下個股 RS 評分極高但已偏離 EMA21 超過 {BUY_ZONE_MAX_EXT:g}%，建議加入自選等縮量回踩先買\n\n"
                    f"{watch_cards}"
                ),
                "color": 15844367,
                "footer": {
                    "text": f"美股數據日：{trade_date} • Minervini模板+IBD RS • 1%風險定倉({ATR_MULTIPLIER:g}xATR) • 🚨財報7日內勿重倉"
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
# 7. 主程式入口
# ==========================================
if __name__ == "__main__":
    prev_history = load_previous_history()
    buy_df, watch_df, macro_header, is_bull, trade_date, funnel_stats = run_stock_screener()
    send_to_discord(buy_df, watch_df, macro_header, is_bull, trade_date, funnel_stats, prev_history)
    save_current_history(buy_df, watch_df, trade_date)
