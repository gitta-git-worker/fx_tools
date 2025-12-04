import os
import json
from datetime import datetime

import requests
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import plotly.express as px

# ==============================
# 設定
# ==============================

API_URL = "https://open.er-api.com/v6/latest"

INSTRUMENTS = [
    "USDJPY", "EURJPY", "GBPJPY", "NZDJPY",
    "EURUSD", "GBPUSD", "NZDUSD",
    "EURNZD",  
##   "GBPNZD", "AUDJPY", "AUDUSD",
]

CONFIG_FILE = "pair_config.json"


# ==============================
# ユーティリティ
# ==============================

FINNHUB_TOKEN = "d4nui4hr01qk2nue1uk0d4nui4hr01qk2nue1ukg"

def get_price(pair: str) -> float:
    """
    レート取得の優先順位:
    1. Twelve Data (リアルタイム / 推奨)
    2. Finnhub (バックアップ)
    3. open.er-api.com (従来の1日1回更新)
    """
    base = pair[:3]
    quote = pair[3:]

    # ============ 1) Twelve Data ============
    td_key = st.session_state.get("td_key", "")
    if td_key:
        try:
            url = "https://api.twelvedata.com/price"
            params = {"symbol": f"{base}/{quote}", "apikey": td_key}
            data = requests.get(url, params=params, timeout=6).json()
            # 正常時: {"price": "150.1234", "symbol": "USD/JPY", ...}
            if "price" in data:
                return float(data["price"])
            # エラーレスポンスだった場合はログだけ残して次へ
            st.warning(f"{pair} TwelveDataエラー: {data}")
        except Exception as e:
            st.warning(f"{pair} TwelveData取得エラー: {e}")

    # ============ 2) Finnhub ============
    fh_key = st.session_state.get("fh_key", "")
    if fh_key:
        for broker in ["OANDA", "FOREXCOM"]:
            try:
                sym = f"{broker}:{base}_{quote}"  # 例: FOREXCOM:USD_JPY
                url = "https://finnhub.io/api/v1/quote"
                params = {"symbol": sym, "token": fh_key}
                data = requests.get(url, params=params, timeout=6).json()
                c = float(data.get("c") or 0)
                if c != 0:
                    return c
            except Exception as e:
                st.warning(f"{pair} Finnhub({broker})取得エラー: {e}")
        # ここまで来たら Finnhub でも有効レートはなかった

    # ============ 3) 従来の open.er-api (日次レート) ============
    try:
        resp = requests.get(f"{API_URL}/{base}", timeout=6)
        jd = resp.json()
        if "rates" in jd and quote in jd["rates"]:
            return float(jd["rates"][quote])
        st.warning(f"{pair} open.er-api レート無し: {jd}")
    except Exception as e:
        st.warning(f"{pair} open.er-api取得エラー: {e}")

    # すべて失敗した場合
    raise RuntimeError("no_valid_price")



def get_pip_factor(pair: str) -> float:
    return 0.01 if "JPY" in pair else 0.0001


def default_config(pair: str) -> dict:
    return {
        "enabled": True,
        "period_min": 15,  # 15分足
        "lookback_n": 20,
        "body_min_pips": 15.0 if "JPY" in pair else 8.0,
        "wick_ratio_max": 2.0,
    }


def load_config() -> dict:
    if not os.path.exists(CONFIG_FILE):
        return {p: default_config(p) for p in INSTRUMENTS}
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)
    cfg = {}
    for p in INSTRUMENTS:
        cfg[p] = raw.get(p, default_config(p))
    return cfg


def save_config(cfg: dict) -> None:
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def update_candles(df: pd.DataFrame, pair: str, period_min: int) -> pd.DataFrame:
    """現在レートから簡易ローソク（period_min分足）を生成"""
    try:
        price = get_price(pair)
    except Exception as e:
        st.warning(f"{pair} 価格取得エラー: {e}")
        return df

    now = datetime.now()

    if df.empty:
        return pd.DataFrame([{
            "time": now,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
        }])

    last_time = df.iloc[-1]["time"]
    mins = (now - last_time).total_seconds() / 60

    if mins < period_min:
        # 同じ足の更新
        df.loc[df.index[-1], "high"] = max(df.iloc[-1]["high"], price)
        df.loc[df.index[-1], "low"] = min(df.iloc[-1]["low"], price)
        df.loc[df.index[-1], "close"] = price
    else:
        # 新しい足
        new_row = {
            "time": now,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    return df.tail(400)


def add_ema(df: pd.DataFrame, span: int, col: str) -> pd.DataFrame:
    df[col] = df["close"].ewm(span=span, adjust=False).mean()
    return df


# ==============================
# シグナル判定 + RR + 強さ
# ==============================

def detect_signal(df: pd.DataFrame, pair: str, cfg: dict) -> dict:
    """
    直近1本前の足からシグナル判定。
    必ず dict を返す（NONE でも）ので、呼び出し側は安全。
    """
    price_now = df.iloc[-1]["close"]

    # 足不足
    if len(df) < cfg["lookback_n"] + 2:
        return {
            "pair": pair,
            "time": None,
            "trend": "FLAT",
            "strength": "☆☆☆☆☆",
            "score": 0,
            "type": "NONE",
            "price": price_now,
            "entry": None,
            "sl": None,
            "tp": None,
            "rr": None,
        }

    pip = get_pip_factor(pair)

    df = add_ema(df.copy(), 20, "ema20")
    df = add_ema(df, 50, "ema50")

    sig = df.iloc[-2]  # シグナル候補足
    prev = df.iloc[-(cfg["lookback_n"] + 2):-2]

    o, h, l, c = sig["open"], sig["high"], sig["low"], sig["close"]
    ema20, ema50 = sig["ema20"], sig["ema50"]

    # トレンド
    if c > ema20 and ema20 > ema50:
        trend = "UP"
    elif c < ema20 and ema20 < ema50:
        trend = "DOWN"
    else:
        trend = "FLAT"

    body = abs(c - o)
    if body == 0:
        body_pips = 0.0
        upper_ratio = 999.0
        lower_ratio = 999.0
    else:
        body_pips = body / pip
        upper_ratio = (h - max(o, c)) / body
        lower_ratio = (min(o, c) - l) / body

    high_prev = prev["high"].max()
    low_prev = prev["low"].min()

    # ---------------------
    # スコアリング
    # ---------------------
    score = 0

    # 実体
    if body_pips >= cfg["body_min_pips"] * 0.8:
        score += 2
    if body_pips >= cfg["body_min_pips"] * 1.3:
        score += 3

    # ヒゲ
    if upper_ratio <= cfg["wick_ratio_max"]:
        score += 1
    if lower_ratio <= cfg["wick_ratio_max"]:
        score += 1

    # ブレイク
    if trend == "UP" and c > high_prev:
        score += 3
    if trend == "DOWN" and c < low_prev:
        score += 3

    star_level = min(5, max(1, score // 2))
    strength = "★" * star_level + "☆" * (5 - star_level)

    # ---------------------
    # エントリー条件 & RR
    # ---------------------
    sig_type = "NONE"
    entry = None
    sl = None
    tp = None
    rr = None

    if trend == "UP" and c > high_prev and body_pips >= cfg["body_min_pips"]:
        sig_type = "BUY"
        entry = c
        sl = l - 2 * pip
        risk = (entry - sl) / pip
        tp = entry + risk * 2 * pip
        if risk > 0:
            rr = (tp - entry) / (entry - sl)
    elif trend == "DOWN" and c < low_prev and body_pips >= cfg["body_min_pips"]:
        sig_type = "SELL"
        entry = c
        sl = h + 2 * pip
        risk = (sl - entry) / pip
        tp = entry - risk * 2 * pip
        if risk > 0:
            rr = (entry - tp) / (sl - entry)

    return {
        "pair": pair,
        "time": sig["time"],
        "trend": trend,
        "strength": strength,
        "score": score,
        "type": sig_type,
        "price": price_now,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "rr": rr,
    }


# ==============================
# 通知関連
# ==============================

def send_line_notify(token: str, message: str):
    url = "https://notify-api.line.me/api/notify"
    headers = {"Authorization": f"Bearer {token}"}
    data = {"message": message}
    try:
        requests.post(url, headers=headers, data=data, timeout=5)
    except Exception as e:
        st.warning(f"LINE通知エラー: {e}")


def send_discord(webhook_url: str, message: str):
    data = {"content": message}
    try:
        requests.post(webhook_url, json=data, timeout=5)
    except Exception as e:
        st.warning(f"Discord通知エラー: {e}")


def maybe_notify(sig: dict,
                 line_token: str,
                 discord_webhook: str,
                 dashboard_url: str,
                 enabled: bool):
    """RR>=2.0 かつ BUY/SELL のときだけ通知。足ごとに1回のみ。"""
    if not enabled:
        return
    if sig["type"] not in ("BUY", "SELL"):
        return
    if sig["rr"] is None or sig["rr"] < 2.0:
        return
    if sig["time"] is None:
        return

    # すでに同じ足で通知済みならスキップ
    key = f"{sig['pair']}::{sig['time']}"
    if "notified" not in st.session_state:
        st.session_state.notified = set()
    if key in st.session_state.notified:
        return

    pair = sig["pair"]
    t = sig["type"]
    price = sig["price"]
    strength = sig["strength"]
    rr = sig["rr"]
    entry = sig["entry"]
    sl = sig["sl"]
    tp = sig["tp"]

    msg = (
        f"📢 FXシグナル\n"
        f"Pair: {pair}  {t}\n"
        f"Price: {price:.3f}\n"
        f"Entry: {entry:.3f} / SL: {sl:.3f} / TP: {tp:.3f}\n"
        f"RR: {rr:.2f}  Strength: {strength}\n"
    )
    if dashboard_url:
        msg += f"Dashboard: {dashboard_url}"

    if line_token:
        send_line_notify(line_token, msg)
    if discord_webhook:
        send_discord(discord_webhook, msg)

    st.session_state.notified.add(key)


# ==============================
# Streamlit UI
# ==============================

st.set_page_config(page_title="FX Multi Pair Dashboard", layout="wide")
st.title("📈 FX Multi Pair Dashboard — エントリーシグナル監視")

cfg = load_config()

# 自動更新（分単位）
default_min = list(cfg.values())[0]["period_min"] if cfg else 15
interval_min = st.sidebar.slider("更新間隔（分）", 1, 60, default_min)
st_autorefresh(interval=interval_min * 60_000, key="refresh")

# レートAPI設定を追加 👇 ここを新しく入れる
st.sidebar.markdown("### 🔑 レートAPI設定")
td_key_input = st.sidebar.text_input(
    "Twelve Data API Key（推奨・リアルタイム）",
    value="",
    type="password",
    help="https://twelvedata.com/ で無料APIキーを取得して入力"
)
fh_key_input = st.sidebar.text_input(
    "Finnhub API Key（任意・バックアップ用）",
    value="",
    type="password",
    help="FXが有効な場合のみ使用"
)

# セッション状態に保存して get_price から参照できるようにする
st.session_state["td_key"] = td_key_input
st.session_state["fh_key"] = fh_key_input

# セッションに前回価格を保存する dict を用意
if "prev_prices" not in st.session_state:
    st.session_state["prev_prices"] = {}

# 通知設定（既存のものはこのまま）
st.sidebar.markdown("### 🔔 通知設定")
notify_enabled = st.sidebar.checkbox("通知を有効にする", value=False)
line_token = st.sidebar.text_input("LINE Notify トークン（任意）", type="password")
discord_webhook = st.sidebar.text_input("Discord Webhook URL（任意）")
dashboard_url = st.sidebar.text_input(
    "ダッシュボードURL（通知用・任意）",
    value="",
    help="例: http://192.168.0.xx:8501"
)

# セッション初期化
if "candles" not in st.session_state:
    st.session_state.candles = {p: pd.DataFrame() for p in INSTRUMENTS}
if "latest" not in st.session_state:
    st.session_state.latest = {}

candles = st.session_state.candles
latest = st.session_state.latest

# ==========================
# 全通貨チェック
# ==========================
rows = []
for pair in INSTRUMENTS:
    if not cfg[pair]["enabled"]:
        continue
    df_pair = candles[pair]
    df_pair = update_candles(df_pair, pair, cfg[pair]["period_min"])
    candles[pair] = df_pair

    if df_pair.empty:
        continue

    sig = detect_signal(df_pair, pair, cfg[pair])
    latest[pair] = sig
    rows.append(sig)

    # 通知条件を満たしていれば通知
    maybe_notify(sig, line_token, discord_webhook, dashboard_url, notify_enabled)

st.session_state.candles = candles
st.session_state.latest = latest

df_table = pd.DataFrame(rows)

if df_table.empty:
    st.warning("まだデータが少ないため、シグナルは表示されていません。少し待つと足がたまってきます。")
    st.stop()

# トレンドアイコン
def trend_to_icon(t: str) -> str:
    if t == "UP":
        return "🟢⬆ 上昇"
    if t == "DOWN":
        return "🔴⬇ 下降"
    return "⚪ー レンジ"

df_table["trend_icon"] = df_table["trend"].map(trend_to_icon)
df_table["rr"] = df_table["rr"].apply(lambda x: round(x, 2) if isinstance(x, (int, float)) else None)

# 強さ順ソート
df_table = df_table.sort_values("score", ascending=False)


arrows = []
for pair, price in zip(df_table["pair"], df_table["price"]):
    prev = st.session_state["prev_prices"].get(pair)

    if prev is None:
        arrow = "→"   # 初回は方向なし
    else:
        if price > prev:
            arrow = "🟢 ↑"
        elif price < prev:
            arrow = "🔻 ↓"
        else:
            arrow = "→"

    arrows.append(arrow)

    # 保存（次回比較用）
    st.session_state["prev_prices"][pair] = price

df_table["arrow"] = arrow


# 表示用
disp = df_table[["pair", "trend_icon", "type", "price", "arrow", "entry", "sl", "tp", "rr", "strength"]].copy()


# ---- 表示だけ丸め（内部計算は丸めない）----
for col in ["price", "entry", "sl", "tp"]:
    disp[col] = disp[col].astype(float).round(3)

disp = disp.rename(columns={
    "pair": "通貨ペア",
    "trend_icon": "トレンド",
    "type": "シグナル",
    "price": "現在値",
    "arrow": "変動",
    "entry": "エントリー",
    "sl": "SL",
    "tp": "TP",
    "rr": "RR(利:損)",
    "strength": "強さ"
})

# 行の色付け
def highlight_signal(row):
    sig = row["シグナル"]
    rr = row["RR(利:損)"]
    base_color = ""
    if sig == "BUY":
        # RRが高いほど濃く
        if isinstance(rr, (int, float)) and rr >= 2.5:
            base_color = "#b3ffb3"  # 少し濃い緑
        else:
            base_color = "#e6ffe6"  # 薄めの緑
    elif sig == "SELL":
        if isinstance(rr, (int, float)) and rr >= 2.5:
            base_color = "#ffb3b3"  # 濃い赤
        else:
            base_color = "#ffe6e6"  # 薄めの赤
    else:
        base_color = "#f6f6f6"  # NONEは薄グレー
    return [f"background-color: {base_color}" for _ in row]

styled = disp.style.apply(highlight_signal, axis=1)

st.subheader("📊 通貨ペア一覧（強さ順）")
st.caption(f"⏱ 最終更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.dataframe(styled, use_container_width=True, height=420)





# ==========================
# 個別チャート
# ==========================
st.subheader("📈 チャート表示する通貨")

selected_pair = st.selectbox("通貨を選択", df_table["pair"].tolist())
df_sel = candles[selected_pair]

if not df_sel.empty:
    fig = px.line(df_sel, x="time", y="close", title=f"{selected_pair} 15分足（簡易）")
    st.plotly_chart(fig, use_container_width=True)

    info = latest[selected_pair]
    st.write(f"シグナル: **{info['type']}** / トレンド: **{info['trend']}** / 強さ: **{info['strength']}**")
    if info["type"] != "NONE":
        rr_str = f"{info['rr']:.2f}" if info["rr"] is not None else "-"
        st.write(
            f"- エントリー: **{info['entry']:.3f}**  / SL: **{info['sl']:.3f}**  / "
            f"TP: **{info['tp']:.3f}**  / RR: **{rr_str}**"
        )

# ==========================
# パラメータ編集
# ==========================
st.subheader("⚙ 通貨ペア別パラメータ編集")

cfg_df = pd.DataFrame.from_dict(cfg, orient="index")
cfg_df.index.name = "pair"
edited = st.data_editor(cfg_df, use_container_width=True)

if st.button("設定を保存", type="primary"):
    new_cfg = {}
    for pair, row in edited.iterrows():
        new_cfg[pair] = {
            "enabled": bool(row.get("enabled", True)),
            "period_min": int(row.get("period_min", 15)),
            "lookback_n": int(row.get("lookback_n", 20)),
            "body_min_pips": float(row.get("body_min_pips", 10.0)),
            "wick_ratio_max": float(row.get("wick_ratio_max", 2.0)),
        }
    save_config(new_cfg)
    st.success("設定を保存しました ✔（再読み込みで反映されます）")
