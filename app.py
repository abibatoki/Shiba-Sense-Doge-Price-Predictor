import os, json, io, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go  # for interactive charts
from datetime import datetime, timezone
from typing import Tuple

import gradio as gr
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# ---- TensorFlow/Keras (inference only) ----
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

# ---------- Paths ----------
ARTIFACT_DIR = "artifacts"
MODEL_WEIGHTS = os.path.join(ARTIFACT_DIR, "best.weights.h5")
SCALER_PATH   = os.path.join(ARTIFACT_DIR, "feature_scaler.joblib")
META_PATH     = os.path.join(ARTIFACT_DIR, "metadata.json")

# ---------- Load metadata & scaler ----------
with open(META_PATH, "r") as f:
    META = json.load(f)

SEQ_LEN     = int(META.get("splits", {}).get("seq_len", 120))
D_MODEL     = int(META.get("model_config", {}).get("d_model", 96))
NUM_HEADS   = int(META.get("model_config", {}).get("num_heads", 4))
NUM_LAYERS  = int(META.get("model_config", {}).get("num_layers", 2))
FF_DIM      = int(META.get("model_config", {}).get("ff_dim", 192))
DROPOUT     = float(META.get("model_config", {}).get("dropout", 0.25))
L2_REG      = float(META.get("model_config", {}).get("l2_reg", 1e-4))
CLIPNORM    = float(META.get("training", {}).get("clipnorm", 1.0))
EMA_BASELINE = int(META.get("baselines", {}).get("ema_span", 10))  # falls back to 10


SCALER = joblib.load(SCALER_PATH)

# ---------- Helpers: indicators ----------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / (loss + 1e-9)
    return 100.0 - (100.0 / (1.0 + rs))

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=1).mean()

# ---------- Robust date parsing (DD-MM-YY or DD-MM-YYYY) ----------
def parse_dates(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    d = pd.to_datetime(s, format="%d-%m-%Y", errors="coerce", utc=True)
    mask = d.isna()
    d[mask] = pd.to_datetime(s[mask], format="%d-%m-%y", errors="coerce", utc=True)
    return d

# ---------- Feature engineering (DOGE) ----------
def add_features_doge(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_close"] = np.log(out["close"].astype(float).clip(lower=1e-12))
    out["log_ret_1d"] = out["log_close"].diff()

    out["ema_20"] = ema(out["close"], 20)
    out["ema_50"] = ema(out["close"], 50)
    out["ratio_close_ema20"] = out["close"] / (out["ema_20"] + 1e-9)
    out["ratio_close_ema50"] = out["close"] / (out["ema_50"] + 1e-9)

    lc_mean = out["log_close"].rolling(60).mean()
    lc_std  = out["log_close"].rolling(60).std()
    out["z_logclose_60"] = (out["log_close"] - lc_mean) / (lc_std + 1e-9)

    out["hl_range_pct"] = (out["high"] - out["low"]) / out["close"].shift(1).replace(0, np.nan)
    out["oc_change_pct"] = (out["close"] - out["open"]) / out["open"].replace(0, np.nan)

    out["ret_mean_5"] = out["log_ret_1d"].rolling(5).mean()
    out["ret_std_5"]  = out["log_ret_1d"].rolling(5).std()
    out["ret_mean_20"] = out["log_ret_1d"].rolling(20).mean()
    out["ret_std_20"]  = out["log_ret_1d"].rolling(20).std()

    out["atr_14"] = atr(out["high"], out["low"], out["close"], 14)
    out["rsi_14"] = rsi(out["close"], 14)
    macd_line, signal_line, hist = macd(out["close"])
    out["macd"] = macd_line
    out["macd_signal"] = signal_line
    out["macd_hist"] = hist

    out["log_volume"] = np.log1p(out["volume"].astype(float).clip(lower=0))
    out["vol_z_20"] = (out["volume"] - out["volume"].rolling(20).mean()) / (out["volume"].rolling(20).std() + 1e-9)

    out["dow"] = out["date"].dt.dayofweek
    out["dow_sin"] = np.sin(2 * np.pi * out["dow"] / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * out["dow"] / 7.0)

    out = out.dropna().reset_index(drop=True)
    return out

# ---------- BTC features ----------
def add_features_btc(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["btc_log_close"] = np.log(out["close"].astype(float).clip(lower=1e-12))
    out["btc_log_ret_1d"] = out["btc_log_close"].diff()

    out["btc_ema_20"] = ema(out["close"], 20)
    out["btc_ema_50"] = ema(out["close"], 50)
    out["btc_ratio_close_ema20"] = out["close"] / (out["btc_ema_20"] + 1e-9)
    out["btc_ratio_close_ema50"] = out["close"] / (out["btc_ema_50"] + 1e-9)

    lc_mean = out["btc_log_close"].rolling(60).mean()
    lc_std  = out["btc_log_close"].rolling(60).std()
    out["btc_z_logclose_60"] = (out["btc_log_close"] - lc_mean) / (lc_std + 1e-9)

    out["btc_ret_mean_5"] = out["btc_log_ret_1d"].rolling(5).mean()
    out["btc_ret_std_20"]  = out["btc_log_ret_1d"].rolling(20).std()

    out["btc_log_volume"] = np.log1p(out["volume"].astype(float).clip(lower=0))
    out["btc_vol_z_20"]   = (out["volume"] - out["volume"].rolling(20).mean()) / (out["volume"].rolling(20).std() + 1e-9)

    keep = [
        "date","btc_log_close","btc_log_ret_1d","btc_ema_20","btc_ema_50",
        "btc_ratio_close_ema20","btc_ratio_close_ema50","btc_z_logclose_60",
        "btc_ret_mean_5","btc_ret_std_20","btc_log_volume","btc_vol_z_20"
    ]
    out = out[keep].dropna().reset_index(drop=True)
    return out

# ---------- Windowing ----------
def make_windows_for_split(split_df, feature_cols, seq_len):
    log_close = split_df["log_close"].values
    y_full = np.empty_like(log_close)
    y_full[:-1] = log_close[1:] - log_close[:-1]
    y_full[-1] = np.nan

    X_raw = split_df[feature_cols].values.astype(np.float32)
    close_arr = split_df["close"].values.astype(np.float32)

    xs, ys, last_close, next_close, dates = [], [], [], [], []
    max_end = len(split_df) - 2
    for idx_end in range(seq_len-1, max_end+1):
        i = idx_end - (seq_len - 1)
        y = y_full[idx_end]
        if np.isnan(y):
            continue
        xs.append(X_raw[i: i+seq_len])
        ys.append(y)
        last_close.append(close_arr[idx_end])
        next_close.append(close_arr[idx_end+1])
        dates.append(split_df["date"].iloc[idx_end+1])
    if not xs:
        return (np.empty((0, seq_len, X_raw.shape[1]), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                pd.to_datetime([]))
    return (np.stack(xs).astype(np.float32),
            np.array(ys, dtype=np.float32),
            np.array(last_close, dtype=np.float32),
            np.array(next_close, dtype=np.float32),
            pd.to_datetime(dates))

def transform_windows(X_raw, scaler):
    n, T, F = X_raw.shape
    X_flat = X_raw.reshape(-1, F)
    X_scaled = scaler.transform(X_flat).astype(np.float32)
    return X_scaled.reshape(n, T, F)

# ---------- Model ----------
class PositionalEmbedding(layers.Layer):
    def __init__(self, seq_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.d_model = d_model
    def build(self, input_shape):
        self.pos_emb = self.add_weight(
            shape=(1, self.seq_len, self.d_model),
            initializer="random_normal",
            trainable=True,
            name="pos_embedding"
        )
    def call(self, x):
        return x + self.pos_emb

def transformer_encoder_block(d_model, num_heads, ff_dim, dropout, l2_reg):
    inputs = layers.Input(shape=(None, d_model))
    attn_out = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model, dropout=dropout)(inputs, inputs)
    attn_out = layers.Dropout(dropout)(attn_out)
    x = layers.LayerNormalization(epsilon=1e-6)(inputs + attn_out)
    ff = keras.Sequential([
        layers.Dense(ff_dim, activation="relu", kernel_regularizer=regularizers.l2(l2_reg)),
        layers.Dropout(dropout),
        layers.Dense(d_model, kernel_regularizer=regularizers.l2(l2_reg))
    ])
    ff_out = ff(x)
    ff_out = layers.Dropout(dropout)(ff_out)
    x = layers.LayerNormalization(epsilon=1e-6)(x + ff_out)
    return keras.Model(inputs, x)

def build_model(seq_len, n_features, d_model, num_heads, num_layers, ff_dim, dropout, l2_reg):
    inputs = layers.Input(shape=(seq_len, n_features))
    x = layers.Dense(d_model, kernel_regularizer=regularizers.l2(l2_reg))(inputs)
    x = PositionalEmbedding(seq_len, d_model)(x)
    for _ in range(num_layers):
        blk = transformer_encoder_block(d_model, num_heads, ff_dim, dropout, l2_reg)
        x = blk(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, name="log_return_next")(x)
    model = keras.Model(inputs, outputs)
    opt = keras.optimizers.Adam(learning_rate=1e-3, clipnorm=CLIPNORM)
    model.compile(optimizer=opt, loss=keras.losses.Huber(delta=1.0))
    return model

# ---------- Metrics ----------
def smape(y_true, y_pred):
    return 100.0 * np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-9))

def direction_accuracy(true_rets, pred_rets):
    return np.mean(np.sign(true_rets) == np.sign(pred_rets)) * 100.0

# ---------- Core inference ----------
def run_eval(doge_csv, btc_csv, start_date, end_date, use_btc=True, ema_span=10, seq_len=None):
    seq_len = seq_len or SEQ_LEN

    # ---- Load DOGE ----
    doge = pd.read_csv(doge_csv)
    doge.columns = [c.strip().replace(' ', '_').lower() for c in doge.columns]
    if "date" not in doge.columns:
        raise ValueError("DOGE file needs a 'date' column.")
    doge["date"] = parse_dates(doge["date"])
    doge = (doge.dropna(subset=["date","open","high","low","close","volume"])
                 .sort_values("date").drop_duplicates("date").reset_index(drop=True))

    # Basic sanity fixes
    bad = doge["high"] < doge["low"]
    if bad.any():
        doge.loc[bad, ["high","low"]] = doge.loc[bad, ["low","high"]].values

    # ---- Features (DOGE) ----
    feat_doge = add_features_doge(doge)

    # ---- Optionally add BTC features ----
    feature_cols = [
        'close','open','high','low','volume',
        'ema_20','ema_50','ratio_close_ema20','ratio_close_ema50','z_logclose_60',
        'log_ret_1d','hl_range_pct','oc_change_pct',
        'ret_mean_5','ret_std_5','ret_mean_20','ret_std_20',
        'atr_14','rsi_14','macd','macd_signal','macd_hist',
        'log_volume','vol_z_20','dow_sin','dow_cos'
    ]

    if use_btc and btc_csv is not None:
        btc = pd.read_csv(btc_csv)
        btc.columns = [c.strip().replace(' ', '_').lower() for c in btc.columns]
        if "date" not in btc.columns:
            raise ValueError("BTC file needs a 'date' column.")
        btc["date"] = parse_dates(btc["date"])
        btc = (btc.dropna(subset=["date","open","high","low","close","volume"])
                    .sort_values("date").drop_duplicates("date").reset_index(drop=True))
        badb = btc["high"] < btc["low"]
        if badb.any():
            btc.loc[badb, ["high","low"]] = btc.loc[badb, ["low","high"]].values

        feat_btc = add_features_btc(btc)

        merged = feat_doge.merge(feat_btc, on="date", how="inner")
        merged["log_spread_doge_btc"] = merged["log_close"] - merged["btc_log_close"]
        merged["ret_corr_20"] = merged["log_ret_1d"].rolling(20).corr(merged["btc_log_ret_1d"])
        feat_df = merged.dropna().reset_index(drop=True)

        btc_cols = [
            "btc_log_ret_1d","btc_ret_mean_5","btc_ret_std_20",
            "btc_ratio_close_ema20","btc_ratio_close_ema50","btc_z_logclose_60",
            "btc_log_volume","btc_vol_z_20"
        ]
        cross_cols = ["log_spread_doge_btc","ret_corr_20"]
        feature_cols = feature_cols + btc_cols + cross_cols
    else:
        feat_df = feat_doge

    # ---- Filter by requested eval window with context ----
    sd = pd.to_datetime(start_date, utc=True)
    ed = pd.to_datetime(end_date, utc=True)

    context = feat_df.loc[feat_df["date"] < sd].tail(seq_len).copy()
    eval_part = feat_df[(feat_df["date"] >= sd) & (feat_df["date"] <= ed)].copy()
    if len(eval_part) == 0:
        raise ValueError("No rows in the selected date range.")
    df_concat = pd.concat([context, eval_part], ignore_index=True)

    # ---- Windows for this segment ----
    X_raw, y_ret, lastC, nextC, dates = make_windows_for_split(df_concat, feature_cols, seq_len)

    # Keep only predictions whose date is within [sd, ed]
    mask_keep = (dates >= sd) & (dates <= ed)
    X_raw, y_ret, lastC, nextC, dates = X_raw[mask_keep], y_ret[mask_keep], lastC[mask_keep], nextC[mask_keep], dates[mask_keep]

    # ---- Scale and predict ----
    X = transform_windows(X_raw, SCALER)
    ds = tf.data.Dataset.from_tensor_slices(X).batch(128)

    # Rebuild model with correct input dims
    n_features = X.shape[-1]
    model = build_model(seq_len, n_features, D_MODEL, NUM_HEADS, NUM_LAYERS, FF_DIM, DROPOUT, L2_REG)
    model.load_weights(MODEL_WEIGHTS)

    y_pred_ret = model.predict(ds, verbose=0).flatten()
    pred_close = lastC * np.exp(y_pred_ret)
    actual_close = nextC

    # ---- Baselines ----
    naive_pred = lastC
    # Use the user-provided EMA span for the baseline calculation
    ema_series = df_concat["close"].ewm(span=ema_span, adjust=False).mean().values
    ema_pred = ema_series[seq_len-1 : len(ema_series)-1]
    ema_pred = ema_pred[mask_keep]

    # ---- Metrics ----
    metrics = {
        "MODEL": {
            "MAE": float(mean_absolute_error(actual_close, pred_close)),
            "RMSE": float(np.sqrt(mean_squared_error(actual_close, pred_close))),
            "sMAPE": float(smape(actual_close, pred_close)),
            "Direction_Accuracy_%": float(direction_accuracy(y_ret, y_pred_ret))
        },
        "NAIVE": {
            "MAE": float(mean_absolute_error(actual_close, naive_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(actual_close, naive_pred))),
            "sMAPE": float(smape(actual_close, naive_pred)),
        },
        "EMA10": {
            "MAE": float(mean_absolute_error(actual_close, ema_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(actual_close, ema_pred))),
            "sMAPE": float(smape(actual_close, ema_pred)),
        }
    }
    # ---- Format metrics into a readable table ----
    # Use a dynamic label for the EMA baseline corresponding to the chosen span
    ema_label = f"EMA({ema_span})"
    metrics_df = pd.DataFrame({
        "Model": ["Transformer", "Naïve", ema_label],
        "MAE": [
            metrics["MODEL"]["MAE"],
            metrics["NAIVE"]["MAE"],
            metrics["EMA10"]["MAE"],
        ],
        "RMSE": [
            metrics["MODEL"]["RMSE"],
            metrics["NAIVE"]["RMSE"],
            metrics["EMA10"]["RMSE"],
        ],
        "sMAPE": [
            metrics["MODEL"]["sMAPE"],
            metrics["NAIVE"]["sMAPE"],
            metrics["EMA10"]["sMAPE"],
        ],
        "Dir. Accuracy (%)": [
            metrics["MODEL"]["Direction_Accuracy_%"],
            np.nan,
            np.nan,
        ],
    })

    # ---- Build evaluation DataFrame with dynamic EMA column ----
    eval_df = pd.DataFrame({
        "date": dates,
        "Actual": actual_close,
        "Naïve": naive_pred,
        ema_label: ema_pred,
        "Transformer": pred_close,
    })

    # ---- Define colours for each series (Dogecoin branding) ----
    colours = {
        "Actual": "#1a1a1a",        # dark grey/black for actual data
        "Naïve": "#7f8c8d",        # muted grey for naive baseline
        ema_label: "#DAA520",        # goldenrod for EMA baseline
        "Transformer": "#C0392B",   # red for transformer predictions
    }

    # ---- Overlay plot using Plotly ----
    overlay_fig = go.Figure()
    for key in ["Actual", "Naïve", ema_label, "Transformer"]:
        overlay_fig.add_trace(
            go.Scatter(
                x=eval_df["date"],
                y=eval_df[key],
                mode="lines",
                name=key,
                line=dict(color=colours[key], width=3, dash="dash" if key == "Transformer" else None),
                hovertemplate=f"%{{x|%Y-%m-%d}}<br>{key}: %{{y:.4f}}<extra></extra>",
            )
        )
    overlay_fig.update_layout(
        title=f"DOGE‑USD: Actual vs Forecasts<br><span style='font-size:12px'>{sd.date()} → {ed.date()}</span>",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        legend_title="Series",
        template="plotly_white",
        height=500,
        margin=dict(l=60, r=40, t=80, b=40),
    )

    # ---- Scatter plot using Plotly ----
    scatter_fig = go.Figure()
    scatter_fig.add_trace(
        go.Scatter(
            x=eval_df["Actual"],
            y=eval_df["Transformer"],
            mode="markers",
            name="Predictions",
            marker=dict(size=6, color="#1f77b4"),
            hovertemplate="Actual: %{x:.4f}<br>Predicted: %{y:.4f}<extra></extra>",
        )
    )
    # Identity line
    mn = float(min(eval_df["Actual"].min(), eval_df["Transformer"].min()))
    mx = float(max(eval_df["Actual"].max(), eval_df["Transformer"].max()))
    scatter_fig.add_trace(
        go.Scatter(
            x=[mn, mx],
            y=[mn, mx],
            mode="lines",
            showlegend=False,
            line=dict(color="#7f8c8d", dash="dash"),
            hoverinfo="skip",
        )
    )
    scatter_fig.update_layout(
        title="Predicted vs Actual Price",
        xaxis_title="Actual Price (USD)",
        yaxis_title="Predicted Price (USD)",
        template="plotly_white",
        height=500,
        margin=dict(l=60, r=40, t=80, b=40),
    )

    return metrics_df, overlay_fig, scatter_fig

# ---------- Gradio UI ----------
with gr.Blocks(title="Shiba Sense: Doge Price Predictor") as demo:
    # Apply custom styling for Dogecoin branding and friendly appearance
    gr.HTML(
        """
        <style>
        #app_title { text-align: left !important; margin: 0.25rem 0 0.75rem 0; }
        #app_title h2 { 
            font-size: 34px !important;   /* ← change this number to your taste */
            line-height: 1.2;
            font-weight: 700;
            margin: 0;
        }
        /* Use a friendly font and light background */
        body, .gradio-container {
            font-family: 'Trebuchet MS', sans-serif;
            background-color: #fafafa;
        }
        /* Headings in gold */
        h2, h3, h4, h5 {
            color: #EFB810;
            margin-top: 0.5rem;
            margin-bottom: 0.5rem;
        }
        /* Primary button in gold with darker hover */
        .gr-button-primary {
            background-color: #EFB810 !important;
            border-color: #EFB810 !important;
            color: #1a1a1a !important;
        }
        .gr-button-primary:hover {
            background-color: #d7a400 !important;
            border-color: #d7a400 !important;
        }
        </style>
        """
    )
    # Header with Shiba Inu icon and title
    with gr.Row():

        gr.Markdown("## Shiba Sense: Doge Price Predictor", elem_id="app_title")

    # File inputs side by side
    with gr.Row():
        doge_file = gr.File(label="DOGE CSV (daily OHLCV)", file_types=[".csv"], value="data/doge_usd_2017_2025.csv", scale=1)
        btc_file  = gr.File(label="BTC CSV (daily OHLCV)", file_types=[".csv"], value="data/BTC_2016_2025.csv", scale=1)
    # Start/end dates, checkbox and sliders grouped in one row for compactness
    with gr.Row():
        start = gr.Textbox(label="Start date (YYYY-MM-DD)", value="2025-01-01", scale=1)
        end   = gr.Textbox(label="End date (YYYY-MM-DD)", value="2025-06-30", scale=1)
        use_btc = gr.Checkbox(label="Use BTC features", value=True, interactive=False)
        ema_span = gr.Slider(
            EMA_BASELINE, EMA_BASELINE, value=EMA_BASELINE, step=1,
            label=f"EMA baseline span (fixed at {EMA_BASELINE})",
            interactive=False
        )
        seq_inp = gr.Slider(
            SEQ_LEN, SEQ_LEN, value=SEQ_LEN, step=1,
            label=f"Sequence length (fixed at {SEQ_LEN})",
            interactive=False
        )

    # Run button
    run_btn = gr.Button("Run evaluation", variant="primary")
    # Outputs: metrics table and two plots
    out_metrics = gr.Dataframe(label="Evaluation Metrics", interactive=False)
    out_overlay = gr.Plot(label="DOGE‑USD: Actual vs Forecasts")
    out_scatter = gr.Plot(label="Predicted vs Actual Price")
    notes_md = """
    ### How to Use
    - **Pick dates:** `2025-01-01 → 2025-06-30` (test) or `2025-07-01 → 2025-08-13` (holdout).
    - Click **Run evaluation** to see metrics, overlay plot, and scatter.
    
    ### Notes
    - **What the model predicts:** next–day **log return** of DOGE; we convert to price.
    - **How to read the plots:** dashed line = Transformer; Naïve = yesterday’s price; EMA is a smooth baseline.
    - **Strengths:** low sMAPE on 2025H1; tracks swings better than EMA; sensible on 2025Q3 holdout.
    - **Limitations:** daily direction can be near chance on some regimes; sensitive to window length.
    - **Data:** daily DOGE **and** BTC OHLCV aligned by date **(BTC features required)**; missing days dropped.
    - **Reproducibility:** metrics assume the training **sequence length (120)**; changing it without retraining can shift results.
    - **Improvements:** add macro/crypto factors, try sinusoidal positional encodings, residual-over-Naïve, and rolling re-fit.
    """
    with gr.Accordion("About this app", open=True):
        gr.Markdown(notes_md)

    def _run(doge_csv, btc_csv, start, end, use_btc, ema_span, seq_inp):
        # Resolve file paths: UploadedFile objects have a .name attribute; strings do not
        doge_path = doge_csv.name if hasattr(doge_csv, "name") else doge_csv
        btc_path  = btc_csv.name if hasattr(btc_csv, "name") else btc_csv
        return run_eval(
            doge_path,
            btc_path,
            start,
            end,
            use_btc=True,
            ema_span=EMA_BASELINE,
            seq_len=int(SEQ_LEN),
        )

    run_btn.click(
        _run,
        inputs=[doge_file, btc_file, start, end, use_btc, ema_span, seq_inp],
        outputs=[out_metrics, out_overlay, out_scatter],
    )

if __name__ == "__main__":
    # Use share=True so that the app is accessible if localhost is not available
    demo.launch(share=True)
