# Shiba Sense — Doge Price Predictor

In this project, I built a forecasting pipeline that predicts the next-day DOGE-USD price using a Transformer encoder trained on daily DOGE + BTC features. The app includes baseline comparisons (Naïve & EMA), visual diagnostics, and a deployable Gradio UI.

---

## 🚀 Live Demo

View it here: 

🔗 **(https://huggingface.co/spaces/abibatoki/shiba-sense-doge-price-predictor)**

---  

## 🧭 Objective

- Forecast next-day price by first predicting the next-day log return and then reconstructing the price.
- Compare the model against simple, honest baselines:
  - **Naïve:** “tomorrow ≈ today”
  - **EMA(10):** 10-day exponential moving average
- Provide an interactive app so anyone can select a date window and inspect metrics and plots.

---

## 📂 Data

- **Source:** Kaggle crypto OHLCV dataset (CC0 / Public Domain) sourced from CoinMarketCap.  
- **Symbols used:** DOGE and BTC (BTC features are required).  
- **Frequency:** Daily  
- **Range used:**  
  - DOGE: 2017-01-01 → 2025-08-13  
  - BTC: 2016-01-01 → 2025-08-13  

### Windows for modeling:
- Train: up to 2023-12-31  
- Validation: 2024-01-01 → 2024-12-31  
- Test: 2025-01-01 → 2025-06-30  
- Holdout: 2025-07-01 → 2025-08-13 (never seen during training/tuning)  

### Notes from cleaning/ingest
- The CSV used DD-MM-YYYY; I fixed parsing with `dayfirst=True` and verified no NaTs.  
- Dropped rows with critical NaNs and aligned DOGE and BTC by date.  
- Log-transformed returns (not prices) for the learning target to stabilize scale.  

---

## 📦 Code & Environment

- **Python:** 3.10 on Hugging Face (local dev also on 3.8)  
- **Core libs:** tensorflow, scikit-learn, pandas, numpy, scipy, plotly, gradio, joblib  
- **Deployment:** Hugging Face Spaces (Gradio SDK), `runtime.txt` with python-3.10

---

## 🧪 Features & Engineering

- **DOGE features:** close, open, high, low, volume, rolling EMA/SMA, and daily log return.  
- **BTC exogenous features:** close, volume, daily log return, and a few simple technicals.  
- **Scaling:** RobustScaler fit on train only, then applied to val/test/holdout.  
- **Windowing:** sequences of 120 context days → predict the next day (horizon=1).  
  (Sequence length locked in the app to match saved weights.)  
- **Target:** next-day log return; prediction converted back to price with today’s close.  

---

## 🧠 Model

<img width="535" height="429" alt="image" src="https://github.com/user-attachments/assets/d30dea96-eaca-4e90-bb61-010f379bae0e" />

- **Architecture:** compact Transformer encoder stack (multi-head self-attention) with dropout, residuals, feed-forward blocks, and layer norm.  
- **Loss & metrics during training:** MAE on returns.  
- **Evaluation metrics:** MAE, RMSE, sMAPE on prices plus Direction Accuracy (% days the up/down move was predicted correctly).  
- **Training controls:** learning-rate schedule on plateau, gradient clipping, and early stopping.  

---

## 🧭 What I Tried

- **First pass (the flat line):**  
  Initially saw a near-flat prediction line caused by scaling/label leak. Fixed label alignment and conversions.  

- **Stabilizing the splits & windows:**  
  Standardized train/val/test/holdout splits and locked sequence length to 120.  

- **Baselines first:**  
  Benchmarked Naïve and EMA(10). Kept me honest when tuning the Transformer.  

- **Shrinking the model:**  
  Reduced depth/width for speed, metrics stayed the same.  

- **Adding BTC features:**  
  Didn’t help much on the test window, but improved holdout generalization (direction accuracy ~57%).  

- **From Matplotlib to Plotly + Gradio:**  
  Switched to Plotly for interactivity and themed the Gradio UI with Dogecoin gold/black.    

---

## 📈 Results

Exact values depend on the date window and data snapshot. Below are representative results.

### Test window — 2025-H1 (2025-01-01 → 2025-06-30)

| Model       | MAE    | RMSE   | sMAPE | Direction Acc. |
|-------------|--------|--------|-------|----------------|
| Transformer | 0.0066 | 0.0107 | 3.3%  | ~47.5%         |
| Naïve       | 0.0066 | 0.0104 | 3.3%  | —              |
| EMA(10)     | 0.0117 | 0.0165 | 5.9%  | —              |

### Holdout — 2025-Q3 (Jul–Aug) (2025-07-01 → 2025-08-13)
- Transformer: MAE ≈ 0.0084, RMSE ≈ 0.0107, sMAPE ≈ 3.95%, Direction Acc. ≈ 56.8%  

### Metric meanings
- **MAE (Mean Absolute Error):** average absolute difference in price units (USD).  
- **RMSE (Root Mean Squared Error):** emphasizes larger errors (USD).  
- **sMAPE (symmetric MAPE):** scale-free %, robust when prices vary.  
- **Direction Accuracy:** % of days where the sign of the move (up/down) was predicted correctly.

---

## 📊 Visuals

- Test overlay plot — 2025-01-01 → 2025-06-30

 <img width="1800" height="750" alt="overlay_2025H1" src="https://github.com/user-attachments/assets/7b5f9de7-4326-4a71-8242-902063fcce80" />
 
- Test scatter (Predicted vs Actual)

  <img width="900" height="900" alt="scatter_2025H1" src="https://github.com/user-attachments/assets/40a97f5c-bfd9-42d9-95f2-c1d8db1b5f2a" />

- Holdout overlay — 2025-07-01 → 2025-08-13

  <img width="1800" height="750" alt="overlay_holdout_2025Q3" src="https://github.com/user-attachments/assets/0b9a458d-3f4b-43b6-8f82-cadbd5580e4c" />

- Holdout scatter  

  <img width="900" height="900" alt="scatter_holdout_2025Q3" src="https://github.com/user-attachments/assets/b9ccc707-0d7f-40d4-a024-d6aad56c7950" />

---

## ✅ What Worked / ⚠️ Limitations

**What worked**
- Transformer tracks medium-term swings and beats EMA on sMAPE.  
- Solid holdout direction accuracy (~57%) for Jul–Aug 2025.  
- Clean UI with fixed knobs prevents misconfiguration.  

**Limitations**
- Daily direction near chance during choppy regimes.  
- Results are regime-dependent; retraining or rolling re-fit would help.  
- Not designed for intraday or execution-aware signals.  

---

## 💡 Ideas for Improvement

- Add richer crypto/macro signals (BTC dominance, funding rates, on-chain flows).  
- Try sinusoidal (non-trainable) positional encodings and residual-over-Naïve targets.  
- Explore ensembling (Transformer + linear/ARX) and rolling re-fit.  
- Hyperparameter search focused on sequence length vs. feature set.  

---

## ⚖️ License & Disclaimer

- **Dataset:** CC0 / Public Domain (Kaggle).    
- **Disclaimer:** This project is for research/education. Not financial advice.  

