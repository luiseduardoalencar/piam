# %%
"""
Forecast de sinistralidade Climazon.
  MASTER EMPRESARIAL → LSTM ensemble (30 runs)
  MASTER EXECUTIVO   → sem modelo (dados insuficientes; apenas historico salvo)

Execução:
    python -X utf8 pipelines/climazon/forecast_climazon.py
"""
from __future__ import annotations

import json
import os
import pickle
import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

try:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.models import Sequential
except ImportError as e:
    raise ImportError("TensorFlow necessário: pip install tensorflow>=2.14") from e



# =============================================================================
# %% — Constantes
# =============================================================================

COMPANY   = "climazon"
PARQUET   = ROOT_DIR / "data" / "processed" / COMPANY / "base_analitica_transformada" / "painel_sinistralidade_climazon_v1.parquet"
FORECAST_ROOT = ROOT_DIR / "data" / "processed" / COMPANY / "forecast"

PLANOS = ["MASTER EMPRESARIAL", "MASTER EXECUTIVO"]
PLANO_SLUG = {"MASTER EMPRESARIAL": "MASTER_EMPRESARIAL", "MASTER EXECUTIVO": "MASTER_EXECUTIVO"}

# LSTM (EMPRESARIAL)
LOOK_BACK      = 6
HORIZON        = 12
TRAIN_FRACTION = 0.80
N_RUNS         = 30      # ensemble runs para EMPRESARIAL
EPOCHS         = 100
BATCH_SIZE     = 8
PATIENCE       = 15
DROPOUT        = 0.2
HUBER_DELTA    = 0.1

# Capeamento de outliers no historico do EXECUTIVO (meses COVID com faturamento infimo)
OUTLIER_CAP_QUANTILE = 0.95

RANDOM_BASE = 42


# =============================================================================
# %% — Utilitários gerais
# =============================================================================

def next_version_dir(root: Path) -> tuple[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    max_n = 0
    for p in root.iterdir():
        if p.is_dir():
            m = re.fullmatch(r"v(\d+)", p.name, flags=re.IGNORECASE)
            if m:
                max_n = max(max_n, int(m.group(1)))
    ver = f"v{max_n + 1}"
    out = root / ver
    out.mkdir(parents=True, exist_ok=True)
    return ver, out


def rmse_fn(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape_fn(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-8
    return float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100.0)


# =============================================================================
# %% — Preparação da série por plano
# =============================================================================

def load_plan_series(plano: str) -> pd.DataFrame:
    """
    Deriva série mensal do parquet para o plano especificado.
    Aplica: remoção de zeros iniciais, preenchimento de gaps (interpolação linear),
    e capeamento de outliers (EXECUTIVO).
    Retorna DataFrame com colunas: DATA, COMPETENCIA, SINISTRALIDADE, VALOR_FATURAMENTO, imputado.
    """
    df = pd.read_parquet(PARQUET)
    df["comp"] = pd.to_datetime(df["competencia"].astype(str), errors="coerce")
    sub = df[df["plano"] == plano].copy()

    monthly = (
        sub.groupby("comp", as_index=False)
        .agg(fat=("valor_faturamento", "sum"), sin_ajust=("valor_sinistro_ajustado", "sum"))
    )
    monthly["SINISTRALIDADE"] = monthly["sin_ajust"] / monthly["fat"]
    monthly = monthly.rename(columns={"comp": "DATA", "fat": "VALOR_FATURAMENTO"})
    monthly = monthly.sort_values("DATA").reset_index(drop=True)

    # Remove períodos com sinistralidade zero (início de carteira)
    first_nonzero = monthly[monthly["SINISTRALIDADE"] > 0]["DATA"].min()
    monthly = monthly[monthly["DATA"] >= first_nonzero].reset_index(drop=True)

    # Preenche gaps com interpolação linear
    full_range = pd.date_range(monthly["DATA"].min(), monthly["DATA"].max(), freq="MS")
    monthly = monthly.set_index("DATA").reindex(full_range)
    monthly.index.name = "DATA"
    monthly["imputado"] = monthly["SINISTRALIDADE"].isna()
    monthly["SINISTRALIDADE"] = monthly["SINISTRALIDADE"].interpolate(method="linear")
    monthly["VALOR_FATURAMENTO"] = monthly["VALOR_FATURAMENTO"].interpolate(method="linear")
    monthly = monthly.reset_index()

    # Capeia outliers apenas para EXECUTIVO (meses COVID com faturamento ínfimo)
    if plano == "MASTER EXECUTIVO":
        cap = monthly["SINISTRALIDADE"].quantile(OUTLIER_CAP_QUANTILE)
        n_capeados = int((monthly["SINISTRALIDADE"] > cap).sum())
        monthly.loc[monthly["SINISTRALIDADE"] > cap, "SINISTRALIDADE"] = cap
        if n_capeados:
            print(f"  [{plano}] {n_capeados} meses capeados em {cap:.3f} (p{int(OUTLIER_CAP_QUANTILE*100)})")

    monthly["COMPETENCIA"] = monthly["DATA"].dt.strftime("%Y-%m")
    return monthly[["DATA", "COMPETENCIA", "SINISTRALIDADE", "VALOR_FATURAMENTO", "imputado"]]


# =============================================================================
# %% — LSTM (MASTER EMPRESARIAL)
# =============================================================================

def _make_windows(scaled: np.ndarray, lb: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(lb, len(scaled)):
        X.append(scaled[i - lb: i, 0])
        y.append(float(scaled[i, 0]))
    return np.array(X, dtype=np.float64), np.array(y, dtype=np.float64)


def _log1p(v: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(np.asarray(v, dtype=np.float64), 0.0))


def _inv(scaled_2d: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    return np.expm1(scaler.inverse_transform(np.asarray(scaled_2d).reshape(-1, 1)).flatten())


def _build_lstm(look_back: int) -> Any:
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(look_back, 1)),
        Dropout(DROPOUT),
        LSTM(32, return_sequences=False),
        Dropout(DROPOUT),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss=tf.keras.losses.Huber(delta=HUBER_DELTA))
    return model


def _fit_lstm(X: np.ndarray, y: np.ndarray, seed: int, verbose: int = 0) -> Any:
    tf.keras.utils.set_random_seed(seed)
    cb = EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True, verbose=0)
    model = _build_lstm(LOOK_BACK)
    model.fit(
        X.reshape(-1, LOOK_BACK, 1), y,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_split=0.15, callbacks=[cb],
        verbose=verbose,
    )
    return model


def _rolling_forecast(model: Any, seed_window: np.ndarray, horizon: int, scaler: MinMaxScaler) -> np.ndarray:
    """Forecast autoregressivo a partir da última janela da série."""
    window = seed_window.copy()
    preds_sc: list[float] = []
    for _ in range(horizon):
        p = float(model.predict(window.reshape(1, LOOK_BACK, 1), verbose=0)[0, 0])
        preds_sc.append(p)
        window = np.append(window[1:], p)
    return _inv(np.array(preds_sc).reshape(-1, 1), scaler)


def train_lstm_ensemble(series: pd.DataFrame) -> dict[str, Any]:
    """
    Treina LSTM × N_RUNS para EMPRESARIAL.
    - 1 run de avaliação (train/test split) para métricas.
    - N_RUNS runs na série completa para ensemble de forecast.
    Retorna métricas, ensemble de previsões e o melhor modelo.
    """
    vals = series["SINISTRALIDADE"].values.astype(np.float64)
    n = len(vals)
    train_size = int(n * TRAIN_FRACTION)

    print(f"  Serie: {n} meses | treino={train_size} | teste={n - train_size}")

    # ── Avaliação: 1 run para métricas (treina só no split 80%) ──────────────
    train_log = _log1p(vals[:train_size])
    test_log  = _log1p(vals[train_size:])

    scaler_eval = MinMaxScaler((0, 1))
    train_sc    = scaler_eval.fit_transform(train_log.reshape(-1, 1))
    test_sc     = scaler_eval.transform(test_log.reshape(-1, 1))
    full_eval_sc = np.vstack([train_sc, test_sc])

    X_all, _ = _make_windows(full_eval_sc, LOOK_BACK)
    split_pt  = train_size - LOOK_BACK
    X_tr_e, y_tr_e = X_all[:split_pt], np.array(
        [float(full_eval_sc[i, 0]) for i in range(LOOK_BACK, train_size)]
    )
    X_te_e = X_all[split_pt:]  # 1 janela por mês de teste (sem skip)

    print(f"  [Avaliação] treinando run de referência...")
    model_eval = _fit_lstm(X_tr_e, y_tr_e, seed=RANDOM_BASE, verbose=1)

    pred_te_sc = model_eval.predict(X_te_e.reshape(-1, LOOK_BACK, 1), verbose=0)
    pred_te    = _inv(pred_te_sc, scaler_eval)
    y_te_real  = vals[train_size:]  # todos os meses de teste

    metric_rmse = rmse_fn(y_te_real, pred_te)
    metric_mae  = float(mean_absolute_error(y_te_real, pred_te))
    metric_mape = mape_fn(y_te_real, pred_te)
    print(f"  Metricas: RMSE={metric_rmse:.4f}  MAE={metric_mae:.4f}  MAPE={metric_mape:.2f}%")

    # Mapeia predições do teste a datas (sem pular meses)
    test_dates = series["DATA"].values[train_size:]
    pred_test_map: dict[Any, float] = dict(zip(test_dates, pred_te))

    # ── Ensemble: N_RUNS runs na série completa ───────────────────────────────
    full_log = _log1p(vals)
    scaler_full = MinMaxScaler((0, 1))
    full_sc = scaler_full.fit_transform(full_log.reshape(-1, 1))
    X_full, y_full = _make_windows(full_sc, LOOK_BACK)
    seed_window = full_sc[-LOOK_BACK:, 0].copy()

    ensemble_preds = np.zeros((N_RUNS, HORIZON), dtype=np.float64)
    best_val_loss  = np.inf
    best_model     = None
    best_scaler    = scaler_full

    print(f"  [Ensemble] {N_RUNS} runs...")
    for run in range(N_RUNS):
        seed = RANDOM_BASE + run
        m = _fit_lstm(X_full, y_full, seed=seed, verbose=0)
        ensemble_preds[run] = _rolling_forecast(m, seed_window, HORIZON, scaler_full)
        val_loss = min(m.history.history.get("val_loss", [np.inf]))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model    = m
        if (run + 1) % 5 == 0:
            print(f"    run {run + 1}/{N_RUNS} concluido")

    return {
        "ensemble_preds":  ensemble_preds,   # (N_RUNS, HORIZON)
        "forecast_mean":   ensemble_preds.mean(axis=0),
        "forecast_p05":    np.percentile(ensemble_preds, 5,  axis=0),
        "forecast_p25":    np.percentile(ensemble_preds, 25, axis=0),
        "forecast_p75":    np.percentile(ensemble_preds, 75, axis=0),
        "forecast_p95":    np.percentile(ensemble_preds, 95, axis=0),
        "pred_test_map":   pred_test_map,
        "train_size":      train_size,
        "scaler":          scaler_full,
        "best_model":      best_model,
        "metrics": {
            "rmse": round(metric_rmse, 6),
            "mae":  round(metric_mae,  6),
            "mape_pct": round(metric_mape, 4),
        },
    }



# =============================================================================
# %% — Construção do CSV de saída por plano
# =============================================================================

def build_output_csv(series: pd.DataFrame, result: dict, last_date: pd.Timestamp) -> pd.DataFrame:
    """Monta CSV com historico + teste + previsão, incluindo bandas de incerteza."""
    train_size = result["train_size"]
    pred_test_map = result["pred_test_map"]

    rows: list[dict[str, Any]] = []
    for i, row in series.iterrows():
        d = row["DATA"]
        rec: dict[str, Any] = {
            "DATA":              d,
            "COMPETENCIA":       row["COMPETENCIA"],
            "VALOR_FATURAMENTO": row["VALOR_FATURAMENTO"],
            "SINISTRALIDADE":    row["SINISTRALIDADE"],
            "imputado":          bool(row["imputado"]),
            "SINISTRALIDADE_PREVISTA": np.nan,
            "forecast_p05": np.nan,
            "forecast_p25": np.nan,
            "forecast_p75": np.nan,
            "forecast_p95": np.nan,
        }
        if i < train_size:
            rec["REGISTO"] = "historico"
        else:
            rec["REGISTO"] = "teste"
            rec["SINISTRALIDADE_PREVISTA"] = pred_test_map.get(d, np.nan)
        rows.append(rec)

    future_dates = pd.date_range(last_date + pd.DateOffset(months=1), periods=HORIZON, freq="MS")
    for j, d in enumerate(future_dates):
        rows.append({
            "DATA":              d,
            "COMPETENCIA":       d.strftime("%Y-%m"),
            "VALOR_FATURAMENTO": np.nan,
            "SINISTRALIDADE":    np.nan,
            "imputado":          False,
            "REGISTO":           "previsao",
            "SINISTRALIDADE_PREVISTA": float(result["forecast_mean"][j]),
            "forecast_p05": float(result["forecast_p05"][j]),
            "forecast_p25": float(result["forecast_p25"][j]),
            "forecast_p75": float(result["forecast_p75"][j]),
            "forecast_p95": float(result["forecast_p95"][j]),
        })

    return pd.DataFrame(rows)


# =============================================================================
# %% — Gráfico por plano
# =============================================================================

def plot_plan_forecast(df_out: pd.DataFrame, plano: str, out_path: Path) -> None:
    reg = df_out["REGISTO"].str.lower()
    df_hist = df_out[reg == "historico"]
    df_test = df_out[reg == "teste"]
    df_prev = df_out[reg == "previsao"]

    fig, ax = plt.subplots(figsize=(14, 5))

    # Série histórica
    ax.plot(df_hist["DATA"], df_hist["SINISTRALIDADE"],
            color="steelblue", linewidth=1.4, label="Historico")

    # Observado no teste
    ax.plot(df_test["DATA"], df_test["SINISTRALIDADE"],
            color="steelblue", linewidth=1.4)

    # Predito no teste
    te_pred = df_test.dropna(subset=["SINISTRALIDADE_PREVISTA"])
    if not te_pred.empty:
        ax.plot(te_pred["DATA"], te_pred["SINISTRALIDADE_PREVISTA"],
                color="orange", linestyle="--", linewidth=1.4, label="Previsto (teste)")

    # Forecast futuro (conecta ao último histórico)
    if not df_prev.empty:
        last_pt = df_test.iloc[-1] if not df_test.empty else df_hist.iloc[-1]
        fut_x = [last_pt["DATA"]] + df_prev["DATA"].tolist()
        fut_y = [float(last_pt["SINISTRALIDADE"])] + df_prev["SINISTRALIDADE_PREVISTA"].tolist()
        ax.plot(fut_x, fut_y, color="crimson", linestyle="--", linewidth=1.4, label="Forecast (futuro)")

        # Banda de confiança
        if df_prev["forecast_p05"].notna().any():
            ax.fill_between(
                df_prev["DATA"],
                df_prev["forecast_p05"],
                df_prev["forecast_p95"],
                alpha=0.15, color="crimson", label="IC 90%",
            )
            ax.fill_between(
                df_prev["DATA"],
                df_prev["forecast_p25"],
                df_prev["forecast_p75"],
                alpha=0.25, color="crimson", label="IC 50%",
            )

    # Linha de split
    split_date = df_hist["DATA"].max()
    ax.axvline(pd.to_datetime(split_date), color="gray", linestyle=":", linewidth=1.0,
               label="Split treino/teste")

    ax.set_xlabel("Mes")
    ax.set_ylabel("Sinistralidade")
    ax.set_title(f"Sinistralidade {plano} — Forecast Mensal")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Grafico: {out_path}")


# =============================================================================
# %% — Pipeline principal
# =============================================================================

def run_forecast_pipeline() -> None:
    from datetime import datetime as dt

    ver, out_root = next_version_dir(FORECAST_ROOT)
    print(f"\n{'='*60}")
    print(f"Forecast Climazon — {ver} -> {out_root}")
    print(f"{'='*60}\n")

    all_meta: dict[str, Any] = {
        "company":       COMPANY,
        "versao":        ver,
        "data_execucao": dt.now().strftime("%Y-%m-%d %H:%M"),
        "horizon":       HORIZON,
        "planos":        {},
    }

    plan_results: dict[str, dict] = {}
    plan_series:  dict[str, pd.DataFrame] = {}

    for plano in PLANOS:
        slug = PLANO_SLUG[plano]
        plan_dir = out_root / slug
        plan_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'─'*55}")
        print(f"Plano: {plano}")
        print(f"{'─'*55}")

        # Série por plano
        series = load_plan_series(plano)
        plan_series[plano] = series
        last_date = series["DATA"].max()
        n = len(series)
        gaps = int(series["imputado"].sum())
        print(f"  {n} meses | {series['DATA'].min().strftime('%Y-%m')} a {last_date.strftime('%Y-%m')} | {gaps} meses imputados")

        # Treino
        if plano == "MASTER EMPRESARIAL":
            print(f"  Modelo: LSTM x {N_RUNS} runs (ensemble)")
            result = train_lstm_ensemble(series)

            # Salva melhor modelo e scaler
            model_path  = plan_dir / "lstm_best.keras"
            scaler_path = plan_dir / "minmax_scaler.pkl"
            if result["best_model"] is not None:
                result["best_model"].save(model_path)
            with open(scaler_path, "wb") as f:
                pickle.dump(result["scaler"], f, protocol=pickle.HIGHEST_PROTOCOL)

            # Salva ensemble
            np.save(plan_dir / "ensemble_predictions.npy", result["ensemble_preds"])

            all_meta["planos"][slug] = {
                "modelo":     "LSTM_ensemble",
                "n_runs":     N_RUNS,
                "look_back":  LOOK_BACK,
                "n_treino":   result["train_size"],
                "n_teste":    n - result["train_size"],
                "metricas":   result["metrics"],
            }

        else:  # MASTER EXECUTIVO — sem modelo de forecast
            print(f"  [MASTER EXECUTIVO] Sem modelo de forecast (dados insuficientes).")
            all_meta["planos"][slug] = {
                "modelo":   "nenhum",
                "motivo":   "dados insuficientes / baixa qualidade — plano excluido do forecast",
            }
            # Salva apenas historico (sem linhas de previsao)
            hist_rows_exec: list[dict[str, Any]] = []
            for _, row in series.iterrows():
                hist_rows_exec.append({
                    "DATA":              row["DATA"],
                    "COMPETENCIA":       row["COMPETENCIA"],
                    "VALOR_FATURAMENTO": row["VALOR_FATURAMENTO"],
                    "SINISTRALIDADE":    row["SINISTRALIDADE"],
                    "imputado":          bool(row["imputado"]),
                    "REGISTO":           "historico",
                    "SINISTRALIDADE_PREVISTA": np.nan,
                    "forecast_p05": np.nan,
                    "forecast_p25": np.nan,
                    "forecast_p75": np.nan,
                    "forecast_p95": np.nan,
                })
            df_exec = pd.DataFrame(hist_rows_exec)
            df_exec.to_csv(plan_dir / "sinistralidade_forecast_completo.csv", index=False)
            plan_results[plano] = None
            all_meta["planos"][slug]["n_obs"] = n
            all_meta["planos"][slug]["data_inicio"] = series["DATA"].min().strftime("%Y-%m")
            all_meta["planos"][slug]["data_fim_historico"] = last_date.strftime("%Y-%m")
            continue

        plan_results[plano] = result

        # CSV do plano
        df_out = build_output_csv(series, result, last_date)
        csv_path = plan_dir / "sinistralidade_forecast_completo.csv"
        df_out.to_csv(csv_path, index=False)
        print(f"  CSV: {csv_path}")

        # Gráfico do plano
        plot_plan_forecast(df_out, plano, plan_dir / "forecast_series.png")

        # Meta do plano
        all_meta["planos"][slug]["n_obs"] = n
        all_meta["planos"][slug]["data_inicio"] = series["DATA"].min().strftime("%Y-%m")
        all_meta["planos"][slug]["data_fim_historico"] = last_date.strftime("%Y-%m")
        future_start = (last_date + pd.DateOffset(months=1))
        all_meta["planos"][slug]["data_inicio_previsao"] = future_start.strftime("%Y-%m")
        all_meta["planos"][slug]["data_fim_previsao"] = (last_date + pd.DateOffset(months=HORIZON)).strftime("%Y-%m")

    # ── Consolidado: usa apenas planos com modelo treinado ───────────────────
    planos_com_modelo = [p for p in PLANOS if plan_results.get(p) is not None]

    print(f"\n{'─'*55}")
    print(f"Consolidado ({len(planos_com_modelo)} plano(s) com modelo)")
    print(f"{'─'*55}")

    consol_dir = out_root / "consolidado"
    consol_dir.mkdir(parents=True, exist_ok=True)

    fat_weights: dict[str, float] = {
        p: float(plan_series[p]["VALOR_FATURAMENTO"].mean()) for p in planos_com_modelo
    }
    total_fat = sum(fat_weights.values())
    w = {p: fat_weights[p] / total_fat for p in planos_com_modelo}
    print(f"  Planos: " + " | ".join(f"{p[:15]}={w[p]:.3f}" for p in planos_com_modelo))

    # Série histórica consolidada (só planos com modelo)
    hist_rows: list[dict[str, Any]] = []
    for plano in planos_com_modelo:
        series = plan_series[plano]
        train_size = plan_results[plano]["train_size"]
        for i, row in series.iterrows():
            hist_rows.append({
                "DATA": row["DATA"],
                "COMPETENCIA": row["COMPETENCIA"],
                "SINISTRALIDADE": row["SINISTRALIDADE"],
                "VALOR_FATURAMENTO": row["VALOR_FATURAMENTO"],
                "plano": plano,
                "REGISTO": "historico" if i < train_size else "teste",
                "SINISTRALIDADE_PREVISTA": (
                    plan_results[plano]["pred_test_map"].get(row["DATA"], np.nan)
                    if i >= train_size else np.nan
                ),
            })

    df_hist_all = pd.DataFrame(hist_rows)

    def _wavg(group: pd.DataFrame) -> dict[str, Any]:
        total_f = group["VALOR_FATURAMENTO"].sum()
        sin_pond = float(
            (group["SINISTRALIDADE"] * group["VALOR_FATURAMENTO"]).sum() / total_f
        ) if total_f > 0 else np.nan
        pred_vals = group["SINISTRALIDADE_PREVISTA"].dropna()
        fat_vals  = group.loc[pred_vals.index, "VALOR_FATURAMENTO"]
        sin_pred  = float((pred_vals * fat_vals).sum() / fat_vals.sum()) if (len(pred_vals) > 0 and fat_vals.sum() > 0) else np.nan
        return {
            "DATA":              group["DATA"].iloc[0],
            "COMPETENCIA":       group["COMPETENCIA"].iloc[0],
            "VALOR_FATURAMENTO": total_f,
            "SINISTRALIDADE":    sin_pond,
            "REGISTO":           group["REGISTO"].iloc[0],
            "SINISTRALIDADE_PREVISTA": sin_pred,
            "forecast_p05": np.nan, "forecast_p25": np.nan,
            "forecast_p75": np.nan, "forecast_p95": np.nan,
        }

    consol_rows: list[dict[str, Any]] = [
        _wavg(group) for _, group in df_hist_all.groupby("COMPETENCIA", sort=True)
    ]

    # Forecast futuro consolidado
    last_date_consol = plan_series[planos_com_modelo[0]]["DATA"].max()
    future_dates = pd.date_range(last_date_consol + pd.DateOffset(months=1), periods=HORIZON, freq="MS")
    for j, d in enumerate(future_dates):
        consol_rows.append({
            "DATA":              d,
            "COMPETENCIA":       d.strftime("%Y-%m"),
            "VALOR_FATURAMENTO": np.nan,
            "SINISTRALIDADE":    np.nan,
            "REGISTO":           "previsao",
            "SINISTRALIDADE_PREVISTA": sum(w[p] * float(plan_results[p]["forecast_mean"][j]) for p in planos_com_modelo),
            "forecast_p05": sum(w[p] * float(plan_results[p]["forecast_p05"][j]) for p in planos_com_modelo),
            "forecast_p25": sum(w[p] * float(plan_results[p]["forecast_p25"][j]) for p in planos_com_modelo),
            "forecast_p75": sum(w[p] * float(plan_results[p]["forecast_p75"][j]) for p in planos_com_modelo),
            "forecast_p95": sum(w[p] * float(plan_results[p]["forecast_p95"][j]) for p in planos_com_modelo),
        })

    df_consol = pd.DataFrame(consol_rows).sort_values("DATA").reset_index(drop=True)
    consol_csv = consol_dir / "sinistralidade_forecast_completo.csv"
    df_consol.to_csv(consol_csv, index=False)
    print(f"  CSV consolidado: {consol_csv}")

    # Meta geral
    all_meta["pesos_faturamento"] = {PLANO_SLUG[p]: round(w[p], 4) for p in planos_com_modelo}
    with open(out_root / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(all_meta, f, indent=2, ensure_ascii=False)
    print(f"  Meta: {out_root / 'run_metadata.json'}")

    # Resumo final
    print(f"\n{'='*60}")
    print("Resumo das metricas (conjunto de teste)")
    print(f"{'─'*60}")
    for plano in planos_com_modelo:
        slug = PLANO_SLUG[plano]
        m = all_meta["planos"][slug]["metricas"]
        print(f"  {plano[:25]:<25}  RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  MAPE={m['mape_pct']:.2f}%")
    print(f"{'='*60}\n")


# =============================================================================
# %% — Execução
# =============================================================================

if __name__ == "__main__":
    run_forecast_pipeline()

# %%
