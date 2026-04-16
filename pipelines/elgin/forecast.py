# %%
from __future__ import annotations

import json
import os
import pickle
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Antes do resto do pipeline: carrega .env (paths e credenciais alinhados ao mlflow_config quando corres a célula MLflow).
from dotenv import load_dotenv

load_dotenv(dotenv_path=ROOT_DIR / ".env")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore", category=UserWarning)

# TensorFlow: treino em CPU (padrao). Menos ruído na importação (avisos C++).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

try:
    import tensorflow as tf

    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.models import Sequential
except ImportError as e:
    raise ImportError(
        "TensorFlow é necessário para este pipeline. Instale com: pip install tensorflow>=2.14"
    ) from e


# =============================================================================
# %% — Constantes
# =============================================================================

COMPANY = "elgin"
INPUT_CSV = ROOT_DIR / "data" / "auxiliar" / COMPANY / "serie_historica_sinistralidade_por_dia.csv"
OUTPUT_FORECAST_ROOT = ROOT_DIR / "data" / "processed" / COMPANY / "forecast"

TARGET_COL = "SINISTRALIDADE"
DATE_COL = "DATA"
COMP_COL = "COMPETENCIA"

LOOK_BACK = 14
FORECAST_HORIZON_DAYS = 365
RANDOM_STATE = 42
TRAIN_FRACTION = 0.8

EPOCHS = 120
BATCH_SIZE = 32
# 0 = silencioso; 1 = uma linha por epoca (loss/val_loss); 2 = uma linha por step
FIT_VERBOSE = 1
DROPOUT = 0.2
# Huber no espaço escalado [0,1]: robusto a outliers vs MSE puro
HUBER_DELTA = 0.1

# Experimento MLflow dedicado (não reutiliza ``piam-elgin-predict`` / ``feature-impact``)
MLFLOW_EXPERIMENT_NAME = "piam-elgin-forecast"
MLFLOW_ARTIFACT_CSV_FILENAME = "sinistralidade_forecast_completo.csv"
# Pasta dentro do run MLflow onde o CSV é guardado
MLFLOW_ARTIFACT_PATH = "sinistralidade_diaria"

PLOT_FILENAME = "forecast_series.png"
MODEL_FILENAME = "lstm_forecast.keras"
SCALER_FILENAME = "minmax_scaler.pkl"
META_FILENAME = "meta.json"

# MLflow: o registo (run + artefatos) só corre quando corres a célula «MLflow» no fim do ficheiro.


# =============================================================================
# %% — Definições: treino LSTM, artefatos e métricas (sem MLflow)
# =============================================================================


def next_version_dir(root: Path) -> tuple[str, Path]:
    """Detecta v1, v2, ... e devolve ('vN', root/'vN')."""
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


def make_windows(data_array: np.ndarray, lb: int) -> tuple[np.ndarray, np.ndarray]:
    """Janelas deslizantes: X (n, lb), y (n,)."""
    x_list: list[np.ndarray] = []
    y_list: list[float] = []
    for i in range(lb, len(data_array)):
        x_list.append(data_array[i - lb : i, 0])
        y_list.append(float(data_array[i, 0]))
    return np.array(x_list, dtype=np.float64), np.array(y_list, dtype=np.float64)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    eps = 1e-8
    return float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100.0)


def target_log1p(raw: np.ndarray) -> np.ndarray:
    """Estabiliza variância: log(1+x) com x >= 0 (sinistralidade não negativa)."""
    v = np.asarray(raw, dtype=np.float64)
    v = np.maximum(v, 0.0)
    return np.log1p(v)


def scaled_to_original_y(
    y_scaled_2d: np.ndarray, scaler: MinMaxScaler
) -> np.ndarray:
    """Inverte MinMaxScaler e expm1 para escala original do alvo."""
    inv = scaler.inverse_transform(np.asarray(y_scaled_2d, dtype=np.float64).reshape(-1, 1))
    return np.expm1(inv.flatten())


def build_lstm_model(look_back: int) -> Any:
    model = Sequential(
        [
            LSTM(64, return_sequences=True, input_shape=(look_back, 1)),
            Dropout(DROPOUT),
            LSTM(32, return_sequences=False),
            Dropout(DROPOUT),
            Dense(1),
        ]
    )
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.Huber(delta=HUBER_DELTA),
    )
    return model


def fit_lstm(
    X: np.ndarray,
    y: np.ndarray,
) -> Any:
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=12,
        restore_best_weights=True,
        verbose=1,
    )
    model = build_lstm_model(LOOK_BACK)
    X_ = X.reshape(X.shape[0], LOOK_BACK, 1)
    history = model.fit(
        X_,
        y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=FIT_VERBOSE,
    )
    _ = history
    return model


def load_series(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if DATE_COL not in df.columns or TARGET_COL not in df.columns:
        raise ValueError(f"CSV deve conter {DATE_COL} e {TARGET_COL}: {path}")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL, TARGET_COL]).sort_values(DATE_COL)
    df = df.drop_duplicates(subset=[DATE_COL], keep="last")
    return df.reset_index(drop=True)


@dataclass
class ForecastRunResult:
    """Saídas do pipeline em disco + métricas (consola / meta local; não vão para MLflow)."""

    version_label: str
    output_dir: Path
    csv_path: Path
    metric_rmse: float
    metric_mae: float
    metric_mape: float


# Último treino executado no bloco «Treino»; lido pelo bloco «MLflow».
LAST_FORECAST_RUN: ForecastRunResult | None = None


def run_forecast_pipeline() -> ForecastRunResult:
    """Carrega dados, treina LSTM, grava CSV/PNG/modelo/meta em ``vN``."""
    ver, output_dir = next_version_dir(OUTPUT_FORECAST_ROOT)
    print(f"Versão de saída: {ver} -> {output_dir}")

    df = load_series(INPUT_CSV)
    y_series = df.set_index(DATE_COL)[TARGET_COL].astype(float)
    y_model = y_series.sort_index()

    n = len(y_model)
    train_size = int(n * TRAIN_FRACTION)
    if train_size <= LOOK_BACK + 10 or n - train_size <= LOOK_BACK:
        raise ValueError(
            "Série demasiado curta para o split 80/20 com look_back=14. Aumente os dados."
        )

    train_idx = y_model.index[:train_size]
    test_idx = y_model.index[train_size:]
    train_y = y_model.loc[train_idx]
    test_y = y_model.loc[test_idx]

    train_y_log = target_log1p(train_y.values)
    test_y_log = target_log1p(test_y.values)

    scaler_eval = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler_eval.fit_transform(train_y_log.reshape(-1, 1))
    test_scaled = scaler_eval.transform(test_y_log.reshape(-1, 1))
    full_scaled_eval = np.vstack([train_scaled, test_scaled])

    X_all_e, y_all_e = make_windows(full_scaled_eval, LOOK_BACK)
    split_point = len(train_y) - LOOK_BACK
    X_train_e = X_all_e[:split_point]
    y_train_e = y_all_e[:split_point]
    X_test_e = X_all_e[split_point:]

    model_eval = fit_lstm(X_train_e, y_train_e)
    pred_test_scaled = model_eval.predict(X_test_e.reshape(-1, LOOK_BACK, 1), verbose=0)
    lstm_pred_test = scaled_to_original_y(pred_test_scaled, scaler_eval)
    lstm_pred_series = pd.Series(lstm_pred_test, index=test_y.index)

    y_true = test_y.values
    y_hat = lstm_pred_series.reindex(test_y.index).values
    metric_rmse = rmse(y_true, y_hat)
    metric_mae = float(mean_absolute_error(y_true, y_hat))
    metric_mape = mape_pct(y_true, y_hat)

    y_full_log = target_log1p(y_model.values)
    scaler_full = MinMaxScaler(feature_range=(0, 1))
    full_scaled = scaler_full.fit_transform(y_full_log.reshape(-1, 1))
    X_full, y_full = make_windows(full_scaled, LOOK_BACK)
    model_final = fit_lstm(X_full, y_full)

    window = full_scaled[-LOOK_BACK:, 0].copy()
    future_scaled: list[float] = []
    for _ in range(FORECAST_HORIZON_DAYS):
        x_in = window.reshape(1, LOOK_BACK, 1)
        p = float(model_final.predict(x_in, verbose=0)[0, 0])
        future_scaled.append(p)
        window = np.append(window[1:], p)

    future_vals = scaled_to_original_y(
        np.array(future_scaled, dtype=np.float64).reshape(-1, 1),
        scaler_full,
    )

    last_date = y_model.index.max()
    future_dates = pd.date_range(
        last_date + pd.Timedelta(days=1), periods=FORECAST_HORIZON_DAYS, freq="D"
    )

    out_rows: list[dict[str, Any]] = []
    pred_test_map = lstm_pred_series.to_dict()

    for _, row in df.iterrows():
        d = pd.Timestamp(row[DATE_COL])
        rec: dict[str, Any] = {c: row[c] for c in df.columns}
        if d in train_idx:
            rec["REGISTO"] = "historico"
            rec["SINISTRALIDADE_PREVISTA"] = np.nan
        elif d in test_idx:
            rec["REGISTO"] = "teste"
            rec["SINISTRALIDADE_PREVISTA"] = pred_test_map.get(d, np.nan)
        else:
            rec["REGISTO"] = "historico"
            rec["SINISTRALIDADE_PREVISTA"] = np.nan
        out_rows.append(rec)

    for d, pv in zip(future_dates, future_vals):
        out_rows.append(
            {
                DATE_COL: d,
                COMP_COL: d.strftime("%Y-%m"),
                "VALOR_FATURAMENTO": np.nan,
                "VALOR_SINISTRO": np.nan,
                TARGET_COL: np.nan,
                "REGISTO": "previsao",
                "SINISTRALIDADE_PREVISTA": float(pv),
            }
        )

    df_out = pd.DataFrame(out_rows)
    csv_path = output_dir / MLFLOW_ARTIFACT_CSV_FILENAME
    df_out.to_csv(csv_path, index=False)
    print(f"CSV completo: {csv_path}")

    fig, ax = plt.subplots(figsize=(14, 5))
    obs_mask = df_out["REGISTO"].isin(["historico", "teste"])
    ax.plot(
        pd.to_datetime(df_out.loc[obs_mask, DATE_COL]),
        df_out.loc[obs_mask, TARGET_COL],
        label="Sinistralidade observada",
        color="C0",
        linewidth=1.2,
    )
    prev_mask = df_out["SINISTRALIDADE_PREVISTA"].notna()
    ax.plot(
        pd.to_datetime(df_out.loc[prev_mask, DATE_COL]),
        df_out.loc[prev_mask, "SINISTRALIDADE_PREVISTA"],
        label="Sinistralidade prevista",
        color="C1",
        linewidth=1.2,
        alpha=0.95,
    )
    ax.set_xlabel("Data")
    ax.set_ylabel("Sinistralidade (%)")
    ax.set_title("Série diária — observado vs previsto (teste + horizonte)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = output_dir / PLOT_FILENAME
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Gráfico: {plot_path}")

    model_path = output_dir / MODEL_FILENAME
    scaler_path = output_dir / SCALER_FILENAME
    model_final.save(model_path)
    with open(scaler_path, "wb") as sf:
        pickle.dump(scaler_full, sf, protocol=pickle.HIGHEST_PROTOCOL)

    meta = {
        "version": ver,
        "input_csv": str(INPUT_CSV),
        "look_back": LOOK_BACK,
        "forecast_horizon_days": FORECAST_HORIZON_DAYS,
        "target_transform": "log1p_then_minmax",
        "inverse_transform": "scaler_inverse_then_expm1",
        "loss": f"huber(delta={HUBER_DELTA})",
        "train_rows": int(len(train_y)),
        "test_rows": int(len(test_y)),
        "n_series": int(n),
        "metrics_test": {"rmse": metric_rmse, "mae": metric_mae, "mape_pct": metric_mape},
        "model_file": MODEL_FILENAME,
        "scaler_file": SCALER_FILENAME,
        "csv_file": MLFLOW_ARTIFACT_CSV_FILENAME,
        "plot_file": PLOT_FILENAME,
        "last_observation_date": str(last_date.date()),
        "forecast_until": str(future_dates.max().date()),
    }
    with open(output_dir / META_FILENAME, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Modelo: {model_path} | scaler: {scaler_path} | meta: {output_dir / META_FILENAME}")

    result = ForecastRunResult(
        version_label=ver,
        output_dir=output_dir,
        csv_path=csv_path,
        metric_rmse=metric_rmse,
        metric_mae=metric_mae,
        metric_mape=metric_mape,
    )
    log_metricas_avaliacao_previsoes(result)
    return result


def log_metricas_avaliacao_previsoes(run: ForecastRunResult) -> None:
    """Consola: métricas de holdout (não são enviadas ao MLflow)."""
    print()
    print("=" * 60)
    print("Métricas avaliativas das previsões (conjunto de teste — holdout)")
    print("-" * 60)
    print(f"  RMSE: {run.metric_rmse:.6f}")
    print(f"  MAE:  {run.metric_mae:.6f}")
    print(f"  MAPE: {run.metric_mape:.4f} %")
    print("=" * 60)
    print()


# =============================================================================
# %% — Treino: executa LSTM, grava artefatos e mostra métricas (tudo nesta célula)
# =============================================================================
#
# Correr esta célula = treino completo + CSV/PNG/modelo + tabela de métricas na consola.
# Só depois, se estiveres satisfeito, corre o bloco «MLflow» abaixo.

LAST_FORECAST_RUN = run_forecast_pipeline()


# =============================================================================
# %% — MLflow: registo do CSV (só depois de validares treino e ficheiros em disco)
# =============================================================================
#
# Definição e chamada juntas — não depende de outra célula para «invocar» a função.


def _mlflow_log_forecast_csv(run: ForecastRunResult) -> None:
    """
    Experimento ``MLFLOW_EXPERIMENT_NAME``: regista **somente** o CSV com observações
    e previsões. Sem ``log_param`` / ``log_metric``.
    Importa ``mlflow_config`` antes de ``mlflow`` para o tracking URI do .env aplicar primeiro.
    """
    csv_path = run.csv_path
    if not csv_path.is_file():
        print("\n[MLflow] CSV não gerado — registo ignorado.")
        return

    try:
        from config.mlflow_config import configurar_mlflow
    except ImportError as e:
        print(f"[MLflow] config.mlflow_config indisponível: {e}")
        return

    try:
        import mlflow
    except ImportError:
        print("[MLflow] Pacote ``mlflow`` não instalado; ignorando registo.")
        return

    try:
        configurar_mlflow(MLFLOW_EXPERIMENT_NAME, preparar_experimento=True)
    except EnvironmentError as e:
        print(
            f"[MLflow] Configuração incompleta (.env): {e}\n"
            "[MLflow] Pipeline OK em disco; sem tracking."
        )
        return

    run_name = f"elgin__{run.version_label}"
    try:
        if mlflow.active_run() is not None:
            mlflow.end_run()
        with mlflow.start_run(run_name=run_name):
            mlflow.log_artifact(str(csv_path), artifact_path=MLFLOW_ARTIFACT_PATH)
        print(
            f"\n[OK] MLflow run '{run_name}' | experimento «{MLFLOW_EXPERIMENT_NAME}» "
            f"| artefato único: {MLFLOW_ARTIFACT_CSV_FILENAME} (em `{MLFLOW_ARTIFACT_PATH}/`)"
        )
    except Exception as e:
        print(f"[MLflow] Falha ao registar artefato (ficheiros locais OK): {e}")
        _ename = type(e).__name__
        if _ename == "ClientError":
            try:
                code = e.response.get("Error", {}).get("Code", "?")
                msg = e.response.get("Error", {}).get("Message", "")
                print(f"[MLflow] S3/boto: Code={code} | Message={msg}")
            except Exception:
                pass
        elif any(
            x in _ename or x in str(e)
            for x in ("Credential", "Unable to locate credentials", "AccessDenied", "InvalidAccessKeyId")
        ):
            print(
                "[MLflow] Credenciais AWS: confirme AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, "
                "AWS_DEFAULT_REGION; com SSO/assume role inclua AWS_SESSION_TOKEN no .env."
            )
        import traceback

        traceback.print_exc()


if LAST_FORECAST_RUN is None:
    print(
        "[forecast] MLflow: corra antes o bloco «Treino» para definir LAST_FORECAST_RUN."
    )
else:
    _mlflow_log_forecast_csv(LAST_FORECAST_RUN)

# %%
