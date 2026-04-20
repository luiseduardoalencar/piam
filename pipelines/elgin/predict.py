# %%
"""
Predict — Sinistralidade ELGIN (pipeline v2, saídas versionadas)

Script autónomo (não importa ``predict-agregado.py``).

- Lê a base transformada (``eda_predict``): Parquet em
  ``data/processed/elgin/base_analitica_transformada/``.
- Treina modelo two-stage (hurdle) por plano: lags por beneficiário, regressão ``fair``,
  filtro ``valor_faturamento`` > 0, calibração macro no holdout (``macro_scale`` no .pkl).
- Grava artefatos em ``data/processed/elgin/predict/v{N}/`` (versão auto-incrementada).
- MLflow (opcional, ``.env``): run com params/metrics; **artefato de run** só
  ``catalogo_perfis_top100.json``; o ``.pkl`` **não** é duplicado como artefato — entra apenas
  no pacote ``pyfunc`` (necessário para inferência via Registry). CSV/macro/plot ficam só na pasta vN.
- Geração de perfil (payload JSON): ``_json_safe``, ``serie_para_payload``,
  ``payload_do_parquet_por_indice`` e ``salvar_payload`` neste mesmo ficheiro.

Saídas por execução (pasta ``data/processed/elgin/predict/v{N}/``):
  - ``models/model_{plano}.pkl`` — objeto serializado com ``joblib``
  - ``predicoes_micro.csv`` — features + ``p_sinistro`` + ``sinistralidade_prevista``
  - ``resultado_macro.json`` — agregação macro no holdout
  - ``real_vs_pred.png`` — dispersão real vs previsto (holdout agregado)
  - ``catalogo_perfis_top100.json`` — 100 perfis (payload + metadados para UI / inferência)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore", category=UserWarning)

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
_elgin_dir = ROOT_DIR / "pipelines" / "elgin"
if str(_elgin_dir) not in sys.path:
    sys.path.insert(0, str(_elgin_dir))

# %%
COMPANY = "elgin"

TRANSFORMED_PARQUET_PATH = (
    ROOT_DIR
    / "data"
    / "raw"
    / COMPANY
    / "base_analitica"
    / "painel_sinistralidade_v1.parquet"
)
FEATURE_CATALOG_PATH = ROOT_DIR / "data" / "auxiliar" / COMPANY / "feature_catalog.csv"

OUTPUT_PREDICT_ROOT = ROOT_DIR / "data" / "processed" / COMPANY / "predict"

TARGET_COL = "sinistralidade_final"
TIME_COL = "competencia"
SEGMENT_COL = "plano"
PREMIUM_COL = "valor_faturamento"

LEAKAGE_COLS: frozenset[str] = frozenset(
    {
        TARGET_COL,
        "sinistralidade_raw",
        "valor_sinistro_raw",
        "valor_sinistro_alt_val",
        "valor_sinistro_ajustado",
        "sin_ref",
        "fator_ajuste_m",
        "S_real_m",
        "F_real_m",
    }
)

FEATURES_BLOQUEADAS_INTERVENCAO: frozenset[str] = frozenset(
    {
        PREMIUM_COL,
        TARGET_COL,
        "sinistralidade_raw",
        "valor_sinistro_raw",
        "valor_sinistro_alt_val",
        "valor_sinistro_ajustado",
        "sin_ref",
        "fator_ajuste_m",
        "S_real_m",
        "F_real_m",
        "cod_beneficiario",
        "competencia",
    }
)

HOLDOUT_FRAC = 0.15
N_SPLITS_CV = 5
RANDOM_STATE = 42
# Cap do alvo no treino do estágio 2 (cauda extrema); 0,99 = mais agressivo que 0,995
OUTLIER_CAP_PCT = 0.99
# ID interno para lags (removido de X antes do modelo)
BENEFICIARIO_COL = "cod_beneficiario"

# Catálogo de perfis para UI (top N por valor_sinistro_raw)
TOP_PERFIS_N = 100
FEATURES_INTERVENCAO_FILENAME = "catalogo_features_intervencao.json"

# MLflow — alinhado a ``pipelines/mvp/predict.py`` + ``config/mlflow_config.py``
MLFLOW_EXPERIMENT_NAME = "piam-elgin-predict"
MLFLOW_REGISTERED_MODEL_NAME = "elgin-sinistralidade-two-stage"

try:
    import lightgbm as lgb
except ImportError as e:
    raise ImportError(
        "Instale LightGBM: pip install lightgbm"
    ) from e


# %%
# Geração de perfil — linha do painel → dict JSON (catálogo top100 / inferência)
def _json_safe(v: Any) -> Any:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if hasattr(v, "item") and callable(v.item):
        try:
            return v.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if pd.isna(v):
        return None
    return v


def serie_para_payload(row: pd.Series, *, excluir_alvo: bool = True) -> dict[str, Any]:
    """Converte uma linha do painel em dict para ``inferencia`` / UI; omite nulos e opcionalmente o alvo."""
    payload: dict[str, Any] = {}
    for k, v in row.items():
        if excluir_alvo and k == TARGET_COL:
            continue
        try:
            is_na = bool(pd.isna(v))
        except (ValueError, TypeError):
            is_na = False
        if is_na:
            continue
        safe = _json_safe(v)
        if safe is None:
            continue
        payload[k] = safe
    return payload


def payload_do_parquet_por_indice(
    parquet_path: Path | None = None,
    indice: int = 0,
) -> dict[str, Any]:
    """Lê a linha ``indice`` (0-based) do Parquet transformado e devolve o payload (sem alvo)."""
    path = parquet_path or TRANSFORMED_PARQUET_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Parquet não encontrado: {path}")
    df = pd.read_parquet(path)
    prem_ok = pd.to_numeric(df[PREMIUM_COL], errors="coerce").fillna(0.0) > 0.0
    df = df.loc[prem_ok].copy().reset_index(drop=True)
    if indice < 0 or indice >= len(df):
        raise IndexError(
            f"indice={indice} fora do intervalo [0, {len(df) - 1}] "
            f"(n={len(df):,} linhas após filtro prêmio > 0)."
        )
    row = df.iloc[indice].copy()
    return serie_para_payload(row, excluir_alvo=True)


def salvar_payload(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


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


def plano_slug(plano: str) -> str:
    s = re.sub(r"[^\w]+", "_", str(plano)).strip("_")
    return (s[:80] if s else "plano").upper()


def load_feature_catalog() -> pd.DataFrame:
    if not FEATURE_CATALOG_PATH.is_file():
        raise FileNotFoundError(f"Catálogo inexistente: {FEATURE_CATALOG_PATH}")
    fc = pd.read_csv(FEATURE_CATALOG_PATH, encoding="utf-8-sig")
    for col in ("feature_name", "category", "dtype"):
        if col not in fc.columns:
            raise ValueError(f"feature_catalog.csv deve ter coluna '{col}'.")
    return fc


def catalog_eligible_names(fc: pd.DataFrame) -> list[str]:
    _np = {"target", "identifier", "leakage"}
    _dtype = fc["dtype"].fillna("").astype(str).str.lower()
    _cat = fc["category"].fillna("").astype(str).str.lower()
    mask = (~_dtype.isin(_np)) & (_cat != "derivada")
    return fc.loc[mask, "feature_name"].astype(str).tolist()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engenharia de features: temporal, taxas, e **lags por beneficiário** (histórico).
    ``cod_beneficiario`` é usado só para lags e removido antes do modelo (não entra em X).
    """
    d = df.copy()
    # Preserva a ordem da base de entrada (antes do sort) para alinhar catálogo / indice_parquet.
    d["_pipeline_row_id"] = np.arange(len(d), dtype=np.int64)
    # Mantém o alvo; remove apenas preditores com leakage (não ``TARGET_COL``).
    d = d.drop(
        columns=[c for c in LEAKAGE_COLS if c in d.columns and c != TARGET_COL],
        errors="ignore",
    )

    if "idade" in d.columns:
        _id = pd.to_numeric(d["idade"], errors="coerce")
        d["idade"] = np.ceil(_id).fillna(-1).astype(np.int64)

    if (
        BENEFICIARIO_COL in d.columns
        and TIME_COL in d.columns
        and TARGET_COL in d.columns
    ):
        d = d.sort_values([BENEFICIARIO_COL, TIME_COL]).reset_index(drop=True)
        gb = d.groupby(BENEFICIARIO_COL, sort=False)
        for k in (1, 2, 3):
            d[f"lag_sin_{k}"] = gb[TARGET_COL].shift(k)
        d["roll_mean_sin_3"] = d.groupby(BENEFICIARIO_COL)[TARGET_COL].transform(
            lambda x: x.rolling(3, min_periods=1).mean()
        )
        if "qtd_eventos_sinistro" in d.columns:
            d["lag_qtd_eventos_1"] = gb["qtd_eventos_sinistro"].shift(1)

    if BENEFICIARIO_COL in d.columns:
        d = d.drop(columns=[BENEFICIARIO_COL])

    if TIME_COL in d.columns:
        comp = d[TIME_COL].astype(str)
        d["mes"] = comp.str[5:7].astype(int)
        d["ano"] = comp.str[:4].astype(int)
        d["mes_sin"] = np.sin(2 * np.pi * d["mes"] / 12)
        d["mes_cos"] = np.cos(2 * np.pi * d["mes"] / 12)

    if "idade" in d.columns:
        d["faixa_etaria"] = pd.cut(
            d["idade"].fillna(-1),
            bins=[-1, 0, 5, 12, 18, 30, 45, 60, 200],
            labels=["inf", "0-5", "6-12", "13-18", "19-30", "31-45", "46-60", "60+"],
        ).astype(str)

    fat = d[PREMIUM_COL].replace(0, np.nan) if PREMIUM_COL in d.columns else None
    if fat is not None:
        for c in [col for col in d.columns if col.startswith("qtd_")]:
            d[f"tx_{c}"] = d[c] / fat

    if "tipo_cadastro" in d.columns:
        d["is_titular"] = (d["tipo_cadastro"].astype(str).str.upper() == "TITULAR").astype(np.int8)
    if "sexo" in d.columns:
        d["is_fem"] = (d["sexo"].astype(str).str.upper() == "F").astype(np.int8)

    for c in d.select_dtypes(include=["float", "int"]).columns:
        d[c] = d[c].replace([np.inf, -np.inf], np.nan).fillna(0)
    for c in d.select_dtypes(include=["object"]).columns:
        d[c] = d[c].fillna("missing").astype("category")
    for c in d.select_dtypes(include=["category"]).columns:
        if d[c].isna().any():
            d[c] = d[c].cat.add_categories(["missing"]).fillna("missing")

    return d


def resolve_feature_columns(df: pd.DataFrame, fc: pd.DataFrame) -> list[str]:
    eligible = catalog_eligible_names(fc)
    exclude = (
        LEAKAGE_COLS
        | {TARGET_COL, SEGMENT_COL, "cod_beneficiario"}
        | {TIME_COL, "_pipeline_row_id"}
    )
    candidates = [c for c in df.columns if c in eligible and c not in exclude]
    engineered = [
        c
        for c in df.columns
        if c
        in {
            "mes",
            "ano",
            "mes_sin",
            "mes_cos",
            "faixa_etaria",
            "is_titular",
            "is_fem",
        }
        or (c.startswith("tx_") and "qtd_" in c)
        or c.startswith("lag_sin_")
        or c.startswith("roll_mean_sin")
        or c.startswith("lag_qtd_")
    ]
    for c in engineered:
        if c in df.columns and c not in candidates:
            candidates.append(c)
    return sorted(set(candidates))


def ensure_no_object_dtype(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = X[c].astype("category")
    return X


class TwoStageModel:
    """
    Estágio 1: P(y > 0); estágio 2: E[log(y)|y>0].
    Compatível com ``predict`` para inferência micro.
    """

    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self._smearing = 1.0
        self.macro_scale = 1.0
        self.clf_: lgb.LGBMClassifier | None = None
        self.reg_: lgb.LGBMRegressor | None = None
        self.feature_names_: list[str] = []

    def _clf_params(self) -> dict:
        return dict(
            objective="binary",
            metric="auc",
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=6,
            min_child_samples=40,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            is_unbalance=True,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )

    def _reg_params(self) -> dict:
        # ``fair`` reduz sensibilidade a outliers na cauda (vs ``regression`` puro).
        return dict(
            objective="fair",
            metric="mae",
            fair_c=20.0,
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=6,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=2.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )

    def fit(self, X, y, sample_weight=None, outlier_cap_pct=None) -> TwoStageModel:
        self.feature_names_ = list(X.columns)
        y = y.astype(float).copy()
        if outlier_cap_pct is not None:
            cap_val = float(y.quantile(outlier_cap_pct))
            y = y.clip(upper=cap_val)
        y_bin = (y > 0).astype(int)
        self.clf_ = lgb.LGBMClassifier(**self._clf_params())
        self.clf_.fit(X, y_bin, sample_weight=sample_weight)
        pos_mask = y > 0
        X_pos = X[pos_mask]
        y_pos = np.log(y[pos_mask] + self.eps)
        self.reg_ = lgb.LGBMRegressor(**self._reg_params())
        sw = sample_weight[pos_mask] if sample_weight is not None else None
        self.reg_.fit(X_pos, y_pos, sample_weight=sw)
        residuals = y_pos.values - self.reg_.predict(X_pos)
        self._smearing = float(np.exp(residuals).mean())
        return self

    def _predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        """Predição micro sem calibração macro (``macro_scale``)."""
        p_pos = self.clf_.predict_proba(X)[:, 1]
        log_y = self.reg_.predict(X)
        return p_pos * (np.exp(log_y) * self._smearing)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._predict_raw(X) * float(getattr(self, "macro_scale", 1.0))

    def predict_stages(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        p_pos = self.clf_.predict_proba(X)[:, 1]
        scale = float(getattr(self, "macro_scale", 1.0))
        y_hat = self._predict_raw(X) * scale
        return p_pos, y_hat


def aggregate_sinistralidade_macro(y_pred: np.ndarray, premio: pd.Series) -> float:
    """
    Média ponderada do índice de sinistralidade pelo prêmio: ``sum(y_hat * premio) / sum(premio)``.
    Ver comentário homólogo em ``predict-agregado.py``.
    """
    premio = pd.to_numeric(premio, errors="coerce").fillna(0.0)
    denom = float(premio.sum())
    if denom == 0.0:
        return float("nan")
    y = np.asarray(y_pred, dtype=float).ravel()
    p = np.asarray(premio, dtype=float).ravel()
    if y.shape != p.shape:
        raise ValueError(f"y_pred e premio devem ter o mesmo tamanho: {y.shape} vs {p.shape}")
    return float((y * p).sum() / denom)


def build_catalogo_perfis_top100(
    version_label: str,
    n: int = TOP_PERFIS_N,
    *,
    criterio_col: str = "valor_sinistro_raw",
    df_raw_aligned: pd.DataFrame | None = None,
    df_feat: pd.DataFrame | None = None,
    feature_cols: list[str] | None = None,
) -> dict[str, Any]:
    """
    Top ``n`` perfis por ``criterio_col`` (desc).

    Se ``df_raw_aligned``, ``df_feat`` e ``feature_cols`` forem passados (mesma ordem e
    tamanho que no treino, após filtro prêmio > 0), ``payload`` = linha de entrada do modelo
    (pós-``build_features``), mesma ordem/tipos que o pyfunc MLflow.

    Caso contrário, lê o Parquet local (comportamento legado; ``payload`` = linha analítica).
    ``indice_parquet`` = índice 0-based na base **filtrada** (prêmio > 0), alinhado ao treino.
    """
    if (
        df_raw_aligned is not None
        and df_feat is not None
        and feature_cols is not None
    ):
        if len(df_raw_aligned) != len(df_feat):
            raise ValueError("df_raw_aligned e df_feat devem ter o mesmo número de linhas.")
        if "_pipeline_row_id" not in df_feat.columns:
            raise ValueError("df_feat deve conter _pipeline_row_id (saída de build_features).")

        df_raw_w = df_raw_aligned.reset_index(drop=True)
        key = pd.to_numeric(df_raw_w[criterio_col], errors="coerce").fillna(0.0).values
        n_take = max(1, min(n, len(df_raw_w)))
        top_positions = np.argsort(-key)[:n_take]

        perfis: list[dict[str, Any]] = []
        for rank, pos in enumerate(top_positions, start=1):
            pos = int(pos)
            # df_feat está reordenado por beneficiário/tempo; o id está em _pipeline_row_id.
            pid = pos
            raw_row = df_raw_w.iloc[pos]
            feat_row = df_feat[df_feat["_pipeline_row_id"] == pid]
            if feat_row.empty:
                continue
            payload = dataframe_row_to_payload_inferencia(
                feat_row[feature_cols].iloc[[0]], feature_cols
            )
            cb = raw_row.get("cod_beneficiario")
            comp = raw_row.get(TIME_COL)
            label = f"#{rank} | {cb} | {comp}"
            perfis.append(
                {
                    "rank": rank,
                    "indice_parquet": pid,
                    "label": label,
                    "resumo": {
                        "cod_beneficiario": _json_safe(cb),
                        "competencia": str(comp)
                        if comp is not None and not pd.isna(comp)
                        else "",
                        "valor_sinistro_raw": float(
                            pd.to_numeric(raw_row.get("valor_sinistro_raw"), errors="coerce")
                            or 0.0
                        ),
                        TARGET_COL: float(
                            pd.to_numeric(raw_row.get(TARGET_COL), errors="coerce") or 0.0
                        ),
                        "qtd_eventos_sinistro": float(
                            pd.to_numeric(raw_row.get("qtd_eventos_sinistro"), errors="coerce")
                            or 0.0
                        ),
                        PREMIUM_COL: float(
                            pd.to_numeric(raw_row.get(PREMIUM_COL), errors="coerce") or 0.0
                        ),
                        SEGMENT_COL: str(raw_row.get(SEGMENT_COL, "")),
                    },
                    "payload": payload,
                }
            )

        return {
            "versao_pasta": version_label,
            "criterio_ordenacao": criterio_col,
            "n": len(perfis),
            "descricao": (
                "Perfis reais (top por valor_sinistro_raw). payload = features do modelo "
                "(pós-build_features), ordem alinhada ao pyfunc. indice_parquet = linha na "
                "base filtrada (prêmio > 0), igual ao treino."
            ),
            "perfis": perfis,
        }

    df = pd.read_parquet(TRANSFORMED_PARQUET_PATH)
    prem_ok = pd.to_numeric(df[PREMIUM_COL], errors="coerce").fillna(0.0) > 0.0
    df = df.loc[prem_ok].copy().reset_index(drop=True)
    df.insert(0, "indice_parquet", np.arange(len(df), dtype=np.int64))
    key = pd.to_numeric(df[criterio_col], errors="coerce").fillna(0.0)
    df = df.assign(_sort_key=key).sort_values(
        ["_sort_key", "indice_parquet"], ascending=[False, True]
    ).head(max(1, n))
    df = df.drop(columns=["_sort_key"])

    perfis_legacy: list[dict[str, Any]] = []
    for rank, (_, row) in enumerate(df.iterrows(), start=1):
        row_pay = row.drop(labels=["indice_parquet"], errors="ignore")
        payload = serie_para_payload(row_pay, excluir_alvo=True)
        cb = row.get("cod_beneficiario")
        comp = row.get(TIME_COL)
        label = f"#{rank} | {cb} | {comp}"
        perfis_legacy.append(
            {
                "rank": rank,
                "indice_parquet": int(row["indice_parquet"]),
                "label": label,
                "resumo": {
                    "cod_beneficiario": _json_safe(cb),
                    "competencia": str(comp) if comp is not None and not pd.isna(comp) else "",
                    "valor_sinistro_raw": float(
                        pd.to_numeric(row.get("valor_sinistro_raw"), errors="coerce") or 0.0
                    ),
                    TARGET_COL: float(
                        pd.to_numeric(row.get(TARGET_COL), errors="coerce") or 0.0
                    ),
                    "qtd_eventos_sinistro": float(
                        pd.to_numeric(row.get("qtd_eventos_sinistro"), errors="coerce")
                        or 0.0
                    ),
                    PREMIUM_COL: float(
                        pd.to_numeric(row.get(PREMIUM_COL), errors="coerce") or 0.0
                    ),
                    SEGMENT_COL: str(row.get(SEGMENT_COL, "")),
                },
                "payload": payload,
            }
        )

    return {
        "versao_pasta": version_label,
        "criterio_ordenacao": criterio_col,
        "n": len(perfis_legacy),
        "descricao": (
            "Perfis (modo legado sem features no treino em memória). "
            "Preferir gerar o catálogo a partir do pipeline de treino."
        ),
        "perfis": perfis_legacy,
    }


def _descricao_regra_elegibilidade(feature: str) -> str:
    regras_txt: dict[str, str] = {
        "qtd_servico_CARDIOLOGIA": "idade >= 30",
        "qtd_servico_QUIMIOTERAPIA": "qtd_servico_QUIMIOTERAPIA > 0",
        "qtd_servico_ENDOSCOPIA": "idade >= 40",
        "qtd_servico_CIRURGICO": "idade >= 12",
        "qtd_servico_DIÁRIA": "qtd_conta_INTERNADO > 0",
        "qtd_esp_cardio": "idade >= 30",
        "qtd_esp_ped": "idade <= 18",
        "qtd_esp_gine": "sexo == 'F'",
        "qtd_esp_cirurg": "idade >= 12",
        "qtd_carater_eletivo": "tipo_cadastro == 'TITULAR'",
        "qtd_conta_INTERNADO": "delta>0: qtd_conta_INTERNADO > 0 | delta<0: universal",
    }
    return regras_txt.get(feature, "universal")


def build_catalogo_features_intervencao_mensal(
    *,
    version_label: str,
    df_mes: pd.DataFrame,
    fc: pd.DataFrame,
    competencia_ref: str,
) -> dict[str, Any]:
    from pipelines.predict.what_if.elegibilidade import calcular_elegibilidade

    feature_to_category = (
        fc[["feature_name", "category"]]
        .dropna(subset=["feature_name"])
        .assign(feature_name=lambda x: x["feature_name"].astype(str))
        .set_index("feature_name")["category"]
        .to_dict()
    )
    feature_to_dtype = (
        fc[["feature_name", "dtype"]]
        .dropna(subset=["feature_name"])
        .assign(feature_name=lambda x: x["feature_name"].astype(str))
        .set_index("feature_name")["dtype"]
        .to_dict()
    )

    dtypes_intervencionaveis = {"numeric", "count"}
    elegiveis_catalogo = set(catalog_eligible_names(fc))
    candidatas = [
        c
        for c in sorted(df_mes.columns)
        if (
            c in elegiveis_catalogo
            and c not in FEATURES_BLOQUEADAS_INTERVENCAO
            and str(feature_to_dtype.get(c, "")).lower() in dtypes_intervencionaveis
        )
    ]

    n_total = int(len(df_mes))
    grupos: dict[str, list[dict[str, Any]]] = {}
    for feat in candidatas:
        mask_pos = calcular_elegibilidade(df_mes, feat, 20.0)
        mask_neg = calcular_elegibilidade(df_mes, feat, -20.0)
        serie = pd.to_numeric(df_mes[feat], errors="coerce").fillna(0.0)

        item = {
            "feature": feat,
            "categoria": str(feature_to_category.get(feat, "sem_categoria")),
            "dtype_catalogo": str(feature_to_dtype.get(feat, "n/a")),
            "regra_elegibilidade": _descricao_regra_elegibilidade(feat),
            "n_elegiveis_delta_positivo": int(mask_pos.sum()),
            "n_elegiveis_delta_negativo": int(mask_neg.sum()),
            "n_nao_elegiveis_delta_positivo": int(n_total - int(mask_pos.sum())),
            "n_com_valor_atual_gt_zero_total": int((serie > 0.0).sum()),
            "n_com_valor_atual_gt_zero_elegiveis_delta_positivo": int(((serie > 0.0) & mask_pos).sum()),
            "pct_base_elegivel_delta_positivo": round(float(mask_pos.sum()) / max(1, n_total) * 100.0, 2),
        }
        grupo = item["categoria"]
        grupos.setdefault(grupo, []).append(item)

    grupos_lista: list[dict[str, Any]] = []
    for grupo_nome in sorted(grupos):
        feats = sorted(grupos[grupo_nome], key=lambda x: x["feature"])
        grupos_lista.append({"grupo": grupo_nome, "n_features": len(feats), "features": feats})

    return {
        "versao_pasta": version_label,
        "competencia_referencia": competencia_ref,
        "n_total_base_mes": n_total,
        "n_features_intervencionaveis": len(candidatas),
        "descricao": "Catálogo de features intervencionáveis para what-if no snapshot mensal.",
        "grupos": grupos_lista,
    }


def _mlflow_prepare_input_example(df: pd.DataFrame) -> pd.DataFrame:
    """MLflow exige tipos compatíveis com a assinatura (evita category vs string)."""
    x = df.copy()
    for c in x.columns:
        if isinstance(x[c].dtype, pd.CategoricalDtype):
            x[c] = x[c].astype(str)
    return x


def dataframe_row_to_payload_inferencia(
    X_one: pd.DataFrame, feature_cols: list[str]
) -> dict[str, Any]:
    """Uma linha de features (pós-engenharia) → dict alinhado ao pyfunc MLflow."""
    X = X_one[feature_cols].copy()
    X = ensure_no_object_dtype(X)
    X = _mlflow_prepare_input_example(X)
    row = X.iloc[0]
    return {c: _json_safe(row[c]) for c in feature_cols}


def _mlflow_log_run_elgin(
    *,
    run_dir: Path,
    version_label: str,
    model_path: Path | None,
    input_example: pd.DataFrame | None,
    model_final: TwoStageModel | None,
    params_extra: dict[str, Any],
    metrics_extra: dict[str, float],
) -> None:
    """Segue o padrão ``pipelines/mvp/predict.py`` + ``config/mlflow_config.py``."""
    try:
        import mlflow
        import mlflow.pyfunc
        from mlflow.models.signature import infer_signature
    except ImportError:
        print("[MLflow] Pacote ``mlflow`` não instalado; ignorando registo.")
        return

    sys.path.insert(0, str(ROOT_DIR))
    try:
        from config.mlflow_config import configurar_mlflow
    except ImportError as e:
        print(f"[MLflow] config.mlflow_config indisponível: {e}")
        return

    try:
        configurar_mlflow(MLFLOW_EXPERIMENT_NAME, preparar_experimento=True)
    except EnvironmentError as e:
        print(f"[MLflow] Configuração incompleta (.env): {e}\n[MLflow] Treino local OK; sem tracking.")
        return

    run_name = f"elgin__{version_label}"
    try:
        _mlflow_log_run_elgin_inner(
            run_name=run_name,
            run_dir=run_dir,
            version_label=version_label,
            model_path=model_path,
            input_example=input_example,
            model_final=model_final,
            params_extra=params_extra,
            metrics_extra=metrics_extra,
        )
    except Exception as e:
        print(f"[MLflow] Falha ao registar run/artefatos (treino já gravado em disco): {e}")


def _mlflow_log_run_elgin_inner(
    *,
    run_name: str,
    run_dir: Path,
    version_label: str,
    model_path: Path | None,
    input_example: pd.DataFrame | None,
    model_final: TwoStageModel | None,
    params_extra: dict[str, Any],
    metrics_extra: dict[str, float],
) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError, AttributeError):
            pass

    import mlflow
    import mlflow.pyfunc
    from mlflow.models.signature import infer_signature

    if mlflow.active_run() is not None:
        mlflow.end_run()

    class _ElginTwoStagePyFunc(mlflow.pyfunc.PythonModel):
        """Entrada: DataFrame com as mesmas colunas de features do ``.pkl``."""

        def load_context(self, context: Any) -> None:
            self._model = joblib.load(context.artifacts["model_pkl"])

        def predict(self, context: Any, model_input: pd.DataFrame) -> pd.DataFrame:
            x = ensure_no_object_dtype(model_input.copy())
            p_pos, y_hat = self._model.predict_stages(x)
            return pd.DataFrame(
                {
                    "p_sinistro": np.asarray(p_pos, dtype=float),
                    "sinistralidade_prevista": np.asarray(y_hat, dtype=float),
                }
            )

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("pipeline", "elgin_two_stage")
        mlflow.log_param("versao_pasta", version_label)
        mlflow.log_param("top_perfis_catalogo", TOP_PERFIS_N)
        mlflow.log_param("holdout_frac", HOLDOUT_FRAC)
        mlflow.log_param("outlier_cap_pct", OUTLIER_CAP_PCT)
        mlflow.log_param("n_splits_cv", N_SPLITS_CV)
        for k, v in params_extra.items():
            mlflow.log_param(k, v)

        for k, v in metrics_extra.items():
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                mlflow.log_metric(k, float(v))

        # Só o catálogo top-100 como artefato do run (resto permanece em ``run_dir`` no disco).
        # O ``.pkl`` não é logado em separado: ``mlflow.pyfunc.log_model`` já embute ``model_pkl``
        # no pacote do modelo (necessário para ``models:/.../latest``).
        cat_feat_path = run_dir / FEATURES_INTERVENCAO_FILENAME
        if cat_feat_path.is_file():
            mlflow.log_artifact(str(cat_feat_path), artifact_path="elgin_artifacts")

        if (
            model_path is not None
            and model_path.is_file()
            and model_final is not None
            and input_example is not None
        ):
            x_fit = ensure_no_object_dtype(input_example.copy())
            out_ex = model_final.predict_stages(x_fit)
            # Exemplo para assinatura MLflow: categorias -> string (evita falha pyfunc)
            ex_ml = _mlflow_prepare_input_example(x_fit.copy())
            output_example = pd.DataFrame(
                {
                    "p_sinistro": np.asarray(out_ex[0], dtype=float).ravel(),
                    "sinistralidade_prevista": np.asarray(out_ex[1], dtype=float).ravel(),
                }
            )
            signature = infer_signature(ex_ml, output_example)
            mlflow.pyfunc.log_model(
                python_model=_ElginTwoStagePyFunc(),
                artifact_path="elgin_pyfunc_model",
                artifacts={"model_pkl": str(model_path.resolve())},
                signature=signature,
                input_example=ex_ml,
                registered_model_name=MLFLOW_REGISTERED_MODEL_NAME,
            )
            print(
                f"[MLflow] Modelo pyfunc registado: {MLFLOW_REGISTERED_MODEL_NAME} "
                f"(entrada = features já engenradas, como no treino)."
            )
        else:
            print("[MLflow] Sem modelo final para pyfunc; apenas artefatos logados.")

    print(f"[MLflow] Run concluído no experimento «{MLFLOW_EXPERIMENT_NAME}».")


def run_training_pipeline(*, log_mlflow: bool = False) -> None:
    if not TRANSFORMED_PARQUET_PATH.is_file():
        raise FileNotFoundError(
            f"Base transformada não encontrada: {TRANSFORMED_PARQUET_PATH}\n"
            "Execute ``pipelines/elgin/eda/eda_predict.py`` primeiro."
        )

    df_raw = pd.read_parquet(TRANSFORMED_PARQUET_PATH)
    print(f"[Carga] {TRANSFORMED_PARQUET_PATH} | shape={df_raw.shape}")

    fc = load_feature_catalog()

    n_before_q = len(df_raw)
    prem_ok = pd.to_numeric(df_raw[PREMIUM_COL], errors="coerce").fillna(0.0) > 0.0
    df_raw = df_raw.loc[prem_ok].copy()
    n_drop = n_before_q - len(df_raw)
    if n_drop:
        print(
            f"[Qualidade] Removidas {n_drop} linhas com {PREMIUM_COL} <= 0 ou inválido "
            f"(restam {len(df_raw)})."
        )

    df = build_features(df_raw)
    feature_cols = resolve_feature_columns(df, fc)
    if not feature_cols:
        raise ValueError("Lista de features vazia após catálogo + engenharia.")

    print(f"[Features] n={len(feature_cols)} (sem identificadores / leakage)")

    VERSION_LABEL, RUN_DIR = next_version_dir(OUTPUT_PREDICT_ROOT)
    MODELS_DIR = RUN_DIR / "models"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[Versão] Nova pasta de execução: {RUN_DIR} ({VERSION_LABEL})")

    segment_values = sorted(df[SEGMENT_COL].dropna().astype(str).unique())
    print(f"[Segmentos] {segment_values}")

    all_pred_rows: list[pd.DataFrame] = []
    holdout_y_true: list[np.ndarray] = []
    holdout_y_pred: list[np.ndarray] = []
    holdout_premio: list[pd.Series] = []

    last_model_path: Path | None = None
    last_input_example: pd.DataFrame | None = None
    last_model_final: TwoStageModel | None = None

    metrics_mlflow: dict[str, float] = {}
    trained_segments: list[str] = []

    for plano in segment_values:
        print("\n" + "=" * 60)
        print(f"Plano: {plano}")
        print("=" * 60)

        mask = df[SEGMENT_COL].astype(str) == plano
        df_seg = (
            df.loc[mask]
            .dropna(subset=[TARGET_COL])
            .sort_values(TIME_COL)
            .reset_index(drop=True)
        )
        n_tot = len(df_seg)
        if n_tot < 500:
            print(f"[PULAR] Poucas linhas ({n_tot}); mínimo = 500.")
            continue

        X_all = df_seg[feature_cols].copy()
        y_all = df_seg[TARGET_COL].astype(float)
        fat_all = df_seg[PREMIUM_COL].astype(float)

        X_all = ensure_no_object_dtype(X_all)
        sample_weight_all = fat_all.values

        idx_split = max(1, min(int(n_tot * (1 - HOLDOUT_FRAC)), n_tot - 300))
        X_train_ho = X_all.iloc[:idx_split]
        X_ho = X_all.iloc[idx_split:]
        y_train_ho = y_all.iloc[:idx_split]
        y_ho = y_all.iloc[idx_split:]
        fat_ho = fat_all.iloc[idx_split:]
        pct_ho = 100.0 * (n_tot - idx_split) / n_tot
        w_train_ho = sample_weight_all[:idx_split]

        tsc = TimeSeriesSplit(n_splits=N_SPLITS_CV)
        cv_metrics: list[dict] = []
        for tr, te in tsc.split(X_train_ho):
            m = TwoStageModel()
            m.fit(
                X_train_ho.iloc[tr],
                y_train_ho.iloc[tr],
                sample_weight=w_train_ho[tr],
                outlier_cap_pct=OUTLIER_CAP_PCT,
            )
            yp = m.predict(X_train_ho.iloc[te])
            cv_metrics.append(
                {
                    "mae": mean_absolute_error(y_train_ho.iloc[te], yp),
                    "rmse": root_mean_squared_error(y_train_ho.iloc[te], yp),
                    "r2": r2_score(y_train_ho.iloc[te], yp),
                }
            )

        df_cv = pd.DataFrame(cv_metrics)
        print(
            f"[CV TimeSeriesSplit x{N_SPLITS_CV}] "
            f"MAE={df_cv['mae'].mean():.4f} | "
            f"RMSE={df_cv['rmse'].mean():.4f} | "
            f"R²={df_cv['r2'].mean():.4f}"
        )

        model_ho = TwoStageModel()
        model_ho.fit(
            X_train_ho,
            y_train_ho,
            sample_weight=w_train_ho,
            outlier_cap_pct=OUTLIER_CAP_PCT,
        )
        p_pos_ho, y_ho_pred = model_ho.predict_stages(X_ho)
        y_ho_np = np.asarray(y_ho, dtype=float)
        y_pred_np = np.asarray(y_ho_pred, dtype=float)

        mae_ho = float(mean_absolute_error(y_ho_np, y_pred_np))
        rmse_ho = float(root_mean_squared_error(y_ho_np, y_pred_np))
        r2_ho = float(r2_score(y_ho_np, y_pred_np))
        print(
            f"[Holdout ~{pct_ho:.1f}%] MAE={mae_ho:.4f} | RMSE={rmse_ho:.4f} | R²={r2_ho:.4f}"
        )

        cap_eval = np.percentile(y_ho_np, 99)
        mask_no_ext = y_ho_np <= cap_eval
        r2_no_ext = float(r2_score(y_ho_np[mask_no_ext], y_pred_np[mask_no_ext]))
        print(f"  R² holdout (excl. acima do p99 do real): {r2_no_ext:.4f}")

        y_bin_ho = (y_ho_np > 0).astype(int)
        auc_ho = float(roc_auc_score(y_bin_ho, p_pos_ho))
        ap_ho = float(average_precision_score(y_bin_ho, p_pos_ho))
        print(f"  Classificador: AUC={auc_ho:.4f} | AP={ap_ho:.4f}")

        sin_real = aggregate_sinistralidade_macro(y_ho_np, fat_ho)
        sin_pred = aggregate_sinistralidade_macro(y_pred_np, fat_ho)
        erro_macro = (
            abs(sin_pred - sin_real) / sin_real
            if sin_real and not np.isnan(sin_real)
            else float("nan")
        )
        print(f"  Macro holdout: real={sin_real:.6f} | pred={sin_pred:.6f} | erro_rel={erro_macro:.2%}")

        model_final = TwoStageModel()
        model_final.fit(
            X_all,
            y_all,
            sample_weight=sample_weight_all,
            outlier_cap_pct=OUTLIER_CAP_PCT,
        )

        # Calibração macro SEM leakage: usar predição bruta do modelo de holdout
        # (treinado apenas em treino) sobre X_ho.
        raw_ho = model_ho._predict_raw(X_ho)
        sin_real_h = aggregate_sinistralidade_macro(y_ho.values, fat_ho)
        sin_pred_raw_h = aggregate_sinistralidade_macro(raw_ho, fat_ho)
        macro_scale = 1.0
        if (
            sin_pred_raw_h
            and not np.isnan(sin_pred_raw_h)
            and abs(sin_pred_raw_h) > 1e-15
        ):
            macro_scale = float(sin_real_h / sin_pred_raw_h)
        model_final.macro_scale = macro_scale
        print(
            f"[Calibração macro] fator holdout (real/pred bruto, sem leakage): "
            f"{model_final.macro_scale:.4f}"
        )

        y_ho_final = model_final.predict(X_ho)
        mae_ho_fin = float(mean_absolute_error(y_ho_np, y_ho_final))
        print(f"[Holdout modelo final + calibração] MAE={mae_ho_fin:.4f}")

        p_full, y_hat_full = model_final.predict_stages(X_all)
        out_micro = X_all.copy()
        out_micro.insert(0, SEGMENT_COL, df_seg[SEGMENT_COL].values)
        out_micro["p_sinistro"] = p_full
        out_micro["sinistralidade_prevista"] = y_hat_full
        all_pred_rows.append(out_micro)

        holdout_y_true.append(y_ho_np)
        holdout_y_pred.append(y_ho_final)
        holdout_premio.append(fat_ho.reset_index(drop=True))

        slug = plano_slug(plano)
        model_path = MODELS_DIR / f"model_{slug}.pkl"
        joblib.dump(model_final, model_path)
        print(f"[OK] Modelo salvo: {model_path}")

        last_model_path = model_path
        last_input_example = X_all.iloc[[0]].copy()
        last_model_final = model_final
        trained_segments.append(str(plano))

    if not all_pred_rows:
        print("Nenhum segmento treinado; não há artefatos de predição.")
    else:
        pred_micro = pd.concat(all_pred_rows, axis=0, ignore_index=True)
        micro_path = RUN_DIR / "predicoes_micro.csv"
        pred_micro.to_csv(micro_path, index=False, encoding="utf-8-sig")
        print(f"\n[Artefato] {micro_path}")

        yt = np.concatenate(holdout_y_true)
        yp = np.concatenate(holdout_y_pred)
        pr = pd.concat(holdout_premio, axis=0, ignore_index=True)

        sin_real_g = aggregate_sinistralidade_macro(yt, pr)
        sin_pred_g = aggregate_sinistralidade_macro(yp, pr)
        erro_rel = (
            abs(sin_pred_g - sin_real_g) / sin_real_g
            if sin_real_g and not np.isnan(sin_real_g)
            else float("nan")
        )

        macro_payload = {
            "versao": VERSION_LABEL,
            "sinistralidade_real": float(sin_real_g),
            "sinistralidade_prevista": float(sin_pred_g),
            "erro_relativo": float(erro_rel),
        }
        macro_path = RUN_DIR / "resultado_macro.json"
        with open(macro_path, "w", encoding="utf-8") as f:
            json.dump(macro_payload, f, indent=2, ensure_ascii=False)
        print(f"[Artefato] {macro_path}")
        print(
            f"[Macro agregado holdout] real={sin_real_g:.6f} | pred={sin_pred_g:.6f} | erro_rel={erro_rel:.4f}"
        )

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(yt, yp, alpha=0.25, s=8)
        lims = [float(min(yt.min(), yp.min())), float(max(yt.max(), yp.max()))]
        ax.plot(lims, lims, "k--", lw=1)
        ax.set_xlabel("Sinistralidade real (holdout)")
        ax.set_ylabel("Sinistralidade prevista (holdout)")
        ax.set_title(f"ELGIN real vs pred — {VERSION_LABEL}")
        plt.tight_layout()
        plot_path = RUN_DIR / "real_vs_pred.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"[Artefato] {plot_path}")

        metrics_mlflow["holdout_macro_sin_real"] = float(sin_real_g)
        metrics_mlflow["holdout_macro_sin_pred"] = float(sin_pred_g)
        metrics_mlflow["holdout_macro_erro_relativo"] = float(erro_rel)

    try:
        comp_ref = str(pd.to_datetime(df_raw[TIME_COL], errors="coerce").max().strftime("%Y-%m"))
        comp_dt = pd.to_datetime(df_raw[TIME_COL], errors="coerce")
        mask_comp = comp_dt.dt.to_period("M") == pd.Period(comp_ref, freq="M")
        df_mes_ref = df_raw.loc[mask_comp].copy().reset_index(drop=True)
        cat_feat = build_catalogo_features_intervencao_mensal(
            version_label=VERSION_LABEL,
            df_mes=df_mes_ref,
            fc=fc,
            competencia_ref=comp_ref,
        )
        catalog_feat_path = RUN_DIR / FEATURES_INTERVENCAO_FILENAME
        with open(catalog_feat_path, "w", encoding="utf-8") as f:
            json.dump(cat_feat, f, indent=2, ensure_ascii=False)
        print(f"[Artefato] {catalog_feat_path}")
    except Exception as e:
        print(f"[aviso] catalogo_features_intervencao: {e}")

    if last_model_final is not None and last_model_path is not None:
        metrics_mlflow["macro_scale"] = float(
            getattr(last_model_final, "macro_scale", 1.0)
        )

    if log_mlflow:
        _mlflow_log_run_elgin(
            run_dir=RUN_DIR,
            version_label=VERSION_LABEL,
            model_path=last_model_path,
            input_example=last_input_example,
            model_final=last_model_final,
            params_extra={
                "n_features": len(feature_cols),
                "n_linhas_pos_filtro_premio": len(df_raw),
                "segmentos_treinados": ",".join(trained_segments),
            },
            metrics_extra=metrics_mlflow,
        )

    print(f"\n=== Concluído — saída em {RUN_DIR} ===")


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treino mensal ELGIN (sinistralidade)")
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Após gravar artefatos locais, registar run/modelo no MLflow",
    )
    args = parser.parse_args()
    run_training_pipeline(log_mlflow=args.mlflow)

# %%
