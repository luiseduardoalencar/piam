# %%
"""
Predict Mensal — Sinistralidade CLIMAZON (Two-Stage Temporal)

Treina modelo two-stage (hurdle) por plano usando TimeSeriesSplit.
Sem MLflow — artefatos gravados apenas em disco local.

Saídas em data/processed/climazon/predict/vN/:
  models/model_{PLANO_SLUG}.pkl          — modelo serializado (joblib)
  catalogo_perfis_top100.json            — ✅ consumido pelo app
  catalogo_features_intervencao.json     — ✅ consumido pelo app
  predicoes_micro.csv                    — features + p_sinistro + sinistralidade_prevista
  resultado_macro.json                   — métricas agregadas no holdout
  real_vs_pred.png                       — dispersão real × previsto
  run_metadata.json                      — data, versão, métricas, features usadas

Execução: célula a célula (VS Code / Cursor) ou python predict_climazon.py
"""
#%%

# %%
from __future__ import annotations

import json
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore", category=UserWarning)

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    import lightgbm as lgb
except ImportError as e:
    raise ImportError("Instale LightGBM: pip install lightgbm") from e


# =============================================================================
# %% — Constantes
# =============================================================================

COMPANY = "climazon"

TRANSFORMED_PARQUET_PATH = (
    ROOT_DIR / "data" / "processed" / COMPANY
    / "base_analitica_transformada"
    / "painel_sinistralidade_climazon_v1.parquet"
)
FEATURE_CATALOG_PATH = ROOT_DIR / "data" / "auxiliar" / COMPANY / "feature_catalog.csv"
OUTPUT_PREDICT_ROOT  = ROOT_DIR / "data" / "processed" / COMPANY / "predict"

TARGET_COL       = "sinistralidade_final"
TIME_COL         = "competencia"
SEGMENT_COL      = "plano"
PREMIUM_COL      = "valor_faturamento"
BENEFICIARIO_COL = "cod_beneficiario"

PLANOS_CANONICOS = ["MASTER EMPRESARIAL", "MASTER EXECUTIVO"]

LEAKAGE_COLS: frozenset[str] = frozenset({
    TARGET_COL,
    "sinistralidade_raw",
    "valor_sinistro_raw",
    "valor_sinistro_alt_val",
    "valor_sinistro_ajustado",
    "sin_ref",
    "fator_ajuste_m",
    "S_real_m",
    "F_real_m",
})

QUASI_LEAKAGE_COLS: frozenset[str] = frozenset({
    "qtd_eventos_sinistro",
    "qtd_carater_eletivo",
    "qtd_carater_urgencia",
})

FEATURES_BLOQUEADAS_INTERVENCAO: frozenset[str] = (
    LEAKAGE_COLS
    | QUASI_LEAKAGE_COLS
    | {PREMIUM_COL, BENEFICIARIO_COL, TIME_COL, SEGMENT_COL}
)

HOLDOUT_FRAC      = 0.15
N_SPLITS_CV       = 5
RANDOM_STATE      = 42
OUTLIER_CAP_PCT   = 0.999  # menos agressivo: modelo vê mais da cauda no treino
MIN_ROWS_TEMPORAL = 200    # reduzido vs Elgin (500) para incluir MASTER EXECUTIVO
TOP_PERFIS_N      = 100

# Dispositivo de treino do LightGBM.
# "cpu"  → sempre funciona
# "cuda" → requer build CUDA: pip install lightgbm --extra-index-url https://pypi.nvidia.com
# "gpu"  → requer OpenCL (não disponível no WSL)
DEVICE = "cpu"


# =============================================================================
# %% — Utilitários gerais
# =============================================================================

def _json_safe(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if hasattr(v, "item"):
        try:
            return v.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


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


def plano_slug(plano: str) -> str:
    s = re.sub(r"[^\w]+", "_", str(plano)).strip("_")
    return (s[:80] if s else "plano").upper()


def ensure_no_object_dtype(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype("category")
    return X


# =============================================================================
# %% — Catálogo de features
# =============================================================================

def load_feature_catalog() -> pd.DataFrame:
    if not FEATURE_CATALOG_PATH.is_file():
        raise FileNotFoundError(f"Catálogo inexistente: {FEATURE_CATALOG_PATH}")
    fc = pd.read_csv(FEATURE_CATALOG_PATH, encoding="utf-8-sig")
    for col in ("feature_name", "category", "dtype"):
        if col not in fc.columns:
            raise ValueError(f"feature_catalog.csv deve ter coluna '{col}'.")
    return fc


def catalog_eligible_names(fc: pd.DataFrame) -> list[str]:
    if "include_in_model" in fc.columns:
        mask = fc["include_in_model"].astype(str).str.lower().isin({"true", "1", "yes"})
        return fc.loc[mask, "feature_name"].astype(str).tolist()
    # fallback: mesma lógica do Elgin
    _blocked = {"target", "identifier", "leakage", "quasi_leakage"}
    mask = ~fc["dtype"].fillna("").astype(str).str.lower().isin(_blocked)
    return fc.loc[mask, "feature_name"].astype(str).tolist()


# =============================================================================
# %% — Feature engineering
# =============================================================================

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engenharia de features: lags por beneficiário, taxas, sazonalidade."""
    d = df.copy()
    d["_pipeline_row_id"] = np.arange(len(d), dtype=np.int64)
    d = d.drop(
        columns=[c for c in LEAKAGE_COLS | QUASI_LEAKAGE_COLS
                 if c in d.columns and c != TARGET_COL],
        errors="ignore",
    )

    if "idade" in d.columns:
        d["idade"] = (
            np.ceil(pd.to_numeric(d["idade"], errors="coerce"))
            .fillna(-1)
            .astype(np.int64)
        )

    if (BENEFICIARIO_COL in d.columns
            and TIME_COL in d.columns
            and TARGET_COL in d.columns):
        d = d.sort_values([BENEFICIARIO_COL, TIME_COL]).reset_index(drop=True)
        gb = d.groupby(BENEFICIARIO_COL, sort=False)
        for k in (1, 2, 3):
            d[f"lag_sin_{k}"] = gb[TARGET_COL].shift(k)
        d["roll_mean_sin_3"] = d.groupby(BENEFICIARIO_COL)[TARGET_COL].transform(
            lambda x: x.rolling(3, min_periods=1).mean()
        )

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

    if PREMIUM_COL in d.columns:
        fat = d[PREMIUM_COL].replace(0, np.nan)
        for c in [col for col in d.columns if col.startswith("qtd_")]:
            d[f"tx_{c}"] = d[c] / fat

    if "tipo_cadastro" in d.columns:
        d["is_titular"] = (
            d["tipo_cadastro"].astype(str).str.upper() == "TITULAR"
        ).astype(np.int8)
    if "sexo" in d.columns:
        d["is_fem"] = (
            d["sexo"].astype(str).str.upper() == "F"
        ).astype(np.int8)

    for c in d.select_dtypes(include=["float", "int"]).columns:
        d[c] = d[c].replace([np.inf, -np.inf], np.nan).fillna(0)
    # converte object E string (pandas StringDtype) para category
    for c in d.columns:
        if not pd.api.types.is_numeric_dtype(d[c]) and not isinstance(d[c].dtype, pd.CategoricalDtype):
            d[c] = d[c].fillna("missing").astype("category")
    for c in d.select_dtypes(include=["category"]).columns:
        if d[c].isna().any():
            d[c] = d[c].cat.add_categories(["missing"]).fillna("missing")

    return d


def resolve_feature_columns(df: pd.DataFrame, fc: pd.DataFrame) -> list[str]:
    eligible = set(catalog_eligible_names(fc))
    exclude = (
        LEAKAGE_COLS
        | QUASI_LEAKAGE_COLS
        | {TARGET_COL, SEGMENT_COL, BENEFICIARIO_COL, TIME_COL, "_pipeline_row_id"}
    )
    candidates = [c for c in df.columns if c in eligible and c not in exclude]
    engineered = [
        c for c in df.columns
        if c in {"mes", "ano", "mes_sin", "mes_cos", "faixa_etaria", "is_titular", "is_fem"}
        or (c.startswith("tx_") and "qtd_" in c)
        or c.startswith("lag_sin_")
        or c.startswith("roll_mean_sin")
    ]
    for c in engineered:
        if c in df.columns and c not in candidates:
            candidates.append(c)
    return sorted(set(candidates))


# =============================================================================
# %% — TwoStageModel
# =============================================================================

class TwoStageModel:
    """
    Estágio 1: LGBMClassifier P(y > 0).
    Estágio 2: LGBMRegressor E[log(y) | y > 0].
    Predição: p_pos * exp(log_y) * smearing * macro_scale.
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
            device=DEVICE,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )

    def _reg_params(self) -> dict:
        return dict(
            objective="fair",
            metric="mae",
            fair_c=100.0,        # aumentado: menos robusto a outliers → aprende mais a cauda
            n_estimators=800,    # mais árvores para capturar padrões da cauda
            learning_rate=0.03,
            num_leaves=31,
            max_depth=6,
            min_child_samples=35,  # equilibrado: menos overfitting em segmentos pequenos
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=2.0,
            device=DEVICE,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: np.ndarray | None = None,
        outlier_cap_pct: float | None = None,
    ) -> "TwoStageModel":
        self.feature_names_ = list(X.columns)
        y = y.astype(float).copy()
        if outlier_cap_pct is not None:
            y = y.clip(upper=float(y.quantile(outlier_cap_pct)))
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
        p_pos = self.clf_.predict_proba(X)[:, 1]
        log_y = self.reg_.predict(X)
        return p_pos * (np.exp(log_y) * self._smearing)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._predict_raw(X) * float(getattr(self, "macro_scale", 1.0))

    def predict_stages(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        p_pos = self.clf_.predict_proba(X)[:, 1]
        y_hat = self._predict_raw(X) * float(getattr(self, "macro_scale", 1.0))
        return p_pos, y_hat


# =============================================================================
# %% — Agregação macro
# =============================================================================

def aggregate_sinistralidade_macro(
    y_pred: np.ndarray, premio: pd.Series
) -> float:
    """Média ponderada do índice de sinistralidade pelo prêmio."""
    premio = pd.to_numeric(premio, errors="coerce").fillna(0.0)
    denom = float(premio.sum())
    if denom == 0.0:
        return float("nan")
    y = np.asarray(y_pred, dtype=float).ravel()
    p = np.asarray(premio, dtype=float).ravel()
    return float((y * p).sum() / denom)


# =============================================================================
# %% — Catálogo de perfis top-100
# =============================================================================

def build_catalogo_perfis_top100(
    df_raw: pd.DataFrame,
    version_label: str,
    n: int = TOP_PERFIS_N,
) -> dict[str, Any]:
    criterio_col = (
        "valor_sinistro_raw"
        if "valor_sinistro_raw" in df_raw.columns
        else TARGET_COL
    )
    prem_ok = pd.to_numeric(df_raw[PREMIUM_COL], errors="coerce").fillna(0.0) > 0.0
    df_w = df_raw.loc[prem_ok].copy().reset_index(drop=True)
    df_w.insert(0, "indice_parquet", np.arange(len(df_w), dtype=np.int64))
    key = pd.to_numeric(df_w[criterio_col], errors="coerce").fillna(0.0)
    df_top = (
        df_w.assign(_sort_key=key)
        .sort_values(["_sort_key", "indice_parquet"], ascending=[False, True])
        .head(max(1, n))
        .drop(columns=["_sort_key"])
    )
    perfis: list[dict[str, Any]] = []
    for rank, (_, row) in enumerate(df_top.iterrows(), start=1):
        cb   = row.get(BENEFICIARIO_COL)
        comp = row.get(TIME_COL)
        label = f"#{rank} | {cb} | {comp}"
        payload: dict[str, Any] = {}
        for k, v in row.items():
            if k in {"indice_parquet", TARGET_COL}:
                continue
            safe = _json_safe(v)
            if safe is not None:
                payload[k] = safe
        perfis.append({
            "rank": rank,
            "indice_parquet": int(row["indice_parquet"]),
            "label": label,
            "resumo": {
                "cod_beneficiario": _json_safe(cb),
                "competencia": str(comp) if comp is not None else "",
                "valor_sinistro_raw": float(
                    pd.to_numeric(row.get("valor_sinistro_raw", 0), errors="coerce") or 0.0
                ),
                TARGET_COL: float(
                    pd.to_numeric(row.get(TARGET_COL, 0), errors="coerce") or 0.0
                ),
                PREMIUM_COL: float(
                    pd.to_numeric(row.get(PREMIUM_COL, 0), errors="coerce") or 0.0
                ),
                SEGMENT_COL: str(row.get(SEGMENT_COL, "")),
            },
            "payload": payload,
        })
    return {
        "versao_pasta": version_label,
        "company": COMPANY,
        "criterio_ordenacao": criterio_col,
        "n": len(perfis),
        "perfis": perfis,
    }


# =============================================================================
# %% — Catálogo de features intervencionáveis (snapshot mensal)
# =============================================================================

def _calcular_elegibilidade(
    df: pd.DataFrame, feature: str, delta_pct: float
) -> pd.Series:
    """Regra simples: delta > 0 → feature < p90; delta < 0 → feature > 0."""
    if feature not in df.columns:
        return pd.Series(False, index=df.index)
    serie = pd.to_numeric(df[feature], errors="coerce").fillna(0.0)
    if delta_pct > 0:
        return serie < float(serie.quantile(0.90))
    return serie > 0


def build_catalogo_features_intervencao(
    df_mes: pd.DataFrame,
    fc: pd.DataFrame,
    version_label: str,
    competencia_ref: str,
) -> dict[str, Any]:
    eligible_set = set(catalog_eligible_names(fc))
    dtypes_ok = {"numeric", "count"}
    feature_to_dtype = (
        fc[["feature_name", "dtype"]]
        .dropna(subset=["feature_name"])
        .set_index("feature_name")["dtype"]
        .to_dict()
    )
    feature_to_cat = (
        fc[["feature_name", "category"]]
        .dropna(subset=["feature_name"])
        .set_index("feature_name")["category"]
        .to_dict()
    )
    candidatas = [
        c for c in sorted(df_mes.columns)
        if c in eligible_set
        and c not in FEATURES_BLOQUEADAS_INTERVENCAO
        and str(feature_to_dtype.get(c, "")).lower() in dtypes_ok
    ]
    n_total = int(len(df_mes))
    grupos: dict[str, list[dict[str, Any]]] = {}
    for feat in candidatas:
        mask_pos = _calcular_elegibilidade(df_mes, feat, +20.0)
        mask_neg = _calcular_elegibilidade(df_mes, feat, -20.0)
        item: dict[str, Any] = {
            "feature": feat,
            "categoria": str(feature_to_cat.get(feat, "sem_categoria")),
            "dtype_catalogo": str(feature_to_dtype.get(feat, "n/a")),
            "n_elegiveis_delta_positivo": int(mask_pos.sum()),
            "n_elegiveis_delta_negativo": int(mask_neg.sum()),
            "pct_base_elegivel_delta_positivo": round(
                float(mask_pos.sum()) / max(1, n_total) * 100.0, 2
            ),
        }
        grupos.setdefault(str(feature_to_cat.get(feat, "sem_categoria")), []).append(item)
    grupos_lista = [
        {"grupo": g, "n_features": len(feats), "features": feats}
        for g, feats in sorted(grupos.items())
    ]
    return {
        "versao_pasta": version_label,
        "company": COMPANY,
        "competencia_referencia": competencia_ref,
        "n_total_base_mes": n_total,
        "n_features_intervencionaveis": len(candidatas),
        "grupos": grupos_lista,
    }


# =============================================================================
# %% — Pipeline de treino
# =============================================================================

def run_training_pipeline() -> tuple[str, Path]:
    if not TRANSFORMED_PARQUET_PATH.is_file():
        raise FileNotFoundError(
            f"Base transformada não encontrada: {TRANSFORMED_PARQUET_PATH}"
        )

    df_raw = pd.read_parquet(TRANSFORMED_PARQUET_PATH)
    print(f"[Carga] shape={df_raw.shape}")

    fc = load_feature_catalog()

    prem_ok = pd.to_numeric(df_raw[PREMIUM_COL], errors="coerce").fillna(0.0) > 0.0
    n_drop = int((~prem_ok).sum())
    df_raw = df_raw.loc[prem_ok].copy()
    if n_drop:
        print(f"[Qualidade] {n_drop} linhas com {PREMIUM_COL} <= 0 removidas (restam {len(df_raw):,})")

    df_raw = df_raw[df_raw[SEGMENT_COL].isin(PLANOS_CANONICOS)].copy()
    print(f"[Filtro] Planos: {df_raw[SEGMENT_COL].unique().tolist()} | n={len(df_raw):,}")

    df = build_features(df_raw)
    feature_cols = resolve_feature_columns(df, fc)
    print(f"[Features] n={len(feature_cols)}")

    VERSION_LABEL, RUN_DIR = next_version_dir(OUTPUT_PREDICT_ROOT)
    MODELS_DIR = RUN_DIR / "models"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[Versão] {RUN_DIR}")

    all_pred_rows: list[pd.DataFrame] = []
    holdout_y_true: list[np.ndarray] = []
    holdout_y_pred: list[np.ndarray] = []
    holdout_premio: list[pd.Series] = []
    metricas: dict[str, Any] = {}
    trained_segments: list[str] = []

    for plano in PLANOS_CANONICOS:
        print(f"\n{'='*60}\nPlano: {plano}\n{'='*60}")

        mask = df[SEGMENT_COL].astype(str) == plano
        df_seg = (
            df.loc[mask]
            .dropna(subset=[TARGET_COL])
            .sort_values(TIME_COL)
            .reset_index(drop=True)
        )
        n_tot = len(df_seg)
        if n_tot < MIN_ROWS_TEMPORAL:
            print(f"[PULAR] Poucas linhas ({n_tot}); mínimo = {MIN_ROWS_TEMPORAL}.")
            continue

        X_all = ensure_no_object_dtype(df_seg[feature_cols].copy())
        y_all = df_seg[TARGET_COL].astype(float)
        fat_all = df_seg[PREMIUM_COL].astype(float)
        w_all = fat_all.values

        idx_split = max(1, min(int(n_tot * (1 - HOLDOUT_FRAC)), n_tot - 100))
        X_tr, X_ho = X_all.iloc[:idx_split], X_all.iloc[idx_split:]
        y_tr, y_ho = y_all.iloc[:idx_split], y_all.iloc[idx_split:]
        fat_ho = fat_all.iloc[idx_split:]
        w_tr = w_all[:idx_split]
        pct_ho = 100.0 * (n_tot - idx_split) / n_tot

        # Cross-validation TimeSeriesSplit
        tsc = TimeSeriesSplit(n_splits=N_SPLITS_CV)
        cv_metrics: list[dict] = []
        for tr_idx, te_idx in tsc.split(X_tr):
            m_cv = TwoStageModel()
            m_cv.fit(
                X_tr.iloc[tr_idx], y_tr.iloc[tr_idx],
                sample_weight=w_tr[tr_idx],
                outlier_cap_pct=OUTLIER_CAP_PCT,
            )
            yp_cv = m_cv.predict(X_tr.iloc[te_idx])
            cv_metrics.append({
                "mae": mean_absolute_error(y_tr.iloc[te_idx], yp_cv),
                "r2":  r2_score(y_tr.iloc[te_idx], yp_cv),
            })
        df_cv = pd.DataFrame(cv_metrics)
        print(
            f"[CV x{N_SPLITS_CV}] MAE={df_cv['mae'].mean():.4f} | "
            f"R²={df_cv['r2'].mean():.4f}"
        )

        # Modelo holdout — para calibração macro sem leakage
        model_ho = TwoStageModel()
        model_ho.fit(X_tr, y_tr, sample_weight=w_tr, outlier_cap_pct=OUTLIER_CAP_PCT)
        p_pos_ho, y_ho_pred = model_ho.predict_stages(X_ho)
        y_ho_np   = np.asarray(y_ho, dtype=float)
        y_pred_np = np.asarray(y_ho_pred, dtype=float)

        mae_ho = float(mean_absolute_error(y_ho_np, y_pred_np))
        r2_ho  = float(r2_score(y_ho_np, y_pred_np))
        y_bin_ho = (y_ho_np > 0).astype(int)
        auc_ho = (
            float(roc_auc_score(y_bin_ho, p_pos_ho))
            if y_bin_ho.sum() > 0 and y_bin_ho.sum() < len(y_bin_ho)
            else float("nan")
        )
        ap_ho = (
            float(average_precision_score(y_bin_ho, p_pos_ho))
            if y_bin_ho.sum() > 0
            else float("nan")
        )
        print(
            f"[Holdout ~{pct_ho:.1f}%] MAE={mae_ho:.4f} | R²={r2_ho:.4f} | "
            f"AUC={auc_ho:.4f} | AP={ap_ho:.4f}"
        )

        sin_real = aggregate_sinistralidade_macro(y_ho_np, fat_ho)
        sin_pred = aggregate_sinistralidade_macro(y_pred_np, fat_ho)
        erro_macro = (
            abs(sin_pred - sin_real) / sin_real
            if sin_real and not np.isnan(sin_real)
            else float("nan")
        )
        print(
            f"  Macro: real={sin_real:.6f} | pred={sin_pred:.6f} | "
            f"erro_rel={erro_macro:.2%}"
        )

        # Modelo final (treino em todo o dataset)
        model_final = TwoStageModel()
        model_final.fit(X_all, y_all, sample_weight=w_all, outlier_cap_pct=OUTLIER_CAP_PCT)

        # Calibração macro SEM leakage — model_ho sobre X_ho
        raw_ho = model_ho._predict_raw(X_ho)
        sin_real_h    = aggregate_sinistralidade_macro(y_ho.values, fat_ho)
        sin_pred_raw_h = aggregate_sinistralidade_macro(raw_ho, fat_ho)
        macro_scale = 1.0
        if (sin_pred_raw_h
                and not np.isnan(sin_pred_raw_h)
                and abs(sin_pred_raw_h) > 1e-15):
            macro_scale = float(sin_real_h / sin_pred_raw_h)
        model_final.macro_scale = macro_scale
        print(f"[Calibração macro] fator={macro_scale:.4f}")

        y_ho_final = model_final.predict(X_ho)
        mae_final  = float(mean_absolute_error(y_ho_np, y_ho_final))
        print(f"[Holdout final+calib] MAE={mae_final:.4f}")

        # Predições completas
        p_full, y_hat_full = model_final.predict_stages(X_all)
        out_micro = X_all.copy()
        out_micro.insert(0, SEGMENT_COL, df_seg[SEGMENT_COL].values)
        out_micro["p_sinistro"]             = p_full
        out_micro["sinistralidade_prevista"] = y_hat_full
        all_pred_rows.append(out_micro)

        holdout_y_true.append(y_ho_np)
        holdout_y_pred.append(np.asarray(y_ho_final, dtype=float))
        holdout_premio.append(fat_ho.reset_index(drop=True))

        slug = plano_slug(plano)
        model_path = MODELS_DIR / f"model_{slug}.pkl"
        joblib.dump(model_final, model_path)
        print(f"[OK] Modelo salvo: {model_path}")
        trained_segments.append(plano)

        metricas[slug] = {
            "mae_cv":          float(df_cv["mae"].mean()),
            "r2_cv":           float(df_cv["r2"].mean()),
            "mae_holdout":     mae_ho,
            "r2_holdout":      r2_ho,
            "auc_holdout":     auc_ho,
            "ap_holdout":      ap_ho,
            "sin_real_holdout": sin_real,
            "sin_pred_holdout": sin_pred,
            "erro_macro_holdout": erro_macro,
            "macro_scale":     macro_scale,
            "n_total":         n_tot,
        }

    # ── Artefatos globais ─────────────────────────────────────────────────────

    if not all_pred_rows:
        print("Nenhum segmento treinado; sem artefatos de predição.")
        return VERSION_LABEL, RUN_DIR

    pred_micro = pd.concat(all_pred_rows, axis=0, ignore_index=True)
    pred_micro.to_csv(RUN_DIR / "predicoes_micro.csv", index=False, encoding="utf-8-sig")
    print(f"\n[Artefato] predicoes_micro.csv")

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
        "versao":   VERSION_LABEL,
        "company":  COMPANY,
        "sinistralidade_real":    float(sin_real_g),
        "sinistralidade_prevista": float(sin_pred_g),
        "erro_relativo":          float(erro_rel),
    }
    with open(RUN_DIR / "resultado_macro.json", "w", encoding="utf-8") as f:
        json.dump(macro_payload, f, indent=2, ensure_ascii=False)
    print(f"[Artefato] resultado_macro.json | erro_rel={erro_rel:.4f}")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(yt, yp, alpha=0.25, s=8)
    lims = [float(min(yt.min(), yp.min())), float(max(yt.max(), yp.max()))]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlabel("Real (holdout)")
    ax.set_ylabel("Previsto (holdout)")
    ax.set_title(f"CLIMAZON real vs pred — {VERSION_LABEL}")
    plt.tight_layout()
    fig.savefig(RUN_DIR / "real_vs_pred.png", dpi=150)
    plt.close(fig)
    print("[Artefato] real_vs_pred.png")

    # Catálogo de perfis top-100
    try:
        cat_perfis = build_catalogo_perfis_top100(df_raw, VERSION_LABEL)
        with open(RUN_DIR / "catalogo_perfis_top100.json", "w", encoding="utf-8") as f:
            json.dump(cat_perfis, f, indent=2, ensure_ascii=False, default=str)
        print(f"[Artefato] catalogo_perfis_top100.json | n={cat_perfis['n']}")
    except Exception as e:
        print(f"[aviso] catalogo_perfis: {e}")

    # Catálogo de features intervencionáveis (última competência)
    try:
        comp_dt  = pd.to_datetime(df_raw[TIME_COL], errors="coerce")
        comp_ref = str(comp_dt.max().strftime("%Y-%m"))
        mask_comp = comp_dt.dt.to_period("M") == pd.Period(comp_ref, freq="M")
        df_mes_ref = df_raw.loc[mask_comp].copy().reset_index(drop=True)
        cat_feat = build_catalogo_features_intervencao(df_mes_ref, fc, VERSION_LABEL, comp_ref)
        with open(RUN_DIR / "catalogo_features_intervencao.json", "w", encoding="utf-8") as f:
            json.dump(cat_feat, f, indent=2, ensure_ascii=False)
        print(
            f"[Artefato] catalogo_features_intervencao.json | "
            f"n_features={cat_feat['n_features_intervencionaveis']}"
        )
    except Exception as e:
        print(f"[aviso] catalogo_features_intervencao: {e}")

    # run_metadata
    meta = {
        "company":          COMPANY,
        "versao":           VERSION_LABEL,
        "data_execucao":    datetime.now().strftime("%Y-%m-%d %H:%M"),
        "parquet_origem":   str(TRANSFORMED_PARQUET_PATH),
        "target_col":       TARGET_COL,
        "holdout_frac":     HOLDOUT_FRAC,
        "n_splits_cv":      N_SPLITS_CV,
        "outlier_cap_pct":  OUTLIER_CAP_PCT,
        "min_rows_temporal": MIN_ROWS_TEMPORAL,
        "planos":           PLANOS_CANONICOS,
        "planos_treinados": trained_segments,
        "n_features":       len(feature_cols),
        "features_usadas":  feature_cols,
        "metricas":         metricas,
    }
    with open(RUN_DIR / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)
    print("[Artefato] run_metadata.json")

    print(f"\n=== Concluído — saída em {RUN_DIR} ===")
    return VERSION_LABEL, RUN_DIR


# %%
if __name__ == "__main__":
    run_training_pipeline()

# %%
#%%