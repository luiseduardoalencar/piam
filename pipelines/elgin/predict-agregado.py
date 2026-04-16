# %%
"""
Predict agregado por beneficiario — Sinistralidade ELGIN.

Script autónomo (não importa ``predict.py``): treino com uma linha por beneficiário,
saídas versionadas e registo MLflow com as mesmas convenções de naming do pipeline mensal.

Execução (Spyder / Jupyter): correr os blocos ``# %%`` em ordem; o bloco de treino
inclui o registo MLflow no final — correr essa célula inteira para treinar e registar.

Geração de perfil: ``_json_safe``, ``serie_para_payload``, ``payload_do_parquet_por_indice_agregado``,
``salvar_payload`` (tudo neste ficheiro; sem módulo ``perfil`` à parte).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# --- Constantes alinhadas ao pipeline mensal (cópia local; altere aqui se necessário) ---
COMPANY = "elgin"

TRANSFORMED_PARQUET_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / COMPANY
    / "base_analitica_transformada"
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

# Agregado: sem holdout temporal (valor só para consistência nos params MLflow)
HOLDOUT_FRAC = 0.0
N_SPLITS_CV = 5
MIN_ROWS_SEGMENT = 200
RANDOM_STATE = 42
OUTLIER_CAP_PCT = 0.99
BENEFICIARIO_COL = "cod_beneficiario"
N_MESES_COL = "n_meses_obs"

TOP_PERFIS_N = 100

MLFLOW_EXPERIMENT_NAME = "piam-elgin-predict"
MLFLOW_REGISTERED_MODEL_NAME = "elgin-sinistralidade-two-stage"

PLANO_MAP = {
    "EMPRESARIAL MASTER": "MASTER EMPRESARIAL",
    "COLETIVO EMPRESARIAL MASTER - PROTOCOLO ANS: 414538991": "MASTER EMPRESARIAL",
}

try:
    import lightgbm as lgb
except ImportError as e:
    raise ImportError(
        "Instale LightGBM: pip install lightgbm"
    ) from e

# %%
# Geração de perfil — linha agregada por beneficiário → dict JSON (catálogo top100)
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
    """Converte uma linha agregada em dict para inferência / UI; omite nulos e opcionalmente o alvo."""
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


def resolve_feature_columns(df: pd.DataFrame, fc: pd.DataFrame) -> list[str]:
    eligible = catalog_eligible_names(fc)
    exclude = (
        LEAKAGE_COLS
        | {TARGET_COL, SEGMENT_COL, "cod_beneficiario"}
        | {TIME_COL}
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
    Mesma estrutura que o pipeline mensal (inferência micro compatível).
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
    premio = pd.to_numeric(premio, errors="coerce").fillna(0.0)
    denom = float(premio.sum())
    if denom == 0.0:
        return float("nan")
    return float(np.asarray(y_pred, dtype=float).sum() / denom)


def _mlflow_prepare_input_example(df: pd.DataFrame) -> pd.DataFrame:
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
    try:
        import mlflow  # noqa: F401
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

        cat_path = run_dir / "catalogo_perfis_top100.json"
        if cat_path.is_file():
            mlflow.log_artifact(str(cat_path), artifact_path="elgin_artifacts")

        if (
            model_path is not None
            and model_path.is_file()
            and model_final is not None
            and input_example is not None
        ):
            x_fit = ensure_no_object_dtype(input_example.copy())
            out_ex = model_final.predict_stages(x_fit)
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


# %%
# Estado entre blocos (treino -> MLflow)
VERSION_LABEL: str | None = None
RUN_DIR: Path | None = None
FEATURE_COLS: list[str] = []
DF_AGG_N: int = 0
TRAINED_SEGMENTS: list[str] = []
LAST_MODEL_PATH: Path | None = None
LAST_INPUT_EXAMPLE: pd.DataFrame | None = None
LAST_MODEL_FINAL: TwoStageModel | None = None
METRICS_MLFLOW: dict[str, float] = {}


def aggregate_panel_by_beneficiary(df: pd.DataFrame) -> pd.DataFrame:
    if BENEFICIARIO_COL not in df.columns:
        raise ValueError(f"Coluna obrigatoria ausente: {BENEFICIARIO_COL}")
    d = df.copy()
    if TIME_COL in d.columns:
        d[TIME_COL] = pd.to_datetime(d[TIME_COL], errors="coerce")
        d = d.sort_values([BENEFICIARIO_COL, TIME_COL])
    else:
        d = d.sort_values(BENEFICIARIO_COL)

    cols = [c for c in d.columns if c not in (BENEFICIARIO_COL, TIME_COL)]
    agg: dict[str, str] = {}
    for c in cols:
        if c == TARGET_COL or c == "pct_urgencia":
            agg[c] = "mean"
        elif c.startswith("qtd_") or c == PREMIUM_COL:
            agg[c] = "sum"
        elif c in (SEGMENT_COL, "idade", "sexo", "tipo_cadastro", "plano"):
            agg[c] = "first"
        elif "sinistralidade" in c.lower() and c != TARGET_COL:
            agg[c] = "mean"
        elif pd.api.types.is_numeric_dtype(d[c]):
            cl = c.lower()
            if any(x in cl for x in ("valor", "sinistro", "sin_ref", "fator", "ajuste")):
                agg[c] = "sum"
            else:
                agg[c] = "mean"
        else:
            agg[c] = "first"

    d_work = d.drop(columns=[TIME_COL], errors="ignore")
    out = d_work.groupby(BENEFICIARIO_COL, as_index=False, dropna=False).agg(agg)
    sizes = d.groupby(BENEFICIARIO_COL, dropna=False).size().reset_index(name=N_MESES_COL)
    out = out.merge(sizes, on=BENEFICIARIO_COL, how="left")
    if "idade" in out.columns:
        _id = pd.to_numeric(out["idade"], errors="coerce")
        out["idade"] = np.ceil(_id).fillna(-1).astype(np.int64)
    return out


def payload_do_parquet_por_indice_agregado(
    parquet_path: Path | None = None,
    indice: int = 0,
) -> dict[str, Any]:
    """
    Agrega o painel por beneficiário (mesma lógica do treino), filtra prêmio > 0
    e devolve ``serie_para_payload`` para a linha ``indice`` (0-based) desse agregado.
    """
    path = parquet_path or TRANSFORMED_PARQUET_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Parquet não encontrado: {path}")
    df_raw = pd.read_parquet(path)
    df_raw[SEGMENT_COL] = df_raw[SEGMENT_COL].replace(PLANO_MAP)
    df_agg = aggregate_panel_by_beneficiary(df_raw)
    prem_ok = pd.to_numeric(df_agg[PREMIUM_COL], errors="coerce").fillna(0.0) > 0.0
    df_agg = df_agg.loc[prem_ok].copy().reset_index(drop=True)
    if indice < 0 or indice >= len(df_agg):
        raise IndexError(
            f"indice={indice} fora do intervalo [0, {len(df_agg) - 1}] "
            f"(n={len(df_agg):,} linhas agregadas)."
        )
    row = df_agg.iloc[indice]
    return serie_para_payload(row, excluir_alvo=True)


def build_features_agregado(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "idade" in d.columns:
        _id = pd.to_numeric(d["idade"], errors="coerce")
        d["idade"] = np.ceil(_id).fillna(-1).astype(np.int64)
    d = d.drop(
        columns=[c for c in LEAKAGE_COLS if c in d.columns and c != TARGET_COL],
        errors="ignore",
    )
    if BENEFICIARIO_COL in d.columns:
        d = d.drop(columns=[BENEFICIARIO_COL], errors="ignore")
    if TIME_COL in d.columns:
        d = d.drop(columns=[TIME_COL], errors="ignore")

    if "idade" in d.columns:
        d["faixa_etaria"] = pd.cut(
            d["idade"].fillna(-1),
            bins=[-1, 0, 5, 12, 18, 30, 45, 60, 200],
            labels=["inf", "0-5", "6-12", "13-18", "19-30", "31-45", "46-60", "60+"],
        ).astype(str)
    if "tipo_cadastro" in d.columns:
        d["is_titular"] = (d["tipo_cadastro"].astype(str).str.upper() == "TITULAR").astype(np.int8)
    if "sexo" in d.columns:
        d["is_fem"] = (d["sexo"].astype(str).str.upper() == "F").astype(np.int8)

    fat = d[PREMIUM_COL].replace(0, np.nan) if PREMIUM_COL in d.columns else None
    if fat is not None:
        for c in [col for col in d.columns if col.startswith("qtd_")]:
            d[f"tx_{c}"] = d[c] / fat

    for c in d.select_dtypes(include=["float", "int"]).columns:
        d[c] = d[c].replace([np.inf, -np.inf], np.nan).fillna(0)
    for c in d.select_dtypes(include=["object"]).columns:
        d[c] = d[c].fillna("missing").astype("category")
    for c in d.select_dtypes(include=["category"]).columns:
        if d[c].isna().any():
            d[c] = d[c].cat.add_categories(["missing"]).fillna("missing")
    return d


def build_catalogo_perfis_top100_agregado(
    version_label: str,
    df_agg: pd.DataFrame,
    df_feat: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, Any]:
    """
    Top ``TOP_PERFIS_N`` por ``valor_sinistro_raw``; ``payload`` = linha de entrada do modelo
    (pós-``build_features_agregado``), alinhada ao pyfunc. ``indice_parquet`` = índice na
    ``df_agg`` já filtrada (prêmio > 0), mesma ordem do treino.
    """
    if len(df_agg) != len(df_feat):
        raise ValueError("df_agg e df_feat devem ter o mesmo número de linhas.")

    df_agg_w = df_agg.reset_index(drop=True)
    df_feat_w = df_feat.reset_index(drop=True)
    key = pd.to_numeric(df_agg_w.get("valor_sinistro_raw"), errors="coerce").fillna(0.0).values
    n_take = max(1, min(TOP_PERFIS_N, len(df_agg_w)))
    top_positions = np.argsort(-key)[:n_take]

    perfis: list[dict[str, Any]] = []
    for rank, pos in enumerate(top_positions, start=1):
        pos = int(pos)
        row = df_agg_w.iloc[pos]
        payload = dataframe_row_to_payload_inferencia(
            df_feat_w.iloc[[pos]][feature_cols], feature_cols
        )
        cb = row.get(BENEFICIARIO_COL)
        label = f"#{rank} | {cb}"
        perfis.append(
            {
                "rank": rank,
                "indice_parquet": pos,
                "label": label,
                "resumo": {
                    "cod_beneficiario": _json_safe(cb),
                    "valor_sinistro_raw": float(
                        pd.to_numeric(row.get("valor_sinistro_raw"), errors="coerce") or 0.0
                    ),
                    TARGET_COL: float(pd.to_numeric(row.get(TARGET_COL), errors="coerce") or 0.0),
                    PREMIUM_COL: float(pd.to_numeric(row.get(PREMIUM_COL), errors="coerce") or 0.0),
                    SEGMENT_COL: str(row.get(SEGMENT_COL, "")),
                    N_MESES_COL: int(pd.to_numeric(row.get(N_MESES_COL), errors="coerce") or 0),
                },
                "payload": payload,
            }
        )
    return {
        "versao_pasta": version_label,
        "criterio_ordenacao": "valor_sinistro_raw",
        "n": len(perfis),
        "descricao": (
            "Perfis agregados por beneficiário (top por valor_sinistro_raw). "
            "payload = features do modelo (pós-build_features_agregado), ordem alinhada ao pyfunc."
        ),
        "perfis": perfis,
    }


# %%
# Treino, artefatos em disco, métricas e MLflow (mesma célula — registo depende das variáveis do treino)

if not TRANSFORMED_PARQUET_PATH.is_file():
    raise FileNotFoundError(f"Base transformada nao encontrada: {TRANSFORMED_PARQUET_PATH}")

df_raw = pd.read_parquet(TRANSFORMED_PARQUET_PATH)
df_raw[SEGMENT_COL] = df_raw[SEGMENT_COL].replace(PLANO_MAP)
df_agg = aggregate_panel_by_beneficiary(df_raw)
print(f"[Carga] painel={df_raw.shape} -> agregado={df_agg.shape}")

n_before_q = len(df_agg)
prem_ok = pd.to_numeric(df_agg[PREMIUM_COL], errors="coerce").fillna(0.0) > 0.0
df_agg = df_agg.loc[prem_ok].copy()
print(f"[Qualidade] removidas {n_before_q - len(df_agg)} linhas com faturamento <= 0")

fc = load_feature_catalog()
df = build_features_agregado(df_agg)
feature_cols = resolve_feature_columns(df, fc)
feature_cols = [c for c in feature_cols if c not in {N_MESES_COL}]
if not feature_cols:
    raise ValueError("Lista de features vazia apos agregacao.")

version_label, run_dir = next_version_dir(OUTPUT_PREDICT_ROOT)
models_dir = run_dir / "models"
models_dir.mkdir(parents=True, exist_ok=True)
print(f"[Versao] {run_dir}")

all_pred_rows: list[pd.DataFrame] = []
y_true_all: list[np.ndarray] = []
y_pred_all: list[np.ndarray] = []
premio_all: list[pd.Series] = []
trained_segments: list[str] = []

last_model_path: Path | None = None
last_input_example: pd.DataFrame | None = None
last_model_final: TwoStageModel | None = None

for plano in sorted(df[SEGMENT_COL].dropna().astype(str).unique()):
    df_seg = df[df[SEGMENT_COL].astype(str) == plano].dropna(subset=[TARGET_COL]).reset_index(drop=True)
    if len(df_seg) < MIN_ROWS_SEGMENT:
        print(f"[PULAR] {plano}: poucas linhas ({len(df_seg)})")
        continue

    X = ensure_no_object_dtype(df_seg[feature_cols].copy())
    y = df_seg[TARGET_COL].astype(float)
    fat = df_seg[PREMIUM_COL].astype(float)
    sw = fat.values

    cv = KFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)
    cv_mae: list[float] = []
    for tr, te in cv.split(X):
        m = TwoStageModel()
        m.fit(X.iloc[tr], y.iloc[tr], sample_weight=sw[tr], outlier_cap_pct=OUTLIER_CAP_PCT)
        yp = m.predict(X.iloc[te])
        cv_mae.append(float(mean_absolute_error(y.iloc[te], yp)))
    print(
        f"[CV x{N_SPLITS_CV}] {plano} | MAE medio={float(np.mean(cv_mae)):.6f} | "
        f"std_fold={float(np.std(cv_mae, ddof=1)):.6f}"
    )

    model_final = TwoStageModel()
    model_final.fit(X, y, sample_weight=sw, outlier_cap_pct=OUTLIER_CAP_PCT)
    p_pos, y_hat = model_final.predict_stages(X)

    y_np = np.asarray(y, dtype=float)
    y_hat_np = np.asarray(y_hat, dtype=float)
    mae_seg = float(mean_absolute_error(y_np, y_hat_np))
    rmse_seg = float(root_mean_squared_error(y_np, y_hat_np))
    r2_seg = float(r2_score(y_np, y_hat_np))
    sum_y = float(y_np.sum())
    sum_y_hat = float(y_hat_np.sum())
    sum_fat = float(fat.sum())
    print(
        f"  [Fit no segmento — valores assumidos] n={len(y_np)} | "
        f"MAE={mae_seg:.6f} | RMSE={rmse_seg:.6f} | R2={r2_seg:.6f}"
    )
    print(
        f"  [Somas no segmento] sum(y_true)={sum_y:.4f} | sum(y_pred)={sum_y_hat:.4f} | "
        f"sum({PREMIUM_COL})={sum_fat:.4f}"
    )
    y_bin = (y_np > 0).astype(int)
    if y_bin.size > 1 and np.unique(y_bin).size > 1:
        auc_s = float(roc_auc_score(y_bin, p_pos))
        ap_s = float(average_precision_score(y_bin, p_pos))
        print(f"  [Stage 1 — P(y>0) vs presenca real] AUC={auc_s:.6f} | AP={ap_s:.6f}")
    else:
        print("  [Stage 1] AUC/AP omitidos (uma so classe em y>0).")

    out_micro = X.copy()
    out_micro.insert(0, SEGMENT_COL, df_seg[SEGMENT_COL].values)
    out_micro["p_sinistro"] = p_pos
    out_micro["sinistralidade_prevista"] = y_hat
    all_pred_rows.append(out_micro)

    y_true_all.append(y.values)
    y_pred_all.append(np.asarray(y_hat, dtype=float))
    premio_all.append(fat.reset_index(drop=True))

    model_path = models_dir / f"model_{plano_slug(plano)}.pkl"
    joblib.dump(model_final, model_path)
    print(f"[OK] Modelo salvo: {model_path}")

    last_model_path = model_path
    last_input_example = X.iloc[[0]].copy()
    last_model_final = model_final
    trained_segments.append(str(plano))

VERSION_LABEL = version_label
RUN_DIR = run_dir
FEATURE_COLS = feature_cols
DF_AGG_N = len(df_agg)
TRAINED_SEGMENTS = trained_segments
LAST_MODEL_PATH = last_model_path
LAST_INPUT_EXAMPLE = last_input_example
LAST_MODEL_FINAL = last_model_final

if not all_pred_rows:
    print("Nenhum segmento treinado; sem artefatos.")
    METRICS_MLFLOW = {}
else:
    pred_micro = pd.concat(all_pred_rows, axis=0, ignore_index=True)
    micro_path = run_dir / "predicoes_micro.csv"
    pred_micro.to_csv(micro_path, index=False, encoding="utf-8-sig")

    yt = np.concatenate(y_true_all)
    yp = np.concatenate(y_pred_all)
    pr = pd.concat(premio_all, axis=0, ignore_index=True)
    macro_real = aggregate_sinistralidade_macro(yt, pr)
    macro_pred = aggregate_sinistralidade_macro(yp, pr)
    erro_rel = abs(macro_pred - macro_real) / macro_real if macro_real and not np.isnan(macro_real) else float("nan")
    macro_payload = {
        "versao": version_label,
        "sinistralidade_real": float(macro_real),
        "sinistralidade_prevista": float(macro_pred),
        "erro_relativo": float(erro_rel),
        "tipo_treino": "agregado_por_beneficiario",
    }
    macro_path = run_dir / "resultado_macro.json"
    macro_path.write_text(json.dumps(macro_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    catalog_path = run_dir / "catalogo_perfis_top100.json"
    cat = build_catalogo_perfis_top100_agregado(version_label, df_agg, df, feature_cols)
    catalog_path.write_text(json.dumps(cat, indent=2, ensure_ascii=False), encoding="utf-8")

    METRICS_MLFLOW = {
        "mae_global": float(mean_absolute_error(yt, yp)),
        "rmse_global": float(root_mean_squared_error(yt, yp)),
        "r2_global": float(r2_score(yt, yp)),
        "holdout_macro_sin_real": float(macro_real),
        "holdout_macro_sin_pred": float(macro_pred),
        "holdout_macro_erro_relativo": float(erro_rel),
    }
    sum_yt = float(np.asarray(yt, dtype=float).sum())
    sum_yp = float(np.asarray(yp, dtype=float).sum())
    sum_pr = float(pd.to_numeric(pr, errors="coerce").fillna(0.0).sum())
    print("\n" + "=" * 60)
    print("[Metricas globais — calculadas a partir de y_true, y_pred e premio por linha]")
    print("=" * 60)
    print(f"  n_linhas (predicoes micro) = {len(yt):,}")
    print(f"  mae_global               = {METRICS_MLFLOW['mae_global']:.8f}")
    print(f"  rmse_global              = {METRICS_MLFLOW['rmse_global']:.8f}")
    print(f"  r2_global                = {METRICS_MLFLOW['r2_global']:.8f}")
    print(
        "  Macro sinistralidade: sum(y)/sum(premio) e sum(y_pred)/sum(premio); "
        f"sum(premio)={sum_pr:,.4f}"
    )
    print(f"  sum(y_true)              = {sum_yt:,.8f}")
    print(f"  sum(y_pred)              = {sum_yp:,.8f}")
    print(f"  holdout_macro_sin_real   = {METRICS_MLFLOW['holdout_macro_sin_real']:.8f}")
    print(f"  holdout_macro_sin_pred   = {METRICS_MLFLOW['holdout_macro_sin_pred']:.8f}")
    print(f"  holdout_macro_erro_rel  = {METRICS_MLFLOW['holdout_macro_erro_relativo']:.8f}")
    y_bin_g = (np.asarray(yt, dtype=float) > 0).astype(int)
    p_concat = pred_micro["p_sinistro"].values if len(pred_micro) == len(yt) else None
    if p_concat is not None and y_bin_g.size > 1 and np.unique(y_bin_g).size > 1:
        auc_g = float(roc_auc_score(y_bin_g, p_concat))
        ap_g = float(average_precision_score(y_bin_g, p_concat))
        print(f"  auc_global (P(y>0) vs y>0) = {auc_g:.8f}")
        print(f"  ap_global  (P(y>0) vs y>0) = {ap_g:.8f}")
        METRICS_MLFLOW["auc_global"] = auc_g
        METRICS_MLFLOW["ap_global"] = ap_g
    print(
        f"  [Hipoteses numericas] n_splits_cv={N_SPLITS_CV} | min_rows_segmento={MIN_ROWS_SEGMENT} | "
        f"outlier_cap_pct={OUTLIER_CAP_PCT} | random_state={RANDOM_STATE}"
    )
    print("=" * 60)
    print(f"=== Treino e artefatos gravados em {run_dir} ===")

if LAST_MODEL_FINAL is not None and LAST_MODEL_PATH is not None and VERSION_LABEL is not None and RUN_DIR is not None:
    params_ml = {
        "pipeline_variant": "agregado_por_beneficiario",
        "n_features": len(FEATURE_COLS),
        "n_linhas_agregadas": DF_AGG_N,
        "segmentos_treinados": ",".join(TRAINED_SEGMENTS),
    }
    _mlflow_log_run_elgin(
        run_dir=RUN_DIR,
        version_label=VERSION_LABEL,
        model_path=LAST_MODEL_PATH,
        input_example=LAST_INPUT_EXAMPLE,
        model_final=LAST_MODEL_FINAL,
        params_extra=params_ml,
        metrics_extra=METRICS_MLFLOW,
    )
    print("\n[MLflow — consola] Parametros extra registados:")
    for k, v in sorted(params_ml.items()):
        print(f"  {k} = {v}")
    print("[MLflow — consola] Metricas registadas:")
    for k, v in sorted(METRICS_MLFLOW.items()):
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            print(f"  {k} = {v}")
        else:
            print(f"  {k} = {v}")
elif LAST_MODEL_FINAL is None:
    print("[MLflow] Ignorado — nenhum segmento treinado.")

if RUN_DIR is not None:
    print(f"\n=== Concluido — saida em {RUN_DIR} ===")

# %%
