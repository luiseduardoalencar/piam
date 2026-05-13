# %%
"""
Feature Impact + Correlação — Sinistralidade CLIMAZON

Consome o parquet transformado pela etapa de EDA e gera todos os artefatos de
análise de impacto de features. Sem MLflow — saídas apenas em disco local.

Artefato principal consumido pelo app Streamlit (aba "Correlação"):
  feature_correlation_sinistralidade.csv   (Spearman por feature × mês)

Demais artefatos (diagnóstico):
  corr_pearson_{seg}.csv / corr_spearman_{seg}.csv / corr_long_{seg}.csv
  high_corr_pairs_{seg}.csv
  r2_isolado_por_categoria_{seg}.csv       (com IC 95%, KFold-4)
  r2_isolado_prospectivo_{seg}.csv
  perm_global_{seg}.csv / perm_categoria_{seg}.csv
  compare_isolado_vs_global_{seg}.csv
  shap_values_{seg}.csv / shap_summary_{seg}.csv / shap_categoria_{seg}.csv
  pdp/pdp_{feature}_{seg}.csv             (top-8 features numéricas)
  quasi_leakage_corr_{seg}.csv
  heatmaps PNG e gráficos de impacto

Saídas versionadas em data/processed/climazon/feature_impact/vN/

Execução: célula a célula (VS Code / Cursor "Run Cell").
"""


# %%
from __future__ import annotations

import json
import re
import sys
import warnings
from datetime import datetime
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

try:
    import seaborn as sns
except ImportError:
    sns = None

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("[WARN] shap não instalado — bloco SHAP será pulado. pip install shap")

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore", category=UserWarning)


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
OUTPUT_FEATURE_IMPACT_ROOT = ROOT_DIR / "data" / "processed" / COMPANY / "feature_impact"

TARGET_COL       = "sinistralidade_final"
TIME_COL         = "competencia"
SEGMENT_COL      = "plano"
BENEFICIARIO_COL = "cod_beneficiario"

PLANOS_CANONICOS = ["MASTER EMPRESARIAL", "MASTER EXECUTIVO"]

# Leakage hard — derivados matemáticos diretos do alvo
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

# Quasi-leakage — co-ocorrem com o sinistro por definição (Spearman > 0.80 com alvo).
# Excluídos dos modelos preditivos, analisados em bloco separado de diagnóstico.
QUASI_LEAKAGE_COLS: frozenset[str] = frozenset({
    "qtd_eventos_sinistro",
    "qtd_carater_eletivo",
    "qtd_carater_urgencia",
})

# Hiperparâmetros de análise
N_SPLITS                 = 4
RANDOM_STATE             = 42
TOP_K_CORR_HEATMAP       = 25
CORR_TOP_N_RANK          = 40
CORR_HIGH_THRESHOLD      = 0.70   # pares acima disto → high_corr_pairs.csv
PERM_N_REPEATS           = 5
PDP_TOP_N                = 8      # features numéricas para PDP
PDP_GRID_RESOLUTION      = 40

# None = usar todas as linhas; inteiro = limite de linhas para etapas pesadas
KFOLD_MAX_ROWS:            int | None = None
PERM_IMPORTANCE_MAX_ROWS:  int | None = None
SHAP_MAX_ROWS:             int | None = 5_000

# Limiares para KFold adaptativo
KFOLD_MIN_ROWS_ADAPTIVE = 500   # segmentos menores: reduz n_splits para 2
KFOLD_MIN_ROWS_SKIP     = 50    # segmentos menores: pula KFold (retorna NaN)

# Limiar para deduplicação de features com correlação perfeita
DEDUP_SPEARMAN_THRESHOLD = 0.99  # |Spearman| ≥ isto → remove a redundante

AGGREGATE_BY_BENEFICIARIO = True
N_MESES_COL               = "n_meses_obs"

# Nome do arquivo consumido pelo app (correlação por mês)
CORR_ARTIFACT_FILENAME = "feature_correlation_sinistralidade.csv"
FEATURE_IMPACT_TOP_N_PLOT = 10

np.random.seed(RANDOM_STATE)


# =============================================================================
# %% — Utilitários gerais
# =============================================================================

def next_version_dir(root: Path) -> tuple[str, Path]:
    """Detecta v1, v2, … e devolve ('vN', root/'vN')."""
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


def _spearman_vs_target(x: pd.Series, y: pd.Series) -> float | None:
    """Spearman entre preditor e target; categórico via factorize."""
    pair = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(pair) < 3:
        return None
    xs = pair["x"]
    ys = pair["y"].astype(float)
    if pd.api.types.is_numeric_dtype(xs):
        xv = xs.astype(float)
    else:
        xv = pd.Series(pd.factorize(xs.astype(str))[0], index=xs.index).astype(float)
    if xv.nunique() < 2 or ys.nunique() < 2:
        return None
    r = xv.corr(ys, method="spearman")
    return float(r) if not pd.isna(r) else None


def _make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _save_corr_heatmap(corr: pd.DataFrame, title: str, path: Path) -> None:
    size = max(8, len(corr.columns) * 0.35)
    plt.figure(figsize=(size, size))
    if sns is not None:
        sns.heatmap(corr, cmap="vlag", center=0, square=True,
                    annot=len(corr) <= 10, fmt=".2f",
                    cbar_kws={"shrink": 0.6})
    else:
        plt.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        plt.colorbar()
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=7)
        plt.yticks(range(len(corr.index)), corr.index, fontsize=7)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


# =============================================================================
# %% — Catálogo de features
# =============================================================================

def load_feature_catalog() -> pd.DataFrame:
    if not FEATURE_CATALOG_PATH.is_file():
        raise FileNotFoundError(f"Catálogo inexistente: {FEATURE_CATALOG_PATH}")
    fc = pd.read_csv(FEATURE_CATALOG_PATH, encoding="utf-8-sig")
    for col in ("feature_name", "category", "dtype", "include_in_model"):
        if col not in fc.columns:
            raise ValueError(f"feature_catalog.csv deve ter coluna '{col}'.")
    return fc


def catalog_eligible_names(fc: pd.DataFrame) -> list[str]:
    """Features elegíveis para análise de impacto (include_in_model == True)."""
    mask = fc["include_in_model"].astype(str).str.lower().isin({"true", "1", "yes"})
    return fc.loc[mask, "feature_name"].astype(str).tolist()


def aggregate_panel_by_beneficiary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Colapsa o painel mensal em uma linha por cod_beneficiario.
    Remove competencia. Regras de agregação:
      - qtd_* e valor_faturamento: soma
      - sinistralidade_final, pct_urgencia: média
      - idade, sexo, tipo_cadastro, plano: primeiro valor
      - demais numéricos de valor/sinistro: soma
      - outros: média
    Acrescenta n_meses_obs (não entra em X).
    """
    if BENEFICIARIO_COL not in df.columns:
        raise ValueError(f"Coluna obrigatória ausente: {BENEFICIARIO_COL}")

    d = df.copy()
    if TIME_COL in d.columns:
        d[TIME_COL] = pd.to_datetime(d[TIME_COL], errors="coerce")
        d = d.sort_values([BENEFICIARIO_COL, TIME_COL])
    else:
        d = d.sort_values(BENEFICIARIO_COL)

    cols = [c for c in d.columns if c not in (BENEFICIARIO_COL, TIME_COL)]
    agg: dict[str, str] = {}
    for c in cols:
        if c in (TARGET_COL, "pct_urgencia"):
            agg[c] = "mean"
        elif c.startswith("qtd_"):
            agg[c] = "sum"
        elif c == "valor_faturamento":
            agg[c] = "sum"
        elif c in (SEGMENT_COL, "idade", "sexo", "tipo_cadastro"):
            agg[c] = "first"
        elif pd.api.types.is_numeric_dtype(d[c]):
            cl = c.lower()
            if any(x in cl for x in ("valor", "sinistro", "sin_ref", "fator", "ajuste")):
                agg[c] = "sum"
            else:
                agg[c] = "mean"
        else:
            agg[c] = "first"

    drop_time = [TIME_COL] if TIME_COL in d.columns else []
    d_work = d.drop(columns=drop_time, errors="ignore")

    g   = d_work.groupby(BENEFICIARIO_COL, as_index=False, dropna=False)
    out = g.agg(agg)

    sizes = d.groupby(BENEFICIARIO_COL, dropna=False).size().reset_index(name=N_MESES_COL)
    out   = out.merge(sizes, on=BENEFICIARIO_COL, how="left")

    if "idade" in out.columns:
        _id = pd.to_numeric(out["idade"], errors="coerce")
        out["idade"] = np.ceil(_id).fillna(-1).astype(np.int64)

    return out


# Subset prospectivo: features conhecidas antes do fechamento do mês
PROSPECTIVE_FEATURES: frozenset[str] = frozenset({
    "idade",
    "sexo",
    "tipo_cadastro",
    "valor_faturamento",
})


# =============================================================================
# %% — Pipelines sklearn
# =============================================================================

def make_rf_pipeline(
    num_cols: list[str],
    cat_cols: list[str],
    n_estimators: int = 300,
) -> Pipeline:
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", _make_ohe()),
    ])
    prep = ColumnTransformer(
        [("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)],
        remainder="drop",
    )
    return Pipeline([
        ("prep", prep),
        ("model", RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])


def split_num_cat(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    num_cols, cat_cols = [], []
    for c in X.columns:
        (num_cols if pd.api.types.is_numeric_dtype(X[c]) else cat_cols).append(c)
    return num_cols, cat_cols


def extract_base_feature_ohe(
    f: str,
    bases_sorted: list[str],
    feature_to_category: dict[str, str],
) -> str:
    for base in bases_sorted:
        if f == base or f.startswith(base + "_"):
            return base
    return f


def _run_kfold_r2(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    kf: KFold,
    label: str = "",
) -> tuple[float, float, float, float]:
    """Retorna (mean, std, ci_lower, ci_upper) do R² via KFold.
    KFold adaptativo: reduz n_splits para segmentos pequenos; pula se < KFOLD_MIN_ROWS_SKIP."""
    rng_sub = np.random.RandomState(RANDOM_STATE)
    if KFOLD_MAX_ROWS is not None and len(X) > KFOLD_MAX_ROWS:
        idx = rng_sub.choice(len(X), size=KFOLD_MAX_ROWS, replace=False)
        X   = X.iloc[idx].reset_index(drop=True)
        y   = y.iloc[idx].reset_index(drop=True)

    n = len(X)
    if n < KFOLD_MIN_ROWS_SKIP:
        if label:
            print(f"    [{label}] SKIP — apenas {n} linhas (mín={KFOLD_MIN_ROWS_SKIP})")
        return float("nan"), float("nan"), float("nan"), float("nan")

    if n < KFOLD_MIN_ROWS_ADAPTIVE and kf.n_splits > 2:
        kf_eff = KFold(n_splits=2, shuffle=kf.shuffle, random_state=RANDOM_STATE)
        if label:
            print(f"    [{label}] KFold adaptativo: {kf.n_splits} → 2 folds (n={n})")
    else:
        kf_eff = kf

    scores = []
    for fold_i, (tr, te) in enumerate(kf_eff.split(X), 1):
        if label:
            print(f"    [{label}] fold {fold_i}/{kf_eff.n_splits} ...", flush=True)
        pipeline.fit(X.iloc[tr], y.iloc[tr])
        scores.append(r2_score(y.iloc[te], pipeline.predict(X.iloc[te])))

    arr  = np.array(scores)
    mean = float(arr.mean())
    std  = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    ci   = 1.96 * std / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
    return mean, std, float(mean - ci), float(mean + ci)


# =============================================================================
# %% — Análise de correlação por segmento
# =============================================================================

def correlation_analysis_segment(
    df_seg: pd.DataFrame,
    corr_numeric_cols: list[str],
    out_dir: Path,
    segment_slug: str,
) -> dict[str, Any]:
    """
    Exporta: corr_pearson, corr_spearman, corr_long, high_corr_pairs,
    corr_rank_spearman_vs_target, heatmaps por família e top-K com alvo.
    """
    meta: dict[str, Any] = {"segment": segment_slug, "artifacts": [], "metrics": {}}
    if len(corr_numeric_cols) < 2:
        return meta

    dnum  = df_seg[corr_numeric_cols].copy()
    pear  = dnum.corr(method="pearson")
    spear = dnum.corr(method="spearman")

    pear_path  = out_dir / f"corr_pearson_{segment_slug}.csv"
    spear_path = out_dir / f"corr_spearman_{segment_slug}.csv"
    pear.to_csv(pear_path)
    spear.to_csv(spear_path)
    meta["artifacts"].extend([pear_path, spear_path])

    # Formato long/tidy
    pear_long  = pear.stack().reset_index()
    spear_long = spear.stack().reset_index()
    pear_long.columns  = ["feature_a", "feature_b", "pearson"]   # type: ignore[assignment]
    spear_long.columns = ["feature_a", "feature_b", "spearman"]  # type: ignore[assignment]
    corr_long = pear_long.merge(spear_long, on=["feature_a", "feature_b"])
    corr_long["segment"] = segment_slug
    corr_long = corr_long[corr_long["feature_a"] < corr_long["feature_b"]].reset_index(drop=True)
    corr_long_path = out_dir / f"corr_long_{segment_slug}.csv"
    corr_long.to_csv(corr_long_path, index=False)
    meta["artifacts"].append(corr_long_path)

    # Pares de alta correlação (alerta de redundância)
    high_pairs = corr_long[corr_long["spearman"].abs() > CORR_HIGH_THRESHOLD].copy()
    high_pairs = high_pairs.sort_values("spearman", ascending=False, key=abs)
    high_pairs_path = out_dir / f"high_corr_pairs_{segment_slug}.csv"
    high_pairs.to_csv(high_pairs_path, index=False)
    meta["artifacts"].append(high_pairs_path)
    meta["metrics"]["n_high_corr_pairs"] = int(len(high_pairs))

    # Ranking Spearman vs alvo
    target_cols = [c for c in corr_numeric_cols if c != TARGET_COL]
    if TARGET_COL in dnum.columns and target_cols:
        s_spear   = dnum[target_cols].corrwith(dnum[TARGET_COL], method="spearman")
        s_pearson = dnum[target_cols].corrwith(dnum[TARGET_COL], method="pearson")
        rank_abs  = s_spear.abs().sort_values(ascending=False).head(CORR_TOP_N_RANK)
        ranking   = pd.DataFrame({
            "feature":      rank_abs.index,
            "abs_spearman": rank_abs.values,
            "spearman":     s_spear.loc[rank_abs.index].values,
            "pearson":      s_pearson.loc[rank_abs.index].values,
            "segment":      segment_slug,
        })
        rank_path = out_dir / f"corr_rank_spearman_vs_target_{segment_slug}.csv"
        ranking.to_csv(rank_path, index=False)
        meta["artifacts"].append(rank_path)
        meta["metrics"]["max_abs_spearman_vs_target"] = float(ranking["abs_spearman"].iloc[0])
        meta["top_features_by_spearman"] = ranking["feature"].head(PDP_TOP_N).tolist()

    # Heatmaps por família de feature
    families = {
        "qtd_servico": [c for c in corr_numeric_cols if c.startswith("qtd_servico_")],
        "qtd_esp":     [c for c in corr_numeric_cols if c.startswith("qtd_esp_")],
        "qtd_conta":   [c for c in corr_numeric_cols if c.startswith("qtd_conta_")],
        "dem_fin":     [c for c in corr_numeric_cols
                        if c in ("idade", "valor_faturamento", "pct_urgencia", TARGET_COL)],
    }
    for fam, cols in families.items():
        cols_f = [c for c in cols if c in spear.columns]
        if len(cols_f) < 2:
            continue
        sub   = spear.loc[cols_f, cols_f]
        hpath = out_dir / f"heatmap_spearman_{fam}_{segment_slug}.png"
        _save_corr_heatmap(sub, f"Spearman — {fam} — {segment_slug}", hpath)
        meta["artifacts"].append(hpath)

    # Heatmap top-K com alvo
    if TARGET_COL in spear.columns:
        others   = [c for c in spear.columns if c != TARGET_COL]
        top_vars = (
            spear[TARGET_COL].reindex(others).abs()
            .sort_values(ascending=False).head(TOP_K_CORR_HEATMAP).index.tolist()
        )
        idx     = [c for c in [TARGET_COL] + top_vars if c in spear.index]
        sub_top = spear.loc[idx, idx]
        htop    = out_dir / f"heatmap_spearman_top{TOP_K_CORR_HEATMAP}_target_{segment_slug}.png"
        _save_corr_heatmap(sub_top, f"Spearman — top {TOP_K_CORR_HEATMAP} vs alvo — {segment_slug}", htop)
        meta["artifacts"].append(htop)

    return meta


# =============================================================================
# %% — Diagnóstico de quasi-leakage
# =============================================================================

def quasi_leakage_analysis(
    df_seg: pd.DataFrame,
    out_dir: Path,
    segment_slug: str,
) -> dict[str, Any]:
    """
    Calcula correlação das colunas quasi-leakage com o alvo e exporta CSV
    de diagnóstico. Não entram no treino — apenas documentam a relação espúria.
    """
    present = [c for c in QUASI_LEAKAGE_COLS if c in df_seg.columns]
    if not present or TARGET_COL not in df_seg.columns:
        return {}

    rows = []
    for c in present:
        if pd.api.types.is_numeric_dtype(df_seg[c]):
            sp = df_seg[c].corr(df_seg[TARGET_COL], method="spearman")
            pe = df_seg[c].corr(df_seg[TARGET_COL], method="pearson")
            rows.append({
                "feature":             c,
                "spearman_vs_target":  sp,
                "pearson_vs_target":   pe,
                "segment":             segment_slug,
                "motivo_exclusao":     "quasi_leakage",
            })

    if not rows:
        return {}

    df_ql = pd.DataFrame(rows)
    path  = out_dir / f"quasi_leakage_corr_{segment_slug}.csv"
    df_ql.to_csv(path, index=False)
    print(f"  {segment_slug} | quasi-leakage → {path.name}")
    return {"path": path, "data": df_ql}


# =============================================================================
# %% — Deduplicação de features redundantes
# =============================================================================

def deduplicate_features_by_spearman(
    df: pd.DataFrame,
    features: list[str],
    threshold: float = DEDUP_SPEARMAN_THRESHOLD,
    out_csv: Path | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    """
    Identifica pares de features numéricas com |Spearman| ≥ threshold e descarta
    a redundante de cada par. Regra de desempate:
      1. qtd_servico_* vence qtd_esp_* (mesma taxonomia, nomes mais limpos)
      2. nome mais curto vence (mais genérico)
    Retorna (features_dedup, lista_pares_removidos).
    """
    num_feats = [f for f in features if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
    cat_feats = [f for f in features if f not in num_feats]

    if len(num_feats) < 2:
        return features, []

    corr = df[num_feats].corr(method="spearman").abs()

    to_remove: set[str] = set()
    pairs_log: list[dict[str, Any]] = []

    for i, a in enumerate(num_feats):
        if a in to_remove:
            continue
        for b in num_feats[i + 1:]:
            if b in to_remove:
                continue
            val = corr.loc[a, b] if (a in corr.index and b in corr.columns) else 0.0
            if pd.isna(val) or val < threshold:
                continue
            # Regra de desempate: qtd_servico_ vence qtd_esp_; senão nome mais curto vence
            if b.startswith("qtd_esp_") and a.startswith("qtd_servico_"):
                drop, keep = b, a
            elif a.startswith("qtd_esp_") and b.startswith("qtd_servico_"):
                drop, keep = a, b
            elif len(a) <= len(b):
                drop, keep = b, a
            else:
                drop, keep = a, b
            to_remove.add(drop)
            pairs_log.append({"kept": keep, "dropped": drop, "abs_spearman": float(val)})

    deduped_num = [f for f in num_feats if f not in to_remove]
    deduped = deduped_num + cat_feats   # preserva cat features intactas

    if out_csv and pairs_log:
        pd.DataFrame(pairs_log).to_csv(out_csv, index=False)

    return deduped, pairs_log


# =============================================================================
# %% — Impacto por segmento (R², Permutation, SHAP, PDP)
# =============================================================================

def impact_segment(
    df_prod: pd.DataFrame,
    feature_catalog: pd.DataFrame,
    valid_features: list[str],
    valid_categories: list[str],
    feature_to_category: dict[str, str],
    bases_sorted: list[str],
    out_dir: Path,
    segment_slug: str,
    kf: KFold,
    prospective_features: list[str] | None = None,
) -> dict[str, Any]:
    """
    Exporta por segmento:
    - r2_isolado_por_categoria_{seg}.csv   (KFold-4, IC 95%)
    - r2_isolado_prospectivo_{seg}.csv
    - perm_global_{seg}.csv / perm_categoria_{seg}.csv
    - compare_isolado_vs_global_{seg}.csv
    - impacto_global_{seg}.png
    - shap_values_{seg}.csv / shap_summary_{seg}.csv / shap_categoria_{seg}.csv
    - pdp/pdp_{feature}_{seg}.csv
    """
    y      = df_prod[TARGET_COL]
    X_base = df_prod[valid_features].copy()
    for c in X_base.columns:
        if X_base[c].dtype == object:
            X_base[c] = X_base[c].astype("category")

    result: dict[str, Any] = {
        "segment_slug":    segment_slug,
        "paths":           {},
        "r2_global":       None,
        "r2_global_std":   None,
        "r2_global_ci":    None,
        "pipeline_global": None,
    }

    # ── 1. R² isolado por categoria ──────────────────────────────────────────
    r2_isolado_rows: list[dict[str, Any]] = []
    for category in valid_categories:
        feats_cat = feature_catalog[
            (feature_catalog["category"] == category)
            & (feature_catalog["feature_name"].isin(X_base.columns))
        ]["feature_name"].tolist()
        feats_cat = [f for f in feats_cat if f in X_base.columns]
        if not feats_cat:
            continue

        X = X_base[feats_cat]
        num_c, cat_c = split_num_cat(X)
        if not num_c and not cat_c:
            continue

        pipe = make_rf_pipeline(num_c, cat_c, n_estimators=300)
        mean, std, ci_lo, ci_hi = _run_kfold_r2(pipe, X, y, kf, label=f"{category}")
        r2_isolado_rows.append({
            "category":    category,
            "r2_mean":     mean,
            "r2_std":      std,
            "r2_ci_lower": ci_lo,
            "r2_ci_upper": ci_hi,
            "n_features":  len(feats_cat),
            "segment":     segment_slug,
        })
        print(f"  {segment_slug} | {category:20s} | R2 isolado = {mean:.3f} ± {std:.3f}")

    df_r2_isolado = pd.DataFrame(r2_isolado_rows)
    iso_path = out_dir / f"r2_isolado_por_categoria_{segment_slug}.csv"
    df_r2_isolado.to_csv(iso_path, index=False)
    result["paths"]["isolated"] = iso_path

    # ── 2. R² prospectivo ───────────────────────────────────────────────────
    if prospective_features:
        prosp_in_X = [f for f in prospective_features if f in X_base.columns]
        if len(prosp_in_X) >= 2:
            X_prosp      = X_base[prosp_in_X]
            num_p, cat_p = split_num_cat(X_prosp)
            pipe_p       = make_rf_pipeline(num_p, cat_p, n_estimators=300)
            mean_p, std_p, ci_lo_p, ci_hi_p = _run_kfold_r2(
                pipe_p, X_prosp, y, kf, label="prospectivo"
            )
            df_prosp = pd.DataFrame([{
                "modo":        "prospectivo",
                "r2_mean":     mean_p,
                "r2_std":      std_p,
                "r2_ci_lower": ci_lo_p,
                "r2_ci_upper": ci_hi_p,
                "n_features":  len(prosp_in_X),
                "features":    "|".join(prosp_in_X),
                "segment":     segment_slug,
            }])
            prosp_path = out_dir / f"r2_isolado_prospectivo_{segment_slug}.csv"
            df_prosp.to_csv(prosp_path, index=False)
            result["paths"]["prospective"] = prosp_path
            print(f"  {segment_slug} | PROSPECTIVO | R2 = {mean_p:.3f} ± {std_p:.3f}")

    # ── 2b. R² isolado — apenas positivos (sinistralidade > 0) ──────────────
    # Filtra apenas os casos com sinistro real para medir o quanto as features
    # explicam a GRAVIDADE, removendo o ruído dos zeros estruturais.
    mask_pos = y > 0
    n_pos    = int(mask_pos.sum())
    print(f"  {segment_slug} | POSITIVOS: {n_pos:,} de {len(y):,} registros ({100*n_pos/max(len(y),1):.1f}%)")

    if n_pos >= KFOLD_MIN_ROWS_SKIP:
        y_pos  = y[mask_pos].reset_index(drop=True)
        X_pos  = X_base[mask_pos].reset_index(drop=True)
        kf_pos = KFold(
            n_splits=2 if n_pos < KFOLD_MIN_ROWS_ADAPTIVE else N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        r2_pos_rows: list[dict[str, Any]] = []
        for category in valid_categories:
            feats_cat = feature_catalog[
                (feature_catalog["category"] == category)
                & (feature_catalog["feature_name"].isin(X_pos.columns))
            ]["feature_name"].tolist()
            feats_cat = [f for f in feats_cat if f in X_pos.columns]
            if not feats_cat:
                continue
            X_c = X_pos[feats_cat]
            num_c, cat_c = split_num_cat(X_c)
            if not num_c and not cat_c:
                continue
            pipe_pos = make_rf_pipeline(num_c, cat_c, n_estimators=300)
            mean_pos, std_pos, ci_lo_pos, ci_hi_pos = _run_kfold_r2(
                pipe_pos, X_c, y_pos, kf_pos, label=f"{category}_pos"
            )
            r2_pos_rows.append({
                "category":    category,
                "r2_mean":     mean_pos,
                "r2_std":      std_pos,
                "r2_ci_lower": ci_lo_pos,
                "r2_ci_upper": ci_hi_pos,
                "n_features":  len(feats_cat),
                "n_positivos": n_pos,
                "segment":     segment_slug,
            })
            print(f"  {segment_slug} | {category:20s} | R2 positivos = {mean_pos:.3f} ± {std_pos:.3f}")

        df_r2_pos = pd.DataFrame(r2_pos_rows)
        pos_path  = out_dir / f"r2_isolado_positivos_{segment_slug}.csv"
        df_r2_pos.to_csv(pos_path, index=False)
        result["paths"]["isolated_positives"] = pos_path
    else:
        print(f"  {segment_slug} | POSITIVOS: apenas {n_pos} — skip KFold")

    # ── 3. Modelo global ────────────────────────────────────────────────────
    num_cols, cat_cols = split_num_cat(X_base)
    pipeline_global    = make_rf_pipeline(num_cols, cat_cols, n_estimators=300)
    print(f"  {segment_slug} | GLOBAL KFold ...", flush=True)
    r2_mean_g, r2_std_g, ci_lo_g, ci_hi_g = _run_kfold_r2(
        pipeline_global, X_base, y, kf, label="global"
    )
    print(f"  {segment_slug} | GLOBAL treino final ...", flush=True)
    pipeline_global.fit(X_base, y)
    X_proc = pipeline_global.named_steps["prep"].transform(X_base)

    print(f"  {segment_slug} | GLOBAL | R2 = {r2_mean_g:.3f} ± {r2_std_g:.3f}")
    result["r2_global"]      = r2_mean_g
    result["r2_global_std"]  = r2_std_g
    result["r2_global_ci"]   = (ci_lo_g, ci_hi_g)
    result["pipeline_global"] = pipeline_global

    ohe       = pipeline_global.named_steps["prep"].named_transformers_["cat"].named_steps["ohe"]
    cat_names = ohe.get_feature_names_out(cat_cols).tolist() if cat_cols else []
    feature_names_proc = num_cols + list(cat_names)

    # ── 4. Permutation importance ───────────────────────────────────────────
    rng = np.random.RandomState(RANDOM_STATE)
    n_perm  = len(X_base) if PERM_IMPORTANCE_MAX_ROWS is None else min(PERM_IMPORTANCE_MAX_ROWS, len(X_base))
    perm_idx = rng.choice(len(X_base), size=n_perm, replace=False)
    X_perm   = X_proc[perm_idx]
    y_perm   = y.iloc[perm_idx]

    print(f"  {segment_slug} | Permutation importance ({n_perm} linhas, {PERM_N_REPEATS} repeats) ...", flush=True)
    perm = permutation_importance(
        pipeline_global.named_steps["model"],
        X_perm, y_perm,
        n_repeats=PERM_N_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scoring="r2",
    )
    df_perm = pd.DataFrame({
        "feature":    feature_names_proc,
        "importance": perm.importances_mean,
        "std":        perm.importances_std,
        "segment":    segment_slug,
    }).sort_values("importance", ascending=False)
    perm_path = out_dir / f"perm_global_{segment_slug}.csv"
    df_perm.to_csv(perm_path, index=False)
    result["paths"]["perm"] = perm_path

    def _base(f: str) -> str:
        return extract_base_feature_ohe(f, bases_sorted, feature_to_category)

    df_perm["base_feature"] = df_perm["feature"].apply(_base)
    df_perm["category"] = (
        df_perm["base_feature"].map(lambda b: feature_to_category.get(b, "outros")).fillna("outros")
    )
    df_perm_cat = (
        df_perm.groupby("category")["importance"]
        .sum().reset_index()
        .sort_values("importance", ascending=False)
    )
    s = df_perm_cat["importance"].sum()
    df_perm_cat["importance_norm"] = df_perm_cat["importance"] / s if s else 0.0
    df_perm_cat["segment"]         = segment_slug
    perm_cat_path = out_dir / f"perm_categoria_{segment_slug}.csv"
    df_perm_cat.to_csv(perm_cat_path, index=False)
    result["paths"]["perm_cat"] = perm_cat_path

    # Compare isolado × global
    df_compare = (
        df_perm_cat
        .rename(columns={"importance_norm": "impacto_global"})
        .merge(
            df_r2_isolado[["category", "r2_mean", "r2_std", "r2_ci_lower", "r2_ci_upper"]],
            on="category", how="left",
        )
    )
    df_compare.columns = [c.replace("r2_mean", "r2_isolado") for c in df_compare.columns]
    compare_path = out_dir / f"compare_isolado_vs_global_{segment_slug}.csv"
    df_compare.to_csv(compare_path, index=False)
    result["paths"]["compare"] = compare_path

    # Gráfico impacto por categoria
    plt.figure(figsize=(7, max(3, len(df_perm_cat) * 0.5)))
    bars = plt.barh(df_perm_cat["category"], df_perm_cat["importance_norm"], color="#2196F3")
    plt.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    plt.gca().invert_yaxis()
    plt.xlabel("Importância normalizada (permutation)")
    plt.title(f"{segment_slug} — Impacto global por categoria")
    plt.tight_layout()
    plot_path = out_dir / f"impacto_global_{segment_slug}.png"
    plt.savefig(plot_path, dpi=120)
    plt.close()
    result["paths"]["plot"] = plot_path

    # ── 5. SHAP values ──────────────────────────────────────────────────────
    if HAS_SHAP:
        try:
            n_shap   = len(X_base) if SHAP_MAX_ROWS is None else min(SHAP_MAX_ROWS, len(X_base))
            shap_idx = rng.choice(len(X_base), size=n_shap, replace=False)
            X_shap_arr = X_proc[shap_idx]
            y_shap     = y.iloc[shap_idx]

            print(f"  {segment_slug} | SHAP ({n_shap} linhas) ...", flush=True)
            explainer   = shap.TreeExplainer(pipeline_global.named_steps["model"])
            shap_values = explainer.shap_values(X_shap_arr)

            df_shap_vals = pd.DataFrame(shap_values, columns=feature_names_proc)
            df_shap_vals["segment"]              = segment_slug
            df_shap_vals["sinistralidade_real"]  = y_shap.values
            shap_vals_path = out_dir / f"shap_values_{segment_slug}.csv"
            df_shap_vals.to_csv(shap_vals_path, index=False)
            result["paths"]["shap_values"] = shap_vals_path

            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            df_shap_sum   = pd.DataFrame({
                "feature":       feature_names_proc,
                "mean_abs_shap": mean_abs_shap,
                "segment":       segment_slug,
            }).sort_values("mean_abs_shap", ascending=False)
            df_shap_sum["base_feature"] = df_shap_sum["feature"].apply(_base)
            df_shap_sum["category"] = (
                df_shap_sum["base_feature"].map(lambda b: feature_to_category.get(b, "outros")).fillna("outros")
            )
            shap_sum_path = out_dir / f"shap_summary_{segment_slug}.csv"
            df_shap_sum.to_csv(shap_sum_path, index=False)
            result["paths"]["shap_summary"] = shap_sum_path

            df_shap_cat = (
                df_shap_sum.groupby("category")["mean_abs_shap"]
                .sum().reset_index()
                .sort_values("mean_abs_shap", ascending=False)
            )
            s_shap = df_shap_cat["mean_abs_shap"].sum()
            df_shap_cat["shap_norm"] = df_shap_cat["mean_abs_shap"] / s_shap if s_shap else 0.0
            df_shap_cat["segment"]   = segment_slug
            shap_cat_path = out_dir / f"shap_categoria_{segment_slug}.csv"
            df_shap_cat.to_csv(shap_cat_path, index=False)
            result["paths"]["shap_cat"] = shap_cat_path

            print(f"  {segment_slug} | SHAP concluído.")
        except Exception as e:
            print(f"  [WARN] SHAP falhou para {segment_slug}: {e}")

    # ── 6. Partial Dependence Plots ─────────────────────────────────────────
    top_num_features = [
        f for f in df_perm["feature"].head(PDP_TOP_N * 3).tolist()
        if f in num_cols
    ][:PDP_TOP_N]

    print(f"  {segment_slug} | PDP ({len(top_num_features)} features) ...", flush=True)
    pdp_dir = out_dir / "pdp"
    pdp_dir.mkdir(exist_ok=True)
    pdp_paths = []

    for feat in top_num_features:
        if feat not in feature_names_proc:
            continue
        feat_idx = feature_names_proc.index(feat)
        try:
            pd_result = partial_dependence(
                pipeline_global.named_steps["model"],
                X_proc,
                features=[feat_idx],
                grid_resolution=PDP_GRID_RESOLUTION,
                kind="average",
            )
            df_pdp = pd.DataFrame({
                "feature_value":            pd_result["grid_values"][0],
                "predicted_sinistralidade": pd_result["average"][0],
                "feature":                  feat,
                "segment":                  segment_slug,
            })
            safe_feat = re.sub(r"[^\w]", "_", feat)[:60]
            pdp_path  = pdp_dir / f"pdp_{safe_feat}_{segment_slug}.csv"
            df_pdp.to_csv(pdp_path, index=False)
            pdp_paths.append(pdp_path)
        except Exception as e:
            print(f"  [WARN] PDP falhou para {feat}: {e}")

    result["paths"]["pdp"] = pdp_paths
    return result


# =============================================================================
# %% — EXECUÇÃO: Carga de dados e catálogo
# =============================================================================

feature_catalog = load_feature_catalog()
print(f"Catálogo carregado: {len(feature_catalog)} features — {FEATURE_CATALOG_PATH}")

df_raw = pd.read_parquet(TRANSFORMED_PARQUET_PATH)
print(f"Parquet carregado: {df_raw.shape} — {TRANSFORMED_PARQUET_PATH}")

# Features elegíveis (include_in_model == True, excluindo leakage e ids)
eligible_base = [
    f for f in catalog_eligible_names(feature_catalog)
    if f in df_raw.columns
    and f not in LEAKAGE_COLS
    and f not in QUASI_LEAKAGE_COLS
    and f not in (BENEFICIARIO_COL, TIME_COL, SEGMENT_COL)
]
print(f"Features elegíveis para análise: {len(eligible_base)}")
print(f"  {eligible_base}")

# Mapeamento feature → categoria (para agrupamento de resultados)
feature_to_category: dict[str, str] = dict(
    zip(feature_catalog["feature_name"], feature_catalog["category"])
)
valid_categories = sorted(
    feature_catalog.loc[
        feature_catalog["feature_name"].isin(eligible_base), "category"
    ].dropna().unique()
)
bases_sorted = sorted(eligible_base, key=len, reverse=True)


# =============================================================================
# %% — EXECUÇÃO: Versionamento e diretório de saída
# =============================================================================

ver, OUTPUT_DIR = next_version_dir(OUTPUT_FEATURE_IMPACT_ROOT)
print(f"Versão de saída : {ver}")
print(f"Diretório       : {OUTPUT_DIR}")


# =============================================================================
# %% — EXECUÇÃO: Correlação mensal (artefato principal para o app)
# =============================================================================
# Gera feature_correlation_sinistralidade.csv por competência (mês).
# Este é o arquivo consumido pela aba "Correlação" do Streamlit.

if TIME_COL not in df_raw.columns:
    raise ValueError(f"Coluna temporal ausente: {TIME_COL}")

comp        = pd.to_datetime(df_raw[TIME_COL], errors="coerce")
df_raw_comp = df_raw.loc[comp.notna()].copy()
df_raw_comp["_comp_period"] = comp.loc[comp.notna()].dt.to_period("M")
meses = sorted(df_raw_comp["_comp_period"].dropna().unique())

print(f"Competências detectadas: {len(meses)}  ({meses[0]} → {meses[-1]})")

for mes in meses:
    mes_slug = str(mes)
    mes_dir  = OUTPUT_DIR / mes_slug
    mes_dir.mkdir(parents=True, exist_ok=True)

    d = df_raw_comp.loc[df_raw_comp["_comp_period"] == mes].copy()
    y = pd.to_numeric(d[TARGET_COL], errors="coerce")

    rows: list[dict[str, Any]] = []
    for col in eligible_base:
        s = _spearman_vs_target(d[col], y)
        if s is None:
            continue
        rows.append({"feature": col, "spearman": float(s)})

    out = pd.DataFrame(rows)
    if not out.empty:
        out["abs_spearman"] = out["spearman"].abs()
        out = out.sort_values("abs_spearman", ascending=False).drop(columns=["abs_spearman"])

    csv_path = mes_dir / CORR_ARTIFACT_FILENAME
    out.to_csv(csv_path, index=False)
    print(f"  [OK] {mes_slug}: {len(out):,} features → {csv_path.name}")

print(f"\n[OK] Correlação mensal concluída. Artefatos em: {OUTPUT_DIR}")


# =============================================================================
# %% — EXECUÇÃO: Agregação por beneficiário (base para análises retrospectivas)
# =============================================================================

if AGGREGATE_BY_BENEFICIARIO:
    print("Agregando painel por beneficiário ...", flush=True)
    df_agg = aggregate_panel_by_beneficiary(df_raw)
    print(f"  Painel mensal : {df_raw.shape}")
    print(f"  Agregado      : {df_agg.shape} ({df_agg[N_MESES_COL].mean():.1f} meses/benef em média)")
    df_prod = df_agg.copy()
else:
    df_prod = df_raw.copy()

# Verificar planos disponíveis na base agregada
planos_disponiveis = [p for p in PLANOS_CANONICOS if p in df_prod[SEGMENT_COL].values]
print(f"Planos disponíveis na base: {planos_disponiveis}")


# =============================================================================
# %% — EXECUÇÃO: Correlação por segmento (matrizes + high-corr + heatmaps)
# =============================================================================

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

corr_results: dict[str, Any] = {}

for plano in planos_disponiveis:
    slug    = plano_slug(plano)
    df_seg  = df_prod[df_prod[SEGMENT_COL] == plano].copy()
    seg_dir = OUTPUT_DIR / slug
    seg_dir.mkdir(parents=True, exist_ok=True)

    num_cols_corr = [
        c for c in eligible_base
        if c in df_seg.columns and pd.api.types.is_numeric_dtype(df_seg[c])
    ]
    if TARGET_COL in df_seg.columns and TARGET_COL not in num_cols_corr:
        num_cols_corr.append(TARGET_COL)

    print(f"\n{'─'*60}")
    print(f"  Correlação — {plano} (n={len(df_seg):,})")
    res = correlation_analysis_segment(df_seg, num_cols_corr, seg_dir, slug)
    corr_results[slug] = res
    n_pairs = res["metrics"].get("n_high_corr_pairs", 0)
    print(f"  [OK] {slug}: {n_pairs} pares com |Spearman| > {CORR_HIGH_THRESHOLD}")

print("\n[OK] Correlação por segmento concluída.")


# =============================================================================
# %% — EXECUÇÃO: Diagnóstico de quasi-leakage
# =============================================================================

ql_results: dict[str, Any] = {}

for plano in planos_disponiveis:
    slug   = plano_slug(plano)
    df_seg = df_prod[df_prod[SEGMENT_COL] == plano].copy()
    seg_dir = OUTPUT_DIR / slug
    res = quasi_leakage_analysis(df_seg, seg_dir, slug)
    ql_results[slug] = res

print("[OK] Diagnóstico de quasi-leakage concluído.")


# =============================================================================
# %% — EXECUÇÃO: Deduplicação de features redundantes por segmento
# =============================================================================
# Features numéricas com |Spearman| ≥ DEDUP_SPEARMAN_THRESHOLD entre si são
# tratadas como redundantes. Mantemos uma de cada par antes das etapas pesadas
# (R², permutation, SHAP, PDP) para não dividir importância entre colunas idênticas.
# Saídas: features_redundantes.csv + features_redundantes.json por segmento.

dedup_results: dict[str, list[str]] = {}

for plano in planos_disponiveis:
    slug    = plano_slug(plano)
    df_seg  = df_prod[df_prod[SEGMENT_COL] == plano].copy()
    seg_dir = OUTPUT_DIR / slug
    seg_dir.mkdir(parents=True, exist_ok=True)

    feats_seg = [f for f in eligible_base if f in df_seg.columns]

    feats_dedup, pairs = deduplicate_features_by_spearman(
        df_seg, feats_seg,
        threshold=DEDUP_SPEARMAN_THRESHOLD,
        out_csv=seg_dir / "features_redundantes.csv",
    )

    redundantes_json = {
        "threshold":          DEDUP_SPEARMAN_THRESHOLD,
        "n_original":         len(feats_seg),
        "n_apos_dedup":       len(feats_dedup),
        "features_removidas": [p["dropped"] for p in pairs],
        "pares_removidos":    pairs,
    }
    (seg_dir / "features_redundantes.json").write_text(
        json.dumps(redundantes_json, indent=2, default=str), encoding="utf-8"
    )

    dedup_results[slug] = feats_dedup
    n_rem = len(feats_seg) - len(feats_dedup)
    print(f"\n  {slug}: {len(feats_seg)} → {len(feats_dedup)} features ({n_rem} removidas por redundância)")
    for p in pairs:
        print(f"    drop {p['dropped']!r:50s} mantém {p['kept']!r}  (Spearman={p['abs_spearman']:.4f})")

print(f"\n[OK] Deduplicação concluída.")


# =============================================================================
# %% — EXECUÇÃO: Impacto por segmento (R², Permutation, SHAP, PDP)
# =============================================================================
# Etapa mais lenta — KFold por categoria + modelo global + SHAP + PDP.
# Executar após confirmar os blocos anteriores.

impact_results: dict[str, Any] = {}

prosp_features = [f for f in PROSPECTIVE_FEATURES if f in eligible_base]

for plano in planos_disponiveis:
    slug    = plano_slug(plano)
    df_seg  = df_prod[df_prod[SEGMENT_COL] == plano].dropna(subset=[TARGET_COL]).copy()
    seg_dir = OUTPUT_DIR / slug
    seg_dir.mkdir(parents=True, exist_ok=True)

    # Usa lista deduplicada; fallback para elegíveis caso o bloco anterior não tenha rodado
    feats_seg = dedup_results.get(slug, [f for f in eligible_base if f in df_seg.columns])
    feats_seg = [f for f in feats_seg if f in df_seg.columns]
    cats_seg  = sorted(
        feature_catalog.loc[
            feature_catalog["feature_name"].isin(feats_seg), "category"
        ].dropna().unique()
    )

    print(f"\n{'─'*60}")
    print(f"  Impact — {plano} (n={len(df_seg):,}, features={len(feats_seg)})")

    res = impact_segment(
        df_prod            = df_seg,
        feature_catalog    = feature_catalog,
        valid_features     = feats_seg,
        valid_categories   = cats_seg,
        feature_to_category = feature_to_category,
        bases_sorted       = bases_sorted,
        out_dir            = seg_dir,
        segment_slug       = slug,
        kf                 = kf,
        prospective_features = prosp_features,
    )
    impact_results[slug] = res

print("\n[OK] Análise de impacto por segmento concluída.")


# =============================================================================
# %% — EXECUÇÃO: Gráfico de correlação global (top-N)
# =============================================================================
# Consome o CSV do último mês disponível e gera barras horizontais.

ultimo_mes = str(meses[-1])
csv_ultimo = OUTPUT_DIR / ultimo_mes / CORR_ARTIFACT_FILENAME

if csv_ultimo.is_file():
    df_plot = pd.read_csv(csv_ultimo)
    if not df_plot.empty and "spearman" in df_plot.columns:
        df_plot["_abs"] = df_plot["spearman"].abs()
        top = df_plot.nlargest(FEATURE_IMPACT_TOP_N_PLOT, "_abs").sort_values("spearman", ascending=True)
        fig_h = max(4.0, len(top) * 0.55)
        plt.figure(figsize=(9, fig_h))
        bars = plt.barh(top["feature"].astype(str), top["spearman"], color="#2196F3")
        plt.bar_label(bars, fmt="%.3f", padding=4, fontsize=8)
        plt.xlabel("Spearman vs sinistralidade_final")
        plt.xlim(-1.05, 1.05)
        plt.title(f"Associação com sinistralidade — top {len(top)} | {ultimo_mes}")
        plt.tight_layout()
        plot_path = OUTPUT_DIR / "impacto_global_top10.png"
        plt.savefig(plot_path, dpi=120)
        plt.close()
        print(f"[OK] Gráfico salvo: {plot_path}")


# =============================================================================
# %% — EXECUÇÃO: Meta / sumário da execução
# =============================================================================

meta = {
    "company":           COMPANY,
    "versao":            ver,
    "data_execucao":     datetime.now().strftime("%Y-%m-%d %H:%M"),
    "parquet_origem":    str(TRANSFORMED_PARQUET_PATH),
    "n_features_elegiveis": len(eligible_base),
    "features_elegiveis":   eligible_base,
    "planos_analisados":    planos_disponiveis,
    "competencias":      [str(m) for m in meses],
    "n_meses":           len(meses),
    "aggregate_by_beneficiario": AGGREGATE_BY_BENEFICIARIO,
    "r2_global_por_plano": {
        slug: {
            "r2_mean": res.get("r2_global"),
            "r2_std":  res.get("r2_global_std"),
            "r2_ci":   res.get("r2_global_ci"),
        }
        for slug, res in impact_results.items()
    },
    "n_high_corr_pairs_por_plano": {
        slug: res.get("metrics", {}).get("n_high_corr_pairs", 0)
        for slug, res in corr_results.items()
    },
    "dedup_por_plano": {
        slug: {
            "n_original":         len([f for f in eligible_base if f in df_prod.columns]),
            "n_apos_dedup":       len(feats_dedup),
            "n_removidas":        len([f for f in eligible_base if f in df_prod.columns]) - len(feats_dedup),
        }
        for slug, feats_dedup in dedup_results.items()
    },
}

meta_path = OUTPUT_DIR / "run_metadata.json"
meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

print(f"\n{'='*60}")
print(f"  Feature Impact Climazon — {ver} concluído")
print(f"  Diretório : {OUTPUT_DIR}")
print(f"  Meta      : {meta_path.name}")
for slug, res in impact_results.items():
    r2 = res.get("r2_global")
    print(f"  {slug}: R2 global = {r2:.3f}" if r2 is not None else f"  {slug}: R2 global = N/A")
print(f"{'='*60}")

# %%
