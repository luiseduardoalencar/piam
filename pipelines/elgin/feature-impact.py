# %%
"""
Feature Impact + correlação — Sinistralidade ELGIN  (v2)

Constantes de leakage/quasi-leakage espelham a lógica de ``predict.py``; este script
é autocontido (não importa ``predict.py``).

Melhorias em relação à v1
--------------------------
- LEAKAGE EXPANDIDO: qtd_eventos_sinistro, qtd_carater_eletivo e qtd_carater_urgencia
  movidos para QUASI_LEAKAGE_COLS — co-ocorrem com o sinistro por definição e inflavam
  artificialmente o R² e as correlações. Analisados separadamente em bloco próprio.

- MODO PROSPECTIVO (proxy): R² com subset mínimo de colunas do painel que existem sem
  engenharia extra (ver PROSPECTIVE_FEATURES) — aproxima “informação conhecida antes
  do fechamento” apenas com variáveis já na base analítica.

- CORRELAÇÃO LONG/TIDY: além das matrizes wide, exporta corr_long_{seg}.csv com
  colunas (feature_a, feature_b, pearson, spearman, segment) — consumível direto
  em Metabase / Superset / PowerBI sem pivot.

- HIGH-CORR PAIRS: CSV com todos os pares |Spearman| > CORR_HIGH_THRESHOLD,
  sinalizando redundância para remoção de features.

- IC DO R² (KFold): r2_isolado_por_categoria inclui r2_mean, r2_std,
  r2_ci_lower, r2_ci_upper (IC 95%). Idem para o R² global.

- SHAP VALUES: exporta shap_values_{seg}.csv (amostra) e shap_summary_{seg}.csv
  (mean |SHAP| por feature e por categoria) para consumo na plataforma.

- PARTIAL DEPENDENCE: para o top-N features do ranking Spearman, exporta
  pdp_{feature}_{seg}.csv com (feature_value, predicted_sinistralidade).

- OUTPUTS TIDY: todos os CSVs principais têm coluna "segment" para facilitar
  empilhamento multi-segmento na plataforma.

- AGREGAÇÃO POR BENEFICIÁRIO (opcional): uma linha por ``cod_beneficiario``, sem
  ``competencia`` — contagens e prémios somados no período; ``sinistralidade_final``
  como média mensal; atributos estáveis com o primeiro valor temporal. Reduz N e
  remove duplicação mensal para o estudo de impacto (não é previsão mês a mês).

Saídas versionadas em ``data/processed/elgin/feature-impact/vN/``
MLflow (opcional): ``FEATURE_IMPACT_MLFLOW=1`` — **artefato único**
``feature_correlation_sinistralidade.csv`` (Spearman de cada variável com ``sinistralidade_final``,
[-1, 1]); params e métricas completos no run. Gráfico ``impacto_global_top10.png`` (top |Spearman|)
só em disco, não MLflow. ``perm_global_{seg}.csv`` mantém permutation importance do RF (outro conceito).
"""
# %%
from __future__ import annotations

import json
import os
import re
import sys
import warnings
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

import mlflow

from config.mlflow_config import configurar_mlflow

warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# %% — Constantes
# =============================================================================

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
OUTPUT_FEATURE_IMPACT_ROOT = ROOT_DIR / "data" / "processed" / COMPANY / "feature-impact"

TARGET_COL       = "sinistralidade_final"
TIME_COL         = "competencia"
SEGMENT_COL      = "plano"
BENEFICIARIO_COL = "cod_beneficiario"

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
# Excluídos dos modelos preditivos mas analisados em bloco separado de diagnóstico.
QUASI_LEAKAGE_COLS: frozenset[str] = frozenset({
    "qtd_eventos_sinistro",   # total de eventos = proxy direto do sinistro
    "qtd_carater_eletivo",    # contagem de eventos eletivos do próprio período
    "qtd_carater_urgencia",   # idem urgência
})

PLANO_MAP = {
    "EMPRESARIAL MASTER": "MASTER EMPRESARIAL",
    "COLETIVO EMPRESARIAL MASTER - PROTOCOLO ANS: 414538991": "MASTER EMPRESARIAL",
}

EXPERIMENT_NAME          = "piam-elgin-feature-impact"
N_SPLITS                 = 4
RANDOM_STATE             = 42
TOP_K_CORR_HEATMAP       = 25
CORR_TOP_N_RANK          = 40
CORR_HIGH_THRESHOLD = 0.70  # pares acima disto vão para high_corr_pairs.csv
PERM_N_REPEATS      = 5     # repetições na permutation importance
PDP_TOP_N           = 8     # features numéricas para partial dependence (CSV)
PDP_GRID_RESOLUTION = 40

# --- Amostragem de linhas (None = usar todas as linhas do segmento / base) ----------
# O Parquet é sempre lido na íntegra; estes limites aplicam-se só a etapas pesadas.
# Para testes rápidos, defina inteiros (ex.: KFOLD_MAX_ROWS = 20_000).
KFOLD_MAX_ROWS: int | None = None
PERM_IMPORTANCE_MAX_ROWS: int | None = None
SHAP_MAX_ROWS: int | None = None

# Agregar painel mensal → uma linha por beneficiário (remove eixo temporal).
AGGREGATE_BY_BENEFICIARIO = True
# Metadado de agregação (excluído do modelo e da lista de preditores).
N_MESES_COL = "n_meses_obs"

# MLflow: por defeito não corre (testes locais só com CSVs). Para registar:
#   FEATURE_IMPACT_MLFLOW=1 python pipelines/elgin/feature-impact.py
# Ou pôr FORCE_MLFLOW = True.
FORCE_MLFLOW = True
ENABLE_MLFLOW = FORCE_MLFLOW or os.environ.get("FEATURE_IMPACT_MLFLOW", "0").strip().lower() in (
    "1",
    "true",
    "yes",
)

# CSV único para MLflow: associação **bivariada** variável × sinistralidade (Spearman), [-1, 1]
MLFLOW_ARTIFACT_CSV_FILENAME = "feature_correlation_sinistralidade.csv"
# Gráfico local (não vai para MLflow) gerado a partir desse CSV
FEATURE_IMPACT_TOP_N_PLOT = 10
FEATURE_IMPACT_PLOT_FILENAME = "impacto_global_top10.png"

np.random.seed(RANDOM_STATE)


# =============================================================================
# %% — Utilitários gerais
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


def plano_slug(plano: str) -> str:
    s = re.sub(r"[^\w]+", "_", str(plano)).strip("_")
    return (s[:80] if s else "plano").upper()


def _spearman_vs_target(x: pd.Series, y: pd.Series) -> float | None:
    """Spearman entre preditor e ``sinistralidade_final``; numérico direto, categórico via codes."""
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


def write_feature_correlation_sinistralidade_csv(
    df_panel: pd.DataFrame,
    feature_cols: list[str],
    out_dir: Path,
    filename: str = MLFLOW_ARTIFACT_CSV_FILENAME,
) -> Path | None:
    """
    Uma linha por (segmento,) variável: **Spearman** com ``sinistralidade_final`` na base
    (mesma lógica de “quanto a variável se associa ao alvo” no painel). Valores em **[-1, 1]**.
    Ordenado por |Spearman| decrescente (maior associação primeiro).

    Com vários planos, prefixa ``segment_slug::`` ao nome da coluna.
    """
    rows: list[dict[str, Any]] = []
    if SEGMENT_COL not in df_panel.columns or TARGET_COL not in df_panel.columns:
        return None
    segments = sorted(df_panel[SEGMENT_COL].dropna().unique())
    multi_seg = len(segments) > 1
    for seg in segments:
        d = df_panel[df_panel[SEGMENT_COL] == seg].reset_index(drop=True)
        slug = plano_slug(str(seg))
        y = d[TARGET_COL]
        for col in feature_cols:
            if col not in d.columns or col == TARGET_COL:
                continue
            s = _spearman_vs_target(d[col], y)
            if s is None:
                continue
            feat = f"{slug}::{col}" if multi_seg else col
            rows.append({"feature": feat, "spearman": s})
    if not rows:
        return None
    out = pd.DataFrame(rows)
    out["abs_spearman"] = out["spearman"].abs()
    out = (
        out.sort_values("abs_spearman", ascending=False)
        .drop(columns=["abs_spearman"])
        .reset_index(drop=True)
    )
    path = out_dir / filename
    out.to_csv(path, index=False)
    return path


def plot_feature_impact_top_n(
    csv_path: Path,
    out_path: Path,
    n: int = FEATURE_IMPACT_TOP_N_PLOT,
) -> Path | None:
    """
    Lê o CSV de Spearman vs sinistralidade e grava barras horizontais com os **n** maiores |Spearman|.
    Coluna de valor: ``spearman`` (ou ``importance`` legado). Só em disco, não MLflow.
    """
    if not csv_path.is_file():
        return None
    df = pd.read_csv(csv_path)
    if df.empty or "feature" not in df.columns:
        return None
    val_col = "spearman" if "spearman" in df.columns else "importance"
    if val_col not in df.columns:
        return None
    work = df.copy()
    work["_abs"] = work[val_col].abs()
    top = work.nlargest(n, "_abs").drop(columns=["_abs"])
    top = top.sort_values(val_col, ascending=True)
    fig_h = max(4.0, len(top) * 0.55)
    plt.figure(figsize=(9, fig_h))
    bars = plt.barh(top["feature"].astype(str), top[val_col], color="#2196F3")
    plt.bar_label(bars, fmt="%.3f", padding=4, fontsize=8)
    plt.xlabel("Spearman vs sinistralidade_final (associação monotónica)")
    plt.xlim(-1.05, 1.05)
    plt.title(f"Associação com sinistralidade — top {len(top)} por |Spearman|")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    return out_path


def _make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


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
    """Features elegíveis: exclui target, identifier, leakage e derivadas."""
    _excl_dtype = {"target", "identifier", "leakage"}
    _dtype = fc["dtype"].fillna("").astype(str).str.lower()
    _cat   = fc["category"].fillna("").astype(str).str.lower()
    mask = (~_dtype.isin(_excl_dtype)) & (_cat != "derivada")
    return fc.loc[mask, "feature_name"].astype(str).tolist()


def normalize_plano(s: pd.Series) -> pd.Series:
    out = s.astype(str)
    for k, v in PLANO_MAP.items():
        out = out.replace(k, v)
    return out


def aggregate_panel_by_beneficiary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Colapsa o painel mensal numa linha por ``cod_beneficiario``.
    Remove ``competencia``. Regras (período observado = soma ou média das mensais):

    - Contagens ``qtd_*`` e ``valor_faturamento``: soma.
    - ``sinistralidade_final``, ``pct_urgencia``: média mensal.
    - Demografia / contrato (idade, sexo, tipo_cadastro, plano): primeiro valor
      após ordenação por tempo (se existir ``competencia``).
    - Demais numéricos de valor/sinistro (incl. colunas de leakage para diagnóstico):
      soma no período.
    - Outros numéricos: média.

    Acrescenta ``n_meses_obs`` (tamanho do grupo). Essa coluna não entra em X.
    """
    if BENEFICIARIO_COL not in df.columns:
        raise ValueError(f"Coluna obrigatória em falta: {BENEFICIARIO_COL}")

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
        elif c.startswith("qtd_"):
            agg[c] = "sum"
        elif c == "valor_faturamento":
            agg[c] = "sum"
        elif c in (SEGMENT_COL, "idade", "sexo", "tipo_cadastro", "plano"):
            agg[c] = "first"
        elif "sinistralidade" in c.lower() and c != TARGET_COL:
            agg[c] = "mean"
        elif pd.api.types.is_numeric_dtype(d[c]):
            cl = c.lower()
            if any(x in cl for x in ("valor", "sinistro", "sin_ref", "fator", "ajuste")) or c in (
                LEAKAGE_COLS | QUASI_LEAKAGE_COLS
            ) - {TARGET_COL}:
                agg[c] = "sum"
            else:
                agg[c] = "mean"
        else:
            agg[c] = "first"

    drop_time = [TIME_COL] if TIME_COL in d.columns else []
    d_work = d.drop(columns=drop_time, errors="ignore")

    g = d_work.groupby(BENEFICIARIO_COL, as_index=False, dropna=False)
    out = g.agg(agg)

    sizes = d.groupby(BENEFICIARIO_COL, dropna=False).size().reset_index(name=N_MESES_COL)
    out = out.merge(sizes, on=BENEFICIARIO_COL, how="left")
    if "idade" in out.columns:
        _id = pd.to_numeric(out["idade"], errors="coerce")
        out["idade"] = np.ceil(_id).fillna(-1).astype(np.int64)
    return out


# Subset “prospectivo” só com colunas do painel analítico (sem engenharia extra).
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


def cols_numeric_for_corr(df: pd.DataFrame, eligible: list[str]) -> list[str]:
    cols = [c for c in eligible if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if TARGET_COL in df.columns and pd.api.types.is_numeric_dtype(df[TARGET_COL]):
        cols = list(dict.fromkeys(cols + [TARGET_COL]))
    return cols


def extract_base_feature_ohe(
    f: str,
    bases_sorted: list[str],
    feature_to_category: dict[str, str],
) -> str:
    for base in bases_sorted:
        if f == base or f.startswith(base + "_"):
            return base
    if f in feature_to_category:
        return f
    return f  # fallback: retorna o próprio nome em vez de levantar erro


# =============================================================================
# %% — Análise de correlação
# =============================================================================

def correlation_analysis_segment(
    df_seg: pd.DataFrame,
    corr_numeric_cols: list[str],
    out_dir: Path,
    segment_slug: str,
) -> dict[str, Any]:
    """
    Exporta:
    - corr_pearson_{seg}.csv          (matriz wide — backward compat)
    - corr_spearman_{seg}.csv         (matriz wide — backward compat)
    - corr_long_{seg}.csv             (formato tidy: feature_a, feature_b, pearson, spearman, segment)
    - corr_rank_spearman_vs_target_{seg}.csv
    - high_corr_pairs_{seg}.csv       (pares com |Spearman| > CORR_HIGH_THRESHOLD)
    - heatmaps por família e top-K
    """
    meta: dict[str, Any] = {"segment": segment_slug, "artifacts": [], "metrics": {}}
    if len(corr_numeric_cols) < 2:
        return meta

    dnum = df_seg[corr_numeric_cols].copy()
    pear  = dnum.corr(method="pearson")
    spear = dnum.corr(method="spearman")

    # — Matrizes wide (backward compat)
    pear_path  = out_dir / f"corr_pearson_{segment_slug}.csv"
    spear_path = out_dir / f"corr_spearman_{segment_slug}.csv"
    pear.to_csv(pear_path)
    spear.to_csv(spear_path)
    meta["artifacts"].extend([pear_path, spear_path])

    # — Formato long/tidy para plataforma de dados
    pear_long  = pear.stack().reset_index()
    spear_long = spear.stack().reset_index()
    pear_long.columns  = ["feature_a", "feature_b", "pearson"]   # type: ignore[assignment]
    spear_long.columns = ["feature_a", "feature_b", "spearman"]  # type: ignore[assignment]
    corr_long = pear_long.merge(spear_long, on=["feature_a", "feature_b"])
    corr_long["segment"] = segment_slug
    # Mantém apenas triângulo superior (sem diagonal) para evitar duplicatas
    corr_long = corr_long[corr_long["feature_a"] < corr_long["feature_b"]].reset_index(drop=True)
    corr_long_path = out_dir / f"corr_long_{segment_slug}.csv"
    corr_long.to_csv(corr_long_path, index=False)
    meta["artifacts"].append(corr_long_path)

    # — Pares de alta correlação (alerta de redundância)
    high_pairs = corr_long[corr_long["spearman"].abs() > CORR_HIGH_THRESHOLD].copy()
    high_pairs = high_pairs.sort_values("spearman", ascending=False, key=abs)
    high_pairs_path = out_dir / f"high_corr_pairs_{segment_slug}.csv"
    high_pairs.to_csv(high_pairs_path, index=False)
    meta["artifacts"].append(high_pairs_path)
    meta["metrics"]["n_high_corr_pairs"] = int(len(high_pairs))

    # — Ranking Spearman vs alvo
    target_cols = [c for c in corr_numeric_cols if c != TARGET_COL]
    if TARGET_COL in dnum.columns and target_cols:
        s_spear   = dnum[target_cols].corrwith(dnum[TARGET_COL], method="spearman")
        s_pearson = dnum[target_cols].corrwith(dnum[TARGET_COL], method="pearson")
        rank_abs  = s_spear.abs().sort_values(ascending=False).head(CORR_TOP_N_RANK)
        ranking   = pd.DataFrame({
            "feature":       rank_abs.index,
            "abs_spearman":  rank_abs.values,
            "spearman":      s_spear.loc[rank_abs.index].values,
            "pearson":       s_pearson.loc[rank_abs.index].values,
            "segment":       segment_slug,
        })
        rank_path = out_dir / f"corr_rank_spearman_vs_target_{segment_slug}.csv"
        ranking.to_csv(rank_path, index=False)
        meta["artifacts"].append(rank_path)
        meta["metrics"]["max_abs_spearman_vs_target"] = float(ranking["abs_spearman"].iloc[0])
        meta["top_features_by_spearman"] = ranking["feature"].head(PDP_TOP_N).tolist()

    # — Heatmaps por família
    families = {
        "qtd_servico": [c for c in corr_numeric_cols if c.startswith("qtd_servico_")],
        "qtd_esp":     [c for c in corr_numeric_cols if c.startswith("qtd_esp_")],
        "qtd_conta":   [c for c in corr_numeric_cols if c.startswith("qtd_conta_")],
        "dem_fin":     [c for c in corr_numeric_cols if c in ("idade", "valor_faturamento", "pct_urgencia", TARGET_COL)],
    }
    for fam, cols in families.items():
        cols_f = [c for c in cols if c in spear.columns]
        if len(cols_f) < 2:
            continue
        sub   = spear.loc[cols_f, cols_f]
        hpath = out_dir / f"heatmap_spearman_{fam}_{segment_slug}.png"
        _save_corr_heatmap(sub, f"Spearman — {fam} — {segment_slug}", hpath)
        meta["artifacts"].append(hpath)

    # — Heatmap top-K com alvo
    if TARGET_COL in spear.columns:
        others   = [c for c in spear.columns if c != TARGET_COL]
        top_vars = (
            spear[TARGET_COL].reindex(others).abs()
            .sort_values(ascending=False).head(TOP_K_CORR_HEATMAP).index.tolist()
        )
        idx     = [c for c in [TARGET_COL] + top_vars if c in spear.index]
        sub_top = spear.loc[idx, idx]
        htop    = out_dir / f"heatmap_spearman_top{TOP_K_CORR_HEATMAP}_with_target_{segment_slug}.png"
        _save_corr_heatmap(sub_top, f"Spearman — top {TOP_K_CORR_HEATMAP} vs alvo — {segment_slug}", htop)
        meta["artifacts"].append(htop)

    return meta


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
# %% — Análise de quasi-leakage (diagnóstico separado)
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
            rows.append({"feature": c, "spearman_vs_target": sp, "pearson_vs_target": pe,
                         "segment": segment_slug, "motivo_exclusao": "quasi_leakage"})

    if not rows:
        return {}

    df_ql = pd.DataFrame(rows)
    path  = out_dir / f"quasi_leakage_corr_{segment_slug}.csv"
    df_ql.to_csv(path, index=False)
    print(f"{segment_slug} | quasi-leakage correlações salvas -> {path.name}")
    return {"path": path, "data": df_ql}


# =============================================================================
# %% — Impacto por segmento (retrospectivo + prospectivo)
# =============================================================================

def _run_kfold_r2(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    kf: KFold,
    label: str = "",
) -> tuple[float, float, float, float]:
    """
    Retorna (mean, std, ci_lower, ci_upper) do R² via KFold.
    Se `KFOLD_MAX_ROWS` for int, subsampla até esse teto; se `None`, usa todas as linhas.
    """
    rng_sub = np.random.RandomState(RANDOM_STATE)
    if KFOLD_MAX_ROWS is not None and len(X) > KFOLD_MAX_ROWS:
        idx = rng_sub.choice(len(X), size=KFOLD_MAX_ROWS, replace=False)
        X   = X.iloc[idx].reset_index(drop=True)
        y   = y.iloc[idx].reset_index(drop=True)

    scores = []
    for fold_i, (tr, te) in enumerate(kf.split(X), 1):
        if label:
            print(f"    [{label}] fold {fold_i}/{kf.n_splits} ...", flush=True)
        pipeline.fit(X.iloc[tr], y.iloc[tr])
        scores.append(r2_score(y.iloc[te], pipeline.predict(X.iloc[te])))
    arr  = np.array(scores)
    mean = float(arr.mean())
    std  = float(arr.std(ddof=1))
    ci   = 1.96 * std / np.sqrt(len(arr))
    return mean, std, float(mean - ci), float(mean + ci)


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
    - r2_isolado_por_categoria_{seg}.csv        (com IC 95%)
    - r2_isolado_prospectivo_{seg}.csv          (se prospective_features fornecido)
    - perm_global_{seg}.csv
    - perm_categoria_{seg}.csv
    - compare_isolado_vs_global_{seg}.csv
    - impacto_global_{seg}.png
    - shap_values_{seg}.csv                     (se shap disponível)
    - shap_summary_{seg}.csv
    - pdp_{feature}_{seg}.csv                   (top-N features)
    """
    y      = df_prod[TARGET_COL]
    X_base = df_prod[valid_features].copy()
    for c in X_base.columns:
        if X_base[c].dtype == object:
            X_base[c] = X_base[c].astype("category")

    result: dict[str, Any] = {
        "segment_slug":   segment_slug,
        "paths":          {},
        "r2_global":      None,
        "r2_global_std":  None,
        "r2_global_ci":   None,
        "pipeline_global": None,
    }

    # -------------------------------------------------------------------------
    # 1. R² isolado por categoria — modo retrospectivo
    # -------------------------------------------------------------------------
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
            "category":      category,
            "r2_mean":       mean,
            "r2_std":        std,
            "r2_ci_lower":   ci_lo,
            "r2_ci_upper":   ci_hi,
            "n_features":    len(feats_cat),
            "segment":       segment_slug,
        })
        print(f"  {segment_slug} | {category:15s} | R2 isolado = {mean:.3f} ± {std:.3f}")

    df_r2_isolado = pd.DataFrame(r2_isolado_rows)
    iso_path = out_dir / f"r2_isolado_por_categoria_{segment_slug}.csv"
    df_r2_isolado.to_csv(iso_path, index=False)
    result["paths"]["isolated"] = iso_path

    # -------------------------------------------------------------------------
    # 2. R² isolado — modo prospectivo (features disponíveis antes do fechamento)
    # -------------------------------------------------------------------------
    if prospective_features:
        prosp_in_X = [f for f in prospective_features if f in X_base.columns]
        if len(prosp_in_X) >= 2:
            X_prosp  = X_base[prosp_in_X]
            num_p, cat_p = split_num_cat(X_prosp)
            pipe_p   = make_rf_pipeline(num_p, cat_p, n_estimators=300)
            mean_p, std_p, ci_lo_p, ci_hi_p = _run_kfold_r2(pipe_p, X_prosp, y, kf, label="prospectivo")
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

    # -------------------------------------------------------------------------
    # 3. Modelo global — todas as features
    # -------------------------------------------------------------------------
    num_cols, cat_cols = split_num_cat(X_base)
    pipeline_global    = make_rf_pipeline(num_cols, cat_cols, n_estimators=300)
    print(f"  {segment_slug} | GLOBAL KFold iniciando ...", flush=True)
    r2_mean_g, r2_std_g, ci_lo_g, ci_hi_g = _run_kfold_r2(pipeline_global, X_base, y, kf, label="global")

    # Re-treina no dataset completo para SHAP e PDP
    print(f"  {segment_slug} | GLOBAL treino final (dataset completo) ...", flush=True)
    pipeline_global.fit(X_base, y)
    X_proc = pipeline_global.named_steps["prep"].transform(X_base)

    print(f"  {segment_slug} | GLOBAL | R2 = {r2_mean_g:.3f} ± {r2_std_g:.3f}")
    result["r2_global"]     = r2_mean_g
    result["r2_global_std"] = r2_std_g
    result["r2_global_ci"]  = (ci_lo_g, ci_hi_g)
    result["pipeline_global"] = pipeline_global

    # Nomes das features após OHE
    ohe       = pipeline_global.named_steps["prep"].named_transformers_["cat"].named_steps["ohe"]
    cat_names = ohe.get_feature_names_out(cat_cols).tolist() if cat_cols else []
    feature_names_proc = num_cols + list(cat_names)

    # -------------------------------------------------------------------------
    # 4. Permutation importance
    # -------------------------------------------------------------------------
    rng = np.random.RandomState(RANDOM_STATE)
    if PERM_IMPORTANCE_MAX_ROWS is None:
        n_perm = len(X_base)
    else:
        n_perm = min(PERM_IMPORTANCE_MAX_ROWS, len(X_base))
    perm_idx = rng.choice(len(X_base), size=n_perm, replace=False)
    X_perm   = X_proc[perm_idx]
    y_perm   = y.iloc[perm_idx]

    print(f"  {segment_slug} | Permutation importance ({n_perm} linhas, {PERM_N_REPEATS} repeats) ...", flush=True)
    perm = permutation_importance(
        pipeline_global.named_steps["model"],
        X_perm, y_perm,
        n_repeats=PERM_N_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=-1,        # paraleliza os repeats nos núcleos disponíveis
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

    # Agrega por categoria
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
        .merge(df_r2_isolado[["category", "r2_mean", "r2_std", "r2_ci_lower", "r2_ci_upper"]],
               on="category", how="left")
    )
    df_compare.columns = [
        c.replace("r2_mean", "r2_isolado") for c in df_compare.columns
    ]
    compare_path = out_dir / f"compare_isolado_vs_global_{segment_slug}.csv"
    df_compare.to_csv(compare_path, index=False)
    result["paths"]["compare"] = compare_path

    # Plot impacto global
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

    # -------------------------------------------------------------------------
    # 5. SHAP values
    # -------------------------------------------------------------------------
    if HAS_SHAP:
        try:
            if SHAP_MAX_ROWS is None:
                n_shap = len(X_base)
            else:
                n_shap = min(SHAP_MAX_ROWS, len(X_base))
            shap_idx   = rng.choice(len(X_base), size=n_shap, replace=False)
            X_shap_arr = X_proc[shap_idx]
            y_shap     = y.iloc[shap_idx]

            print(f"  {segment_slug} | SHAP ({n_shap} linhas) ...", flush=True)
            explainer   = shap.TreeExplainer(pipeline_global.named_steps["model"])
            shap_values = explainer.shap_values(X_shap_arr)

            # CSV de valores individuais (amostra)
            df_shap_vals = pd.DataFrame(shap_values, columns=feature_names_proc)
            df_shap_vals["segment"]          = segment_slug
            df_shap_vals["sinistralidade_real"] = y_shap.values
            shap_vals_path = out_dir / f"shap_values_{segment_slug}.csv"
            df_shap_vals.to_csv(shap_vals_path, index=False)
            result["paths"]["shap_values"] = shap_vals_path

            # Sumário: mean |SHAP| por feature e por categoria
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            df_shap_sum   = pd.DataFrame({
                "feature":    feature_names_proc,
                "mean_abs_shap": mean_abs_shap,
                "segment":    segment_slug,
            }).sort_values("mean_abs_shap", ascending=False)

            # Mapeamento para categoria
            df_shap_sum["base_feature"] = df_shap_sum["feature"].apply(_base)
            df_shap_sum["category"] = (
                df_shap_sum["base_feature"].map(lambda b: feature_to_category.get(b, "outros")).fillna("outros")
            )
            shap_sum_path = out_dir / f"shap_summary_{segment_slug}.csv"
            df_shap_sum.to_csv(shap_sum_path, index=False)
            result["paths"]["shap_summary"] = shap_sum_path

            # Sumário por categoria
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

            print(f"  {segment_slug} | SHAP calculado ({n_shap} linhas)")
        except Exception as e:
            print(f"  [WARN] SHAP falhou para {segment_slug}: {e}")

    # -------------------------------------------------------------------------
    # 6. Partial Dependence Plots (top-N features numéricas)
    # -------------------------------------------------------------------------
    top_num_features = [
        f for f in df_perm["feature"].head(PDP_TOP_N * 3).tolist()
        if f in num_cols
    ][:PDP_TOP_N]

    print(f"  {segment_slug} | PDP para {len(top_num_features)} features ...", flush=True)
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
                "feature_value":           pd_result["grid_values"][0],
                "predicted_sinistralidade": pd_result["average"][0],
                "feature":                 feat,
                "segment":                 segment_slug,
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
# %% — Carga e preparação
# =============================================================================

feature_catalog = load_feature_catalog()
df_raw = pd.read_parquet(TRANSFORMED_PARQUET_PATH)
print("Shape (painel):", df_raw.shape, "Fonte:", TRANSFORMED_PARQUET_PATH)

n_rows_panel = len(df_raw)
if AGGREGATE_BY_BENEFICIARIO:
    df_raw = aggregate_panel_by_beneficiary(df_raw)
    print(
        f"Agregado por {BENEFICIARIO_COL}: {n_rows_panel} linhas -> {len(df_raw)} beneficiarios"
    )

eligible = catalog_eligible_names(feature_catalog)
exclude_x = (
    LEAKAGE_COLS
    | QUASI_LEAKAGE_COLS
    | {TARGET_COL, BENEFICIARIO_COL, TIME_COL, SEGMENT_COL, N_MESES_COL}
)
valid_features = [
    f for f in eligible
    if f in df_raw.columns and f not in exclude_x
]

df_raw[SEGMENT_COL] = normalize_plano(df_raw[SEGMENT_COL])
segment_values      = sorted(df_raw[SEGMENT_COL].dropna().unique())
print("Segmentos:", segment_values)
print("Features preditivas:", len(valid_features), valid_features[:5], "...")
print("Quasi-leakage excluídas do treino:", sorted(QUASI_LEAKAGE_COLS))
print(
    "Amostragem: KFold=%s | permutation=%s | SHAP=%s"
    % (
        "todas as linhas" if KFOLD_MAX_ROWS is None else f"max {KFOLD_MAX_ROWS}",
        "todas as linhas" if PERM_IMPORTANCE_MAX_ROWS is None else f"max {PERM_IMPORTANCE_MAX_ROWS}",
        "todas as linhas" if SHAP_MAX_ROWS is None else f"max {SHAP_MAX_ROWS}",
    )
)
print(
    "MLflow ao final:",
    "SIM" if ENABLE_MLFLOW else "NAO (defina FEATURE_IMPACT_MLFLOW=1 ou FORCE_MLFLOW)",
)

eligible_for_corr = [
    f for f in catalog_eligible_names(feature_catalog)
    if f in df_raw.columns
    and f not in LEAKAGE_COLS
    and f not in (BENEFICIARIO_COL, TIME_COL)
    # quasi-leakage entra na correlação para diagnóstico, mas marcado separadamente
]

feature_to_category_full = feature_catalog.set_index("feature_name")["category"].astype(str).to_dict()
bases_sorted = sorted(feature_to_category_full.keys(), key=len, reverse=True)

valid_categories = sorted({
    str(c)
    for c in feature_catalog.loc[
        feature_catalog["feature_name"].isin(valid_features), "category"
    ].unique()
})

# Features prospectivas disponíveis neste dataset
prospective_in_data = [
    f for f in PROSPECTIVE_FEATURES
    if f in df_raw.columns and f in valid_features
]

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
ver, OUTPUT_DIR = next_version_dir(OUTPUT_FEATURE_IMPACT_ROOT)
print(f"Versão de saída: {ver} -> {OUTPUT_DIR}")


# =============================================================================
# %% — Loop principal por segmento
# =============================================================================

all_corr_meta:  list[dict[str, Any]] = []
all_impact:     list[dict[str, Any]] = []
all_quasi_leak: list[dict[str, Any]] = []

for seg in segment_values:
    segment_slug = plano_slug(seg)
    df_seg = df_raw[df_raw[SEGMENT_COL] == seg].reset_index(drop=True)

    if len(df_seg) < N_SPLITS * 2:
        print(f"[SKIP] {seg}: poucas linhas ({len(df_seg)})")
        continue

    print(f"\n{'='*60}")
    print(f"Segmento: {seg}  ({len(df_seg)} linhas)")
    print(f"{'='*60}")

    # Correlação
    corr_cols = cols_numeric_for_corr(
        df_seg, [c for c in eligible_for_corr if c in df_seg.columns]
    )
    cmeta = correlation_analysis_segment(df_seg, corr_cols, OUTPUT_DIR, segment_slug)
    all_corr_meta.append(cmeta)

    # Diagnóstico quasi-leakage
    ql = quasi_leakage_analysis(df_seg, OUTPUT_DIR, segment_slug)
    if ql:
        all_quasi_leak.append(ql)

    # Impacto (retro + prospectivo)
    imp = impact_segment(
        df_seg,
        feature_catalog,
        valid_features,
        valid_categories,
        feature_to_category_full,
        bases_sorted,
        OUTPUT_DIR,
        segment_slug,
        kf,
        prospective_features=prospective_in_data,
    )
    all_impact.append(imp)


# =============================================================================
# %% — Outputs consolidados (multi-segmento, prontos para plataforma)
# =============================================================================

# summary R² global com IC
summary_rows = []
for imp in all_impact:
    ci = imp.get("r2_global_ci") or (None, None)
    summary_rows.append({
        "segment":       imp["segment_slug"],
        "r2_global":     imp["r2_global"],
        "r2_global_std": imp["r2_global_std"],
        "r2_ci_lower":   ci[0],
        "r2_ci_upper":   ci[1],
    })
summary_path = OUTPUT_DIR / "summary_r2_global.csv"
pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

# Correlação long empilhada (todos os segmentos num único CSV)
corr_long_all_path = OUTPUT_DIR / "corr_long_all_segments.csv"
_corr_long_parts = []
for cm in all_corr_meta:
    seg = cm.get("segment", "")
    p   = OUTPUT_DIR / f"corr_long_{seg}.csv"
    if p.is_file():
        _corr_long_parts.append(pd.read_csv(p))
if _corr_long_parts:
    pd.concat(_corr_long_parts, ignore_index=True).to_csv(corr_long_all_path, index=False)

# R² isolado empilhado
r2_iso_all_path = OUTPUT_DIR / "r2_isolado_all_segments.csv"
_iso_parts = []
for imp in all_impact:
    p = imp["paths"].get("isolated")
    if p and Path(p).is_file():
        _iso_parts.append(pd.read_csv(p))
if _iso_parts:
    pd.concat(_iso_parts, ignore_index=True).to_csv(r2_iso_all_path, index=False)

# Compare isolado × global empilhado
compare_all_path = OUTPUT_DIR / "compare_all_segments.csv"
_cmp_parts = []
for imp in all_impact:
    p = imp["paths"].get("compare")
    if p and Path(p).is_file():
        _cmp_parts.append(pd.read_csv(p))
if _cmp_parts:
    pd.concat(_cmp_parts, ignore_index=True).to_csv(compare_all_path, index=False)

# High-corr pairs empilhado
hcp_all_path = OUTPUT_DIR / "high_corr_pairs_all_segments.csv"
_hcp_parts = []
for cm in all_corr_meta:
    seg = cm.get("segment", "")
    p   = OUTPUT_DIR / f"high_corr_pairs_{seg}.csv"
    if p.is_file():
        _hcp_parts.append(pd.read_csv(p))
if _hcp_parts:
    pd.concat(_hcp_parts, ignore_index=True).to_csv(hcp_all_path, index=False)

# SHAP summary empilhado
shap_all_path = OUTPUT_DIR / "shap_summary_all_segments.csv"
_shap_parts = []
for imp in all_impact:
    p = imp["paths"].get("shap_summary")
    if p and Path(p).is_file():
        _shap_parts.append(pd.read_csv(p))
if _shap_parts:
    pd.concat(_shap_parts, ignore_index=True).to_csv(shap_all_path, index=False)

# Metadata da run
meta_path = OUTPUT_DIR / "run_metadata.json"
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump({
        "version": ver,
        "parquet": str(TRANSFORMED_PARQUET_PATH),
        "aggregate_by_beneficiary": AGGREGATE_BY_BENEFICIARIO,
        "n_rows_panel": n_rows_panel,
        "n_rows_after_agg": len(df_raw) if AGGREGATE_BY_BENEFICIARIO else n_rows_panel,
        "n_segments": len(all_impact),
        "n_valid_features": len(valid_features),
        "quasi_leakage_cols": sorted(QUASI_LEAKAGE_COLS),
        "corr_high_threshold": CORR_HIGH_THRESHOLD,
        "perm_importance_max_rows": PERM_IMPORTANCE_MAX_ROWS,
        "perm_n_repeats": PERM_N_REPEATS,
        "kfold_max_rows": KFOLD_MAX_ROWS,
        "shap_max_rows": SHAP_MAX_ROWS,
        "pdp_top_n": PDP_TOP_N,
        "has_shap": HAS_SHAP,
    }, f, indent=2)

print(f"\nOutputs consolidados salvos em {OUTPUT_DIR}")

mlflow_artifact_csv_path = write_feature_correlation_sinistralidade_csv(
    df_raw, valid_features, OUTPUT_DIR
)
if mlflow_artifact_csv_path is not None:
    print(
        f"CSV Spearman vs {TARGET_COL} para MLflow/BI: {mlflow_artifact_csv_path.name}"
    )
plot_top_path: Path | None = None
if mlflow_artifact_csv_path is not None:
    plot_top_path = plot_feature_impact_top_n(
        mlflow_artifact_csv_path,
        OUTPUT_DIR / FEATURE_IMPACT_PLOT_FILENAME,
        n=FEATURE_IMPACT_TOP_N_PLOT,
    )
if plot_top_path is not None:
    print(f"Gráfico local top-{FEATURE_IMPACT_TOP_N_PLOT}: {plot_top_path.name}")


# =============================================================================
# %% — MLflow (opcional: artefato = feature_correlation_sinistralidade.csv; params/métricas completos)
# =============================================================================

def _fmt_opt_int(v: int | None) -> str:
    return "full" if v is None else str(v)


if ENABLE_MLFLOW:
    if mlflow_artifact_csv_path is None or not mlflow_artifact_csv_path.is_file():
        print(
            "\n[MLflow] CSV Spearman vs sinistralidade não gerado — registo ignorado."
        )
    else:
        _ = configurar_mlflow(EXPERIMENT_NAME)
        run_name = f"elgin__{ver}"

        params_log: dict[str, str] = {
            "data_path": str(TRANSFORMED_PARQUET_PATH),
            "output_dir": str(OUTPUT_DIR),
            "version": ver,
            "aggregate_by_beneficiary": str(AGGREGATE_BY_BENEFICIARIO),
            "n_splits": str(N_SPLITS),
            "random_state": str(RANDOM_STATE),
            "perm_importance_max_rows": _fmt_opt_int(PERM_IMPORTANCE_MAX_ROWS),
            "perm_n_repeats": str(PERM_N_REPEATS),
            "kfold_max_rows": _fmt_opt_int(KFOLD_MAX_ROWS),
            "shap_max_rows": _fmt_opt_int(SHAP_MAX_ROWS),
            "corr_high_threshold": str(CORR_HIGH_THRESHOLD),
            "quasi_leakage_cols": "|".join(sorted(QUASI_LEAKAGE_COLS)),
            "mlflow_artifact_csv": MLFLOW_ARTIFACT_CSV_FILENAME,
        }

        metrics_log: dict[str, float] = {}
        for imp in all_impact:
            slug = imp["segment_slug"]
            if imp.get("r2_global") is not None:
                metrics_log[f"r2_global_{slug}"] = float(imp["r2_global"])
            if imp.get("r2_global_std") is not None:
                metrics_log[f"r2_global_std_{slug}"] = float(imp["r2_global_std"])

        for cm in all_corr_meta:
            seg = cm.get("segment", "")
            m = cm.get("metrics", {})
            if "max_abs_spearman_vs_target" in m:
                metrics_log[f"max_abs_spearman_{seg}"] = float(m["max_abs_spearman_vs_target"])
            if "n_high_corr_pairs" in m:
                metrics_log[f"n_high_corr_pairs_{seg}"] = float(m["n_high_corr_pairs"])

        with mlflow.start_run(run_name=run_name):
            for k, v in params_log.items():
                mlflow.log_param(k, v)
            for k, v in metrics_log.items():
                mlflow.log_metric(k, v)
            mlflow.log_artifact(
                str(mlflow_artifact_csv_path),
                artifact_path="feature_impact",
            )

        print(
            f"\n[OK] MLflow run '{run_name}' | artefato: {MLFLOW_ARTIFACT_CSV_FILENAME} "
            f"| params/metrics registados (PNG não enviado)"
        )
else:
    print(
        "\n[MLflow] Ignorado. Para registar (CSV + params/métricas): "
        "FEATURE_IMPACT_MLFLOW=1 ou FORCE_MLFLOW=True."
    )
# %%
