#%%
"""
EDA + transformação — painel ELGIN (base analítica)

- Lê a base bruta em ``data/raw/elgin/base_analitica/painel_sinistralidade_v1.parquet``.
- Executa EDA (metadados, describe, resumo por coluna, distribuições opcionais em PNG).
- Padroniza a coluna ``plano`` para um único rótulo (MASTER EMPRESARIAL).
- Grava a base transformada em ``data/processed/elgin/base_analitica_transformada/``.

O pipeline ``pipelines/elgin/predict.py`` consome o Parquet gerado aqui.

Execução: célula a célula (VS Code / Cursor “Run Cell”) ou script completo.
"""

#%%
from __future__ import annotations

import gc
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ``eda_predict.py`` está em pipelines/elgin/eda/ → raiz = parents[3]
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

COMPANY = "elgin"

RAW_PANEL_PATH = (
    ROOT_DIR / "data" / "raw" / COMPANY / "base_analitica" / "painel_sinistralidade_v1.parquet"
)

OUTPUT_DIR = ROOT_DIR / "data" / "processed" / COMPANY / "base_analitica_transformada"
OUTPUT_PARQUET_PATH = OUTPUT_DIR / "painel_sinistralidade_v1.parquet"

SEGMENT_COL = "plano"
PLANO_LABEL_PADRAO = "MASTER EMPRESARIAL"

EDA_SAVE_PNG_TO_DISK = False

sns.set_theme(style="whitegrid")

#%%
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EDA_PLOTS_DIR = OUTPUT_DIR / "plots" / "eda"
EDA_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

#%%
# CARGA DO PAINEL (Parquet)
# Requer engine Parquet (ex.: pyarrow).
if not RAW_PANEL_PATH.is_file():
    raise FileNotFoundError(
        f"Ficheiro não encontrado: {RAW_PANEL_PATH}\n"
        "Gere o Parquet a partir do Excel ou copie o ficheiro para esse caminho."
    )

df_raw = pd.read_parquet(RAW_PANEL_PATH)

print("Fonte:", RAW_PANEL_PATH)
print("Shape:", df_raw.shape)
print("Colunas:", len(df_raw.columns))

#%%
# EDA I: COLUNAS, TIPOS E INFO (LEVE)

df_lista_colunas = pd.DataFrame(
    {"indice": range(1, len(df_raw.columns) + 1), "coluna": df_raw.columns}
)
print("\n--- Lista de colunas (índice e nome) ---")
print(df_lista_colunas.to_string(index=False))

print("\n--- dtypes (pandas) ---")
print(df_raw.dtypes)

print("\n--- info() ---")
df_raw.info(verbose=True, show_counts=True)

#%%
# EDA II: ESTATÍSTICAS NOS LOGS (SEM GRÁFICOS)

eda_describe_num = df_raw.describe(include=[np.number]).T
eda_describe_num.index.name = "coluna"
eda_describe_num = eda_describe_num.rename(
    columns={
        "count": "n_validos",
        "mean": "media",
        "std": "desvio_padrao",
        "min": "minimo",
        "25%": "q25",
        "50%": "mediana",
        "75%": "q75",
        "max": "maximo",
    }
)

non_num_cols = [
    c
    for c in df_raw.columns
    if not (pd.api.types.is_numeric_dtype(df_raw[c]) or pd.api.types.is_bool_dtype(df_raw[c]))
]
if non_num_cols:
    eda_describe_obj = df_raw[non_num_cols].astype("string").describe(include="all").T
    eda_describe_obj.index.name = "coluna"
else:
    eda_describe_obj = pd.DataFrame()

resumo_rows = []
for col in df_raw.columns:
    s = df_raw[col]
    row = {
        "coluna": col,
        "dtype": str(s.dtype),
        "n_linhas": len(s),
        "n_nulos": int(s.isna().sum()),
        "pct_nulos": round(100.0 * float(s.isna().mean()), 4),
        "n_unicos": int(s.nunique(dropna=True)),
    }
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        sn = pd.to_numeric(s, errors="coerce") if s.dtype == object else s
        row["media"] = float(sn.mean()) if sn.notna().any() else np.nan
        row["desvio_padrao"] = float(sn.std()) if sn.notna().sum() > 1 else np.nan
        row["minimo"] = float(sn.min()) if sn.notna().any() else np.nan
        row["q25"] = float(sn.quantile(0.25)) if sn.notna().any() else np.nan
        row["mediana"] = float(sn.median()) if sn.notna().any() else np.nan
        row["q75"] = float(sn.quantile(0.75)) if sn.notna().any() else np.nan
        row["maximo"] = float(sn.max()) if sn.notna().any() else np.nan
        row["moda"] = np.nan
        row["freq_moda"] = np.nan
    else:
        for k in (
            "media",
            "desvio_padrao",
            "minimo",
            "q25",
            "mediana",
            "q75",
            "maximo",
        ):
            row[k] = np.nan
        mode_s = s.mode(dropna=True)
        if len(mode_s):
            top = mode_s.iloc[0]
            row["moda"] = top
            row["freq_moda"] = int((s == top).sum())
        else:
            row["moda"] = np.nan
            row["freq_moda"] = np.nan

    resumo_rows.append(row)

eda_resumo_por_coluna = pd.DataFrame(resumo_rows)

print("\n--- describe (numéricas, transposto) ---")
if not eda_describe_num.empty:
    print(eda_describe_num.to_string())
else:
    print("(sem colunas numéricas)")

if not eda_describe_obj.empty:
    print("\n--- describe (não numéricas) ---")
    print(eda_describe_obj.to_string())

print("\n--- Resumo completo por coluna ---")
print(eda_resumo_por_coluna.to_string(index=False))

eda_describe_num.to_csv(OUTPUT_DIR / "eda_describe_numericas.csv", encoding="utf-8-sig")
if not eda_describe_obj.empty:
    eda_describe_obj.to_csv(OUTPUT_DIR / "eda_describe_nao_numericas.csv", encoding="utf-8-sig")
eda_resumo_por_coluna.to_csv(OUTPUT_DIR / "eda_resumo_por_coluna.csv", index=False, encoding="utf-8-sig")
print(f"\nTabelas EDA II gravadas em: {OUTPUT_DIR}")

#%%
# EDA III: DISTRIBUIÇÕES (SAÍDA INTERATIVA + LOG; PNG opcional em disco)

plt.close("all")
gc.collect()


def _slug_col(name: str) -> str:
    s = pd.Series([str(name)]).str.replace(r"[^\w]+", "_", regex=True).str.strip("_").iloc[0]
    return s[:120] if s else "col"


_cols = list(df_raw.columns)
_n = len(_cols)

for _idx, col in enumerate(_cols, start=1):
    print(f"\n[EDA III] Figura {_idx}/{_n} — {col}")

    s = df_raw[col]
    slug = _slug_col(col)
    out_path = EDA_PLOTS_DIR / f"dist_{slug}.png"

    fig, ax = plt.subplots(figsize=(8, 4))
    data_plot = s.dropna()
    if len(data_plot) == 0:
        ax.text(0.5, 0.5, "Coluna vazia", ha="center", va="center")
        ax.set_title(str(col)[:80])
        plt.tight_layout()
        if EDA_SAVE_PNG_TO_DISK:
            fig.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.show()
        plt.close(fig)
        del fig, ax
        gc.collect()
        continue

    is_num = pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s)
    if is_num:
        v = pd.to_numeric(data_plot, errors="coerce").dropna()
        if len(v) == 0:
            ax.text(0.5, 0.5, "Sem valores numéricos", ha="center", va="center")
        else:
            sns.histplot(v, kde=True, ax=ax, stat="density", edgecolor="white", bins=50)
        ax.set_xlabel(str(col)[:100])
        ax.set_ylabel("densidade")
        ax.set_title(f"{str(col)[:90]}\nhistograma + KDE (n={len(v):,})")
    else:
        vc = data_plot.astype(str).value_counts()
        top_n = min(40, len(vc))
        vc_top = vc.head(top_n)
        sns.barplot(x=vc_top.values, y=vc_top.index, ax=ax, orient="h")
        ax.set_xlabel("frequência")
        ax.set_ylabel("categoria")
        ax.set_title(f"{str(col)[:70]}\nTop {top_n} categorias (barras)")

    plt.tight_layout()
    if EDA_SAVE_PNG_TO_DISK:
        fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    del fig, ax
    gc.collect()

if EDA_SAVE_PNG_TO_DISK:
    print(f"\n[EDA III] PNG gravados em: {EDA_PLOTS_DIR}")
else:
    print(
        "\n[EDA III] Gráficos mostrados na saída interativa (plt.show); "
        "disco desligado. Para gravar PNG, defina EDA_SAVE_PNG_TO_DISK = True."
    )

#%%
# PADRONIZAÇÃO DA COLUNA ``plano``

if SEGMENT_COL not in df_raw.columns:
    raise KeyError(f"Coluna '{SEGMENT_COL}' inexistente; não é possível padronizar.")

print("\n--- Valores de `plano` antes da padronização ---")
print(df_raw[SEGMENT_COL].astype("string").fillna("<NA>").value_counts(dropna=False))

df_raw[SEGMENT_COL] = PLANO_LABEL_PADRAO

print(f"\n--- Após padronização: todos os registos com plano = {PLANO_LABEL_PADRAO!r} ---")
print(df_raw[SEGMENT_COL].value_counts())

#%%
# GRAVAR BASE TRANSFORMADA

df_raw.to_parquet(OUTPUT_PARQUET_PATH, index=False)
print(f"\n[OK] Base transformada gravada em:\n  {OUTPUT_PARQUET_PATH}")

# %%
