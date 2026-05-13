#%%
"""
EDA + Transformação — painel CLIMAZON (base analítica)

Lê a base bruta em:
  data/raw/climazon/analytic_base/painel_sinistralidade_climazon_v2.parquet

Fluxo de blocos:
  [AUDITORIA]     Blocos 1-6  → inspecionar sem alterar nada
  [CHECKPOINT]    Bloco 7     → revisar saídas antes de prosseguir
  [MAPEAMENTOS]   Bloco 8     → definir PLANO_MAP, TIPO_CADASTRO_MAP, QTD_CONTA_MERGE
                               (preencher/ajustar com base na auditoria)
  [TRANSFORMAÇÕES] Blocos 9-13 → aplicar limpeza e normalização
  [OUTPUTS]       Blocos 14-16 → feature_catalog, série mensal, parquet transformado
  [VALIDAÇÕES]    Bloco 17    → asserts finais

Execução: célula a célula (VS Code / Cursor "Run Cell") ou script completo.
O pipeline predict_climazon.py consome o Parquet gerado aqui.
"""

#%%
# ── SETUP: imports, constantes, paths ─────────────────────────────────────────
from __future__ import annotations

import gc
import sys
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# eda_climazon.py está em pipelines/climazon/ → raiz do projeto = parents[2]
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

COMPANY = "climazon"

RAW_PANEL_PATH = (
    ROOT_DIR
    / "data" / "raw" / COMPANY / "analytic_base"
    / "painel_sinistralidade_climazon_v2.parquet"
)

PROCESSED_DIR   = ROOT_DIR / "data" / "processed" / COMPANY
AUXILIAR_DIR    = ROOT_DIR / "data" / "auxiliar"   / COMPANY
OUTPUT_DIR      = PROCESSED_DIR / "base_analitica_transformada"
OUTPUT_PARQUET  = OUTPUT_DIR / "painel_sinistralidade_climazon_v1.parquet"

TARGET_COL  = "sinistralidade_final"
SEGMENT_COL = "plano"
PREMIUM_COL = "valor_faturamento"
TIME_COL    = "competencia"
ID_COL      = "cod_beneficiario"

EDA_SAVE_PNG_TO_DISK = False   # True para gravar PNGs em disco

sns.set_theme(style="whitegrid")
print(f"ROOT_DIR : {ROOT_DIR}")
print(f"RAW_PATH : {RAW_PANEL_PATH}")
print(f"EXISTS   : {RAW_PANEL_PATH.exists()}")

#%%
# ── DIRETÓRIOS DE SAÍDA ────────────────────────────────────────────────────────
for d in [OUTPUT_DIR, AUXILIAR_DIR]:
    d.mkdir(parents=True, exist_ok=True)

EDA_PLOTS_DIR = OUTPUT_DIR / "plots" / "eda"
EDA_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
print(f"[OK] Diretórios criados.")

#%%
# ── CARGA DO PARQUET ───────────────────────────────────────────────────────────
if not RAW_PANEL_PATH.is_file():
    raise FileNotFoundError(
        f"Parquet não encontrado: {RAW_PANEL_PATH}\n"
        "Verifique se o arquivo existe neste caminho."
    )

df_raw = pd.read_parquet(RAW_PANEL_PATH)

print(f"Shape   : {df_raw.shape}")
print(f"Colunas : {len(df_raw.columns)}")
print(f"\n--- Dtypes ---")
print(df_raw.dtypes.to_string())

#%%
# ── AUDITORIA I: NULOS POR COLUNA ─────────────────────────────────────────────
# Objetivo: identificar todas as colunas com nulos e sua magnitude.

nulos = (
    df_raw.isna()
    .sum()
    .rename("n_nulos")
    .to_frame()
    .assign(pct_nulos=lambda x: (x["n_nulos"] / len(df_raw) * 100).round(2))
    .sort_values("n_nulos", ascending=False)
)
nulos = nulos[nulos["n_nulos"] > 0]

print(f"=== Colunas com nulos ({len(nulos)} de {len(df_raw.columns)}) ===")
print(nulos.to_string())

nulos.to_csv(OUTPUT_DIR / "auditoria_nulos.csv", encoding="utf-8-sig")
print(f"\n[OK] auditoria_nulos.csv gravado.")

#%%
# ── AUDITORIA II: VALORES DISTINTOS — COLUNAS CATEGÓRICAS ─────────────────────
# Objetivo: ver TODOS os rótulos reais de cada coluna categórica antes de
# qualquer mapeamento. Os valores exibidos aqui devem ser usados para preencher
# PLANO_MAP e TIPO_CADASTRO_MAP no Bloco 8 (MAPEAMENTOS).

COLUNAS_CATEGORICAS = [SEGMENT_COL, "sexo", "tipo_cadastro"]

for col in COLUNAS_CATEGORICAS:
    if col not in df_raw.columns:
        print(f"\n[AVISO] Coluna '{col}' não encontrada no DataFrame.")
        continue

    vc = df_raw[col].astype("string").fillna("<NA>").value_counts(dropna=False)
    print(f"\n{'='*60}")
    print(f"  COLUNA: {col!r}  |  {vc.sum()} registros  |  {len(vc)} valores únicos")
    print(f"{'='*60}")
    print(vc.to_string())

# competencia: mostrar range e contagem de valores únicos
print(f"\n{'='*60}")
print(f"  COLUNA: {TIME_COL!r}")
print(f"{'='*60}")
comp_sorted = sorted(df_raw[TIME_COL].dropna().unique())
print(f"  Total de competências únicas : {len(comp_sorted)}")
print(f"  Primeira : {comp_sorted[0]}")
print(f"  Última   : {comp_sorted[-1]}")
print(f"  Todas    : {comp_sorted}")

#%%
# ── AUDITORIA III: COLUNAS qtd_conta_* (detectar nomes duplicados) ────────────
# Objetivo: listar todos os nomes de colunas qtd_conta_* para identificar
# variações de maiúsculas/espaçamento que representam o mesmo conceito.
# Usar a saída aqui para definir QTD_CONTA_MERGE no Bloco 8.

qtd_conta_cols = sorted([c for c in df_raw.columns if c.lower().startswith("qtd_conta_")])

print(f"=== Colunas qtd_conta_* encontradas ({len(qtd_conta_cols)}) ===")
for i, c in enumerate(qtd_conta_cols, 1):
    n_nulos  = df_raw[c].isna().sum()
    n_nao_zero = (df_raw[c].fillna(0) > 0).sum()
    soma_total = df_raw[c].fillna(0).sum()
    print(f"  {i:02d}. {c!r:55s} | nulos={n_nulos:6d} | n>0={n_nao_zero:6d} | soma={soma_total:.0f}")

# Agrupar por "conceito" (lowercase sem prefixo) para visualizar duplicatas
from collections import defaultdict
grupos: dict[str, list[str]] = defaultdict(list)
for c in qtd_conta_cols:
    conceito = c.replace("qtd_conta_", "").strip().lower().replace(" ", "_")
    grupos[conceito].append(c)

print(f"\n=== Grupos de possíveis duplicatas ===")
for conceito, cols in sorted(grupos.items()):
    status = "⚠ DUPLICATA" if len(cols) > 1 else "  ok"
    print(f"  {status}  '{conceito}' → {cols}")

# Também listar qtd_servico_* para verificar se há duplicatas lá
qtd_servico_cols = sorted([c for c in df_raw.columns if c.lower().startswith("qtd_servico_")])
print(f"\n=== Colunas qtd_servico_* encontradas ({len(qtd_servico_cols)}) ===")
for c in qtd_servico_cols:
    print(f"  {c!r}")

#%%
# ── AUDITORIA IV: DISTRIBUIÇÃO DO TARGET ──────────────────────────────────────
# Objetivo: entender a forma da distribuição da variável-alvo.

print(f"=== Target: {TARGET_COL!r} ===")
s_target = df_raw[TARGET_COL]
print(f"  dtype          : {s_target.dtype}")
print(f"  n_total        : {len(s_target):,}")
print(f"  n_nulos        : {s_target.isna().sum():,}  ({s_target.isna().mean()*100:.2f}%)")
print(f"  n_zeros        : {(s_target == 0).sum():,}  ({(s_target == 0).mean()*100:.2f}%)")
print(f"  n_positivos    : {(s_target > 0).sum():,}  ({(s_target > 0).mean()*100:.2f}%)")

s_pos = s_target[s_target > 0]
print(f"\n  --- Apenas positivos (n={len(s_pos):,}) ---")
print(f"  mean    : {s_pos.mean():.4f}")
print(f"  median  : {s_pos.median():.4f}")
print(f"  std     : {s_pos.std():.4f}")
print(f"  p95     : {s_pos.quantile(0.95):.4f}")
print(f"  p99     : {s_pos.quantile(0.99):.4f}")
print(f"  max     : {s_pos.max():.4f}")

# Faixas de sinistralidade
bins   = [0, 0.5, 1.0, 1.5, 2.0, 5.0, float("inf")]
labels = ["(0-0.5]", "(0.5-1]", "(1-1.5]", "(1.5-2]", "(2-5]", ">5"]
faixas = pd.cut(s_pos, bins=bins, labels=labels)
print(f"\n  --- Distribuição por faixa (apenas positivos) ---")
print(faixas.value_counts().sort_index().to_string())

#%%
# ── AUDITORIA V: ESTATÍSTICAS NUMÉRICAS ───────────────────────────────────────

num_cols = df_raw.select_dtypes(include=[np.number, "Int64", "Float64"]).columns.tolist()
desc = df_raw[num_cols].describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99]).T
desc.index.name = "coluna"
desc.to_csv(OUTPUT_DIR / "auditoria_describe_numericas.csv", encoding="utf-8-sig")

print(f"=== describe (numéricas) — {len(num_cols)} colunas ===")
print(desc.to_string())
print(f"\n[OK] auditoria_describe_numericas.csv gravado.")

#%%
# ── AUDITORIA VI: DISTRIBUIÇÕES VISUAIS ───────────────────────────────────────
# Histogramas (numéricas) e barras (categóricas).
# Mesmo padrão do EDA elgin — EDA_SAVE_PNG_TO_DISK controla gravação em disco.

def _slug_col(name: str) -> str:
    s = pd.Series([str(name)]).str.replace(r"[^\w]+", "_", regex=True).str.strip("_").iloc[0]
    return s[:120] if s else "col"

plt.close("all")
gc.collect()

_cols = list(df_raw.columns)
_n    = len(_cols)

for _idx, col in enumerate(_cols, start=1):
    print(f"\n[EDA] Figura {_idx}/{_n} — {col}")

    s        = df_raw[col]
    slug     = _slug_col(col)
    out_path = EDA_PLOTS_DIR / f"dist_{slug}.png"

    fig, ax  = plt.subplots(figsize=(8, 4))
    data_plot = s.dropna()

    if len(data_plot) == 0:
        ax.text(0.5, 0.5, "Coluna vazia", ha="center", va="center")
        ax.set_title(str(col)[:80])
        plt.tight_layout()
        if EDA_SAVE_PNG_TO_DISK:
            fig.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.show()
        plt.close(fig); del fig, ax; gc.collect()
        continue

    is_num = pd.api.types.is_numeric_dtype(s) or str(s.dtype).startswith(("Int", "Float"))
    if is_num:
        v = pd.to_numeric(data_plot, errors="coerce").dropna()
        if len(v):
            sns.histplot(v, kde=True, ax=ax, stat="density", edgecolor="white", bins=50)
        ax.set_xlabel(str(col)[:100])
        ax.set_ylabel("densidade")
        ax.set_title(f"{str(col)[:90]}\nhistograma + KDE (n={len(v):,})")
    else:
        vc    = data_plot.astype(str).value_counts()
        top_n = min(40, len(vc))
        sns.barplot(x=vc.head(top_n).values, y=vc.head(top_n).index, ax=ax, orient="h")
        ax.set_xlabel("frequência")
        ax.set_ylabel("categoria")
        ax.set_title(f"{str(col)[:70]}\nTop {top_n} categorias")

    plt.tight_layout()
    if EDA_SAVE_PNG_TO_DISK:
        fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.show()
    plt.close(fig); del fig, ax; gc.collect()

print(
    "\n[EDA] Gráficos exibidos."
    if not EDA_SAVE_PNG_TO_DISK
    else f"\n[EDA] PNG gravados em: {EDA_PLOTS_DIR}"
)

#%%
# ════════════════════════════════════════════════════════════════════════════════
# CHECKPOINT — MAPEAMENTOS JÁ PREENCHIDOS COM OS VALORES REAIS DA AUDITORIA
#
# Os blocos a seguir (Mapeamentos → Transformações → Outputs → Validações)
# estão prontos para execução. Os valores reais foram confirmados na auditoria:
#
#  plano         : 10 valores → 2 canônicos (MASTER EMPRESARIAL / MASTER EXECUTIVO)
#  sexo          : M, F, <NA> → M, F, DESCONHECIDO
#  tipo_cadastro : 12 valores (incl. nulos) → TITULAR, DEPENDENTE, DESCONHECIDO
#  qtd_conta_*   : 12 colunas → 7 colunas canônicas (duplicatas somadas)
#  target        : 2,72% nulos → imputados como 0
#
# ════════════════════════════════════════════════════════════════════════════════
print("CHECKPOINT: mapeamentos confirmados — prosseguindo para transformações.")

#%%
# ── MAPEAMENTOS — preenchidos com os valores reais da Auditoria ────────────────

# --------------------------------------------------------------------------
# PLANO_MAP: 10 valores reais → 2 rótulos canônicos
#
# Nota: "Empresarial Master" aparece duas vezes no value_counts com contagens
# diferentes (19410 e 7327). Provavelmente uma das ocorrências tem espaço
# à direita. Por isso a transformação faz str.strip() antes de .map().
# --------------------------------------------------------------------------
PLANO_MAP: dict[str, str] = {
    # → MASTER EMPRESARIAL
    "MASTER EMPRESARIAL"             : "MASTER EMPRESARIAL",
    "Master Empresarial"             : "MASTER EMPRESARIAL",
    "Empresarial Master"             : "MASTER EMPRESARIAL",
    "Coletivo Empresarial Master"    : "MASTER EMPRESARIAL",
    # → MASTER EXECUTIVO
    "MASTER EXECUTIVO"                              : "MASTER EXECUTIVO",
    "Master Executivo"                              : "MASTER EXECUTIVO",
    "MASTER EXECUTIVO - COLETIVO EMPRESARIAL"       : "MASTER EXECUTIVO",
    "Master Executivo - Protocolo ANS: 478519173"   : "MASTER EXECUTIVO",
}

PLANOS_CANONICOS = sorted(set(PLANO_MAP.values()))
print(f"Planos canônicos definidos: {PLANOS_CANONICOS}")

# --------------------------------------------------------------------------
# TIPO_CADASTRO_MAP: 12 valores reais → 3 categorias canônicas
#
# Atenção: "Titular Beneficiário(a)" (com acento) e "Titular Beneficiario(a)"
# (sem acento) são dois valores distintos — ambos mapeados para TITULAR.
# "Dependente Filho(a) Def.  Mental/Física" tem duplo espaço — mantido exato.
# --------------------------------------------------------------------------
TIPO_CADASTRO_MAP: dict[str, str] = {
    # → TITULAR
    "TITULAR"                                  : "TITULAR",
    "Titular Beneficiário(a)"                  : "TITULAR",   # com acento
    "Titular Beneficiario(a)"                  : "TITULAR",   # sem acento
    # → DEPENDENTE
    "DEPENDENTE"                               : "DEPENDENTE",
    "Dependente Filho(a)"                      : "DEPENDENTE",
    "Dependente Cônjuge"                       : "DEPENDENTE",
    "Dependente Companheiro(a)"                : "DEPENDENTE",
    "Dependente Filho(a) Universitário(a)"     : "DEPENDENTE",
    "Dependente Tutelado(a)"                   : "DEPENDENTE",
    "Dependente Enteado(a)"                    : "DEPENDENTE",
    "Dependente Filho(a) Def.  Mental/Física"  : "DEPENDENTE",   # duplo espaço
    # → DESCONHECIDO (nulos)
    "<NA>"                                     : "DESCONHECIDO",
}

print(f"Categorias tipo_cadastro: {sorted(set(TIPO_CADASTRO_MAP.values()))}")

# --------------------------------------------------------------------------
# QTD_CONTA_MERGE: nomes exatos confirmados na Auditoria III
#
# qtd_conta_urgencia_emergencia: coluna toda zero (soma=0, n>0=0).
# É mantida para preservar a estrutura, mas não contribui com sinal.
# --------------------------------------------------------------------------
QTD_CONTA_MERGE: dict[str, list[str]] = {
    "qtd_conta_internado": [
        "qtd_conta_INTERNADO",
        "qtd_conta_Internado",
        "qtd_conta_internado",      # 1 registro não-zero; soma junto
    ],
    "qtd_conta_externo": [
        "qtd_conta_EXTERNO",
        "qtd_conta_Externo",
    ],
    "qtd_conta_pronto_socorro": [
        "qtd_conta_PRONTO SOCORRO",
        "qtd_conta_Pronto socorro",
    ],
    "qtd_conta_ambulatorial": [
        "qtd_conta_ATENDIMENTO AMBULATORIAL",
        "qtd_conta_Atendimento Ambulatorial",
    ],
    "qtd_conta_eletivo": [
        "qtd_conta_Eletivo",
    ],
    "qtd_conta_na": [
        "qtd_conta_NA",
    ],
    "qtd_conta_urgencia_emergencia": [
        "qtd_conta_Urgência \\ Emergência",   # toda zero — mantida por completude
    ],
}

print(f"Grupos qtd_conta_* a consolidar: {len(QTD_CONTA_MERGE)}")

# Coluna de carga de trabalho para leakage/quasi-leakage
LEAKAGE_COLS = [
    "sinistralidade_raw",
    "valor_sinistro_raw",
    "valor_sinistro_alt_val",
    "valor_sinistro_ajustado",
    "fator_ajuste_m",
    "S_real_m",
    "F_real_m",
    "sin_ref",
]
QUASI_LEAKAGE_COLS = [
    "qtd_eventos_sinistro",
    "qtd_carater_eletivo",
    "qtd_carater_urgencia",
]

#%%
# ── TRANSFORMAÇÃO 1: NORMALIZAR PLANO ─────────────────────────────────────────
df = df_raw.copy()

print("--- plano ANTES ---")
print(df[SEGMENT_COL].value_counts(dropna=False).to_string())

# strip antes do mapeamento: "Empresarial Master" aparece com contagens distintas
# no value_counts, sugerindo que uma variante tem espaço à direita
df[SEGMENT_COL] = df[SEGMENT_COL].astype(str).str.strip()

nao_mapeados_plano = df[SEGMENT_COL][~df[SEGMENT_COL].isin(PLANO_MAP)].unique()
if len(nao_mapeados_plano):
    print(f"\n⚠ Valores NÃO cobertos pelo PLANO_MAP (serão NaN): {nao_mapeados_plano}")
    print("  → Adicione esses valores ao PLANO_MAP no Bloco de Mapeamentos.")

df[SEGMENT_COL] = df[SEGMENT_COL].map(PLANO_MAP)

print("\n--- plano DEPOIS ---")
print(df[SEGMENT_COL].value_counts(dropna=False).to_string())
assert df[SEGMENT_COL].isna().sum() == 0, (
    "Ainda há nulos em 'plano' após mapeamento — adicione os valores faltantes ao PLANO_MAP."
)

#%%
# ── TRANSFORMAÇÃO 2: NORMALIZAR TIPO_CADASTRO ─────────────────────────────────

print("--- tipo_cadastro ANTES ---")
print(df["tipo_cadastro"].value_counts(dropna=False).to_string())

nao_mapeados_tc = (
    df["tipo_cadastro"]
    .astype("string")
    .fillna("<NA>")
    [~df["tipo_cadastro"].astype("string").fillna("<NA>").isin(TIPO_CADASTRO_MAP)]
    .unique()
)
if len(nao_mapeados_tc):
    print(f"\n⚠ Valores NÃO cobertos pelo TIPO_CADASTRO_MAP: {nao_mapeados_tc}")
    print("  → Adicione esses valores ao TIPO_CADASTRO_MAP no Bloco 8.")

df["tipo_cadastro"] = (
    df["tipo_cadastro"]
    .astype("string")
    .fillna("<NA>")
    .map(TIPO_CADASTRO_MAP)
    .fillna("DESCONHECIDO")   # fallback para qualquer valor não mapeado
)

print("\n--- tipo_cadastro DEPOIS ---")
print(df["tipo_cadastro"].value_counts(dropna=False).to_string())

#%%
# ── TRANSFORMAÇÃO 3: NORMALIZAR SEXO ──────────────────────────────────────────
# sexo tem os mesmos 17477 nulos de tipo_cadastro (mesmas linhas).
# Valores confirmados: M, F, <NA>.
# Nulos recebem "DESCONHECIDO" para preservar as linhas e permitir encoding.

print("--- sexo ANTES ---")
print(df["sexo"].value_counts(dropna=False).to_string())

df["sexo"] = df["sexo"].astype("string").fillna("DESCONHECIDO")

print("\n--- sexo DEPOIS ---")
print(df["sexo"].value_counts(dropna=False).to_string())
assert df["sexo"].isna().sum() == 0, "Ainda há nulos em 'sexo' após tratamento."

#%%
# ── TRANSFORMAÇÃO 4: CONSOLIDAR qtd_conta_* DUPLICADAS ────────────────────────

# Para cada grupo definido em QTD_CONTA_MERGE:
#   1. Identificar colunas existentes no DataFrame
#   2. Somar (fillna=0) → nova coluna com nome canônico
#   3. Remover as colunas originais
colunas_originais_a_remover: list[str] = []

for col_final, cols_origem in QTD_CONTA_MERGE.items():
    existentes = [c for c in cols_origem if c in df.columns]
    ausentes   = [c for c in cols_origem if c not in df.columns]

    if ausentes:
        print(f"  [AVISO] Colunas ausentes (serão ignoradas): {ausentes}")

    if not existentes:
        print(f"  [AVISO] Nenhuma coluna encontrada para '{col_final}' — pulando.")
        continue

    df[col_final] = df[existentes].fillna(0).sum(axis=1)
    colunas_originais_a_remover.extend(existentes)
    print(f"  [OK] '{col_final}' ← soma de {existentes}")

# Remover colunas originais que foram consolidadas
# (exceto as que já têm o nome canônico — seriam sobrescritas corretamente)
cols_para_remover = [
    c for c in colunas_originais_a_remover
    if c not in QTD_CONTA_MERGE  # não remover se já é um nome canônico
]
df = df.drop(columns=list(set(cols_para_remover)), errors="ignore")
print(f"\n[OK] {len(set(cols_para_remover))} colunas originais removidas.")
print(f"  qtd_conta_* restantes: {sorted(c for c in df.columns if c.startswith('qtd_conta_'))}")

#%%
# ── TRANSFORMAÇÃO 5: TRATAR NULOS NO TARGET ───────────────────────────────────

n_nulos_antes = df[TARGET_COL].isna().sum()
print(f"Nulos em '{TARGET_COL}' antes: {n_nulos_antes}")

# Nulos no target = ausência de sinistro → imputar como 0
df[TARGET_COL] = df[TARGET_COL].fillna(0.0)

print(f"Nulos em '{TARGET_COL}' depois: {df[TARGET_COL].isna().sum()}")

#%%
# ── TRANSFORMAÇÃO 6: CONVERTER Int64 NULLABLE → float64 ───────────────────────
# Necessário para compatibilidade com sklearn (não aceita pd.Int64Dtype).

int64_cols = [c for c in df.columns if str(df[c].dtype) == "Int64"]
print(f"Colunas Int64 a converter ({len(int64_cols)}): {int64_cols}")

for c in int64_cols:
    df[c] = df[c].astype("float64")

print("[OK] Conversão concluída.")

#%%
# ── TRANSFORMAÇÃO 7: FILTRO valor_faturamento > 0 ─────────────────────────────

n_antes = len(df)
df = df[df[PREMIUM_COL] > 0].copy()
n_depois = len(df)
print(f"Registros removidos (faturamento <= 0): {n_antes - n_depois}")
print(f"Shape final: {df.shape}")

#%%
# ── OUTPUT 1: FEATURE CATALOG ─────────────────────────────────────────────────
# Catálogo de todas as colunas com categoria e dtype para uso nos pipelines.

CATEGORY_MAP: dict[str, tuple[str, str, bool]] = {
    # coluna: (category, dtype, include_in_model)
    "cod_beneficiario"             : ("identificacao",  "identifier", False),
    "competencia"                  : ("contratual",     "identifier", False),
    "plano"                        : ("contratual",     "categorical", True),
    "tipo_cadastro"                : ("contratual",     "categorical", True),
    "sexo"                         : ("demografico",    "categorical", True),
    "idade"                        : ("demografico",    "numeric",    True),
    "valor_faturamento"            : ("financeiro",     "monetary",   True),
    "sinistralidade_final"         : ("resultado",      "target",     False),  # target
    "sinistralidade_raw"           : ("resultado",      "leakage",    False),
    "valor_sinistro_raw"           : ("resultado",      "leakage",    False),
    "valor_sinistro_alt_val"       : ("resultado",      "leakage",    False),
    "valor_sinistro_ajustado"      : ("resultado",      "leakage",    False),
    "fator_ajuste_m"               : ("resultado",      "leakage",    False),
    "S_real_m"                     : ("resultado",      "leakage",    False),
    "F_real_m"                     : ("resultado",      "leakage",    False),
    "sin_ref"                      : ("resultado",      "leakage",    False),
    "qtd_eventos_sinistro"         : ("utilizacao",     "quasi_leakage", False),
    "qtd_carater_eletivo"          : ("utilizacao",     "quasi_leakage", False),
    "qtd_carater_urgencia"         : ("utilizacao",     "quasi_leakage", False),
    "pct_urgencia"                 : ("utilizacao",     "numeric",    True),
}

# Inferir categoria para colunas restantes
catalog_rows = []
for col in df.columns:
    if col in CATEGORY_MAP:
        cat, dtype_, include = CATEGORY_MAP[col]
    elif col.startswith("qtd_conta_"):
        cat, dtype_, include = "utilizacao", "count", True
    elif col.startswith("qtd_servico_"):
        cat, dtype_, include = "utilizacao", "count", True
    elif col.startswith("qtd_esp_"):
        cat, dtype_, include = "utilizacao", "count", True
    else:
        cat, dtype_, include = "desconhecido", str(df[col].dtype), True

    catalog_rows.append({
        "feature_name"      : col,
        "category"          : cat,
        "dtype"             : dtype_,
        "include_in_model"  : include,
        "pandas_dtype"      : str(df[col].dtype),
        "n_nulos"           : int(df[col].isna().sum()),
        "n_unicos"          : int(df[col].nunique()),
    })

feature_catalog = pd.DataFrame(catalog_rows)
catalog_path = AUXILIAR_DIR / "feature_catalog.csv"
feature_catalog.to_csv(catalog_path, index=False, encoding="utf-8-sig")
print(f"[OK] feature_catalog.csv gravado: {catalog_path}")
print(f"  Total de features    : {len(feature_catalog)}")
print(f"  Include in model     : {feature_catalog['include_in_model'].sum()}")
print(f"  Por categoria:\n{feature_catalog['category'].value_counts().to_string()}")

#%%
# ── OUTPUT 2: SÉRIE HISTÓRICA MENSAL (entrada para o forecast) ────────────────
# Agrega o painel por competência → sinistralidade da carteira a cada mês.

serie = (
    df.groupby(TIME_COL)
    .agg(
        VALOR_FATURAMENTO=("valor_faturamento",       "sum"),
        VALOR_SINISTRO   =("valor_sinistro_ajustado", "sum"),
    )
    .reset_index()
    .rename(columns={TIME_COL: "COMPETENCIA"})
)
serie["SINISTRALIDADE"] = serie["VALOR_SINISTRO"] / serie["VALOR_FATURAMENTO"]
serie["DATA"]           = pd.to_datetime(serie["COMPETENCIA"] + "-01")
serie = serie.sort_values("DATA").reset_index(drop=True)

serie_path = AUXILIAR_DIR / "serie_historica_sinistralidade_mensal.csv"
serie.to_csv(serie_path, index=False, encoding="utf-8-sig")

print(f"[OK] Série mensal gravada: {serie_path}")
print(f"  Período : {serie['COMPETENCIA'].iloc[0]} → {serie['COMPETENCIA'].iloc[-1]}")
print(f"  Meses   : {len(serie)}")
print(f"\n  Sinistralidade média  : {serie['SINISTRALIDADE'].mean():.4f}")
print(f"  Sinistralidade min    : {serie['SINISTRALIDADE'].min():.4f}")
print(f"  Sinistralidade max    : {serie['SINISTRALIDADE'].max():.4f}")
print(serie.tail(6).to_string(index=False))

#%%
# ── OUTPUT 3: SALVAR PARQUET TRANSFORMADO ─────────────────────────────────────

df.to_parquet(OUTPUT_PARQUET, index=False)
print(f"[OK] Base transformada gravada em:\n  {OUTPUT_PARQUET}")
print(f"  Shape final: {df.shape}")

#%%
# ── VALIDAÇÕES FINAIS ──────────────────────────────────────────────────────────

erros: list[str] = []

if df[TARGET_COL].isna().sum() > 0:
    erros.append(f"Target '{TARGET_COL}' ainda tem nulos.")

if df[PREMIUM_COL].min() <= 0:
    erros.append(f"'{PREMIUM_COL}' tem valores <= 0.")

if not df[SEGMENT_COL].isin(PLANOS_CANONICOS).all():
    valores_inesperados = df[SEGMENT_COL][~df[SEGMENT_COL].isin(PLANOS_CANONICOS)].unique()
    erros.append(f"'plano' com valores não canônicos: {valores_inesperados}")

if df["sexo"].isna().sum() > 0:
    erros.append("'sexo' ainda tem nulos após tratamento.")

if df["tipo_cadastro"].isna().sum() > 0:
    erros.append("'tipo_cadastro' ainda tem nulos após tratamento.")

qtd_conta_final = [c for c in df.columns if c.startswith("qtd_conta_")]
nomes_originais_restantes = [
    c for c in qtd_conta_final
    if any(c in cols_orig for cols_orig in QTD_CONTA_MERGE.values() if c not in QTD_CONTA_MERGE)
]
if nomes_originais_restantes:
    erros.append(f"Colunas originais qtd_conta_* ainda presentes: {nomes_originais_restantes}")

if not OUTPUT_PARQUET.exists():
    erros.append("Parquet de saída não foi criado.")

if erros:
    print("\n❌ VALIDAÇÕES COM ERRO:")
    for e in erros:
        print(f"  - {e}")
    raise AssertionError("Pipeline concluído com erros. Corrija antes de prosseguir.")
else:
    print("\n✅ Todas as validações passaram.")
    print(f"   Parquet  : {OUTPUT_PARQUET}")
    print(f"   Catalog  : {AUXILIAR_DIR / 'feature_catalog.csv'}")
    print(f"   Série    : {AUXILIAR_DIR / 'serie_historica_sinistralidade_mensal.csv'}")
    print(f"   Shape    : {df.shape}")
    print(f"   Gerado   : {datetime.now().strftime('%Y-%m-%d %H:%M')}")

#%%
