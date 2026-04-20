# Plano Técnico Final: Simulador de Intervenção What-If
## Sistema de Análise de Impacto na Sinistralidade — ELGIN

> Versão final com decisões arquiteturais fechadas.

---

## 1. Premissas Fechadas

| Decisão | Resolução |
|---|---|
| Base de intervenção | MVP com base estática carregada uma vez por execução |
| Elegibilidade | Calculada sobre `df_agg` (dados brutos pré-feature engineering) |
| Reconstrução de `faixa_etaria` | Obrigatória quando `idade` é alterada; mesmos bins do treino |
| Reconstrução de `pct_urgencia` | Obrigatória quando `qtd_carater_urgencia` ou `qtd_eventos_sinistro` são alteradas |
| Reconstrução de `tx_*` | Responsabilidade da camada de API que orquestra a inferência |
| Valores negativos | Clipados em `0.0` (sem sentido semântico para o usuário) |
| `valor_faturamento` | Permanece inalterado — denominador da média ponderada não é intervenção de comportamento |
| Features de leakage/target | Bloqueadas para intervenção sem exceção |
| Persistência de metadados | `n_individuos_afetados` e `n_individuos_nao_elegiveis` incluídos no relatório |
| MLflow | Bloco `# %%` isolado ao final, após toda persistência em disco |

---

## 2. Visão Geral do Sistema

O módulo responde à pergunta:

> *"Se eu alterar em X% a característica Y da minha base, qual será o impacto na sinistralidade final da carteira?"*

**Saída consolidada obrigatória:**

```json
{
  "sinistralidade_antes": 0.821,
  "sinistralidade_depois": 0.874,
  "delta_absoluto": 0.053,
  "delta_relativo_pct": 6.46,
  "n_individuos_afetados": 4102,
  "n_individuos_nao_elegiveis": 4207,
  "intervencao": {
    "feature": "qtd_servico_CARDIOLOGIA",
    "delta_pct": 20.0
  },
  "versao_modelo": "v1"
}
```

---

## 3. Estrutura de Arquivos

### Módulos novos

```
pipelines/predict/what_if/
├── __init__.py
├── what_if_engine.py      # orquestração central da intervenção
├── elegibilidade.py       # regras de elegibilidade por feature (opera sobre df_agg)
├── correlacoes.py         # correlações para modo envelhecimento correlacionado
└── runners/
    └── run_what_if.py     # script executável (células # %%)
```

### Saídas versionadas (alinhadas ao padrão `vN/`)

```
data/processed/elgin/predict/vN/
└── what_if/
    ├── resultado_what_if.json          # resultado consolidado da simulação
    ├── predicoes_micro_antes.csv       # sinistralidade_prevista por beneficiário (base original)
    ├── predicoes_micro_depois.csv      # sinistralidade_prevista por beneficiário (base modificada)
    └── relatorio_intervencao.json      # metadados: n_afetados, n_não_elegíveis, % da base atingida
```

> O bloco MLflow é sempre executado **após** todos esses arquivos estarem gravados em disco.

---

## 4. Catálogo de Features Intervencionáveis

### 4.1 — Critério de inclusão

Feature é intervencionável se satisfaz simultaneamente no `feature_catalog.csv`:
- `dtype` ∉ `{leakage, identifier, target}`
- `category` ≠ `derivada`

Features bloqueadas por definição (mesmo que passem no catálogo):
```
valor_faturamento   # denominador da média ponderada — intocável
sinistralidade_final, sinistralidade_raw, valor_sinistro_raw,
valor_sinistro_alt_val, valor_sinistro_ajustado, sin_ref,
fator_ajuste_m, S_real_m, F_real_m, cod_beneficiario, competencia
```

### 4.2 — Tabela de elegibilidade por feature

A elegibilidade é avaliada **sobre `df_agg`** (dados brutos do beneficiário, antes do feature engineering). Isso garante que as regras reflitam o estado real cadastral e clínico do indivíduo, não valores transformados.

| Feature | Categoria | Regra de Elegibilidade sobre `df_agg` |
|---|---|---|
| `idade` | demografico | Universal — veja seção 5.1 |
| `qtd_eventos_sinistro` | utilizacao | Universal |
| `qtd_carater_eletivo` | utilizacao | `tipo_cadastro == "TITULAR"` |
| `qtd_carater_urgencia` | utilizacao | Universal |
| `qtd_conta_ATENDIMENTO AMBULATORIAL` | utilizacao | Universal |
| `qtd_conta_ELETIVO` | utilizacao | Universal |
| `qtd_conta_EXTERNO` | utilizacao | Universal |
| `qtd_conta_INTERNADO` | utilizacao | Para **aumento**: `qtd_conta_INTERNADO > 0`; para **redução**: Universal |
| `qtd_conta_PRONTO SOCORRO` | utilizacao | Universal |
| `qtd_conta_URGÊNCIA \ EMERGÊNCIA` | utilizacao | Universal |
| `qtd_servico_ACUPUNTURA` | utilizacao | Universal |
| `qtd_servico_CARDIOLOGIA` | utilizacao | `idade >= 30` |
| `qtd_servico_CIRURGICO` | utilizacao | `idade >= 12` |
| `qtd_servico_CLÍNICO` | utilizacao | Universal |
| `qtd_servico_CONSULTA` | utilizacao | Universal |
| `qtd_servico_DIÁRIA` | utilizacao | `qtd_conta_INTERNADO > 0` |
| `qtd_servico_ENDOSCOPIA` | utilizacao | `idade >= 40` |
| `qtd_servico_FISIOTERAPIA` | utilizacao | Universal |
| `qtd_servico_LABORATÓRIO` | utilizacao | Universal |
| `qtd_servico_QUIMIOTERAPIA` | utilizacao | `qtd_servico_QUIMIOTERAPIA > 0` (somente usuários ativos) |
| `qtd_servico_RADIOLOGIA` | utilizacao | Universal |
| `qtd_servico_RESSONÂNCIA MAGNÉTICA` | utilizacao | Universal |
| `qtd_servico_TERAPIA` | utilizacao | Universal |
| `qtd_servico_TOMOGRAFIA` | utilizacao | Universal |
| `qtd_servico_ULTRA-SONOGRAFIA` | utilizacao | Universal |
| `qtd_servico___OUTROS__` | utilizacao | Universal |
| `qtd_esp_cardio` | utilizacao | `idade >= 30` |
| `qtd_esp_cirurg` | utilizacao | `idade >= 12` |
| `qtd_esp_clin_geral` | utilizacao | Universal |
| `qtd_esp_derm` | utilizacao | Universal |
| `qtd_esp_gine` | utilizacao | `sexo == "F"` |
| `qtd_esp_lab_imagem` | utilizacao | Universal |
| `qtd_esp_neuro` | utilizacao | Universal |
| `qtd_esp_oftal` | utilizacao | Universal |
| `qtd_esp_orto` | utilizacao | Universal |
| `qtd_esp_otorrino` | utilizacao | Universal |
| `qtd_esp_outros` | utilizacao | Universal |
| `qtd_esp_ped` | utilizacao | `idade <= 18` |
| `qtd_esp_psiq` | utilizacao | Universal |
| `pct_urgencia` | utilizacao | Universal — mas recalculado como derivada quando fontes são alteradas |
| `n_meses_obs` | contratual | Universal — simulação de maturidade da carteira |

---

## 5. Casos Especiais e Decisões Técnicas

### 5.1 — Intervenção em `idade`

**Dois modos disponíveis, selecionados pelo usuário:**

**`modo=simples`**
Somente `idade` é alterada. `faixa_etaria` é **obrigatoriamente recalculada** com os mesmos bins do treino antes de qualquer inferência:

```python
bins   = [-1, 0, 5, 12, 18, 30, 45, 60, 200]
labels = ["inf", "0-5", "6-12", "13-18", "19-30", "31-45", "46-60", "60+"]
```

**Justificativa:** o modelo foi treinado com `faixa_etaria` derivada deterministicamente de `idade` via esses exatos bins. Alterar `idade` sem recalcular `faixa_etaria` envia ao modelo um estado que nunca existiu no espaço de treinamento, comprometendo a confiabilidade da predição.

**`modo=correlacionado`**
Além de `idade` e `faixa_etaria`, aplica deltas proporcionais com amortecimento nas features de utilização historicamente correlacionadas com envelhecimento (definidas em `correlacoes.py`). O amortecimento evita que a correlação seja tratada como causalidade 1:1.

### 5.2 — Reconstrução de `pct_urgencia`

`pct_urgencia = qtd_carater_urgencia / qtd_eventos_sinistro`

Quando `qtd_carater_urgencia` ou `qtd_eventos_sinistro` forem alteradas, `pct_urgencia` é recalculada obrigatoriamente sobre os valores modificados. Divisão por zero resulta em `0.0`.

**Justificativa:** `pct_urgencia` é uma feature usada diretamente pelo modelo. A correlação entre ela e suas fontes é determinística e estrutural — não recalcular introduz inconsistência interna no vetor de entrada.

### 5.3 — Reconstrução de `tx_*` (responsabilidade da API)

As features `tx_qtd_*` são derivadas por `valor_faturamento` dentro de `build_features_agregado`. Quando uma `qtd_*` é alterada, as `tx_*` correspondentes precisam ser recalculadas. **Essa responsabilidade pertence à camada de API** que orquestra a chamada de inferência — não ao script `run_what_if.py`. O script entrega o estado bruto modificado; a API aplica a engenharia antes de submeter ao modelo.

### 5.4 — Clip em zero

Todo valor resultante de uma intervenção é clipado em `0.0`:

```python
valor_modificado = (valor_original * delta_fator).clip(lower=0.0)
```

Valores negativos não têm interpretação semântica válida para contagens de eventos ou taxas de utilização.

### 5.5 — `valor_faturamento` como denominador

O denominador da média ponderada (`valor_faturamento`) **não é alterado em nenhuma circunstância** neste módulo. A intervenção simula mudanças de comportamento de utilização da carteira, não reajuste contratual. Alterar o denominador mudaria a natureza do índice calculado, tornando o resultado incomparável com a baseline.

### 5.6 — Intervenções simultâneas

A estrutura `INTERVENCOES` aceita lista de dicionários. Cada intervenção é aplicada sequencialmente sobre o mesmo `df_mod`. Reconstruções de derivadas (`faixa_etaria`, `pct_urgencia`) são executadas **após o loop completo de intervenções**, não entre cada item.

---

## 6. Código — Células `# %%`

### `run_what_if.py` — estrutura completa

```python
# %%
"""
What-If Engine — Simulador de Intervenção em Features.
Sinistralidade ELGIN.

Execução: rodar os blocos # %% em ordem.
Bloco MLflow ao final — executar apenas após todos os artefatos gravados em disco.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

# Reutiliza todas as constantes e funções de predict-agregado.py
# (importar ou manter no mesmo namespace Spyder/Jupyter)
# ROOT_DIR, TRANSFORMED_PARQUET_PATH, FEATURE_CATALOG_PATH, OUTPUT_PREDICT_ROOT,
# TARGET_COL, PREMIUM_COL, SEGMENT_COL, BENEFICIARIO_COL, N_MESES_COL,
# LEAKAGE_COLS, PLANO_MAP, VERSION_LABEL, RUN_DIR, FEATURE_COLS,
# LAST_MODEL_PATH, TwoStageModel, aggregate_panel_by_beneficiary,
# build_features_agregado, resolve_feature_columns, ensure_no_object_dtype,
# load_feature_catalog, aggregate_sinistralidade_macro

from what_if.elegibilidade import calcular_elegibilidade
from what_if.correlacoes   import CORRELACOES_ENVELHECIMENTO, aplicar_correlacoes_idade


# %%
# ===========================================================================
# [Passo 1 — Definição da Intervenção]
# Editar este bloco para cada simulação.
# ===========================================================================

INTERVENCOES: list[dict[str, Any]] = [
    {
        "feature":   "qtd_servico_CARDIOLOGIA",
        "delta_pct": +20.0,
    },
]

# Para intervenção em idade, definir o modo:
# "simples"       → apenas idade + faixa_etaria recalculada
# "correlacionado"→ idade + faixa_etaria + features de envelhecimento ajustadas
MODO_IDADE: str = "simples"


# %%
# ===========================================================================
# [Passo 2 — Carregamento da Base e do Modelo]
# ===========================================================================

df_raw = pd.read_parquet(TRANSFORMED_PARQUET_PATH)
df_raw[SEGMENT_COL] = df_raw[SEGMENT_COL].replace(PLANO_MAP)
df_agg = aggregate_panel_by_beneficiary(df_raw)

prem_ok = pd.to_numeric(df_agg[PREMIUM_COL], errors="coerce").fillna(0.0) > 0.0
df_agg  = df_agg.loc[prem_ok].copy().reset_index(drop=True)

fc          = load_feature_catalog()
df_feat     = build_features_agregado(df_agg)
feature_cols = resolve_feature_columns(df_feat, fc)
feature_cols = [c for c in feature_cols if c not in {N_MESES_COL}]

model: TwoStageModel = joblib.load(LAST_MODEL_PATH)
fat_base = df_agg[PREMIUM_COL].astype(float).reset_index(drop=True)

print(f"[Carga] n_beneficiarios={len(df_agg):,} | n_features={len(feature_cols)}")


# %%
# ===========================================================================
# [Passo 3 — Inferência de Referência (Baseline)]
# ===========================================================================

X_base = ensure_no_object_dtype(df_feat[feature_cols].copy())
_, y_hat_antes = model.predict_stages(X_base)
sin_antes = aggregate_sinistralidade_macro(y_hat_antes, fat_base)

print(f"[Baseline] sinistralidade_antes = {sin_antes:.6f}")


# %%
# ===========================================================================
# [Passo 4 — Aplicação das Intervenções]
# ===========================================================================

# Cópias de trabalho — originais não são modificados
df_mod_agg  = df_agg.copy()   # dados brutos modificáveis
df_mod_feat = df_feat.copy()  # features modificáveis

mascaras_por_intervencao: list[dict] = []
_idade_alterada             = False
_urgencia_alterada          = False
_delta_pct_idade            = 0.0

for interv in INTERVENCOES:
    feature_alvo = interv["feature"]
    delta_pct    = float(interv["delta_pct"])
    delta_fator  = 1.0 + delta_pct / 100.0

    # --- Elegibilidade calculada sobre df_agg (dados brutos) ---
    mask = calcular_elegibilidade(df_mod_agg, feature_alvo, delta_pct)
    n_elegiveis     = int(mask.sum())
    n_nao_elegiveis = len(df_mod_agg) - n_elegiveis

    mascaras_por_intervencao.append({
        "feature":          feature_alvo,
        "delta_pct":        delta_pct,
        "n_elegiveis":      n_elegiveis,
        "n_nao_elegiveis":  n_nao_elegiveis,
    })

    # --- Aplicar delta na coluna raw de df_mod_agg ---
    if feature_alvo in df_mod_agg.columns:
        df_mod_agg.loc[mask, feature_alvo] = (
            df_mod_agg.loc[mask, feature_alvo] * delta_fator
        ).clip(lower=0.0)

    # --- Flags para reconstruções pós-loop ---
    if feature_alvo == "idade":
        _idade_alterada  = True
        _delta_pct_idade = delta_pct
        _mask_idade      = mask

    if feature_alvo in {"qtd_carater_urgencia", "qtd_eventos_sinistro"}:
        _urgencia_alterada = True

    print(
        f"[Intervenção] {feature_alvo} | delta={delta_pct:+.1f}% | "
        f"elegíveis={n_elegiveis:,} | não elegíveis={n_nao_elegiveis:,}"
    )


# %%
# ===========================================================================
# [Passo 5 — Reconstrução de Derivadas Pós-Loop]
# Ordem importa: urgencia → idade → correlações de idade
# ===========================================================================

# 5a — Reconstruir pct_urgencia se fontes foram alteradas
if _urgencia_alterada and "pct_urgencia" in df_mod_feat.columns:
    with np.errstate(divide="ignore", invalid="ignore"):
        df_mod_feat["pct_urgencia"] = np.where(
            df_mod_agg["qtd_eventos_sinistro"] > 0,
            df_mod_agg["qtd_carater_urgencia"] / df_mod_agg["qtd_eventos_sinistro"],
            0.0,
        )
    print("[Reconstrução] pct_urgencia recalculada.")

# 5b — Reconstruir faixa_etaria se idade foi alterada
if _idade_alterada and "faixa_etaria" in df_mod_feat.columns:
    _id_num = pd.to_numeric(df_mod_agg["idade"], errors="coerce")
    df_mod_feat["faixa_etaria"] = pd.cut(
        _id_num.fillna(-1),
        bins=[-1, 0, 5, 12, 18, 30, 45, 60, 200],
        labels=["inf", "0-5", "6-12", "13-18", "19-30", "31-45", "46-60", "60+"],
    ).astype(str)
    print("[Reconstrução] faixa_etaria recalculada a partir de idade modificada.")

    # 5c — Correlações de envelhecimento (somente modo correlacionado)
    if MODO_IDADE == "correlacionado":
        df_mod_feat = aplicar_correlacoes_idade(
            df_mod_feat   = df_mod_feat,
            df_mod_agg    = df_mod_agg,
            mask_elegivel = _mask_idade,
            delta_pct_idade = _delta_pct_idade,
            feature_cols  = feature_cols,
        )
        print("[Reconstrução] Correlações de envelhecimento aplicadas.")


# %%
# ===========================================================================
# [Passo 6 — Inferência Modificada e Consolidação]
# ===========================================================================

X_mod = ensure_no_object_dtype(df_mod_feat[feature_cols].copy())
_, y_hat_depois = model.predict_stages(X_mod)
sin_depois = aggregate_sinistralidade_macro(y_hat_depois, fat_base)

delta_abs     = sin_depois - sin_antes
delta_rel_pct = (delta_abs / sin_antes * 100.0) if sin_antes != 0 else float("nan")

resultado = {
    "sinistralidade_antes":       round(float(sin_antes), 6),
    "sinistralidade_depois":      round(float(sin_depois), 6),
    "delta_absoluto":             round(float(delta_abs), 6),
    "delta_relativo_pct":         round(float(delta_rel_pct), 4),
    "n_individuos_afetados":      sum(m["n_elegiveis"]     for m in mascaras_por_intervencao),
    "n_individuos_nao_elegiveis": sum(m["n_nao_elegiveis"] for m in mascaras_por_intervencao),
    "intervencoes":               INTERVENCOES,
    "modo_idade":                 MODO_IDADE,
    "versao_modelo":              VERSION_LABEL,
}

print("\n" + "=" * 60)
print("[Resultado]")
print(f"  sinistralidade_antes  = {sin_antes:.6f}")
print(f"  sinistralidade_depois = {sin_depois:.6f}")
print(f"  delta_absoluto        = {delta_abs:+.6f}")
print(f"  delta_relativo_pct    = {delta_rel_pct:+.4f}%")
print("=" * 60)


# %%
# ===========================================================================
# [Passo 7 — Persistência em Disco]
# Todo arquivo deve ser gravado ANTES do bloco MLflow.
# ===========================================================================

what_if_dir = RUN_DIR / "what_if"
what_if_dir.mkdir(parents=True, exist_ok=True)

# 7a — Resultado consolidado
(what_if_dir / "resultado_what_if.json").write_text(
    json.dumps(resultado, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

# 7b — Predições micro — antes
_n = len(df_agg)
pred_antes = pd.DataFrame({
    BENEFICIARIO_COL:              df_agg[BENEFICIARIO_COL].values,
    PREMIUM_COL:                   fat_base.values,
    "sinistralidade_prevista_antes": np.asarray(y_hat_antes, dtype=float),
})
pred_antes.to_csv(
    what_if_dir / "predicoes_micro_antes.csv",
    index=False, encoding="utf-8-sig",
)

# 7c — Predições micro — depois (com máscara de elegibilidade agregada)
mask_afetados = np.zeros(_n, dtype=bool)
for m in mascaras_por_intervencao:
    mask_intervencao = calcular_elegibilidade(
        df_agg, m["feature"], m["delta_pct"]
    ).values
    mask_afetados |= mask_intervencao

pred_depois = pred_antes.copy()
pred_depois["sinistralidade_prevista_depois"] = np.asarray(y_hat_depois, dtype=float)
pred_depois["delta_individual"] = (
    pred_depois["sinistralidade_prevista_depois"]
    - pred_depois["sinistralidade_prevista_antes"]
)
pred_depois["foi_afetado"] = mask_afetados.astype(int)
pred_depois.to_csv(
    what_if_dir / "predicoes_micro_depois.csv",
    index=False, encoding="utf-8-sig",
)

# 7d — Relatório de intervenção
relatorio = {
    "versao":             VERSION_LABEL,
    "n_total_base":       _n,
    "intervencoes":       mascaras_por_intervencao,
    "pct_base_afetada":   round(float(mask_afetados.sum()) / _n * 100, 2),
    "modo_idade":         MODO_IDADE,
}
(what_if_dir / "relatorio_intervencao.json").write_text(
    json.dumps(relatorio, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print(f"[Disco] Artefatos gravados em {what_if_dir}")


# %%
# ===========================================================================
# [Passo 8 — Registro MLflow]
# Executar SOMENTE após confirmação de que todos os arquivos de Passo 7
# foram gravados com sucesso. Este bloco é isolado por design.
# ===========================================================================

def _what_if_mlflow_log(
    *,
    run_dir:    Path,
    version_label: str,
    resultado:  dict,
    relatorio:  dict,
) -> None:
    try:
        import mlflow
    except ImportError:
        print("[MLflow] Pacote não instalado; ignorando registro.")
        return

    import sys
    sys.path.insert(0, str(ROOT_DIR))
    try:
        from config.mlflow_config import configurar_mlflow
    except ImportError as e:
        print(f"[MLflow] config.mlflow_config indisponível: {e}")
        return

    try:
        configurar_mlflow(MLFLOW_EXPERIMENT_NAME, preparar_experimento=True)
    except EnvironmentError as e:
        print(f"[MLflow] Configuração incompleta: {e}\n[MLflow] Artefatos em disco OK; sem tracking.")
        return

    run_name = f"elgin__what_if__{version_label}"
    what_if_dir_local = run_dir / "what_if"

    if mlflow.active_run() is not None:
        mlflow.end_run()

    with mlflow.start_run(run_name=run_name):
        # Parâmetros da intervenção
        mlflow.log_param("pipeline",       "elgin_what_if")
        mlflow.log_param("versao_pasta",   version_label)
        mlflow.log_param("n_intervencoes", len(INTERVENCOES))
        mlflow.log_param("modo_idade",     MODO_IDADE)
        for i, interv in enumerate(INTERVENCOES):
            mlflow.log_param(f"interv_{i}_feature",   interv["feature"])
            mlflow.log_param(f"interv_{i}_delta_pct", interv["delta_pct"])

        # Métricas do resultado
        mlflow.log_metric("sinistralidade_antes",       resultado["sinistralidade_antes"])
        mlflow.log_metric("sinistralidade_depois",      resultado["sinistralidade_depois"])
        mlflow.log_metric("delta_absoluto",             resultado["delta_absoluto"])
        mlflow.log_metric("delta_relativo_pct",         resultado["delta_relativo_pct"])
        mlflow.log_metric("n_individuos_afetados",      resultado["n_individuos_afetados"])
        mlflow.log_metric("n_individuos_nao_elegiveis", resultado["n_individuos_nao_elegiveis"])

        # Artefatos — somente o que já está em disco
        for fname in [
            "resultado_what_if.json",
            "relatorio_intervencao.json",
            "predicoes_micro_antes.csv",
            "predicoes_micro_depois.csv",
        ]:
            fpath = what_if_dir_local / fname
            if fpath.is_file():
                mlflow.log_artifact(str(fpath), artifact_path="what_if_artifacts")

    print(f"[MLflow] Run what-if registrado: {run_name}")


_what_if_mlflow_log(
    run_dir=RUN_DIR,
    version_label=VERSION_LABEL,
    resultado=resultado,
    relatorio=relatorio,
)
```

---

## 7. Módulo `elegibilidade.py`

```python
# what_if/elegibilidade.py
"""
Regras de elegibilidade para intervenção.
Operam exclusivamente sobre df_agg (dados brutos pré-feature engineering).
"""

from __future__ import annotations

import pandas as pd
import numpy as np

# Regras indexadas por feature.
# Cada callable recebe (df_agg, delta_pct) e retorna pd.Series[bool].
# delta_pct é necessário para regras assimétricas (ex: internação).

def _universal(df: pd.DataFrame, delta_pct: float) -> pd.Series:
    return pd.Series([True] * len(df), index=df.index)

def _internado(df: pd.DataFrame, delta_pct: float) -> pd.Series:
    # Aumento só para quem já tem registro de internação; redução é universal
    if delta_pct > 0:
        return df["qtd_conta_INTERNADO"] > 0
    return _universal(df, delta_pct)

def _quimio(df: pd.DataFrame, delta_pct: float) -> pd.Series:
    # Qualquer direção: apenas usuários ativos do serviço
    return df["qtd_servico_QUIMIOTERAPIA"] > 0

REGRAS_ELEGIBILIDADE: dict[str, callable] = {
    "qtd_servico_CARDIOLOGIA":                lambda df, d: df["idade"] >= 30,
    "qtd_servico_QUIMIOTERAPIA":               _quimio,
    "qtd_servico_ENDOSCOPIA":                  lambda df, d: df["idade"] >= 40,
    "qtd_servico_CIRURGICO":                   lambda df, d: df["idade"] >= 12,
    "qtd_servico_DIÁRIA":                      lambda df, d: df["qtd_conta_INTERNADO"] > 0,
    "qtd_esp_cardio":                          lambda df, d: df["idade"] >= 30,
    "qtd_esp_ped":                             lambda df, d: df["idade"] <= 18,
    "qtd_esp_gine":                            lambda df, d: df["sexo"].astype(str).str.upper() == "F",
    "qtd_esp_cirurg":                          lambda df, d: df["idade"] >= 12,
    "qtd_carater_eletivo":                     lambda df, d: df["tipo_cadastro"].astype(str).str.upper() == "TITULAR",
    "qtd_conta_INTERNADO":                     _internado,
}

_DEFAULT = _universal


def calcular_elegibilidade(
    df_agg: pd.DataFrame,
    feature: str,
    delta_pct: float = 0.0,
) -> pd.Series:
    """
    Retorna máscara booleana de elegibilidade para a feature e direção de delta.
    Avalia sobre df_agg (dados brutos).
    """
    regra = REGRAS_ELEGIBILIDADE.get(feature, _DEFAULT)
    mask  = regra(df_agg, delta_pct)
    if isinstance(mask, pd.Series):
        mask = mask.fillna(False).astype(bool)
        mask.index = df_agg.index
    return mask
```

---

## 8. Módulo `correlacoes.py`

```python
# what_if/correlacoes.py
"""
Correlações de features com envelhecimento da carteira.
Usadas apenas quando modo_idade == "correlacionado".

fator_amortecimento:
  0.0 = nenhum impacto na feature
  1.0 = impacto proporcional ao delta percentual de idade
  Valores intermediários refletem correlação histórica não-causal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Mapa: feature_raw → fator de amortecimento
CORRELACOES_ENVELHECIMENTO: dict[str, float] = {
    "qtd_esp_cardio":           0.70,
    "qtd_servico_CARDIOLOGIA":  0.65,
    "qtd_esp_orto":             0.50,
    "qtd_servico_FISIOTERAPIA": 0.40,
    "qtd_servico_LABORATÓRIO":  0.35,
    "qtd_conta_INTERNADO":      0.30,
    "qtd_esp_oftal":            0.30,
    "qtd_carater_urgencia":     0.25,
    "qtd_esp_neuro":            0.20,
}


def aplicar_correlacoes_idade(
    df_mod_feat:     pd.DataFrame,
    df_mod_agg:      pd.DataFrame,
    mask_elegivel:   pd.Series,
    delta_pct_idade: float,
    feature_cols:    list[str],
) -> pd.DataFrame:
    """
    Aplica deltas amortecidos nas features correlacionadas com envelhecimento.
    Modifica df_mod_agg in-place para manter consistência com reconstruções de tx_*.
    Retorna df_mod_feat atualizado.
    """
    df_out = df_mod_feat.copy()

    for feat, amort in CORRELACOES_ENVELHECIMENTO.items():
        if feat not in df_mod_agg.columns:
            continue

        delta_efetivo = (delta_pct_idade / 100.0) * amort
        df_mod_agg.loc[mask_elegivel, feat] = (
            df_mod_agg.loc[mask_elegivel, feat] * (1.0 + delta_efetivo)
        ).clip(lower=0.0)

        # Nota: reconstrução de tx_* correspondentes é responsabilidade da API de inferência
        # conforme premissa arquitetural definida neste plano.

    return df_out
```

---

## 9. Checklist de Execução

**Antes de rodar o Passo 8 (MLflow), verificar:**

- [ ] `resultado_what_if.json` existe e tem tamanho > 0
- [ ] `relatorio_intervencao.json` existe e tem tamanho > 0
- [ ] `predicoes_micro_antes.csv` existe e tem `len == n_beneficiarios`
- [ ] `predicoes_micro_depois.csv` existe e tem `len == n_beneficiarios`
- [ ] `sin_antes` e `sin_depois` são valores finitos (não `nan`, não `inf`)
- [ ] `n_individuos_afetados + n_individuos_nao_elegiveis == n_total_base`
