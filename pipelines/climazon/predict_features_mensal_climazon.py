# %%
"""
What-If Mensal — Sinistralidade CLIMAZON (CLI para Subprocess)

Script CLI chamado pelo Streamlit via subprocess para simular uma intervenção
em uma feature no mês de referência.

Uso:
    python pipelines/climazon/predict_features_mensal_climazon.py \
        --versao v1 \
        --feature qtd_conta_internado \
        --delta-pct -30.0 \
        --competencia-ref 2024-12 \
        --plano "MASTER EMPRESARIAL"

Saídas em data/processed/climazon/predict/vN/what_if_mensal/:
  resultado_what_if.json       — ✅ consumido pelo app
  comparacao_antes_depois.csv  — detalhes por beneficiário
"""

# %%
from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
import types
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

ROOT_DIR = Path(__file__).resolve().parents[2]
_climazon_dir = ROOT_DIR / "pipelines" / "climazon"
for _p in (str(ROOT_DIR), str(_climazon_dir)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import predict_climazon as pc  # noqa: E402


# =============================================================================
# %% — Constantes
# =============================================================================

COMPANY                  = pc.COMPANY
OUTPUT_PREDICT_ROOT      = pc.OUTPUT_PREDICT_ROOT
TRANSFORMED_PARQUET_PATH = pc.TRANSFORMED_PARQUET_PATH
TARGET_COL               = pc.TARGET_COL
TIME_COL                 = pc.TIME_COL
SEGMENT_COL              = pc.SEGMENT_COL
PREMIUM_COL              = pc.PREMIUM_COL
BENEFICIARIO_COL         = pc.BENEFICIARIO_COL
PLANOS_CANONICOS         = pc.PLANOS_CANONICOS


# =============================================================================
# %% — Utilitários
# =============================================================================


@contextlib.contextmanager
def _shim_main_for_joblib():
    """Injeta TwoStageModel no __main__ para que joblib.load desserialize corretamente.

    O modelo foi salvo quando predict_climazon.py era __main__, então o pickle
    referencia __main__.TwoStageModel. Este shim resolve a classe de onde quer
    que o what-if seja chamado.
    """
    real_main = sys.modules.get("__main__")
    shim = types.ModuleType("__main__")
    shim.TwoStageModel = pc.TwoStageModel  # type: ignore[attr-defined]
    sys.modules["__main__"] = shim
    try:
        yield
    finally:
        if real_main is not None:
            sys.modules["__main__"] = real_main
        else:
            sys.modules.pop("__main__", None)


def _find_version_dir(versao: str | None) -> tuple[str, Path]:
    root = OUTPUT_PREDICT_ROOT
    if not root.is_dir():
        raise FileNotFoundError(f"Pasta predict não encontrada: {root}")
    if versao:
        ver_dir = root / versao
        if not ver_dir.is_dir():
            raise FileNotFoundError(f"Versão '{versao}' não encontrada em {root}")
        return versao, ver_dir
    max_n = 0
    for p in root.iterdir():
        if p.is_dir():
            m = re.fullmatch(r"v(\d+)", p.name, flags=re.IGNORECASE)
            if m:
                max_n = max(max_n, int(m.group(1)))
    if max_n == 0:
        raise FileNotFoundError(f"Nenhuma versão vN em {root}")
    ver = f"v{max_n}"
    return ver, root / ver


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


# =============================================================================
# %% — Lógica de what-if por plano
# =============================================================================

def _run_what_if_plano(
    df_feat_mes: pd.DataFrame,
    model: Any,
    feature: str,
    delta_pct: float,
    plano: str,
    modo_idade: str,
) -> dict[str, Any] | None:
    """Executa o what-if para um plano específico. Retorna None se sem dados."""
    mask_plano = df_feat_mes[SEGMENT_COL].astype(str) == plano if SEGMENT_COL in df_feat_mes.columns else pd.Series(True, index=df_feat_mes.index)
    df_seg = df_feat_mes.loc[mask_plano].copy().reset_index(drop=True)
    if len(df_seg) == 0:
        return None

    feature_cols = [c for c in model.feature_names_ if c in df_seg.columns]
    if not feature_cols:
        return None

    fat = (
        pd.to_numeric(df_seg[PREMIUM_COL], errors="coerce").fillna(1.0)
        if PREMIUM_COL in df_seg.columns
        else pd.Series(1.0, index=df_seg.index)
    )

    # Baseline
    X_base = pc.ensure_no_object_dtype(df_seg[feature_cols])
    y_base = model.predict(X_base)
    sin_antes = pc.aggregate_sinistralidade_macro(y_base, fat)

    # Elegibilidade — calculada na coluna original (se presente em df_feat)
    elegiveis = _calcular_elegibilidade(df_seg, feature, delta_pct)

    # Intervenção
    df_int = df_seg.copy()

    if feature == "idade" and modo_idade == "correlacionado":
        # Efeitos secundários correlacionados com envelhecimento
        for col_sec, peso in [("pct_urgencia", 0.3), ("qtd_conta_pronto_socorro", 0.2)]:
            if col_sec in df_int.columns:
                df_int.loc[elegiveis, col_sec] = (
                    pd.to_numeric(df_int.loc[elegiveis, col_sec], errors="coerce").fillna(0)
                    * (1 + delta_pct / 100 * peso)
                )

    if feature in df_int.columns:
        df_int.loc[elegiveis, feature] = (
            pd.to_numeric(df_int.loc[elegiveis, feature], errors="coerce").fillna(0)
            * (1 + delta_pct / 100)
        )
        # Propaga ao tx_ correspondente
        tx_col = f"tx_{feature}"
        if tx_col in df_int.columns and PREMIUM_COL in df_int.columns:
            fat_col = pd.to_numeric(df_int[PREMIUM_COL], errors="coerce").replace(0, np.nan)
            df_int[tx_col] = (
                pd.to_numeric(df_int[feature], errors="coerce").fillna(0) / fat_col
            )

    X_int = pc.ensure_no_object_dtype(df_int[feature_cols])
    y_int = model.predict(X_int)
    sin_depois = pc.aggregate_sinistralidade_macro(y_int, fat)

    n_afetados = int(elegiveis.sum())
    n_total = len(df_seg)
    delta_abs = sin_depois - sin_antes
    delta_rel = (delta_abs / sin_antes * 100) if sin_antes and not np.isnan(sin_antes) else 0.0

    print(
        f"  [{plano}] antes={sin_antes:.4f} | depois={sin_depois:.4f} | "
        f"delta={delta_rel:+.2f}% | n_afetados={n_afetados}/{n_total}"
    )

    return {
        "plano": plano,
        "n_individuos_total": n_total,
        "n_individuos_afetados": n_afetados,
        "pct_base_afetada": round(100.0 * n_afetados / max(1, n_total), 2),
        "sinistralidade_antes": float(sin_antes),
        "sinistralidade_depois": float(sin_depois),
        "delta_absoluto": float(delta_abs),
        "delta_relativo_pct": float(delta_rel),
        "_y_base": y_base,
        "_y_int": y_int,
        "_fat": fat,
    }


# =============================================================================
# %% — Execução principal
# =============================================================================

def run_what_if(
    feature: str,
    delta_pct: float,
    versao: str | None = None,
    competencia_ref: str | None = None,
    plano: str | None = None,
    modo_idade: str = "simples",
) -> dict[str, Any]:
    ver, ver_dir = _find_version_dir(versao)
    print(f"[What-If] versao={ver} | feature={feature} | delta_pct={delta_pct:+.1f}%")

    # Carga e filtro básico
    df_full = pd.read_parquet(TRANSFORMED_PARQUET_PATH)
    prem_ok = pd.to_numeric(df_full[PREMIUM_COL], errors="coerce").fillna(0.0) > 0.0
    df_full = df_full.loc[
        prem_ok & df_full[SEGMENT_COL].isin(PLANOS_CANONICOS)
    ].copy()

    planos_alvo = [plano] if plano else PLANOS_CANONICOS
    df_full = df_full[df_full[SEGMENT_COL].isin(planos_alvo)].copy()

    # Validações
    if feature not in df_full.columns:
        raise ValueError(f"Feature '{feature}' não existe na base")
    if not (-90 <= delta_pct <= 200):
        raise ValueError(f"delta_pct={delta_pct} fora do range permitido [-90, 200]")

    # Competência de referência
    comp_dt = pd.to_datetime(df_full[TIME_COL], errors="coerce")
    if competencia_ref is None:
        competencia_ref = str(comp_dt.max().strftime("%Y-%m"))
    print(f"[What-If] competencia_ref={competencia_ref} | planos={planos_alvo}")

    # Constrói features sobre o histórico completo (necessário para lags corretos)
    df_feat_full = pc.build_features(df_full)

    # Filtra para a competência de referência
    if TIME_COL in df_feat_full.columns:
        comp_feat_dt = pd.to_datetime(df_feat_full[TIME_COL], errors="coerce")
        mask_comp = comp_feat_dt.dt.to_period("M") == pd.Period(competencia_ref, freq="M")
    else:
        mask_comp = pd.Series(True, index=df_feat_full.index)
    df_feat_mes = df_feat_full.loc[mask_comp].copy().reset_index(drop=True)

    if len(df_feat_mes) == 0:
        raise ValueError(
            f"Nenhuma linha para competencia_ref='{competencia_ref}' "
            f"nos planos {planos_alvo}"
        )
    print(f"[What-If] n_linhas_mes={len(df_feat_mes):,}")

    # Executa what-if por plano
    resultados_raw: list[dict[str, Any]] = []
    for plano_iter in planos_alvo:
        slug = pc.plano_slug(plano_iter)
        model_path = ver_dir / "models" / f"model_{slug}.pkl"
        if not model_path.exists():
            print(f"[PULAR] Modelo não encontrado: {model_path}", file=sys.stderr)
            continue
        with _shim_main_for_joblib():
            model = joblib.load(model_path)
        res = _run_what_if_plano(
            df_feat_mes=df_feat_mes,
            model=model,
            feature=feature,
            delta_pct=delta_pct,
            plano=plano_iter,
            modo_idade=modo_idade,
        )
        if res is not None:
            resultados_raw.append(res)

    if not resultados_raw:
        raise RuntimeError("Nenhum plano pôde ser processado.")

    # Remove arrays numpy do resultado (não serializáveis)
    resultados_limpos = [
        {k: v for k, v in r.items() if not k.startswith("_")}
        for r in resultados_raw
    ]

    # Consolida (agrega planos se mais de um)
    if len(resultados_raw) == 1:
        res_global = {k: v for k, v in resultados_raw[0].items() if not k.startswith("_")}
    else:
        # Média ponderada por n_individuos_total
        n_tot_all = sum(r["n_individuos_total"] for r in resultados_raw)
        n_afet_all = sum(r["n_individuos_afetados"] for r in resultados_raw)
        # Macro ponderado pelo prêmio (soma dos arrays)
        y_base_all = np.concatenate([r["_y_base"] for r in resultados_raw])
        y_int_all  = np.concatenate([r["_y_int"]  for r in resultados_raw])
        fat_all    = pd.concat([r["_fat"] for r in resultados_raw], ignore_index=True)
        sin_a = pc.aggregate_sinistralidade_macro(y_base_all, fat_all)
        sin_d = pc.aggregate_sinistralidade_macro(y_int_all, fat_all)
        delta_a = sin_d - sin_a
        delta_r = (delta_a / sin_a * 100) if sin_a and not np.isnan(sin_a) else 0.0
        res_global = {
            "plano": "TODOS",
            "n_individuos_total":    n_tot_all,
            "n_individuos_afetados": n_afet_all,
            "pct_base_afetada":      round(100.0 * n_afet_all / max(1, n_tot_all), 2),
            "sinistralidade_antes":  float(sin_a),
            "sinistralidade_depois": float(sin_d),
            "delta_absoluto":        float(delta_a),
            "delta_relativo_pct":    float(delta_r),
        }

    resultado_final: dict[str, Any] = {
        "company":               COMPANY,
        "versao":                ver,
        "data_execucao":         datetime.now().strftime("%Y-%m-%d %H:%M"),
        "competencia_referencia": competencia_ref,
        "planos_alvo":           planos_alvo,
        "feature_intervencionada": feature,
        "delta_pct_aplicado":    delta_pct,
        "modo_idade":            modo_idade,
        "intervencoes":          [{"feature": feature, "delta_pct": delta_pct}],
        "por_plano":             resultados_limpos,
        **res_global,
    }

    # Salva saídas
    out_dir = ver_dir / "what_if_mensal"
    out_dir.mkdir(parents=True, exist_ok=True)

    resultado_path = out_dir / "resultado_what_if.json"
    with open(resultado_path, "w", encoding="utf-8") as f:
        json.dump(resultado_final, f, indent=2, ensure_ascii=False, default=str)
    print(f"[OK] {resultado_path}")

    # comparacao_antes_depois.csv
    try:
        rows_comp: list[dict[str, Any]] = []
        for r in resultados_raw:
            p = r["plano"]
            slug_p = pc.plano_slug(p)
            model_path = ver_dir / "models" / f"model_{slug_p}.pkl"
            if not model_path.exists():
                continue
            mask_p = (
                df_feat_mes[SEGMENT_COL].astype(str) == p
                if SEGMENT_COL in df_feat_mes.columns
                else pd.Series(True, index=df_feat_mes.index)
            )
            df_p = df_feat_mes.loc[mask_p].reset_index(drop=True)
            for i in range(len(df_p)):
                rows_comp.append({
                    "plano": p,
                    "sinistralidade_prevista_antes": float(r["_y_base"][i]),
                    "sinistralidade_prevista_depois": float(r["_y_int"][i]),
                    "delta_abs": float(r["_y_int"][i] - r["_y_base"][i]),
                })
        if rows_comp:
            pd.DataFrame(rows_comp).to_csv(
                out_dir / "comparacao_antes_depois.csv",
                index=False, encoding="utf-8-sig",
            )
    except Exception as e:
        print(f"[aviso] comparacao_antes_depois: {e}")

    return resultado_final


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="What-If mensal CLIMAZON — simula intervenção em uma feature"
    )
    parser.add_argument(
        "--versao", default=None,
        help="Versão do modelo (ex: v1). Padrão: mais recente",
    )
    parser.add_argument("--feature", required=True, help="Feature a intervir")
    parser.add_argument(
        "--delta-pct", type=float, required=True, dest="delta_pct",
        help="Delta percentual [-90, 200]",
    )
    parser.add_argument(
        "--competencia-ref", default=None, dest="competencia_ref",
        help="Mês de referência YYYY-MM. Padrão: última competência disponível",
    )
    parser.add_argument(
        "--plano", default=None,
        help="Filtrar por plano específico. Padrão: todos os planos canônicos",
    )
    parser.add_argument(
        "--modo-idade", default="simples", dest="modo_idade",
        choices=["simples", "correlacionado"],
        help="Para feature=idade: aplicar efeitos secundários correlacionados",
    )
    args = parser.parse_args()

    if not (-90 <= args.delta_pct <= 200):
        print(
            f"ERRO: delta_pct={args.delta_pct} fora do range permitido [-90, 200]",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        resultado = run_what_if(
            feature=args.feature,
            delta_pct=args.delta_pct,
            versao=args.versao,
            competencia_ref=args.competencia_ref,
            plano=args.plano,
            modo_idade=args.modo_idade,
        )
        print(json.dumps(resultado, indent=2, ensure_ascii=False, default=str))
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"ERRO: {e}", file=sys.stderr)
        sys.exit(1)

# %%
