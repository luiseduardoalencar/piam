"""
What-if mensal ELGIN (snapshot da competência de referência).

Executa intervenção em features na base mensal (não agregada por beneficiário ao longo do histórico),
reconstrói features com ``predict.py`` e infere usando os modelos locais de ``data/processed/elgin/predict/vN``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ROOT_DIR / "pipelines" / "predict") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "pipelines" / "predict"))
if str(ROOT_DIR / "pipelines" / "elgin") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "pipelines" / "elgin"))

import predict as ep
from what_if.correlacoes import aplicar_correlacoes_idade
from what_if.elegibilidade import calcular_elegibilidade


def _load_intervencoes_json(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        raise ValueError(f"Ficheiro vazio: {path}")
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("O JSON deve ser uma lista de objetos {feature, delta_pct}.")
    return data


def _intervencoes_from_args(args: argparse.Namespace) -> list[dict[str, Any]]:
    has_file = args.intervencoes is not None
    has_cli = args.feature is not None or args.delta_pct is not None
    if has_file and has_cli:
        raise SystemExit("Use apenas --intervencoes OU (--feature e --delta-pct).")
    if has_cli:
        if args.feature is None or args.delta_pct is None:
            raise SystemExit("--feature e --delta-pct são obrigatórios em conjunto.")
        return [{"feature": args.feature, "delta_pct": float(args.delta_pct)}]
    if not has_file:
        raise SystemExit("Informe --intervencoes arquivo.json ou --feature e --delta-pct.")
    return _load_intervencoes_json(args.intervencoes)


def _load_two_stage_model(path: Path) -> Any:
    real_main = sys.modules["__main__"]
    sys.modules["__main__"] = ep
    try:
        return joblib.load(path)
    finally:
        sys.modules["__main__"] = real_main


def predict_stages_por_plano(
    df_feat: pd.DataFrame,
    feature_cols: list[str],
    models_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    p_all = np.zeros(len(df_feat), dtype=float)
    y_all = np.zeros(len(df_feat), dtype=float)
    for plano in sorted(df_feat[ep.SEGMENT_COL].dropna().astype(str).unique()):
        mask = (df_feat[ep.SEGMENT_COL].astype(str) == plano).to_numpy()
        if not mask.any():
            continue
        slug = ep.plano_slug(plano)
        mpath = models_dir / f"model_{slug}.pkl"
        if not mpath.is_file():
            raise FileNotFoundError(f"Modelo ausente para plano={plano!r}: {mpath}")
        model = _load_two_stage_model(mpath)
        X = ep.ensure_no_object_dtype(df_feat.loc[mask, feature_cols].copy())
        p_pos, y_hat = model.predict_stages(X)
        p_all[mask] = np.asarray(p_pos, dtype=float).ravel()
        y_all[mask] = np.asarray(y_hat, dtype=float).ravel()
    return p_all, y_all


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="What-if mensal ELGIN")
    ap.add_argument("--versao", required=True, help="Pasta vN em data/processed/elgin/predict")
    ap.add_argument("--competencia", default=None, help="Competência YYYY-MM (default: última da base)")
    ap.add_argument("--intervencoes", type=Path, default=None)
    ap.add_argument("--feature", default=None)
    ap.add_argument("--delta-pct", type=float, default=None, metavar="PCT")
    ap.add_argument("--modo-idade", choices=("simples", "correlacionado"), default="simples")
    ap.add_argument("--skip-mlflow", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    intervencoes = _intervencoes_from_args(args)
    run_dir = ep.OUTPUT_PREDICT_ROOT / args.versao
    models_dir = run_dir / "models"
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Pasta de versão inexistente: {run_dir}")
    if not models_dir.is_dir():
        raise FileNotFoundError(f"Pasta de modelos inexistente: {models_dir}")

    fc = ep.load_feature_catalog()
    elegiveis = set(ep.catalog_eligible_names(fc))
    for i in intervencoes:
        f = str(i["feature"])
        if f not in elegiveis:
            raise ValueError(f"Feature não intervencionável pelo catálogo: {f}")

    df_raw = pd.read_parquet(ep.TRANSFORMED_PARQUET_PATH)
    prem_ok = pd.to_numeric(df_raw[ep.PREMIUM_COL], errors="coerce").fillna(0.0) > 0.0
    df_raw = df_raw.loc[prem_ok].copy().reset_index(drop=True)

    comp = pd.to_datetime(df_raw[ep.TIME_COL], errors="coerce")
    if args.competencia:
        comp_ref = pd.Period(args.competencia, freq="M")
    else:
        comp_ref = comp.max().to_period("M")
    comp_period = comp.dt.to_period("M")
    mask_comp = comp_period == comp_ref
    df_mes = df_raw.loc[mask_comp].copy().reset_index(drop=True)
    if df_mes.empty:
        raise ValueError(f"Nenhuma linha para competência {comp_ref}.")

    # IMPORTANTE: lags precisam do histórico completo. Calculamos features em toda base
    # e só depois filtramos a competência-alvo.
    df_feat_full = ep.build_features(df_raw)
    df_feat = df_feat_full.loc[mask_comp].copy().reset_index(drop=True)
    feature_cols = ep.resolve_feature_columns(df_feat_full, fc)
    fat_base = pd.to_numeric(df_mes[ep.PREMIUM_COL], errors="coerce").fillna(0.0).reset_index(drop=True)

    print(f"[Carga] competencia={comp_ref} | n_beneficiarios_mes={len(df_mes):,} | n_features={len(feature_cols)}")
    _, y_hat_antes = predict_stages_por_plano(df_feat, feature_cols, models_dir)
    sin_antes = ep.aggregate_sinistralidade_macro(y_hat_antes, fat_base)
    print(f"[Baseline] sinistralidade_antes = {sin_antes:.6f}")

    df_mod = df_mes.copy()
    mascaras: list[dict[str, Any]] = []
    idade_alterada = False
    urg_alterada = False
    delta_idade = 0.0
    mask_idade = pd.Series([False] * len(df_mod), index=df_mod.index)

    for interv in intervencoes:
        feat = str(interv["feature"])
        delta_pct = float(interv["delta_pct"])
        fator = 1.0 + delta_pct / 100.0
        mask = calcular_elegibilidade(df_mod, feat, delta_pct)

        mascaras.append(
            {
                "feature": feat,
                "delta_pct": delta_pct,
                "n_elegiveis": int(mask.sum()),
                "n_nao_elegiveis": int(len(df_mod) - int(mask.sum())),
            }
        )
        if feat in df_mod.columns:
            df_mod[feat] = pd.to_numeric(df_mod[feat], errors="coerce").fillna(0.0).astype(float)
            df_mod.loc[mask, feat] = (df_mod.loc[mask, feat] * fator).clip(lower=0.0)

        if feat == "idade":
            idade_alterada = True
            delta_idade = delta_pct
            mask_idade = mask
        if feat in {"qtd_carater_urgencia", "qtd_eventos_sinistro"}:
            urg_alterada = True

        print(
            f"[Intervenção] {feat} | delta={delta_pct:+.1f}% | "
            f"elegíveis={int(mask.sum()):,} | não elegíveis={int(len(df_mod)-int(mask.sum())):,}"
        )

    if urg_alterada and "pct_urgencia" in df_mod.columns:
        q_ev = pd.to_numeric(df_mod["qtd_eventos_sinistro"], errors="coerce").fillna(0.0)
        q_ur = pd.to_numeric(df_mod["qtd_carater_urgencia"], errors="coerce").fillna(0.0)
        df_mod["pct_urgencia"] = np.where(q_ev > 0, q_ur / q_ev, 0.0)
    if idade_alterada and args.modo_idade == "correlacionado":
        aplicar_correlacoes_idade(df_mod, mask_idade, delta_idade)

    # Reconstrói features do mês modificado preservando histórico para lags.
    df_hist = df_raw.loc[~mask_comp].copy()
    df_mod_full = pd.concat([df_hist, df_mod], axis=0, ignore_index=True)
    df_mod_feat_full = ep.build_features(df_mod_full)
    comp_mod = pd.to_datetime(df_mod_feat_full[ep.TIME_COL], errors="coerce").dt.to_period("M")
    df_mod_feat = df_mod_feat_full.loc[comp_mod == comp_ref].copy().reset_index(drop=True)
    missing = [c for c in feature_cols if c not in df_mod_feat.columns]
    if missing:
        raise ValueError(f"Colunas ausentes após intervenção: {missing[:20]}")

    _, y_hat_depois = predict_stages_por_plano(df_mod_feat, feature_cols, models_dir)
    sin_depois = ep.aggregate_sinistralidade_macro(y_hat_depois, fat_base)
    delta_abs = sin_depois - sin_antes
    delta_rel = (delta_abs / sin_antes * 100.0) if sin_antes != 0 else float("nan")

    resultado = {
        "competencia_referencia": str(comp_ref),
        "sinistralidade_antes": round(float(sin_antes), 6),
        "sinistralidade_depois": round(float(sin_depois), 6),
        "delta_absoluto": round(float(delta_abs), 8),
        "delta_relativo_pct": round(float(delta_rel), 4),
        "n_individuos_afetados": sum(m["n_elegiveis"] for m in mascaras),
        "n_individuos_nao_elegiveis": sum(m["n_nao_elegiveis"] for m in mascaras),
        "intervencoes": intervencoes,
        "modo_idade": args.modo_idade,
        "versao_modelo": args.versao,
    }

    out_dir = run_dir / "what_if_mensal"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "resultado_what_if.json").write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    pred = pd.DataFrame(
        {
            ep.BENEFICIARIO_COL: df_mes[ep.BENEFICIARIO_COL].values,
            ep.TIME_COL: df_mes[ep.TIME_COL].astype(str).values,
            ep.PREMIUM_COL: fat_base.values,
            "sinistralidade_prevista_antes": np.asarray(y_hat_antes, dtype=float),
            "sinistralidade_prevista_depois": np.asarray(y_hat_depois, dtype=float),
        }
    )
    pred["delta_individual"] = pred["sinistralidade_prevista_depois"] - pred["sinistralidade_prevista_antes"]
    mask_af = np.zeros(len(df_mes), dtype=bool)
    for m in mascaras:
        mask_af |= calcular_elegibilidade(df_mes, m["feature"], m["delta_pct"]).values
    pred["foi_afetado"] = mask_af.astype(int)
    pred.to_csv(out_dir / "predicoes_micro_depois.csv", index=False, encoding="utf-8-sig")

    relatorio = {
        "versao": args.versao,
        "competencia_referencia": str(comp_ref),
        "n_total_base_mes": int(len(df_mes)),
        "intervencoes": mascaras,
        "pct_base_afetada": round(float(mask_af.sum()) / max(1, len(df_mes)) * 100.0, 2),
        "modo_idade": args.modo_idade,
    }
    (out_dir / "relatorio_intervencao.json").write_text(
        json.dumps(relatorio, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[Disco] Artefatos gravados em {out_dir}")


if __name__ == "__main__":
    main()
