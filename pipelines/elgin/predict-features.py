# %%
"""
What-If Engine — Simulador de Intervenção em Features.
Sinistralidade ELGIN (agregado por beneficiário).

Execução: ``python pipelines/elgin/predict-features.py --help`` ou blocos ``# %%`` em ordem.
Bloco MLflow ao final — executar apenas após todos os artefatos gravados em disco.

Carrega modelos ``model_<plano>.pkl`` da pasta ``--versao`` (vN) e aplica intervenções
via ``--intervencoes`` (JSON) **ou** ``--feature`` + ``--delta-pct`` (uma intervenção).

**Git Bash / bash:** não use ``echo [...] > f.json`` — o ``[`` é interpretado pelo
comando ``test``. Prefira ``--feature``/``--delta-pct`` ou::

    printf '%s' '[{"feature":"qtd_servico_CARDIOLOGIA","delta_pct":20}]' > tmp_interv.json
"""

from __future__ import annotations

import argparse
import importlib.util
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

from what_if.correlacoes import aplicar_correlacoes_idade
from what_if.elegibilidade import calcular_elegibilidade

_PA_PATH = ROOT_DIR / "pipelines" / "elgin" / "predict-agregado.py"


def _load_predict_agregado():
    spec = importlib.util.spec_from_file_location("predict_agregado_what_if", _PA_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Não foi possível carregar {_PA_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pa = _load_predict_agregado()

FEATURES_BLOQUEADAS: frozenset[str] = frozenset(
    {
        pa.PREMIUM_COL,
        pa.TARGET_COL,
        "sinistralidade_raw",
        "valor_sinistro_raw",
        "valor_sinistro_alt_val",
        "valor_sinistro_ajustado",
        "sin_ref",
        "fator_ajuste_m",
        "S_real_m",
        "F_real_m",
        pa.BENEFICIARIO_COL,
        pa.TIME_COL,
    }
)


# joblib de um TwoStageModel treinado costuma ser >> 1 KiB; ficheiros de poucos bytes são inválidos.
_MIN_PKL_BYTES = 512


def _load_two_stage_model(path: Path) -> Any:
    sz = path.stat().st_size
    if sz < _MIN_PKL_BYTES:
        raise OSError(
            f"Modelo inválido ou truncado ({sz} bytes < {_MIN_PKL_BYTES}): {path}\n"
            "Volte a correr o treino agregado para gerar .pkl válidos:\n"
            "  python pipelines/elgin/predict-agregado.py\n"
            "ou use outra pasta --versao em data/processed/elgin/predict/ com models/model_*.pkl íntegros."
        )
    real_main = sys.modules["__main__"]
    sys.modules["__main__"] = pa
    try:
        return joblib.load(path)
    finally:
        sys.modules["__main__"] = real_main


def predict_stages_agregado_por_plano(
    df_feat: pd.DataFrame,
    feature_cols: list[str],
    models_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    p_all = np.zeros(len(df_feat), dtype=float)
    y_all = np.zeros(len(df_feat), dtype=float)
    for plano in sorted(df_feat[pa.SEGMENT_COL].dropna().astype(str).unique()):
        mask = (df_feat[pa.SEGMENT_COL].astype(str) == plano).to_numpy()
        if not mask.any():
            continue
        slug = pa.plano_slug(plano)
        mpath = models_dir / f"model_{slug}.pkl"
        if not mpath.is_file():
            raise FileNotFoundError(
                f"Modelo em falta para plano={plano!r} (slug={slug}): {mpath}"
            )
        model = _load_two_stage_model(mpath)
        X = pa.ensure_no_object_dtype(df_feat.loc[mask, feature_cols].copy())
        p_pos, y_hat = model.predict_stages(X)
        p_all[mask] = np.asarray(p_pos, dtype=float).ravel()
        y_all[mask] = np.asarray(y_hat, dtype=float).ravel()
    return p_all, y_all


def _load_intervencoes_json(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8-sig")
    text = raw.strip()
    if not text:
        raise ValueError(f"Ficheiro vazio: {path}")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        preview = text[:160].replace("\r", "\\r").replace("\n", "\\n")
        raise ValueError(
            f"JSON inválido em {path}: {e}\n"
            f"Primeiros caracteres: {preview!r}\n"
            "Dica: no Git Bash não use echo com [ ]; use --feature e --delta-pct, ou "
            "printf com aspas simples em volta do JSON."
        ) from e
    if not isinstance(data, list):
        raise ValueError("O JSON deve ser uma lista, ex.: [{\"feature\": \"...\", \"delta_pct\": 20}]")
    return data


def _intervencoes_from_args(args: argparse.Namespace) -> list[dict[str, Any]]:
    has_file = args.intervencoes is not None
    has_cli = args.feature is not None or args.delta_pct is not None
    if has_file and has_cli:
        raise SystemExit("Use apenas --intervencoes OU (--feature e --delta-pct), não ambos.")
    if has_cli:
        if args.feature is None or args.delta_pct is None:
            raise SystemExit("--feature e --delta-pct são obrigatórios em conjunto.")
        return [{"feature": args.feature, "delta_pct": float(args.delta_pct)}]
    if not has_file:
        raise SystemExit(
            "Indique --intervencoes ficheiro.json ou --feature e --delta-pct "
            "(ex.: --feature qtd_servico_CARDIOLOGIA --delta-pct 20)."
        )
    return _load_intervencoes_json(args.intervencoes)


def _validar_intervencoes(
    intervencoes: list[dict[str, Any]],
    fc: pd.DataFrame,
) -> None:
    eligible = set(pa.catalog_eligible_names(fc))
    for interv in intervencoes:
        feat = interv["feature"]
        if feat in FEATURES_BLOQUEADAS:
            raise ValueError(f"Feature bloqueada para intervenção: {feat!r}")
        if feat not in eligible:
            raise ValueError(
                f"Feature não intervencionável (catálogo / tipo): {feat!r}"
            )


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="What-if ELGIN — intervenções em features agregadas")
    ap.add_argument(
        "--versao",
        type=str,
        required=True,
        help="Pasta vN em data/processed/elgin/predict (ex.: v3) com models/model_*.pkl",
    )
    ap.add_argument(
        "--intervencoes",
        type=Path,
        default=None,
        help='JSON: lista de {"feature": "...", "delta_pct": number}',
    )
    ap.add_argument(
        "--feature",
        default=None,
        help="Uma feature alvo (alternativa a --intervencoes; requer --delta-pct)",
    )
    ap.add_argument(
        "--delta-pct",
        type=float,
        default=None,
        metavar="PCT",
        help="Delta em %% para --feature (ex.: 20 para +20%%)",
    )
    ap.add_argument(
        "--modo-idade",
        choices=("simples", "correlacionado"),
        default="simples",
        help="Só relevante se houver intervenção em idade",
    )
    ap.add_argument(
        "--skip-mlflow",
        action="store_true",
        help="Não executar o bloco de registo MLflow (Passo 8)",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    modo_idade: str = args.modo_idade

    intervencoes: list[dict[str, Any]] = _intervencoes_from_args(args)
    if not intervencoes:
        raise ValueError("Lista INTERVENCOES vazia.")

    run_dir = pa.OUTPUT_PREDICT_ROOT / args.versao
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Pasta de versão inexistente: {run_dir}")
    models_dir = run_dir / "models"
    if not models_dir.is_dir():
        raise FileNotFoundError(f"Pasta de modelos inexistente: {models_dir}")

    version_label = args.versao.strip()

    fc = pa.load_feature_catalog()
    _validar_intervencoes(intervencoes, fc)

    # %%
    # [Passo 2 — Carregamento da Base]
    df_raw = pd.read_parquet(pa.TRANSFORMED_PARQUET_PATH)
    df_raw[pa.SEGMENT_COL] = df_raw[pa.SEGMENT_COL].replace(pa.PLANO_MAP)
    df_agg = pa.aggregate_panel_by_beneficiary(df_raw)

    prem_ok = pd.to_numeric(df_agg[pa.PREMIUM_COL], errors="coerce").fillna(0.0) > 0.0
    df_agg = df_agg.loc[prem_ok].copy().reset_index(drop=True)

    df_feat = pa.build_features_agregado(df_agg)
    feature_cols = pa.resolve_feature_columns(df_feat, fc)
    feature_cols = [c for c in feature_cols if c not in {pa.N_MESES_COL}]

    fat_base = df_agg[pa.PREMIUM_COL].astype(float).reset_index(drop=True)

    print(
        f"[Carga] n_beneficiarios={len(df_agg):,} | n_features={len(feature_cols)} | "
        f"versao={version_label}"
    )

    # %%
    # [Passo 3 — Baseline]
    _, y_hat_antes = predict_stages_agregado_por_plano(df_feat, feature_cols, models_dir)
    sin_antes = pa.aggregate_sinistralidade_macro(y_hat_antes, fat_base)
    print(f"[Baseline] sinistralidade_antes = {sin_antes:.6f}")

    # %%
    # [Passo 4 — Intervenções]
    df_mod_agg = df_agg.copy()
    mascaras_por_intervencao: list[dict[str, Any]] = []
    idade_alterada = False
    urgencia_alterada = False
    delta_pct_idade = 0.0
    mask_idade = pd.Series([False] * len(df_mod_agg), index=df_mod_agg.index)

    for interv in intervencoes:
        feature_alvo = interv["feature"]
        delta_pct = float(interv["delta_pct"])
        delta_fator = 1.0 + delta_pct / 100.0

        mask = calcular_elegibilidade(df_mod_agg, feature_alvo, delta_pct)
        n_elegiveis = int(mask.sum())
        n_nao_elegiveis = len(df_mod_agg) - n_elegiveis

        mascaras_por_intervencao.append(
            {
                "feature": feature_alvo,
                "delta_pct": delta_pct,
                "n_elegiveis": n_elegiveis,
                "n_nao_elegiveis": n_nao_elegiveis,
            }
        )

        if feature_alvo in df_mod_agg.columns:
            # Evita erro/future-warning ao atribuir float em colunas inteiras (Int64/int64).
            df_mod_agg[feature_alvo] = pd.to_numeric(
                df_mod_agg[feature_alvo], errors="coerce"
            ).fillna(0.0).astype(float)
            df_mod_agg.loc[mask, feature_alvo] = (
                df_mod_agg.loc[mask, feature_alvo] * delta_fator
            ).clip(lower=0.0)
        else:
            print(
                f"[aviso] Coluna {feature_alvo!r} inexistente em df_agg; "
                "nenhum valor foi alterado (verifique nomes no Parquet agregado)."
            )

        if feature_alvo == "idade":
            idade_alterada = True
            delta_pct_idade = delta_pct
            mask_idade = mask

        if feature_alvo in {"qtd_carater_urgencia", "qtd_eventos_sinistro"}:
            urgencia_alterada = True

        print(
            f"[Intervenção] {feature_alvo} | delta={delta_pct:+.1f}% | "
            f"elegíveis={n_elegiveis:,} | não elegíveis={n_nao_elegiveis:,}"
        )

    # %%
    # [Passo 5 — Derivadas pós-loop]
    if urgencia_alterada and "pct_urgencia" in df_mod_agg.columns:
        q_ev = pd.to_numeric(df_mod_agg["qtd_eventos_sinistro"], errors="coerce").fillna(0.0)
        q_ur = pd.to_numeric(df_mod_agg["qtd_carater_urgencia"], errors="coerce").fillna(0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            df_mod_agg["pct_urgencia"] = np.where(q_ev > 0, q_ur / q_ev, 0.0)
        print("[Reconstrução] pct_urgencia recalculada em df_agg.")

    if idade_alterada and modo_idade == "correlacionado":
        aplicar_correlacoes_idade(df_mod_agg, mask_idade, delta_pct_idade)
        print("[Reconstrução] Correlações de envelhecimento aplicadas em df_agg.")

    df_mod_feat = pa.build_features_agregado(df_mod_agg)
    missing = [c for c in feature_cols if c not in df_mod_feat.columns]
    if missing:
        raise ValueError(f"Colunas em falta após rebuild: {missing[:20]}")

    # %%
    # [Passo 6 — Inferência modificada]
    _, y_hat_depois = predict_stages_agregado_por_plano(
        df_mod_feat, feature_cols, models_dir
    )
    sin_depois = pa.aggregate_sinistralidade_macro(y_hat_depois, fat_base)

    delta_abs = sin_depois - sin_antes
    delta_rel_pct = (delta_abs / sin_antes * 100.0) if sin_antes != 0 else float("nan")

    premio_tot = float(pd.to_numeric(fat_base, errors="coerce").fillna(0.0).sum())
    sum_y_antes = float(np.asarray(y_hat_antes, dtype=float).sum())
    sum_y_depois = float(np.asarray(y_hat_depois, dtype=float).sum())
    delta_sum_y = sum_y_depois - sum_y_antes

    resultado: dict[str, Any] = {
        "sinistralidade_antes": round(float(sin_antes), 6),
        "sinistralidade_depois": round(float(sin_depois), 6),
        "delta_absoluto": round(float(delta_abs), 6),
        "delta_relativo_pct": round(float(delta_rel_pct), 4),
        "premio_total_carteira": round(premio_tot, 2),
        "sum_sinistralidade_prevista_antes": round(sum_y_antes, 4),
        "sum_sinistralidade_prevista_depois": round(sum_y_depois, 4),
        "delta_sum_sinistralidade_prevista": round(delta_sum_y, 4),
        "n_individuos_afetados": sum(m["n_elegiveis"] for m in mascaras_por_intervencao),
        "n_individuos_nao_elegiveis": sum(m["n_nao_elegiveis"] for m in mascaras_por_intervencao),
        "intervencoes": intervencoes,
        "modo_idade": modo_idade,
        "versao_modelo": version_label,
    }

    print("\n" + "=" * 60)
    print("[Resultado]")
    print(f"  sinistralidade_antes  = {sin_antes:.6f}")
    print(f"  sinistralidade_depois = {sin_depois:.6f}")
    print(
        f"  delta_absoluto (índice)= {delta_abs:+.10f}  (macro = sum(y_hat*prêmio)/sum(prêmio))"
    )
    print(f"  delta_relativo_pct     = {delta_rel_pct:+.4f}%")
    print(
        f"  [Somas] sum(prêmio)={premio_tot:,.2f} | sum(y_hat) antes={sum_y_antes:,.4f} | "
        f"depois={sum_y_depois:,.4f} | delta_sum(y_hat)={delta_sum_y:+.4f}"
    )
    print(
        "  Nota: com prêmio total muito grande, um delta no numerador pode mudar o índice só "
        "na 5.ª–8.ª casa decimal; ver predicoes_micro_depois.csv (delta_individual, foi_afetado)."
    )
    print("=" * 60)

    # %%
    # [Passo 7 — Disco]
    what_if_dir = run_dir / "what_if"
    what_if_dir.mkdir(parents=True, exist_ok=True)

    (what_if_dir / "resultado_what_if.json").write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    _n = len(df_agg)
    pred_antes = pd.DataFrame(
        {
            pa.BENEFICIARIO_COL: df_agg[pa.BENEFICIARIO_COL].values,
            pa.PREMIUM_COL: fat_base.values,
            "sinistralidade_prevista_antes": np.asarray(y_hat_antes, dtype=float),
        }
    )
    pred_antes.to_csv(
        what_if_dir / "predicoes_micro_antes.csv",
        index=False,
        encoding="utf-8-sig",
    )

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
        index=False,
        encoding="utf-8-sig",
    )

    relatorio = {
        "versao": version_label,
        "n_total_base": _n,
        "intervencoes": mascaras_por_intervencao,
        "pct_base_afetada": round(float(mask_afetados.sum()) / _n * 100, 2),
        "modo_idade": modo_idade,
    }
    (what_if_dir / "relatorio_intervencao.json").write_text(
        json.dumps(relatorio, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[Disco] Artefatos gravados em {what_if_dir}")

    # %%
    # [Passo 8 — MLflow]
    if not args.skip_mlflow:
        _what_if_mlflow_log(
            run_dir=run_dir,
            version_label=version_label,
            resultado=resultado,
            relatorio=relatorio,
            intervencoes=intervencoes,
            modo_idade=modo_idade,
        )


def _what_if_mlflow_log(
    *,
    run_dir: Path,
    version_label: str,
    resultado: dict,
    relatorio: dict,
    intervencoes: list[dict[str, Any]],
    modo_idade: str,
) -> None:
    try:
        import mlflow
    except ImportError:
        print("[MLflow] Pacote não instalado; ignorando registro.")
        return

    sys.path.insert(0, str(ROOT_DIR))
    try:
        from config.mlflow_config import configurar_mlflow
    except ImportError as e:
        print(f"[MLflow] config.mlflow_config indisponível: {e}")
        return

    try:
        configurar_mlflow(pa.MLFLOW_EXPERIMENT_NAME, preparar_experimento=True)
    except EnvironmentError as e:
        print(
            f"[MLflow] Configuração incompleta: {e}\n[MLflow] Artefatos em disco OK; sem tracking."
        )
        return

    run_name = f"elgin__what_if__{version_label}"
    what_if_dir_local = run_dir / "what_if"

    if mlflow.active_run() is not None:
        mlflow.end_run()

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("pipeline", "elgin_what_if")
        mlflow.log_param("versao_pasta", version_label)
        mlflow.log_param("n_intervencoes", len(intervencoes))
        mlflow.log_param("modo_idade", modo_idade)
        for i, interv in enumerate(intervencoes):
            mlflow.log_param(f"interv_{i}_feature", interv["feature"])
            mlflow.log_param(f"interv_{i}_delta_pct", interv["delta_pct"])

        mlflow.log_metric("sinistralidade_antes", resultado["sinistralidade_antes"])
        mlflow.log_metric("sinistralidade_depois", resultado["sinistralidade_depois"])
        mlflow.log_metric("delta_absoluto", resultado["delta_absoluto"])
        mlflow.log_metric("delta_relativo_pct", resultado["delta_relativo_pct"])
        mlflow.log_metric("n_individuos_afetados", resultado["n_individuos_afetados"])
        mlflow.log_metric(
            "n_individuos_nao_elegiveis", resultado["n_individuos_nao_elegiveis"]
        )

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


if __name__ == "__main__":
    main()

# %%
