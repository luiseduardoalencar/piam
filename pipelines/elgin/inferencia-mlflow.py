"""
Inferência ELGIN via **último modelo registado** no MLflow (pyfunc).

**Não regista nada no MLflow:** usa ``configurar_mlflow(..., preparar_experimento=False)`` —
só URI/credenciais para ler o Registry e artefatos; não cria experimento nem run.

- Carrega ``models:/<nome>/latest`` (pyfunc), escolhe **aleatoriamente** um perfil entre os 100
  do ``catalogo_perfis_top100.json`` do run desse modelo (``runs:/...``, leitura).
- Modo predefinido: lista também artefatos do run de predição e o último run de
  ``piam-elgin-feature-impact`` (só leitura / preview em memória).
- ``--somente-inferencia``: apenas carrega o modelo e executa a predição (sem listagens extra).
- ``--carteira N`` (ex.: ``7``): ``N`` “utilizadores” com perfis **aleatórios** entre os do
  ``catalogo_perfis_top100.json`` do run do modelo; pesos **aleatórios** que somam **100%**
  (Dirichlet); só inferência — **não treina, não regista** nada (igual ao resto do script).

Requer ``.env`` como em ``config/mlflow_config.py``.

Uso (na raiz do repositório)::

    python pipelines/elgin/inferencia-mlflow.py --somente-inferencia
    python pipelines/elgin/inferencia-mlflow.py --somente-inferencia --carteira 7 --seed 42
    python pipelines/elgin/inferencia-mlflow.py
    python pipelines/elgin/inferencia-mlflow.py --seed 42
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "pipelines" / "elgin"))
sys.path.insert(0, str(ROOT_DIR))

import predict as ep  # noqa: E402

EXPERIMENT_FEATURE_IMPACT = "piam-elgin-feature-impact"
ARTIFACT_CATALOG_REL = "elgin_artifacts/catalogo_perfis_top100.json"
ARTIFACT_FEATURE_CSV_REL = "feature_impact/feature_correlation_sinistralidade.csv"


@contextlib.contextmanager
def _predict_module_as_main_for_joblib() -> Iterator[None]:
    real_main = sys.modules["__main__"]
    sys.modules["__main__"] = ep
    try:
        yield
    finally:
        sys.modules["__main__"] = real_main


def _load_predict_agregado():
    path = ROOT_DIR / "pipelines" / "elgin" / "predict-agregado.py"
    spec = importlib.util.spec_from_file_location("predict_agregado_dyn", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Nao foi possivel carregar {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _feature_names_from_pyfunc(model: Any) -> list[str]:
    meta = model.metadata
    if meta is None or meta.signature is None or meta.signature.inputs is None:
        raise ValueError("Modelo sem signature de entrada.")
    from mlflow.types.schema import ColSpec, TensorSpec

    names: list[str] = []
    for col in meta.signature.inputs:
        if isinstance(col, ColSpec) and col.name:
            names.append(str(col.name))
        elif isinstance(col, TensorSpec):
            raise ValueError("Entrada tensor nao suportada.")
    if not names:
        raise ValueError("Signature sem colunas nomeadas.")
    return names


def _prepare_X_mlflow(df_feat: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    missing = [c for c in feature_names if c not in df_feat.columns]
    if missing:
        raise ValueError("Faltam colunas apos engenharia.\n" f"Exemplos: {missing[:15]}")
    X = df_feat[feature_names].copy()
    X = ep.ensure_no_object_dtype(X)
    return ep._mlflow_prepare_input_example(X)


def _artifact_uri_run(run_id: str, rel_path: str) -> str:
    rel_path = rel_path.replace("\\", "/").lstrip("/")
    return f"runs:/{run_id}/{rel_path}"


def _load_text_artifact(run_id: str, rel_path: str) -> str:
    uri = _artifact_uri_run(run_id, rel_path)
    try:
        import mlflow.artifacts

        return mlflow.artifacts.load_text(uri)
    except Exception:
        import tempfile

        client = MlflowClient()
        with tempfile.TemporaryDirectory() as tmp:
            local = client.download_artifacts(run_id, rel_path, dst_path=tmp)
            return Path(local).read_text(encoding="utf-8-sig")


def _print_artifact_tree(client: MlflowClient, run_id: str, path: str = "") -> None:
    """Lista artefatos (nome, tamanho, dir) sem download para pasta do utilizador."""

    try:
        entries = client.list_artifacts(run_id, path if path else None)
    except Exception as e:
        print(f"  [erro ao listar path={path!r}] {e}")
        return
    for f in sorted(entries, key=lambda x: x.path):
        sub = f"{path}/{f.path}" if path else f.path
        uri = _artifact_uri_run(run_id, sub)
        if getattr(f, "is_dir", False):
            print(f"  [dir ] {uri}/")
            _print_artifact_tree(client, run_id, sub)
        else:
            sz = getattr(f, "file_size", None)
            sz_s = f"{sz} B" if sz is not None else "?"
            print(f"  [file] {uri}  ({sz_s})")


def _latest_model_version(client: MlflowClient, registered_name: str) -> Any:
    versions = client.search_model_versions(f"name={repr(registered_name)}")
    if not versions:
        raise FileNotFoundError(
            f"Nenhuma versao registada para {registered_name!r}. Registe um modelo (predict / predict-agregado)."
        )
    return max(versions, key=lambda v: int(v.version))


def _pipeline_variant_from_run(client: MlflowClient, run_id: str) -> str:
    run = client.get_run(run_id)
    return (run.data.params or {}).get("pipeline_variant", "") or ""


def _df_raw_for_profile(
    *,
    pipeline_variant: str,
    indice_parquet: int,
    escolhido: dict[str, Any],
) -> pd.DataFrame:
    if pipeline_variant.strip() == "agregado_por_beneficiario":
        pa = _load_predict_agregado()
        df_raw_p = pd.read_parquet(ep.TRANSFORMED_PARQUET_PATH)
        df_raw_p[ep.SEGMENT_COL] = df_raw_p[ep.SEGMENT_COL].replace(ep.PLANO_MAP)
        df_agg = pa.aggregate_panel_by_beneficiary(df_raw_p)
        prem_ok = pd.to_numeric(df_agg[ep.PREMIUM_COL], errors="coerce").fillna(0.0) > 0.0
        df_agg = df_agg.loc[prem_ok].copy().reset_index(drop=True)
        if indice_parquet < 0 or indice_parquet >= len(df_agg):
            raise IndexError(
                f"indice_parquet={indice_parquet} fora do intervalo [0, {len(df_agg) - 1}] (agregado)."
            )
        df_linha = df_agg.iloc[[indice_parquet]].copy()
    else:
        df_full = pd.read_parquet(ep.TRANSFORMED_PARQUET_PATH)
        df_full[ep.SEGMENT_COL] = df_full[ep.SEGMENT_COL].replace(ep.PLANO_MAP)
        prem_ok = (
            pd.to_numeric(df_full[ep.PREMIUM_COL], errors="coerce").fillna(0.0) > 0.0
        )
        df_full = df_full.loc[prem_ok].copy().reset_index(drop=True)
        if indice_parquet < 0 or indice_parquet >= len(df_full):
            raise IndexError(
                f"indice_parquet={indice_parquet} fora do intervalo [0, {len(df_full) - 1}] "
                f"(mensal, base filtrada prêmio > 0)."
            )
        df_linha = df_full.iloc[[indice_parquet]].copy()

    d = df_linha.copy()
    if ep.TARGET_COL in d.columns:
        d = d.drop(columns=[ep.TARGET_COL])
    return d


def _build_features_for_variant(pipeline_variant: str, df_raw: pd.DataFrame) -> pd.DataFrame:
    if pipeline_variant.strip() == "agregado_por_beneficiario":
        pa = _load_predict_agregado()
        return pa.build_features_agregado(df_raw)
    return ep.build_features(df_raw)


def _infer_one_profile(
    model: Any,
    feat_names: list[str],
    pipeline_variant: str,
    escolhido: dict[str, Any],
) -> tuple[float, float]:
    """Uma linha → sinistralidade_prevista, p_sinistro (usa o pyfunc já carregado)."""
    indice_parquet = int(escolhido["indice_parquet"])
    df_raw_use = _df_raw_for_profile(
        pipeline_variant=pipeline_variant,
        indice_parquet=indice_parquet,
        escolhido=escolhido,
    )
    df_feat = _build_features_for_variant(pipeline_variant, df_raw_use)
    for c in feat_names:
        if c not in df_feat.columns:
            df_feat[c] = 0.0
    fc = ep.load_feature_catalog()
    _ = ep.resolve_feature_columns(df_feat, fc)
    X = _prepare_X_mlflow(df_feat, feat_names)
    out_df = model.predict(X)
    if not isinstance(out_df, pd.DataFrame):
        raise TypeError(f"Saida pyfunc inesperada: {type(out_df)}")
    return (
        float(out_df["sinistralidade_prevista"].iloc[0]),
        float(out_df["p_sinistro"].iloc[0]),
    )


def _sample_profiles(perfis: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    """N perfis distintos se possível; senão repete (choices)."""
    if n <= 0:
        raise ValueError("n deve ser >= 1")
    if len(perfis) >= n:
        return random.sample(perfis, k=n)
    return [random.choice(perfis) for _ in range(n)]


def _pesos_aleatorios_soma_um(n: int, rng: np.random.Generator) -> list[float]:
    """Vetor aleatório estritamente positivo com soma 1 (partição de 100%)."""
    if n <= 0:
        raise ValueError("n deve ser >= 1")
    w = rng.dirichlet(np.ones(n)).tolist()
    s = float(sum(w))
    return [float(x / s) for x in w]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inferencia no ultimo modelo MLflow + listagem de artefatos (predict + feature-impact)"
    )
    parser.add_argument(
        "--registered-model",
        default=ep.MLFLOW_REGISTERED_MODEL_NAME,
        help="Nome no Model Registry",
    )
    parser.add_argument("--model-version", type=int, default=None, help="Versao fixa; omitir = latest")
    parser.add_argument("--seed", type=int, default=None, help="Semente para perfil aleatorio")
    parser.add_argument(
        "--somente-inferencia",
        action="store_true",
        help="So carrega o modelo e executa predicao (sem listar artefatos nem feature-impact).",
    )
    parser.add_argument(
        "--carteira",
        type=int,
        default=None,
        metavar="N",
        help=(
            "N utilizadores: perfis aleatorios do top100 do catalogo do run + pesos aleatorios "
            "somando 100%%. Implica leitura MLflow/registry; nao treina nem regista."
        ),
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    from config.mlflow_config import configurar_mlflow

    configurar_mlflow(ep.MLFLOW_EXPERIMENT_NAME, preparar_experimento=False)
    client = MlflowClient()

    if args.model_version is not None:
        mv = client.get_model_version(args.registered_model, str(args.model_version))
    else:
        mv = _latest_model_version(client, args.registered_model)

    run_id_predict = mv.run_id
    model_uri = f"models:/{args.registered_model}/{mv.version}"

    listagens = not args.somente_inferencia and args.carteira is None
    run_id_fi: str | None = None
    if listagens:
        print("\n" + "=" * 60)
        print("[Modelo registado]")
        print("=" * 60)
        print(f"  name            : {args.registered_model}")
        print(f"  version         : {mv.version}")
        print(f"  run_id (treino) : {run_id_predict}")
        print(f"  model URI       : {model_uri}")
        print(f"  artefact source : {getattr(mv, 'source', '(n/d)')}")

        print("\n" + "=" * 60)
        print(f"[Artefatos do run de predicao] (listagem; URI runs:/{run_id_predict}/...)")
        print("=" * 60)
        _print_artifact_tree(client, run_id_predict, "")

        print("\n" + "=" * 60)
        print(f"[Experimento] {EXPERIMENT_FEATURE_IMPACT} — ultimo run")
        print("=" * 60)
        exp_fi = client.get_experiment_by_name(EXPERIMENT_FEATURE_IMPACT)
        if exp_fi is None:
            print(f"  Experimento {EXPERIMENT_FEATURE_IMPACT!r} nao encontrado no tracking.")
        else:
            runs = client.search_runs(
                experiment_ids=[exp_fi.experiment_id],
                order_by=["start_time DESC"],
                max_results=1,
            )
            if not runs:
                print("  Nenhum run neste experimento.")
            else:
                run_id_fi = runs[0].info.run_id
                print(f"  run_id : {run_id_fi}")
                _print_artifact_tree(client, run_id_fi, "")

                csv_uri = _artifact_uri_run(run_id_fi, ARTIFACT_FEATURE_CSV_REL)
                print(f"\n  CSV esperado (feature-impact): {csv_uri}")
                try:
                    txt = _load_text_artifact(run_id_fi, ARTIFACT_FEATURE_CSV_REL)
                    lines = txt.splitlines()
                    preview = "\n".join(lines[:15])
                    print("  [preview primeiras linhas, em memoria]\n")
                    print(preview)
                    if len(lines) > 15:
                        print(f"  ... ({len(lines) - 15} linhas a mais)")
                except Exception as e:
                    print(f"  [aviso] Nao foi possivel ler o CSV: {e}")

        print("\n" + "=" * 60)
        print("[Catalogo top100 do run de predicao] -> perfil aleatorio")
        print("=" * 60)
    catalog_uri = _artifact_uri_run(run_id_predict, ARTIFACT_CATALOG_REL)
    if listagens:
        print(f"  URI: {catalog_uri}")
    try:
        cat_text = _load_text_artifact(run_id_predict, ARTIFACT_CATALOG_REL)
    except Exception as e:
        raise FileNotFoundError(
            f"Artefato {ARTIFACT_CATALOG_REL!r} inexistente ou inacessivel no run {run_id_predict}. {e}"
        ) from e

    catalog = json.loads(cat_text)
    perfis: list[dict[str, Any]] = catalog.get("perfis") or []
    if not perfis:
        raise ValueError("Lista perfis vazia no catalogo JSON.")
    pipeline_variant = _pipeline_variant_from_run(client, run_id_predict)

    try:
        import mlflow.pyfunc
    except ImportError as e:
        raise ImportError("pip install mlflow") from e

    with _predict_module_as_main_for_joblib():
        model = mlflow.pyfunc.load_model(model_uri)

    feat_names = _feature_names_from_pyfunc(model)

    if args.carteira is not None:
        n = int(args.carteira)
        if n < 1:
            raise ValueError("--carteira N requer N >= 1")
        escolhidos = _sample_profiles(perfis, n)
        pesos = _pesos_aleatorios_soma_um(n, rng)
        usuarios: list[dict[str, Any]] = []
        sins: list[float] = []
        ps: list[float] = []
        for i, esc in enumerate(escolhidos):
            sin_i, p_i = _infer_one_profile(model, feat_names, pipeline_variant, esc)
            sins.append(sin_i)
            ps.append(p_i)
            usuarios.append(
                {
                    "i": i,
                    "peso": pesos[i],
                    "peso_pct": round(100.0 * pesos[i], 6),
                    "perfil_catalogo": {
                        "rank": esc.get("rank"),
                        "label": esc.get("label"),
                        "indice_parquet": esc.get("indice_parquet"),
                        "resumo": esc.get("resumo"),
                    },
                    "sinistralidade_prevista": sin_i,
                    "p_sinistro": p_i,
                }
            )
        sin_pond = float(sum(pesos[j] * sins[j] for j in range(n)))
        p_pond = float(sum(pesos[j] * ps[j] for j in range(n)))
        resultado = {
            "modo": "carteira_n_usuarios",
            "n_usuarios": n,
            "pesos_somam": float(sum(pesos)),
            "nota_pesos": "Pesos aleatorios (Dirichlet) somando 100%; perfis aleatorios do catalogo top100 deste run.",
            "sinistralidade_prevista_media_ponderada": sin_pond,
            "p_sinistro_medio_ponderado": p_pond,
            "usuarios": usuarios,
            "model_uri": model_uri,
            "registered_model": args.registered_model,
            "model_version": int(mv.version),
            "run_id_predict": run_id_predict,
            "pipeline_variant": pipeline_variant,
            "catalogo_artifact_uri": catalog_uri,
        }
    else:
        escolhido = random.choice(perfis)
        indice_parquet = int(escolhido["indice_parquet"])
        if listagens:
            print(f"  pipeline_variant (param do run): {pipeline_variant!r}")
            print(f"  perfil escolhido (rank/label): {escolhido.get('rank')} | {escolhido.get('label')}")
            print(f"  indice_parquet: {indice_parquet}")

        sin_pred, p_s = _infer_one_profile(model, feat_names, pipeline_variant, escolhido)

        resultado = {
            "sinistralidade_prevista": sin_pred,
            "p_sinistro": p_s,
            "model_uri": model_uri,
            "registered_model": args.registered_model,
            "model_version": int(mv.version),
            "run_id_predict": run_id_predict,
            "pipeline_variant": pipeline_variant,
            "perfil_catalogo": {
                "rank": escolhido.get("rank"),
                "label": escolhido.get("label"),
                "indice_parquet": escolhido.get("indice_parquet"),
                "resumo": escolhido.get("resumo"),
            },
            "catalogo_artifact_uri": catalog_uri,
            "feature_impact_run_id": run_id_fi,
            "feature_impact_csv_uri": _artifact_uri_run(run_id_fi, ARTIFACT_FEATURE_CSV_REL)
            if run_id_fi
            else None,
        }

    quiet = args.somente_inferencia or args.carteira is not None
    if quiet:
        print(json.dumps(resultado, indent=2, ensure_ascii=False))
    else:
        print("\n" + "=" * 60)
        print("[Resultado inferencia]")
        print("=" * 60)
        print(json.dumps(resultado, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
