"""
Inferência ELGIN via modelo registado no MLflow (pyfunc).

Toda a descoberta de runs, versões e caminhos de artefatos faz-se através da API do
MLflow (``MlflowClient`` + listagens); não há ``run_id`` nem caminhos de artefatos
fixos no código. Nomes de modelo e de experimentos vêm de variáveis de ambiente
(ver ``.env`` / ``config/mlflow_config.py``) ou de argumentos CLI.

- Carrega ``models:/<nome>/<versão>`` (pyfunc), resolve o run de origem e localiza
  o catálogo de perfis por pesquisa recursiva nos artefatos do run.
- Escolhe aleatoriamente um perfil entre os listados no JSON do catálogo.
- ``--somente-inferencia``: só carrega o modelo e executa a predição.
- ``--carteira N``: N perfis aleatórios do catálogo e pesos Dirichlet somando 100%%.

Requer credenciais/URI como em ``config/mlflow_config.py`` (``.env``).

Uso (na raiz do repositório)::

    set MLFLOW_REGISTERED_MODEL_NAME=elgin-sinistralidade-two-stage
    python pipelines/elgin/inferencia-mlflow.py --somente-inferencia
    python pipelines/elgin/inferencia-mlflow.py --somente-inferencia --carteira 7 --seed 42
    python pipelines/elgin/inferencia-mlflow.py
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
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


def _registered_model_name(cli_value: str | None) -> str:
    name = (cli_value or "").strip() or (os.getenv("MLFLOW_REGISTERED_MODEL_NAME") or "").strip()
    if not name:
        raise SystemExit(
            "Defina o modelo registado: argumento --registered-model ou "
            "variável de ambiente MLFLOW_REGISTERED_MODEL_NAME."
        )
    return name


def _experiment_feature_impact_name(cli_value: str | None) -> str | None:
    v = (cli_value or "").strip() or (os.getenv("MLFLOW_EXPERIMENT_FEATURE_IMPACT") or "").strip()
    return v or None


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


def _find_artifact_rel_path(
    client: MlflowClient,
    run_id: str,
    *,
    basename: str,
) -> str | None:
    """Procura ``basename`` nos artefatos do run (pesquisa recursiva via API)."""

    def walk(path: str | None) -> str | None:
        try:
            entries = client.list_artifacts(run_id, path if path else None)
        except Exception:
            return None
        for f in entries:
            sub = f"{path}/{f.path}" if path else f.path
            sub = sub.replace("\\", "/").strip("/")
            if getattr(f, "is_dir", False):
                found = walk(sub)
                if found:
                    return found
            elif Path(f.path).name == basename or sub.endswith(basename):
                return sub
        return None

    return walk(None)


def _print_artifact_tree(client: MlflowClient, run_id: str, path: str = "") -> None:
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
    if n <= 0:
        raise ValueError("n deve ser >= 1")
    if len(perfis) >= n:
        return random.sample(perfis, k=n)
    return [random.choice(perfis) for _ in range(n)]


def _pesos_aleatorios_soma_um(n: int, rng: np.random.Generator) -> list[float]:
    if n <= 0:
        raise ValueError("n deve ser >= 1")
    w = rng.dirichlet(np.ones(n)).tolist()
    s = float(sum(w))
    return [float(x / s) for x in w]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inferencia no modelo registado MLflow + listagem opcional de artefatos"
    )
    parser.add_argument(
        "--registered-model",
        default=None,
        help="Nome no Model Registry (alternativa a MLFLOW_REGISTERED_MODEL_NAME)",
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
            "N utilizadores: perfis aleatorios do catalogo do run do modelo + pesos aleatorios "
            "somando 100%%."
        ),
    )
    parser.add_argument(
        "--experiment-feature-impact",
        default=None,
        help="Nome do experimento MLflow para feature-impact (ou MLFLOW_EXPERIMENT_FEATURE_IMPACT)",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    registered_name = _registered_model_name(args.registered_model)
    exp_fi_name: str | None = _experiment_feature_impact_name(args.experiment_feature_impact)

    from config.mlflow_config import configurar_mlflow

    exp_main = (os.getenv("MLFLOW_EXPERIMENT_NAME") or "").strip()
    configurar_mlflow(exp_main, preparar_experimento=False)
    client = MlflowClient()

    if args.model_version is not None:
        mv = client.get_model_version(registered_name, str(args.model_version))
    else:
        mv = _latest_model_version(client, registered_name)

    run_id_predict = mv.run_id
    model_uri = f"models:/{registered_name}/{mv.version}"

    listagens = not args.somente_inferencia and args.carteira is None
    run_id_fi: str | None = None
    fi_csv_rel: str | None = None

    if listagens:
        print("\n" + "=" * 60)
        print("[Modelo registado]")
        print("=" * 60)
        print(f"  name            : {registered_name}")
        print(f"  version         : {mv.version}")
        print(f"  run_id (treino) : {run_id_predict}")
        print(f"  model URI       : {model_uri}")
        print(f"  artefact source : {getattr(mv, 'source', '(n/d)')}")

        print("\n" + "=" * 60)
        print(f"[Artefatos do run de predicao] (listagem; URI runs:/{run_id_predict}/...)")
        print("=" * 60)
        _print_artifact_tree(client, run_id_predict, "")

        print("\n" + "=" * 60)
        if not exp_fi_name:
            print("[Experimento feature-impact] omitido (defina MLFLOW_EXPERIMENT_FEATURE_IMPACT ou --experiment-feature-impact)")
        else:
            print(f"[Experimento] {exp_fi_name} — ultimo run")
        print("=" * 60)
        exp_fi = client.get_experiment_by_name(exp_fi_name) if exp_fi_name else None
        if exp_fi_name and exp_fi is None:
            print(f"  Experimento {exp_fi_name!r} nao encontrado no tracking.")
        elif exp_fi is not None:
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

                fi_csv_rel = _find_artifact_rel_path(
                    client, run_id_fi, basename="feature_correlation_sinistralidade.csv"
                )
                if fi_csv_rel:
                    csv_uri = _artifact_uri_run(run_id_fi, fi_csv_rel)
                    print(f"\n  CSV feature-impact (descoberto via API): {csv_uri}")
                    try:
                        txt = _load_text_artifact(run_id_fi, fi_csv_rel)
                        lines = txt.splitlines()
                        preview = "\n".join(lines[:15])
                        print("  [preview primeiras linhas, em memoria]\n")
                        print(preview)
                        if len(lines) > 15:
                            print(f"  ... ({len(lines) - 15} linhas a mais)")
                    except Exception as e:
                        print(f"  [aviso] Nao foi possivel ler o CSV: {e}")
                else:
                    print("  [aviso] CSV feature_correlation_sinistralidade.csv nao encontrado no run.")

        print("\n" + "=" * 60)
        print("[Catalogo de perfis no run de predicao] -> perfil aleatorio")
        print("=" * 60)

    cat_rel = _find_artifact_rel_path(client, run_id_predict, basename="catalogo_perfis_top100.json")
    if not cat_rel:
        raise FileNotFoundError(
            "Nao foi possivel localizar 'catalogo_perfis_top100.json' nos artefatos do run "
            f"{run_id_predict} (pesquisa via MlflowClient.list_artifacts)."
        )
    catalog_uri = _artifact_uri_run(run_id_predict, cat_rel)
    if listagens:
        print(f"  URI: {catalog_uri}")

    try:
        cat_text = _load_text_artifact(run_id_predict, cat_rel)
    except Exception as e:
        raise FileNotFoundError(
            f"Artefato inacessivel no run {run_id_predict}: {cat_rel!r}. {e}"
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
            "nota_pesos": "Pesos aleatorios (Dirichlet) somando 100%; perfis aleatorios do catalogo deste run.",
            "sinistralidade_prevista_media_ponderada": sin_pond,
            "p_sinistro_medio_ponderado": p_pond,
            "usuarios": usuarios,
            "model_uri": model_uri,
            "registered_model": registered_name,
            "model_version": int(mv.version),
            "run_id_predict": run_id_predict,
            "pipeline_variant": pipeline_variant,
            "catalogo_artifact_uri": catalog_uri,
            "catalogo_artifact_rel_path": cat_rel,
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
            "registered_model": registered_name,
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
            "catalogo_artifact_rel_path": cat_rel,
            "feature_impact_run_id": run_id_fi,
            "feature_impact_csv_uri": _artifact_uri_run(run_id_fi, fi_csv_rel)
            if run_id_fi and fi_csv_rel
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
