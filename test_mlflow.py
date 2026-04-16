"""
Teste de conectividade MLflow: experimentos, runs, modelos registados e inferência (pyfunc).

Executar na raiz do projeto:
    python test_mlflow.py

Requer .env com as mesmas variáveis que config/mlflow_config.py (tracking + AWS).

Opcional: ``MLFLOW_HTTP_REQUEST_TIMEOUT`` (segundos por pedido HTTP; predefinição 60).
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv

load_dotenv(dotenv_path=ROOT_DIR / ".env")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "60")

import mlflow
import mlflow.pyfunc
import pandas as pd
from mlflow.tracking import MlflowClient
from mlflow.types import DataType
from mlflow.types.schema import ColSpec, Schema, TensorSpec

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("mlflow_health")


def _setup_tracking() -> None:
    required = [
        "MLFLOW_TRACKING_URI",
        "MLFLOW_TRACKING_USERNAME",
        "MLFLOW_TRACKING_PASSWORD",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
    ]
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        raise EnvironmentError(
            "Variáveis ausentes no .env:\n" + "\n".join(f"  - {v}" for v in missing)
        )

    uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(uri)
    os.environ["MLFLOW_REGISTRY_URI"] = uri
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME", "")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "")


def _colspec_to_value(spec: ColSpec) -> Any:
    t = spec.type
    if t == DataType.string:
        return "x"
    if t in (DataType.long, DataType.integer):
        return 1
    if t == DataType.boolean:
        return False
    if t == DataType.datetime:
        return "2020-01-01"
    return 0.0


def _dataframe_one_row_from_schema(schema: Schema) -> pd.DataFrame | None:
    row: dict[str, Any] = {}
    for col in schema.inputs:
        if isinstance(col, TensorSpec):
            # pyfunc sklearn costuma usar ColSpec; tensores exigem outro formato
            return None
        if isinstance(col, ColSpec):
            if not col.name:
                return None
            row[col.name] = _colspec_to_value(col)
    if not row:
        return None
    return pd.DataFrame([row])


def _infer_sample_dataframe(model: mlflow.pyfunc.PyFuncModel) -> pd.DataFrame | None:
    meta = model.metadata
    if meta is None or meta.signature is None or meta.signature.inputs is None:
        return None
    try:
        return _dataframe_one_row_from_schema(meta.signature.inputs)
    except Exception as e:
        log.warning("Não foi possível montar DataFrame a partir da assinatura: %s", e)
        return None


def _stage_label(stage: str | None) -> str:
    if not stage:
        return "None"
    return stage


def main() -> int:
    log.info("=== Início do teste MLflow ===")
    log.info("Diretório do projeto: %s", ROOT_DIR)

    try:
        _setup_tracking()
    except EnvironmentError as e:
        log.error("%s", e)
        return 2

    uri = os.getenv("MLFLOW_TRACKING_URI")
    log.info("Tracking URI: %s", uri)
    log.info(
        "Timeout HTTP (MLFLOW_HTTP_REQUEST_TIMEOUT): %ss",
        os.getenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "n/d"),
    )

    client = MlflowClient(tracking_uri=uri, registry_uri=uri)

    # --- Experimentos e runs ---
    try:
        experiments = client.search_experiments()
    except Exception as e:
        log.exception(
            "Falha ao contactar o servidor MLflow (rede/VPN, URI ou credenciais). Detalhe: %s",
            e,
        )
        return 3
    log.info("--- EXPERIMENTOS (%d encontrados) ---", len(experiments))
    total_runs = 0
    for exp in experiments:
        exp_id = exp.experiment_id
        name = exp.name
        lifecycle = getattr(exp, "lifecycle_stage", "unknown")
        art = getattr(exp, "artifact_location", "")
        runs = client.search_runs(
            experiment_ids=[exp_id],
            max_results=5000,
        )
        n_runs = len(runs)
        total_runs += n_runs
        state = "ativo" if lifecycle == "active" else f"inativo ({lifecycle})"
        log.info(
            "  [exp] id=%s | nome=%r | %s | runs (até 5000)=%d | artifact_location=%s",
            exp_id,
            name,
            state,
            n_runs,
            art[:120] + ("..." if len(art) > 120 else ""),
        )
    log.info("Total de runs contabilizados (soma por experimento, cap 5000/exp): %d", total_runs)

    # --- Modelos registados ---
    try:
        registered = client.search_registered_models(max_results=500)
    except Exception as e:
        log.exception("Falha ao listar modelos registados: %s", e)
        return 3
    log.info("--- MODELOS REGISTADOS (%d) ---", len(registered))

    inference_results: list[tuple[str, int, str, str]] = []

    for rm in registered:
        mname = rm.name
        log.info("  Modelo: %r", mname)
        versions = client.search_model_versions(filter_string=f"name='{mname}'")
        if not versions:
            log.warning("    Nenhuma versão listada (inesperado).")
            continue

        for mv in sorted(versions, key=lambda v: int(v.version)):
            ver = mv.version
            status = mv.status or "UNKNOWN"
            stage = mv.current_stage or "None"
            run_id = mv.run_id or ""
            # READY = artefactos OK; FAILED = falha no registo
            status_ok = status.upper() == "READY"
            archived = stage.upper() == "ARCHIVED"
            # "Ativo" para servir: READY e não arquivado (convensão comum)
            servivel = status_ok and not archived
            log.info(
                "    versão=%s | status=%s | stage=%s | run_id=%s | servível_inferência=%s",
                ver,
                status,
                _stage_label(stage),
                run_id[:12] + "..." if len(run_id) > 12 else run_id,
                servivel,
            )

            if not status_ok:
                inference_results.append((mname, int(ver), "SKIP", f"status={status}"))
                continue

            uri_model = f"models:/{mname}/{ver}"
            try:
                pyf = mlflow.pyfunc.load_model(uri_model)
            except Exception as e:
                log.error("    [LOAD FALHOU] %s — %s", uri_model, e)
                inference_results.append((mname, int(ver), "LOAD_FAIL", str(e)[:200]))
                continue

            log.info("    [LOAD OK] %s", uri_model)

            df_in = _infer_sample_dataframe(pyf)
            if df_in is None:
                log.warning("    [INFERÊNCIA] sem assinatura/input schema — apenas load testado.")
                inference_results.append((mname, int(ver), "LOAD_ONLY", "no_signature"))
                continue

            try:
                out = pyf.predict(df_in)
                log.info(
                    "    [INFERÊNCIA OK] predict(1 linha) -> tipo=%s shape=%s",
                    type(out),
                    getattr(out, "shape", "n/a"),
                )
                inference_results.append((mname, int(ver), "PREDICT_OK", ""))
            except Exception as e:
                log.error("    [INFERÊNCIA FALHOU] %s", e)
                inference_results.append((mname, int(ver), "PREDICT_FAIL", str(e)[:200]))

    # --- Resumo ---
    log.info("=== RESUMO INFERÊNCIA ===")
    ok = sum(1 for _, _, s, _ in inference_results if s == "PREDICT_OK")
    load_only = sum(1 for _, _, s, _ in inference_results if s == "LOAD_ONLY")
    fails = [x for x in inference_results if x[2] not in ("PREDICT_OK", "LOAD_ONLY", "SKIP")]
    log.info(
        "Predições OK: %d | só load (sem assinatura para dummy): %d | outros: %d",
        ok,
        load_only,
        len(fails),
    )
    for mname, ver, code, detail in fails:
        log.warning("  Falha: %s v%s — %s — %s", mname, ver, code, detail)

    # --- Latest por modelo (rota típica models:/name/latest) ---
    log.info("--- TESTE models:/<nome>/latest (um por modelo) ---")
    latest_failures: list[str] = []
    for rm in registered:
        mname = rm.name
        uri_latest = f"models:/{mname}/latest"
        try:
            mlflow.pyfunc.load_model(uri_latest)
            log.info("  [OK] %s", uri_latest)
        except Exception as e:
            log.error("  [FALHOU] %s — %s", uri_latest, e)
            latest_failures.append(f"{uri_latest}: {e}")

    log.info("=== Fim do teste MLflow ===")
    if fails or latest_failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
