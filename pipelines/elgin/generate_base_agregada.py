"""
Gera a base agregada por beneficiário (uma linha por ``cod_beneficiario``) a partir do Parquet
transformado, e grava CSV em ``data/auxiliar/elgin/``.

A regra de agregação é a mesma de ``aggregate_panel_by_beneficiary`` em ``predict-agregado.py``;
este script não importa esse módulo para evitar executar o bloco de treino ao importar.

Uso (na raiz do repositório)::

    python pipelines/elgin/generate_base_agregada.py
    python pipelines/elgin/generate_base_agregada.py --output data/auxiliar/elgin/minha_base.csv
    python pipelines/elgin/generate_base_agregada.py --parquet caminho/para/painel.parquet --sem-filtro-premio
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

COMPANY = "elgin"
TRANSFORMED_PARQUET_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / COMPANY
    / "base_analitica_transformada"
    / "painel_sinistralidade_v1.parquet"
)
OUTPUT_DEFAULT = ROOT_DIR / "data" / "auxiliar" / COMPANY / "base_agregada_beneficiarios.csv"

BENEFICIARIO_COL = "cod_beneficiario"
TIME_COL = "competencia"
TARGET_COL = "sinistralidade_final"
SEGMENT_COL = "plano"
PREMIUM_COL = "valor_faturamento"
N_MESES_COL = "n_meses_obs"

PLANO_MAP = {
    "EMPRESARIAL MASTER": "MASTER EMPRESARIAL",
    "COLETIVO EMPRESARIAL MASTER - PROTOCOLO ANS: 414538991": "MASTER EMPRESARIAL",
}


def aggregate_panel_by_beneficiary(df: pd.DataFrame) -> pd.DataFrame:
    """Alinhado a ``predict-agregado.aggregate_panel_by_beneficiary``."""
    if BENEFICIARIO_COL not in df.columns:
        raise ValueError(f"Coluna obrigatória ausente: {BENEFICIARIO_COL}")
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
        elif c.startswith("qtd_") or c == PREMIUM_COL:
            agg[c] = "sum"
        elif c in (SEGMENT_COL, "idade", "sexo", "tipo_cadastro", "plano"):
            agg[c] = "first"
        elif "sinistralidade" in c.lower() and c != TARGET_COL:
            agg[c] = "mean"
        elif pd.api.types.is_numeric_dtype(d[c]):
            cl = c.lower()
            if any(x in cl for x in ("valor", "sinistro", "sin_ref", "fator", "ajuste")):
                agg[c] = "sum"
            else:
                agg[c] = "mean"
        else:
            agg[c] = "first"

    d_work = d.drop(columns=[TIME_COL], errors="ignore")
    out = d_work.groupby(BENEFICIARIO_COL, as_index=False, dropna=False).agg(agg)
    sizes = d.groupby(BENEFICIARIO_COL, dropna=False).size().reset_index(name=N_MESES_COL)
    out = out.merge(sizes, on=BENEFICIARIO_COL, how="left")
    if "idade" in out.columns:
        _id = pd.to_numeric(out["idade"], errors="coerce")
        out["idade"] = np.ceil(_id).fillna(-1).astype(np.int64)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera CSV da base agregada por beneficiário")
    parser.add_argument(
        "--parquet",
        type=Path,
        default=TRANSFORMED_PARQUET_PATH,
        help="Parquet transformado (painel mensal); padrão = base analítica ELGIN",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DEFAULT,
        help=f"CSV de saída (padrão: {OUTPUT_DEFAULT})",
    )
    parser.add_argument(
        "--sem-filtro-premio",
        action="store_true",
        help="Não remove linhas com valor_faturamento <= 0 (o treino agregado filtra por padrão)",
    )
    args = parser.parse_args()

    path_in = args.parquet.resolve()
    if not path_in.is_file():
        raise FileNotFoundError(f"Parquet não encontrado: {path_in}")

    df_raw = pd.read_parquet(path_in)
    if SEGMENT_COL not in df_raw.columns:
        raise ValueError(f"Coluna ausente: {SEGMENT_COL}")
    df_raw[SEGMENT_COL] = df_raw[SEGMENT_COL].replace(PLANO_MAP)

    df_agg = aggregate_panel_by_beneficiary(df_raw)
    n_in = len(df_agg)

    if not args.sem_filtro_premio and PREMIUM_COL in df_agg.columns:
        prem_ok = pd.to_numeric(df_agg[PREMIUM_COL], errors="coerce").fillna(0.0) > 0.0
        df_agg = df_agg.loc[prem_ok].copy()
        n_drop = n_in - len(df_agg)
        print(f"[Filtro] Removidas {n_drop} linhas com {PREMIUM_COL} <= 0 (restam {len(df_agg):,}).")
    else:
        print("[Filtro] Sem filtro de prêmio (--sem-filtro-premio ou coluna ausente).")

    out_path = args.output
    if not out_path.is_absolute():
        out_path = (ROOT_DIR / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_agg.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"[OK] Entrada: {path_in} | painel={df_raw.shape}")
    print(f"[OK] Agregado: {df_agg.shape} -> {out_path}")


if __name__ == "__main__":
    main()
