"""
Simulação de carteira homogênea — perfil JSON replicado em toda a base.

Cenário: "e se todos os registros da base fossem deste tipo de usuário?"
Calcula a sinistralidade **global** (macro) ponderada por ``valor_faturamento``:

    sum(sinistralidade_prevista * premio) / sum(premio)

Modos:
  - homogeneo: N = tamanho da base; cada linha é o mesmo perfil (JSON + template).
  - premio_real: mesmo perfil em todas as colunas, mas ``valor_faturamento`` copiado
    linha a linha da base real (mantém o volume de prêmio observado).

Uso (na raiz do repo)::

    python pipelines/elgin/simulacao_carteira.py --versao v1 --json data/auxiliar/elgin/exemplo_inferencia.json
    python pipelines/elgin/simulacao_carteira.py --versao v1 --json perfil.json --modo premio_real

Perfil = linha real da base transformada (índice 0 = primeira linha do Parquet na ordem do pandas)::

    python pipelines/elgin/simulacao_carteira.py --versao v1 --perfil-indice 15000 --modo premio_real
    python pipelines/elgin/simulacao_carteira.py --versao v1 --perfil-indice 15000 --export-perfil data/auxiliar/elgin/perfil_linha.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "pipelines" / "elgin"))

import predict as ep  # noqa: E402


def _latest_version_dir(root: Path) -> Path | None:
    max_n = 0
    best: Path | None = None
    for p in root.iterdir():
        if not p.is_dir():
            continue
        m = re.fullmatch(r"v(\d+)", p.name, flags=re.IGNORECASE)
        if m:
            n = int(m.group(1))
            if n >= max_n:
                max_n = n
                best = p
    return best


def _resolve_version_dir(versao: str | None) -> Path:
    root = ep.OUTPUT_PREDICT_ROOT
    if versao:
        d = root / versao
        if not d.is_dir():
            raise FileNotFoundError(f"Pasta de versão não encontrada: {d}")
        return d
    latest = _latest_version_dir(root)
    if latest is None:
        raise FileNotFoundError(
            f"Nenhuma pasta vN em {root}. Rode ``python pipelines/elgin/predict.py`` antes."
        )
    return latest


def _load_model(mpath: Path) -> ep.TwoStageModel:
    _real_main = sys.modules["__main__"]
    sys.modules["__main__"] = ep
    try:
        return joblib.load(mpath)
    finally:
        sys.modules["__main__"] = _real_main


def _template_row() -> pd.DataFrame:
    if not ep.TRANSFORMED_PARQUET_PATH.is_file():
        raise FileNotFoundError(
            f"Base transformada inexistente: {ep.TRANSFORMED_PARQUET_PATH}"
        )
    t = pd.read_parquet(ep.TRANSFORMED_PARQUET_PATH).iloc[[0]]
    t[ep.SEGMENT_COL] = t[ep.SEGMENT_COL].replace(ep.PLANO_MAP)
    return t


def _apply_payload(template: pd.DataFrame, payload: dict) -> pd.DataFrame:
    row = template.iloc[[0]].copy()
    for k, v in payload.items():
        if k in row.columns:
            row[k] = v
        else:
            print(f"[aviso] chave ignorada (sem coluna na base): {k!r}", file=sys.stderr)
    return row


def _prepare_X(df_feat: pd.DataFrame, model: ep.TwoStageModel) -> pd.DataFrame:
    expected = list(model.feature_names_)
    missing = [c for c in expected if c not in df_feat.columns]
    if missing:
        raise ValueError(f"Colunas faltando após engenharia (ex.): {missing[:20]}")
    return ep.ensure_no_object_dtype(df_feat[expected].copy())


def _unique_beneficiarios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cada linha precisa de um ``cod_beneficiario`` distinto para que lags em
    ``build_features`` não misturem histórico entre linhas replicadas.
    """
    d = df.copy()
    col = ep.BENEFICIARIO_COL
    if col in d.columns:
        d[col] = np.arange(len(d), dtype=np.int64)
    return d


def _macro_ponderado(y_pred: np.ndarray, premio: np.ndarray | pd.Series) -> float:
    p = np.asarray(premio, dtype=float).ravel()
    y = np.asarray(y_pred, dtype=float).ravel()
    s = float(np.nansum(p))
    if s == 0.0:
        return float("nan")
    return float(np.nansum(y * p) / s)


def montar_cenario_homogeneo(
    base: pd.DataFrame,
    payload: dict,
    template: pd.DataFrame,
) -> pd.DataFrame:
    n = len(base)
    one = _apply_payload(template, payload)
    return pd.concat([one] * n, ignore_index=True)


def montar_cenario_premio_real(
    base: pd.DataFrame,
    payload: dict,
    template: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for i in range(len(base)):
        r = _apply_payload(template, payload)
        r[ep.PREMIUM_COL] = base.iloc[i][ep.PREMIUM_COL]
        rows.append(r)
    return pd.concat(rows, ignore_index=True)


def rodar_simulacao(
    payload: dict,
    *,
    versao: str | None = None,
    plano_label: str = "MASTER EMPRESARIAL",
    modo: str = "homogeneo",
    max_linhas: int | None = None,
) -> dict:
    if ep.TARGET_COL in payload:
        payload = {k: v for k, v in payload.items() if k != ep.TARGET_COL}

    vdir = _resolve_version_dir(versao)
    slug = ep.plano_slug(plano_label)
    mpath = vdir / "models" / f"model_{slug}.pkl"
    if not mpath.is_file():
        raise FileNotFoundError(f"Modelo não encontrado: {mpath}")

    model = _load_model(mpath)
    template = _template_row()

    base = pd.read_parquet(ep.TRANSFORMED_PARQUET_PATH)
    base[ep.SEGMENT_COL] = base[ep.SEGMENT_COL].replace(ep.PLANO_MAP)
    if max_linhas is not None:
        base = base.iloc[: max_linhas].copy()

    if modo == "homogeneo":
        df_raw = montar_cenario_homogeneo(base, payload, template)
    elif modo == "premio_real":
        df_raw = montar_cenario_premio_real(base, payload, template)
    else:
        raise ValueError("modo deve ser 'homogeneo' ou 'premio_real'")

    df_raw = _unique_beneficiarios(df_raw)
    df_feat = ep.build_features(df_raw)
    fc = ep.load_feature_catalog()
    _ = ep.resolve_feature_columns(df_feat, fc)
    X = _prepare_X(df_feat, model)

    # Prêmio na mesma ordem das linhas (para macro); vem da matriz de features se estiver no modelo
    if ep.PREMIUM_COL not in X.columns:
        raise KeyError(
            f"'{ep.PREMIUM_COL}' precisa estar entre as features do modelo para ponderar o macro."
        )
    premio = X[ep.PREMIUM_COL].to_numpy(dtype=float)
    y_pred = model.predict(X)

    sin_global = _macro_ponderado(y_pred, premio)

    out = {
        "modo": modo,
        "versao_modelo": vdir.name,
        "n_linhas": int(len(X)),
        "premio_total": float(np.nansum(premio)),
        "sinistralidade_global_cenario": sin_global,
        "sinistralidade_prevista_media": float(np.nanmean(y_pred)),
        "modelo": str(mpath.resolve()),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Simulação: carteira inteira = perfil JSON")
    ap.add_argument("--versao", default=None, help="vN em data/processed/elgin/predict (padrão: última)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--json", type=Path, default=None, help="Arquivo JSON do perfil")
    src.add_argument(
        "--perfil-indice",
        type=int,
        default=None,
        help="Linha N da base transformada (Parquet) como perfil",
    )
    ap.add_argument(
        "--export-perfil",
        type=Path,
        default=None,
        help="Com --perfil-indice, grava o JSON extraído neste caminho",
    )
    ap.add_argument(
        "--modo",
        choices=("homogeneo", "premio_real"),
        default="homogeneo",
        help="homogeneo: mesmo perfil em todas as linhas; premio_real: prêmio vem da base linha a linha",
    )
    ap.add_argument("--plano", default="MASTER EMPRESARIAL", help="Segmento do modelo salvo")
    ap.add_argument(
        "--max-linhas",
        type=int,
        default=None,
        help="Limita linhas da base (teste rápido; padrão = base completa)",
    )
    args = ap.parse_args()

    if args.perfil_indice is not None:
        payload = ep.payload_do_parquet_por_indice(indice=args.perfil_indice)
        if args.export_perfil:
            ep.salvar_payload(args.export_perfil, payload)
            print(f"[ok] perfil exportado: {args.export_perfil}", file=sys.stderr)
    else:
        payload = json.loads(Path(args.json).read_text(encoding="utf-8-sig"))
    out = rodar_simulacao(
        payload,
        versao=args.versao,
        plano_label=args.plano,
        modo=args.modo,
        max_linhas=args.max_linhas,
    )
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
