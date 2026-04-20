"""
Regras de elegibilidade para intervenção.
Operam exclusivamente sobre df_agg (dados brutos pré-feature engineering).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _universal(df: pd.DataFrame, delta_pct: float) -> pd.Series:
    return pd.Series([True] * len(df), index=df.index)


def _internado(df: pd.DataFrame, delta_pct: float) -> pd.Series:
    if delta_pct > 0:
        return df["qtd_conta_INTERNADO"] > 0
    return _universal(df, delta_pct)


def _quimio(df: pd.DataFrame, delta_pct: float) -> pd.Series:
    return df["qtd_servico_QUIMIOTERAPIA"] > 0


REGRAS_ELEGIBILIDADE: dict[str, callable] = {
    "qtd_servico_CARDIOLOGIA": lambda df, d: df["idade"] >= 30,
    "qtd_servico_QUIMIOTERAPIA": _quimio,
    "qtd_servico_ENDOSCOPIA": lambda df, d: df["idade"] >= 40,
    "qtd_servico_CIRURGICO": lambda df, d: df["idade"] >= 12,
    "qtd_servico_DIÁRIA": lambda df, d: df["qtd_conta_INTERNADO"] > 0,
    "qtd_esp_cardio": lambda df, d: df["idade"] >= 30,
    "qtd_esp_ped": lambda df, d: df["idade"] <= 18,
    "qtd_esp_gine": lambda df, d: df["sexo"].astype(str).str.upper() == "F",
    "qtd_esp_cirurg": lambda df, d: df["idade"] >= 12,
    "qtd_carater_eletivo": lambda df, d: df["tipo_cadastro"].astype(str).str.upper()
    == "TITULAR",
    "qtd_conta_INTERNADO": _internado,
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
    mask = regra(df_agg, delta_pct)
    if isinstance(mask, pd.Series):
        mask = mask.fillna(False).astype(bool)
        mask.index = df_agg.index
    else:
        mask = pd.Series(np.asarray(mask, dtype=bool), index=df_agg.index)
    return mask
