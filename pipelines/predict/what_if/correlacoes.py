"""
Correlações de features com envelhecimento da carteira.
Usadas apenas quando modo_idade == "correlacionado".

fator_amortecimento:
  0.0 = nenhum impacto na feature
  1.0 = impacto proporcional ao delta percentual de idade
  Valores intermediários refletem correlação histórica não-causal.
"""

from __future__ import annotations

import pandas as pd

CORRELACOES_ENVELHECIMENTO: dict[str, float] = {
    "qtd_esp_cardio": 0.70,
    "qtd_servico_CARDIOLOGIA": 0.65,
    "qtd_esp_orto": 0.50,
    "qtd_servico_FISIOTERAPIA": 0.40,
    "qtd_servico_LABORATÓRIO": 0.35,
    "qtd_conta_INTERNADO": 0.30,
    "qtd_esp_oftal": 0.30,
    "qtd_carater_urgencia": 0.25,
    "qtd_esp_neuro": 0.20,
}


def aplicar_correlacoes_idade(
    df_mod_agg: pd.DataFrame,
    mask_elegivel: pd.Series,
    delta_pct_idade: float,
) -> None:
    """
    Aplica deltas amortecidos nas features correlacionadas com envelhecimento.
    Modifica df_mod_agg in-place.
    """
    for feat, amort in CORRELACOES_ENVELHECIMENTO.items():
        if feat not in df_mod_agg.columns:
            continue
        # Força tipo numérico/float para aceitar deltas percentuais sem erro de cast.
        df_mod_agg[feat] = pd.to_numeric(df_mod_agg[feat], errors="coerce").fillna(0.0).astype(float)
        delta_efetivo = (delta_pct_idade / 100.0) * amort
        df_mod_agg.loc[mask_elegivel, feat] = (
            df_mod_agg.loc[mask_elegivel, feat] * (1.0 + delta_efetivo)
        ).clip(lower=0.0)
