"""
Mapeamento centralizado dos nomes técnicos dos atributos da carteira
para os rótulos exibidos na UI dos apps Streamlit (ELGIN e CLIMAZON).

Convenções:
- qtd_conta_*   → regime do atendimento (Atendimentos / Internações)
- qtd_servico_* → o que foi feito (Procedimentos / Exames / Sessões / Consultas)
- qtd_esp_*     → quem atendeu (Atendimentos com [Especialidade])

Os dois pipelines usam convenções diferentes para o mesmo conceito
(ex.: ELGIN salva "qtd_conta_PRONTO SOCORRO" e CLIMAZON salva
"qtd_conta_pronto_socorro"). Por isso ambas as variantes estão
mapeadas para o mesmo rótulo.
"""
from __future__ import annotations

FEATURE_LABELS: dict[str, str] = {
    # ---- Demografia / cadastro ----
    "idade": "Idade",
    "sexo": "Sexo",
    "tipo_cadastro": "Tipo de cadastro",
    "valor_faturamento": "Faturamento mensal",
    "pct_urgencia": "Percentual de uso em urgência",

    # ---- qtd_conta_* — regime do atendimento ----
    # variante ELGIN (UPPERCASE com espaço)
    "qtd_conta_ATENDIMENTO AMBULATORIAL": "Atendimentos ambulatoriais",
    "qtd_conta_ELETIVO": "Atendimentos eletivos",
    "qtd_conta_EXTERNO": "Atendimentos externos",
    "qtd_conta_INTERNADO": "Internações hospitalares",
    "qtd_conta_NA": "Atendimentos não classificados",
    "qtd_conta_PRONTO SOCORRO": "Atendimentos em pronto-socorro",
    "qtd_conta_URGÊNCIA \\ EMERGÊNCIA": "Atendimentos de urgência/emergência",
    # variante CLIMAZON (lowercase com underscore)
    "qtd_conta_ambulatorial": "Atendimentos ambulatoriais",
    "qtd_conta_eletivo": "Atendimentos eletivos",
    "qtd_conta_externo": "Atendimentos externos",
    "qtd_conta_internado": "Internações hospitalares",
    "qtd_conta_pronto_socorro": "Atendimentos em pronto-socorro",
    "qtd_conta_urgencia_emergencia": "Atendimentos de urgência/emergência",

    # ---- qtd_servico_* — o que foi feito ----
    "qtd_servico_CONSULTA": "Consultas médicas",
    "qtd_servico_CARDIOLOGIA": "Procedimentos cardiológicos",
    "qtd_servico_CIRURGICO": "Procedimentos cirúrgicos",
    "qtd_servico_CLÍNICO": "Procedimentos clínicos",
    "qtd_servico_LABORATÓRIO": "Exames laboratoriais",
    "qtd_servico_RADIOLOGIA": "Exames de radiologia",
    "qtd_servico_TOMOGRAFIA": "Exames de tomografia",
    "qtd_servico_ULTRA-SONOGRAFIA": "Exames de ultrassonografia",
    "qtd_servico_RESSONÂNCIA MAGNÉTICA": "Exames de ressonância magnética",
    "qtd_servico_QUIMIOTERAPIA": "Sessões de quimioterapia",
    "qtd_servico_FISIOTERAPIA": "Sessões de fisioterapia",
    "qtd_servico_HEMODIÁLISE": "Sessões de hemodiálise",
    "qtd_servico_ACUPUNTURA": "Sessões de acupuntura",
    "qtd_servico_ENDOSCOPIA": "Exames de endoscopia",
    "qtd_servico_DIÁRIA": "Diárias hospitalares",
    "qtd_servico_TERAPIA": "Sessões de terapia",
    "qtd_servico___OUTROS__": "Outros procedimentos médicos",

    # ---- qtd_esp_* — quem atendeu (especialidade do profissional) ----
    "qtd_esp_clin_geral": "Atendimentos com Clínico Geral",
    "qtd_esp_cardio": "Atendimentos com Cardiologista",
    "qtd_esp_gine": "Atendimentos com Ginecologista",
    "qtd_esp_orto": "Atendimentos com Ortopedista",
    "qtd_esp_oftal": "Atendimentos com Oftalmologista",
    "qtd_esp_neuro": "Atendimentos com Neurologista",
    "qtd_esp_ped": "Atendimentos com Pediatra",
    "qtd_esp_derm": "Atendimentos com Dermatologista",
    "qtd_esp_otorrino": "Atendimentos com Otorrinolaringologista",
    "qtd_esp_psiq": "Atendimentos com Psiquiatra",
    "qtd_esp_cirurg": "Atendimentos com Cirurgião",
    "qtd_esp_lab_imagem": "Atendimentos em laboratório/imagem",
    "qtd_esp_outros": "Atendimentos em outras especialidades",
}


def format_feature_name(feature: str) -> str:
    """Devolve o rótulo amigável de um atributo.

    Se o nome técnico não estiver em FEATURE_LABELS, gera um fallback
    a partir do próprio nome (remove prefixos `qtd_`, troca underscores
    por espaço, capitaliza as três primeiras palavras).
    """
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]
    label = (
        feature.replace("tx_", "taxa ")
        .replace("qtd_", "")
        .replace("_", " ")
        .strip()
    )
    words = [p.capitalize() for p in label.split() if p]
    return " ".join(words[:3]) if words else feature
