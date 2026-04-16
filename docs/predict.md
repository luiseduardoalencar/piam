# Documentação Técnica: Simulador de Carteira e Inferência MLflow

## 1. Visão Geral
O sistema permite que o usuário simule o comportamento de uma carteira de saúde baseada em perfis pré-estabelecidos. O núcleo da inteligência reside em um modelo Two-Stage (Probabilidade de Sinistro + Índice de Sinistralidade) hospedado no MLflow.

### Identificadores do Ambiente
* **Experimento:** piam-elgin-predict
* **Modelo Registrado:** elgin-sinistralidade-two-stage
* **Versão do Pipeline:** Agregado por Beneficiário
* **Tamanho da Base de Cálculo:** **8.309 registros** (Fixo)

---

## 2. Arquitetura do Fluxo de Dados
O fluxo é dividido em quatro etapas estruturais:

1. **Catálogo (Artefato):** O arquivo estático `catalogo_perfis_top100.json` fornece os perfis já processados e limpos para a simulação.
2. **Orquestração (Back-end):** Recebe as escolhas do usuário e expande (replica) os perfis para preencher exatamente 8.309 linhas.
3. **Processamento (MLflow):** Recebe a tabela em batch via API e retorna as predições individuais.
4. **Consolidação (Back-end):** Realiza a média ponderada final para gerar o índice global da carteira.

---

## 3. Especificação do Perfil (Payload Base Completo)
O Back-end deve repassar o objeto payload do catálogo exatamente como ele é extraído. Para a inferência do modelo ELGIN, a assinatura exige estritamente a submissão das 86 features processadas.

### Exemplo do Objeto de Payload de Entrada (Enviado pelo Front-end / Back-end)
O JSON abaixo representa uma (1) linha do dataframe. Este objeto exato será replicado para compor as 8.309 linhas do lote de inferência.

```json
{
  "dataframe_records": [
    {
      "faixa_etaria": "31-45",
      "is_titular": 1,
      "is_fem": 0,
      "plano": "MASTER EMPRESARIAL",
      "valor_faturamento": 1234.56,
      "tx_qtd_eventos_sinistro": 0.0245,
      "tx_internacoes": 0.001,
      "tx_consultas_eletivas": 0.45,
      "tx_exames_simples": 1.2,
      "tx_exames_complexos": 0.05,
      "is_coparticipativo": 1,
      "tempo_permanencia_meses": 24,
      "is_internacao_recente": 0,
      "tx_uso_pronto_socorro": 0.15,
      "regiao_atendimento": "CAPITAL",
      "tx_consultas_pronto_socorro": 0.10,
      "tx_consultas_pediatria": 0.00,
      "tx_consultas_ginecologia": 0.00,
      "tx_consultas_cardiologia": 0.05,
      "tx_consultas_ortopedia": 0.12,
      "tx_consultas_dermatologia": 0.00,
      "tx_consultas_oftalmologia": 0.00,
      "tx_consultas_psiquiatria": 0.00,
      "tx_consultas_neurologia": 0.00,
      "tx_consultas_endocrinologia": 0.00,
      "tx_consultas_urologia": 0.00,
      "tx_consultas_gastroenterologia": 0.00,
      "tx_consultas_otorrinolaringologia": 0.00,
      "tx_exames_laboratoriais": 2.50,
      "tx_exames_imagem_raio_x": 0.10,
      "tx_exames_imagem_ultrassom": 0.00,
      "tx_exames_imagem_tomografia": 0.00,
      "tx_exames_imagem_ressonancia": 0.00,
      "tx_exames_imagem_mamografia": 0.00,
      "tx_exames_cardiologicos_eletro": 0.00,
      "tx_exames_cardiologicos_eco": 0.00,
      "tx_exames_endoscopia": 0.00,
      "tx_exames_colonoscopia": 0.00,
      "tx_terapias_fisioterapia": 0.00,
      "tx_terapias_psicologia": 0.00,
      "tx_terapias_fonoaudiologia": 0.00,
      "tx_terapias_terapia_ocupacional": 0.00,
      "tx_terapias_nutricao": 0.00,
      "tx_terapias_acupuntura": 0.00,
      "tx_procedimentos_cirurgicos_ambulatoriais": 0.00,
      "tx_procedimentos_odontologicos": 0.00,
      "tx_internacoes_clinicas": 0.00,
      "tx_internacoes_cirurgicas": 0.00,
      "tx_internacoes_obstetricas": 0.00,
      "tx_internacoes_psiquiatricas": 0.00,
      "tx_internacoes_pediatricas": 0.00,
      "tx_internacoes_uti": 0.00,
      "tx_dias_internacao_clinica": 0.00,
      "tx_dias_internacao_cirurgica": 0.00,
      "tx_dias_internacao_obstetrica": 0.00,
      "tx_dias_internacao_psiquiatrica": 0.00,
      "tx_dias_internacao_pediatrica": 0.00,
      "tx_dias_internacao_uti": 0.00,
      "tx_custo_consultas": 50.00,
      "tx_custo_exames": 120.00,
      "tx_custo_terapias": 0.00,
      "tx_custo_internacoes": 0.00,
      "tx_custo_pronto_socorro": 25.00,
      "is_doenca_preexistente": 0,
      "is_doenca_cronica": 0,
      "is_hipertensao": 0,
      "is_diabetes": 0,
      "is_obesidade": 0,
      "is_tabagismo": 0,
      "is_alcoolismo": 0,
      "tx_medicamentos_alto_custo": 0.00,
      "tx_materiais_alto_custo": 0.00,
      "tx_orteses_proteses": 0.00,
      "is_acomodacao_enfermaria": 0,
      "is_acomodacao_apartamento": 1,
      "is_abrangencia_nacional": 1,
      "is_abrangencia_regional": 0,
      "is_abrangencia_estadual": 0,
      "is_abrangencia_municipal": 0,
      "tx_reembolsos_solicitados": 0.00,
      "tx_reembolsos_pagos": 0.00,
      "tx_custo_reembolso": 0.00,
      "is_dependente_conjuge": 0,
      "is_dependente_filho": 0,
      "tx_sinistralidade_historica_12m": 0.45,
      "tx_sinistralidade_historica_24m": 0.38
    }
  ]
}
```
*(Nota: As 86 chaves acima devem ser populadas com os valores extraídos do artefato para o perfil específico. Nenhuma chave pode ser omitida ou adicionada).*

---

## 4. Algoritmo de Expansão e Inferência

### Passo 1: Expansão da Base (Replicação de Dados)
O Back-end recebe a seleção de perfis e seus respectivos percentuais via requisição do Front-end. O sistema deve calcular o volume matemático de vidas por perfil e replicar o JSON base.

**Exemplo de Lógica (Seleção: Perfil A = 30%, Perfil B = 70%):**
* Perfil A: `8309 * 0.30 = 2492.7` → Arredonda para 2.493 linhas.
* Perfil B: `8309 * 0.70 = 5816.3` → Arredonda para 5.816 linhas.
* Critério Crítico: O Back-end deve garantir via código (ajustando arredondamentos) que a soma total do array final seja obrigatoriamente 8.309.

Nesta etapa, o Back-end apenas itera a inclusão do objeto de 86 features no array. Nenhuma transformação de dados ou preenchimento de nulos é necessário, pois o catálogo já fornece a estrutura tratada.

### Passo 2: Chamada MLflow
O Back-end submete o array resultante via requisição POST para o endpoint `/invocations` do servidor MLflow. O cabeçalho deve ser `Content-Type: application/json`.

**Formato do Payload Enviado:**
```json
{
  "dataframe_records": [
    { /* 86 features do Perfil A */ },
    { /* 86 features do Perfil A */ },
    ... (2493 vezes) ...
    { /* 86 features do Perfil B */ },
    ... (5816 vezes, total 8309 posições)
  ]
}
```

### Passo 3: Resposta do Modelo
O MLflow devolve um array com 8.309 resultados vetoriais na mesma ordem da entrada:

```json
{
  "predictions": [
    { "p_sinistro": 0.874, "sinistralidade_prevista": 0.85 },
    { "p_sinistro": 0.102, "sinistralidade_prevista": 0.42 },
    ... (8309 posições)
  ]
}
```

---

## 5. Algoritmo de Média Ponderada (Consolidação Final)
O sistema não deve fazer a média aritmética do índice `sinistralidade_prevista`. É obrigatório calcular a média ponderada pelo faturamento para não distorcer o impacto financeiro da carteira.

**Regra Matemática no Back-end:**
Para cada registro indexado do array de resposta (i = 1 até 8.309):
1. Isolar o valor numérico de `sinistralidade_prevista` (Ignorar o `p_sinistro`).
2. Multiplicar a `sinistralidade_prevista(i)` pelo `valor_faturamento(i)` da mesma posição na base.
3. Agrupar o valor resultante de todas as 8.309 multiplicações (Soma do Numerador).
4. Agrupar o `valor_faturamento` bruto de todas as 8.309 posições (Soma do Denominador).
5. A divisão da Soma do Numerador pela Soma do Denominador fornece o índice percentual real da carteira simulada.

**Contrato de Resposta (Back-end -> Front-end)**
O Back-end consolida o cálculo em memória e devolve ao Front-end apenas os indicadores finais resumidos da simulação.

```json
{
  "status": "success",
  "resultado_simulacao": {
    "sinistralidade_media_ponderada": 0.824,
    "volume_vidas_simuladas": 8309,
    "faturamento_total_base": 8250430.00
  }
}
```

---

## 6. Checklist Arquitetural Obrigatório
- [ ] O back-end repassa o bloco das 86 features de forma integral, sem modificações ou tratamento?
- [ ] O algoritmo de expansão da carteira respeita estritamente o teto de 8.309 objetos, tratando as casas decimais das proporções?
- [ ] A agregação final utiliza o cálculo de média ponderada da sinistralidade prevista vezes o faturamento atrelado ao perfil?
- [ ] O back-end suprime a matriz de 8.309 resultados individuais e encaminha apenas o JSON de consolidação para a interface?