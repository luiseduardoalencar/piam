# 📈 API de Forecast de Sinistralidade — MVP (12 meses)

## 🔗 Rota da API
**POST** `https://api.homo.piam.life/v1/forecast`  
> ⚙️ Homologação (trocar para `https://api.piam.life/v1/forecast` em produção)

---

## 🧩 Descrição
Rota que recebe **exatamente 12 meses consecutivos** de dados agregados de um produto (receitas/despesas), gera **features temporais** e retorna o **forecast de sinistralidade (%)** para os próximos `horizon` meses, usando o modelo registrado no **MLflow**:

```

Nome do modelo: sinistralidade-forecast
Versão atual:   v1

````

---

## 📥 Corpo da Requisição (JSON)

### Regras obrigatórias
- `historico` **deve ter exatamente 12 itens** (12 meses consecutivos).
- Cada item precisa de:  
  `competencia` (formato `YYYY-MM` ou `YYYYMM`), `receita_total` (número), `despesa_total` (número).
- Campo `sinistralidade` **é opcional**; se ausente, será calculado como `(despesa_total / receita_total) * 100`.
- `produto` é o nome comercial do plano (ex.: `"Uniplam Família"`).
- `horizon` é o **número de meses futuros** a prever (ex.: `6`).

### Estrutura
| Campo       | Tipo   | Obrigatório | Descrição                                                                 |
|-------------|--------|-------------|---------------------------------------------------------------------------|
| `produto`   | string | ✅          | Nome do produto/plano                                                     |
| `horizon`   | int    | ✅          | Meses a prever à frente (ex.: `6`)                                        |
| `historico` | array  | ✅          | **12 itens** com dados mensais agregados (consecutivos)                   |

### Exemplo de entrada
```json
{
  "produto": "Uniplam Família",
  "horizon": 6,
  "historico": [
    {"competencia": "2023-01", "receita_total": 500000, "despesa_total": 330000},
    {"competencia": "2023-02", "receita_total": 510000, "despesa_total": 340000},
    {"competencia": "2023-03", "receita_total": 530000, "despesa_total": 390000},
    {"competencia": "2023-04", "receita_total": 525000, "despesa_total": 410000},
    {"competencia": "2023-05", "receita_total": 540000, "despesa_total": 360000},
    {"competencia": "2023-06", "receita_total": 550000, "despesa_total": 395000},
    {"competencia": "2023-07", "receita_total": 565000, "despesa_total": 415000},
    {"competencia": "2023-08", "receita_total": 570000, "despesa_total": 420000},
    {"competencia": "2023-09", "receita_total": 590000, "despesa_total": 450000},
    {"competencia": "2023-10", "receita_total": 600000, "despesa_total": 470000},
    {"competencia": "2023-11", "receita_total": 605000, "despesa_total": 475000},
    {"competencia": "2023-12", "receita_total": 610000, "despesa_total": 480000}
  ]
}
````

---

## 📤 Resposta (JSON)

| Campo            | Tipo   | Descrição                                                       |
| ---------------- | ------ | --------------------------------------------------------------- |
| `produto`        | string | Produto analisado                                               |
| `horizon`        | int    | Horizonte solicitado                                            |
| `diagnosticos`   | object | Checagens do payload (se são 12 meses, período detectado, etc.) |
| `forecast`       | array  | Lista com as **previsões futuras** (tamanho = `horizon`)        |
| `mlflow`         | object | Metadados do experimento/run (útil para auditoria)              |
| `model_registry` | object | Modelo utilizado (nome/versão/stage)                            |

> **Observação:** por ser MVP com 12 meses fixos, **não** retornamos métricas de teste.
> (Opcionalmente, o backend pode calcular métricas se o cliente enviar `y_true` do `mês+1`….)

### Exemplo de saída

```json
{
  "produto": "Uniplam Família",
  "horizon": 6,
  "diagnosticos": {
    "qtde_meses_recebidos": 12,
    "periodo_detectado": { "inicio": "2023-01", "fim": "2023-12" },
    "consecutivo": true
  },
  "forecast": [
    {"ds": "2024-01-01", "y_pred": 77.2},
    {"ds": "2024-02-01", "y_pred": 76.4},
    {"ds": "2024-03-01", "y_pred": 75.8},
    {"ds": "2024-04-01", "y_pred": 76.1},
    {"ds": "2024-05-01", "y_pred": 77.0},
    {"ds": "2024-06-01", "y_pred": 77.6}
  ],
  "mlflow": {
    "tracking_uri": "https://mlflow.homo.piam.life",
    "experiment": "sinistralidade-forecast",
    "experiment_id": "4",
    "run_id": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    "run_url": "https://mlflow.homo.piam.life/#/experiments/4/runs/xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
  },
  "model_registry": {
    "name": "sinistralidade-forecast",
    "version": 1,
    "stage": "None"
  }
}
```

---

## ⚙️ Comportamento esperado no backend (resumo técnico)

1. **Validar**: `historico` tem **12 itens** e competências **consecutivas** (mês a mês).
2. **Calcular** `sinistralidade` se não vier no JSON: `(despesa_total / receita_total) * 100`.
3. **Feature engineering**:

   * Lags: `lag_1`…`lag_12` da série `sinistralidade`
   * Médias móveis: `rollmean_3`, `rollmean_6`, `rollmean_12`
   * Calendário: `ano` (**int32**), `mes` (**int32**) e dummies `mes_2`…`mes_12` (**bool**)
4. **Alinhar ao schema do modelo** (ordem/nomes/tipos) e **carregar** o modelo do Registry:
   `models:/sinistralidade-forecast/Production` *(ou versão fixa, ex. `.../1`)*
5. **Gerar o forecast futuro** para `horizon` meses (estratégia recursiva de lags) e devolver em `forecast`.

> **Tipos importantes** para evitar erro de schema no MLflow:
> `ano`, `mes` → **int32** | `mes_2..mes_12` → **bool** | demais → **float64**

---

## ⚠️ Códigos de Resposta

| Código | Situação                                                                 |
| ------ | ------------------------------------------------------------------------ |
| `200`  | Execução concluída com sucesso                                           |
| `400`  | JSON inválido (campos ausentes, tipos errados, datas não parseáveis)     |
| `422`  | Rejeitado por regra de **MVP** (histórico ≠ 12 itens ou não consecutivo) |
| `500`  | Erro interno do servidor/modelo MLflow                                   |

---

## 🧪 Testes rápidos (curl / Postman)

### curl

```bash
curl -X POST "https://api.homo.piam.life/v1/forecast" \
  -H "Content-Type: application/json" \
  -d '{
    "produto": "Uniplam Família",
    "horizon": 6,
    "historico": [
      {"competencia": "2023-01", "receita_total": 500000, "despesa_total": 330000},
      {"competencia": "2023-02", "receita_total": 510000, "despesa_total": 340000},
      {"competencia": "2023-03", "receita_total": 530000, "despesa_total": 390000},
      {"competencia": "2023-04", "receita_total": 525000, "despesa_total": 410000},
      {"competencia": "2023-05", "receita_total": 540000, "despesa_total": 360000},
      {"competencia": "2023-06", "receita_total": 550000, "despesa_total": 395000},
      {"competencia": "2023-07", "receita_total": 565000, "despesa_total": 415000},
      {"competencia": "2023-08", "receita_total": 570000, "despesa_total": 420000},
      {"competencia": "2023-09", "receita_total": 590000, "despesa_total": 450000},
      {"competencia": "2023-10", "receita_total": 600000, "despesa_total": 470000},
      {"competencia": "2023-11", "receita_total": 605000, "despesa_total": 475000},
      {"competencia": "2023-12", "receita_total": 610000, "despesa_total": 480000}
    ]
  }'
```

### Axios (JS)

```javascript
const payload = {
  produto: "Uniplam Família",
  horizon: 6,
  historico: [
    { competencia: "2023-01", receita_total: 500000, despesa_total: 330000 },
    // ...até "2023-12" (12 itens consecutivos)
  ]
};

axios.post("https://api.homo.piam.life/v1/forecast", payload)
  .then(res => console.log(res.data))
  .catch(err => console.error(err.response?.data || err.message));
```

---

## 🧾 Versão do Modelo

| Item            | Valor                            |
| --------------- | -------------------------------- |
| **Nome**        | `sinistralidade-forecast`        |
| **Versão**      | `1`                              |
| **Registry**    | MLflow                           |
| **Framework**   | scikit-learn                     |
| **Base**        | RandomForestRegressor            |
| **Pipeline**    | `models/forecasting/pipeline.py` |
| **Treinado em** | 2025-11-10                       |

---

> 🧠 **Resumo:** Envie **12 meses consecutivos** com `competencia`, `receita_total`, `despesa_total` (e opcional `sinistralidade`).
> A API retorna as **previsões para os próximos `horizon` meses**, mais metadados do run/modelo.

