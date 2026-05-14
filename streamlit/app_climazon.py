from __future__ import annotations

import html
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from streamlit_option_menu import option_menu

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=ROOT_DIR / ".env")

PREDICT_ROOT      = ROOT_DIR / "data" / "processed" / "climazon" / "predict"
FEAT_IMPACT_ROOT  = ROOT_DIR / "data" / "processed" / "climazon" / "feature_impact"
FORECAST_ROOT     = ROOT_DIR / "data" / "processed" / "climazon" / "forecast"
RAW_PARQUET       = ROOT_DIR / "data" / "processed" / "climazon" / "base_analitica_transformada" / "painel_sinistralidade_climazon_v1.parquet"

AUTH_EMAIL    = os.getenv("CLIMAZON_AUTH_EMAIL", "")
AUTH_PASSWORD = os.getenv("CLIMAZON_AUTH_PASSWORD", "")

PLANOS_CANONICOS = ["MASTER EMPRESARIAL", "MASTER EXECUTIVO"]
PLANO_SLUG = {"MASTER EMPRESARIAL": "MASTER_EMPRESARIAL", "MASTER EXECUTIVO": "MASTER_EXECUTIVO"}

MODO_IDADE_LABEL: dict[str, str] = {
    "simples": "Só idade (efeito isolado)",
    "correlacionado": "Idade + correlacionados (cenário plausível)",
}
MODO_IDADE_HELP = (
    "**Só idade:** altera apenas a coluna `idade`; o restante do painel permanece igual. "
    "Útil para auditoria e para ver o efeito marginal da idade.\n\n"
    "**Idade + correlacionados:** além da idade, aplica ajustes amortecidos em variáveis associadas. "
    "Tende a ser mais rico em narrativa operacional."
)


# =============================================================================
# Autenticação
# =============================================================================

def require_login() -> bool:
    if st.session_state.get("climazon_authenticated_user"):
        return True

    st.title("Acesso à plataforma")
    st.caption("Informe login e senha para continuar.")

    with st.form("login_form", clear_on_submit=False):
        email    = st.text_input("Login", value="")
        password = st.text_input("Senha", value="", type="password")
        submitted = st.form_submit_button("Entrar", type="primary")

    if submitted:
        if email.strip().lower() == AUTH_EMAIL.strip().lower() and password == AUTH_PASSWORD:
            st.session_state["climazon_authenticated_user"] = email.strip().lower()
            st.success("Login realizado com sucesso.")
            st.rerun()
        else:
            st.error("Login ou senha inválidos.")

    return False


# =============================================================================
# Helpers
# =============================================================================

def _version_key(name: str) -> int:
    m = re.search(r"v(\d+)", str(name), flags=re.IGNORECASE)
    return int(m.group(1)) if m else -1


def _fmt_pct(value: Any, nd: int = 2) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "-"
    if np.isnan(x):
        return "-"
    return f"{x * 100.0:.{nd}f}%"


def _colored_metric(container, label: str, value: str, *, is_good: bool, help_text: str = "") -> None:
    """
    Métrica com o big number colorido. Verde quando is_good=True (queda na sinistralidade),
    vermelho caso contrário. Tooltip nativo do navegador via atributo HTML title.
    """
    color = "#2e7d32" if is_good else "#c62828"
    info_html = (
        f'<span title="{html.escape(help_text)}" '
        f'style="color:#9aa0a6;cursor:help;margin-left:6px;font-size:0.85rem;">ⓘ</span>'
        if help_text else ""
    )
    container.markdown(
        f"""
<div style="margin-bottom:1rem;">
  <div style="font-size:0.875rem;color:rgba(49,51,63,0.6);line-height:1.2;padding-bottom:4px;">
    {html.escape(label)}{info_html}
  </div>
  <div style="font-size:2.25rem;font-weight:700;color:{color};line-height:1.2;">
    {html.escape(value)}
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def _sinistralidade_status(valor: float) -> tuple[str, str]:
    """
    Faixas de referência (operadora): aceitável ~70-75%.
    Recebe valor em escala 0-1 (ratio). Retorna (rótulo, cor CSS).
    """
    if valor < 0:
        return "Valor negativo", "#c62828"
    if valor < 0.70:
        return "Boa sinistralidade", "#2e7d32"
    if valor <= 0.75:
        return "Estável", "#1565c0"
    return "Alta sinistralidade", "#c62828"


def _format_feature_name(feature: str) -> str:
    mapping = {
        "idade": "Idade",
        "sexo": "Sexo",
        "tipo_cadastro": "Cadastro",
        "pct_urgencia": "Pct. urgência",
        "qtd_servico_CONSULTA": "Qtd. consulta",
        "qtd_esp_clin_geral": "Clínico geral",
        "qtd_conta_pronto_socorro": "Pronto socorro",
        "qtd_esp_outros": "Esp. outros",
        "qtd_conta_ambulatorial": "Atend. amb.",
        "qtd_servico_LABORATÓRIO": "Laboratório",
        "qtd_conta_externo": "Conta externa",
        "qtd_servico_ULTRA-SONOGRAFIA": "Ultrassom",
        "qtd_servico_RADIOLOGIA": "Radiologia",
        "qtd_esp_lab_imagem": "Lab. imagem",
        "qtd_servico___OUTROS__": "Serv. outros",
        "qtd_esp_cardio": "Esp. cardio",
        "qtd_esp_oftal": "Esp. oftalmo",
        "qtd_esp_neuro": "Esp. neuro",
        "qtd_servico_TOMOGRAFIA": "Tomografia",
        "qtd_conta_internado": "Conta internação",
        "qtd_servico_CIRURGICO": "Serv. cirúrgico",
        "qtd_servico_FISIOTERAPIA": "Fisioterapia",
        "valor_faturamento": "Faturamento",
        "qtd_servico_CLÍNICO": "Serv. clínico",
        "qtd_conta_urgencia_emergencia": "Urgência/emergência",
        "qtd_servico_RESSONÂNCIA MAGNÉTICA": "Ressonância",
        "qtd_servico_QUIMIOTERAPIA": "Quimioterapia",
        "qtd_servico_HEMODIÁLISE": "Hemodiálise",
    }
    if feature in mapping:
        return mapping[feature]
    label = feature.replace("tx_", "taxa ").replace("qtd_", "").replace("_", " ").strip()
    words = [p.capitalize() for p in label.split() if p]
    return " ".join(words[:3]) if words else feature


# =============================================================================
# Funções de dados
# =============================================================================

def list_predict_versions() -> list[str]:
    if not PREDICT_ROOT.is_dir():
        return []
    vers = [p.name for p in PREDICT_ROOT.iterdir() if p.is_dir() and re.fullmatch(r"v\d+", p.name, re.IGNORECASE)]
    return sorted(vers, key=_version_key, reverse=True)


def list_forecast_versions() -> list[str]:
    if not FORECAST_ROOT.is_dir():
        return []
    vers = [p.name for p in FORECAST_ROOT.iterdir() if p.is_dir() and re.fullmatch(r"v\d+", p.name, re.IGNORECASE)]
    return sorted(vers, key=_version_key, reverse=True)


def _latest_feat_impact_ver_dir() -> Path:
    if not FEAT_IMPACT_ROOT.is_dir():
        raise FileNotFoundError(f"Diretório não encontrado: {FEAT_IMPACT_ROOT}")
    candidates = [p for p in FEAT_IMPACT_ROOT.iterdir() if p.is_dir() and re.fullmatch(r"v\d+", p.name, re.IGNORECASE)]
    if not candidates:
        raise FileNotFoundError("Nenhuma versão vN em feature_impact")
    return max(candidates, key=lambda p: _version_key(p.name))


@st.cache_data(show_spinner=False)
def load_features_catalog(version_label: str) -> dict[str, Any]:
    path = PREDICT_ROOT / version_label / "catalogo_features_intervencao.json"
    if not path.is_file():
        raise FileNotFoundError(f"Não encontrado: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "grupos" not in payload:
        raise ValueError(f"Campo 'grupos' ausente em {path}")
    return payload


@st.cache_data(show_spinner=False)
def load_feature_impact(plano: str | None, competencia: str | None = None) -> tuple[pd.DataFrame, str]:
    ver_dir = _latest_feat_impact_ver_dir()

    if plano and plano in PLANOS_CANONICOS:
        slug = PLANO_SLUG[plano]
        csv_path = ver_dir / slug / f"corr_rank_spearman_vs_target_{slug}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Não encontrado: {csv_path}")
        df = pd.read_csv(csv_path)
        df = df.rename(columns={"abs_spearman": "abs_spearman"})
        if "spearman" not in df.columns and "abs_spearman" in df.columns:
            df["spearman"] = df["abs_spearman"]
        df["abs_spearman"] = df["spearman"].abs()
        return df.sort_values("abs_spearman", ascending=False).reset_index(drop=True), plano

    # Todos os planos, por competência
    comps = sorted(
        [p.name for p in ver_dir.iterdir() if p.is_dir() and re.fullmatch(r"\d{4}-\d{2}", p.name)],
    )
    if not comps:
        raise FileNotFoundError("Nenhuma competência em feature_impact")
    comp_sel = competencia if competencia in comps else comps[-1]
    csv_path = ver_dir / comp_sel / "feature_correlation_sinistralidade.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Não encontrado: {csv_path}")
    df = pd.read_csv(csv_path)
    df["abs_spearman"] = df["spearman"].abs()
    return df.sort_values("abs_spearman", ascending=False).reset_index(drop=True), comp_sel


def list_feat_impact_competencias() -> list[str]:
    try:
        ver_dir = _latest_feat_impact_ver_dir()
    except FileNotFoundError:
        return []
    return sorted([p.name for p in ver_dir.iterdir() if p.is_dir() and re.fullmatch(r"\d{4}-\d{2}", p.name)])


@st.cache_data(show_spinner=False)
def load_kpi_mes_atual() -> dict[str, Any]:
    if not RAW_PARQUET.is_file():
        raise FileNotFoundError(f"Parquet não encontrado: {RAW_PARQUET}")
    cols = ["competencia", "valor_faturamento", "sinistralidade_final"]
    df = pd.read_parquet(RAW_PARQUET, columns=cols)
    comp = pd.to_datetime(df["competencia"].astype(str), errors="coerce")
    comp_ref = comp.max().to_period("M")
    mask = comp.dt.to_period("M") == comp_ref
    d = df.loc[mask].copy()
    fat = pd.to_numeric(d["valor_faturamento"], errors="coerce").fillna(0.0)
    sin = pd.to_numeric(d["sinistralidade_final"], errors="coerce").fillna(0.0)
    denom = float(fat.sum())
    sin_pond = float((sin * fat).sum() / denom) if denom > 0 else float("nan")
    return {
        "competencia_referencia": str(comp_ref),
        "sinistralidade_mes": sin_pond,
        "n_vidas_mes": int(len(d)),
        "faturamento_mes": denom,
    }


def load_latest_what_if_result(version_label: str) -> dict[str, Any] | None:
    path = PREDICT_ROOT / version_label / "what_if_mensal" / "resultado_what_if.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_forecast_csv(version_label: str, plano: str | None = None) -> pd.DataFrame:
    """
    Carrega CSV de forecast.
    - plano=None (ou 'Todos')  → consolidado (média ponderada)
    - plano='MASTER EMPRESARIAL' | 'MASTER EXECUTIVO' → CSV específico do plano
    Suporta o formato antigo (v1, arquivo na raiz da versão) e o novo (v2+, subpastas).
    """
    slug_map = {"MASTER EMPRESARIAL": "MASTER_EMPRESARIAL", "MASTER EXECUTIVO": "MASTER_EXECUTIVO"}

    if plano and plano in slug_map:
        slug = slug_map[plano]
        csv_path = FORECAST_ROOT / version_label / slug / "sinistralidade_forecast_completo.csv"
    else:
        # Novo formato: consolidado/
        csv_path = FORECAST_ROOT / version_label / "consolidado" / "sinistralidade_forecast_completo.csv"
        if not csv_path.is_file():
            # Fallback: formato v1 (arquivo na raiz)
            csv_path = FORECAST_ROOT / version_label / "sinistralidade_forecast_completo.csv"

    if not csv_path.is_file():
        raise FileNotFoundError(f"Não encontrado: {csv_path}")
    df = pd.read_csv(csv_path)
    df["DATA"] = pd.to_datetime(df.get("DATA"), errors="coerce")
    return df.dropna(subset=["DATA"]).sort_values("DATA").reset_index(drop=True)


def _run_what_if_subprocess(
    *,
    version_label: str,
    feature: str,
    delta_pct: float,
    competencia_ref: str,
    plano: str | None,
    modo_idade: str,
) -> tuple[bool, str]:
    cmd = [
        sys.executable,
        str(ROOT_DIR / "pipelines" / "climazon" / "predict_features_mensal_climazon.py"),
        "--versao", version_label,
        "--feature", feature,
        "--delta-pct", str(delta_pct),
        "--competencia-ref", competencia_ref,
        "--modo-idade", modo_idade,
    ]
    if plano:
        cmd += ["--plano", plano]
    proc = subprocess.run(cmd, cwd=str(ROOT_DIR), text=True, capture_output=True, timeout=180)
    output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    return proc.returncode == 0, output.strip()


# =============================================================================
# Abas
# =============================================================================

def render_correlacao_tab() -> None:
    st.header("Correlação")
    st.caption("Ranking Spearman do impacto das features na sinistralidade.")

    plano_sel = st.selectbox("Plano", options=PLANOS_CANONICOS, key="corr_plano")

    plano_filter = plano_sel

    comp_opts: list[str] = []
    comp_sel: str | None = None
    if plano_filter is None:
        comp_opts = list_feat_impact_competencias()
        if comp_opts:
            comp_sel = st.selectbox("Competência", options=comp_opts, index=len(comp_opts) - 1, key="corr_comp")

    try:
        df, label = load_feature_impact(plano_filter, comp_sel)
    except Exception as e:
        st.error(f"Erro ao carregar feature impact: {e}")
        return

    if "plano" in df.columns or "segment" in df.columns:
        pass  # já filtrado pelo CSV do plano

    df = df.copy()
    df["feature_label"] = df["feature"].map(_format_feature_name)

    top_n_max = min(30, len(df))
    top_n = st.slider("Top N features", min_value=5, max_value=top_n_max, value=min(15, top_n_max))

    st.metric("Total de features analisadas", len(df))
    st.caption(f"Referência: **{label}**")

    st.dataframe(
        df[["feature_label", "spearman"]].rename(
            columns={"feature_label": "Feature", "spearman": "Grau de impacto"}
        ),
        hide_index=True,
        use_container_width=True,
    )

    top = df.head(top_n).sort_values("spearman")
    top["sentido"] = np.where(top["spearman"] >= 0, "Positivo", "Negativo")

    fig = px.bar(
        top,
        x="spearman",
        y="feature_label",
        orientation="h",
        color="sentido",
        color_discrete_map={"Positivo": "#2e7d32", "Negativo": "#c62828"},
        title=f"Top {top_n} features por Grau de Impacto",
        labels={"feature_label": "Feature", "spearman": "Grau de impacto", "sentido": "Impacto"},
        hover_data={"feature": True, "feature_label": False, "spearman": ":.4f"},
    )
    fig.update_layout(
        height=max(420, top_n * 28),
        yaxis={"categoryorder": "array", "categoryarray": top["feature_label"].tolist()},
        legend_title_text="Impacto",
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
    )
    fig.update_traces(marker_line_width=0)
    st.plotly_chart(fig, use_container_width=True)


def render_predicao_tab() -> None:
    st.header("Predição por features")
    st.caption("Simule o impacto de alterar a frequência de um comportamento na sinistralidade agregada.")

    kpi: dict[str, Any] | None = None
    try:
        kpi = load_kpi_mes_atual()
        k1, k2, k3, k4 = st.columns(4)
        k1.metric(
            "Mês de referência",
            kpi["competencia_referencia"],
            help="Mês mais recente disponível na base de dados.",
        )
        k2.metric(
            "Sinistralidade real do mês",
            _fmt_pct(kpi["sinistralidade_mes"]),
            help="O que de fato aconteceu no mês: total de sinistros pagos dividido pelo total faturado, em %.",
        )
        k3.metric(
            "Beneficiários no mês",
            f"{int(kpi['n_vidas_mes']):,}",
            help="Total de beneficiários ativos no mês de referência.",
        )
        k4.metric(
            "Faturamento do mês",
            f"R$ {float(kpi['faturamento_mes']):,.0f}",
            help="Soma do prêmio (valor faturado) de todos os beneficiários no mês.",
        )
    except Exception as e:
        st.warning(f"Não foi possível carregar KPI mensal: {e}")

    versions = list_predict_versions()
    if not versions:
        st.error("Nenhuma pasta vN encontrada em data/processed/climazon/predict.")
        return

    version_label = versions[0]
    comp_ref = kpi["competencia_referencia"] if kpi else "2025-10"

    # ---- baseline calibração ----
    ultimo_resultado = load_latest_what_if_result(version_label)
    baseline_inicial: float | None = None
    if ultimo_resultado and "sinistralidade_antes" in ultimo_resultado:
        baseline_inicial = float(ultimo_resultado["sinistralidade_antes"])
    if st.session_state.get("climazon_pred_hide_baseline"):
        baseline_inicial = None

    if baseline_inicial is not None and not np.isnan(baseline_inicial):
        real_val = float((kpi or {}).get("sinistralidade_mes", float("nan")))
        st.markdown("### Calibração do mês")
        if not np.isnan(real_val) and real_val != 0:
            gap = (baseline_inicial - real_val) / real_val * 100.0
            erro_text = f"{gap:+.2f}% (positivo = modelo superestimou; negativo = subestimou)"
        else:
            erro_text = "indisponível (sinistralidade real do mês é zero ou ausente)"
        st.metric(
            "Sinistralidade prevista pelo modelo",
            _fmt_pct(baseline_inicial),
            help=(
                "O que o modelo estima para o mês atual, em %. "
                f"Sinistralidade real do mês: {_fmt_pct(real_val)}. "
                f"Erro do modelo: {erro_text}."
            ),
        )

    # ---- controles ----
    plano_sel = st.selectbox("Plano", options=PLANOS_CANONICOS, key="pred_plano")

    try:
        catalog = load_features_catalog(version_label)
    except Exception as e:
        st.error(f"Falha ao carregar catálogo de features ({version_label}): {e}")
        return

    grupos = catalog.get("grupos", [])
    if not grupos:
        st.warning("Catálogo sem grupos/features disponíveis.")
        return

    grupo_nomes = [str(g.get("grupo", "sem_grupo")) for g in grupos]
    grupo_sel = st.selectbox("Grupo de características", options=grupo_nomes, key="pred_grupo")
    grupo_obj = next((g for g in grupos if str(g.get("grupo")) == grupo_sel), grupos[0])

    features = grupo_obj.get("features", [])
    if not features:
        st.warning("Grupo sem features disponíveis.")
        return

    feat_labels = [
        f"{_format_feature_name(str(f.get('feature', '')))}  | elegíveis(+): {f.get('n_elegiveis_delta_positivo', 0):,}"
        for f in features
    ]
    feat_idx = st.selectbox(
        "Feature para intervenção",
        options=list(range(len(features))),
        format_func=lambda i: feat_labels[i],
        key="pred_feat_idx",
    )
    feat = features[feat_idx]
    feat_name = str(feat.get("feature"))

    delta_pct = st.slider("Delta da intervenção (%)", min_value=-90.0, max_value=200.0, value=20.0, step=1.0)

    modo_idade = "simples"
    if feat_name == "idade":
        modo_idade = st.radio(
            "Modo de idade",
            options=["simples", "correlacionado"],
            format_func=lambda k: MODO_IDADE_LABEL.get(k, k),
            horizontal=True,
            help=MODO_IDADE_HELP,
        )

    b1, b2 = st.columns(2)
    with b1:
        run_clicked = st.button("Executar simulação", type="primary", use_container_width=True)
    with b2:
        clear_clicked = st.button("Limpar resultado", type="secondary", use_container_width=True)

    if clear_clicked:
        for k in ("climazon_pred_last_output", "climazon_pred_last_ok", "climazon_pred_last_ver"):
            st.session_state.pop(k, None)
        st.session_state["climazon_pred_hide_baseline"] = True
        st.rerun()

    if run_clicked:
        with st.spinner("Executando simulação..."):
            ok, output = _run_what_if_subprocess(
                version_label=version_label,
                feature=feat_name,
                delta_pct=float(delta_pct),
                competencia_ref=comp_ref,
                plano=plano_sel,
                modo_idade=modo_idade,
            )
        st.session_state["climazon_pred_last_output"] = output
        st.session_state["climazon_pred_last_ok"] = ok
        st.session_state["climazon_pred_last_ver"] = version_label
        if ok:
            st.session_state["climazon_pred_hide_baseline"] = False
        st.rerun()

    if "climazon_pred_last_ok" not in st.session_state:
        return

    ok = bool(st.session_state["climazon_pred_last_ok"])
    out = str(st.session_state.get("climazon_pred_last_output", ""))
    last_ver = str(st.session_state.get("climazon_pred_last_ver", version_label))

    if ok:
        st.success("Simulação executada com sucesso.")
    else:
        st.error("Falha ao executar simulação.")

    resultado = load_latest_what_if_result(last_ver) or {}

    if resultado:
        st.subheader("Resultado da Simulação")
        rr1, rr2, rr3, rr4 = st.columns(4)
        rr1.metric(
            "Sinistralidade antes da simulação",
            _fmt_pct(resultado.get("sinistralidade_antes", 0.0)),
            help="Índice de sinistralidade que o modelo prevê para o mês atual, sem aplicar a intervenção.",
        )
        rr2.metric(
            "Sinistralidade depois da simulação",
            _fmt_pct(resultado.get("sinistralidade_depois", 0.0)),
            help="Índice de sinistralidade que o modelo prevê após aplicar a intervenção escolhida.",
        )
        d_abs = float(resultado.get("delta_absoluto", 0.0))
        d_rel = float(resultado.get("delta_relativo_pct", 0.0))
        _colored_metric(
            rr3,
            "Variação no índice",
            f"{d_abs * 100.0:+.2f} p.p.",
            is_good=(d_abs <= 0),
            help_text=(
                "Diferença bruta entre depois e antes, em pontos percentuais (p.p.). "
                "Verde = sinistralidade caiu; vermelho = subiu. "
                "Exemplo: se a sinistralidade caiu de 100% para 95%, a variação é -5 p.p."
            ),
        )
        _colored_metric(
            rr4,
            "Variação proporcional",
            f"{d_rel:+.2f}%",
            is_good=(d_rel <= 0),
            help_text=(
                "Quanto a sinistralidade caiu ou subiu em termos relativos: "
                "(depois − antes) ÷ antes × 100. Verde = sinistralidade caiu; vermelho = subiu. "
                "Exemplo: cair de 100% para 95% representa -5% de redução proporcional."
            ),
        )

        ee1, ee2 = st.columns(2)
        ee1.metric(
            "Beneficiários afetados",
            f"{int(resultado.get('n_individuos_afetados', 0)):,}",
            help="Quantidade de beneficiários elegíveis à intervenção, ou seja, que tiveram a feature modificada na simulação.",
        )
        ee2.metric(
            "Beneficiários fora do escopo",
            f"{int(resultado.get('n_individuos_nao_elegiveis', 0)):,}",
            help="Beneficiários que não atendem à regra de elegibilidade da intervenção e portanto não tiveram seus valores alterados.",
        )

        por_plano = resultado.get("por_plano", [])
        if isinstance(por_plano, list) and len(por_plano) > 1:
            st.markdown("**Detalhamento por plano**")
            pp_rows = [
                {
                    "Plano": r.get("plano", "-"),
                    "Antes": _fmt_pct(r.get("sinistralidade_antes")),
                    "Depois": _fmt_pct(r.get("sinistralidade_depois")),
                    "Delta p.p.": f"{float(r.get('delta_absoluto', 0)) * 100:+.2f}",
                }
                for r in por_plano
            ]
            st.dataframe(pd.DataFrame(pp_rows), hide_index=True, use_container_width=True)

        intervs = resultado.get("intervencoes", [])
        if isinstance(intervs, list) and intervs:
            st.markdown("**Intervenções aplicadas**")
            iv_df = pd.DataFrame(intervs)
            iv_df["feature_label"] = iv_df.get("feature", iv_df.iloc[:, 0]).astype(str).map(_format_feature_name)
            keep = [c for c in ["feature_label", "feature", "delta_pct"] if c in iv_df.columns]
            st.dataframe(
                iv_df[keep].rename(columns={"feature_label": "Feature", "feature": "Coluna", "delta_pct": "Delta (%)"}),
                hide_index=True,
                use_container_width=True,
            )

    with st.expander("Detalhes técnicos da execução", expanded=not ok):
        st.code(out or "(sem saída)")


PLANOS_SEM_FORECAST = {"MASTER EXECUTIVO"}
_AVISO_SEM_FORECAST = (
    "Forecast indisponível para **MASTER EXECUTIVO**. "
    "O volume de dados deste plano é insuficiente e a qualidade das informações "
    "não permite a geração de um modelo de previsão confiável."
)


def render_previsao_tab() -> None:
    st.header("Previsão")
    st.caption("Forecast mensal (LSTM ensemble, MASTER EMPRESARIAL).")

    vers = list_forecast_versions()
    if not vers:
        st.warning("Nenhum forecast encontrado em data/processed/climazon/forecast.")
        return

    ver = vers[0]

    # Selectbox de plano
    plano_sel_prev = st.selectbox("Visualizar", options=PLANOS_CANONICOS, key="prev_plano")

    # Bloqueia EXECUTIVO
    if plano_sel_prev in PLANOS_SEM_FORECAST:
        st.warning(_AVISO_SEM_FORECAST)
        return

    plano_filter = plano_sel_prev

    try:
        df = load_forecast_csv(ver, plano=plano_filter)
    except Exception as e:
        st.error(f"Erro ao carregar forecast: {e}")
        return

    df = df.copy()
    reg = df.get("REGISTO", pd.Series("", index=df.index)).astype(str).str.lower()
    df["SINISTRALIDADE"] = pd.to_numeric(df.get("SINISTRALIDADE"), errors="coerce")
    df["SINISTRALIDADE_PREVISTA"] = pd.to_numeric(df.get("SINISTRALIDADE_PREVISTA"), errors="coerce")
    hist_mask = reg.isin(["historico", "teste"])
    prev_mask = reg == "previsao"

    df_hist = df[hist_mask & df["SINISTRALIDADE"].notna()].copy()
    df_prev = df[prev_mask & df["SINISTRALIDADE_PREVISTA"].notna()].copy()

    # Próximos 3 meses
    proximos = df_prev.sort_values("DATA").head(3)
    if not proximos.empty:
        st.subheader("Próximos 3 meses")
        cols = st.columns(len(proximos))
        for col, (_, row) in zip(cols, proximos.iterrows()):
            val = float(row["SINISTRALIDADE_PREVISTA"])
            label, color = _sinistralidade_status(val)
            mes_str = pd.Timestamp(row["DATA"]).strftime("%m/%Y")
            # banda de incerteza
            p05 = row.get("forecast_p05", np.nan)
            p95 = row.get("forecast_p95", np.nan)
            ci_str = (
                f"Intervalo provável: {float(p05):.1%} a {float(p95):.1%}"
                if not (pd.isna(p05) or pd.isna(p95)) else ""
            )
            with col:
                st.markdown(
                    f"""
<div style="border:1px solid #e6e6e6;border-radius:12px;padding:1rem;text-align:center;">
  <div style="font-size:0.95rem;font-weight:600;color:#666;">{mes_str}</div>
  <div style="font-size:2rem;font-weight:700;color:{color};margin-top:0.35rem;">{val:.2%}</div>
  <div style="font-size:0.95rem;font-weight:600;color:{color};margin-top:0.25rem;">{label}</div>
  <div style="font-size:0.78rem;color:#888;margin-top:0.15rem;">{ci_str}</div>
</div>
""",
                    unsafe_allow_html=True,
                )

    # Limita forecast aos próximos 3 meses
    df_prev = df_prev.sort_values("DATA").head(3).reset_index(drop=True)

    # Gráfico
    fig = go.Figure()

    # Série histórica completa (historico + teste = observado real)
    df_hist_sorted = df_hist.sort_values("DATA")
    fig.add_trace(go.Scatter(
        x=df_hist_sorted["DATA"],
        y=df_hist_sorted["SINISTRALIDADE"],
        mode="lines+markers",
        name="Histórico",
        line={"color": "#1f77b4", "width": 3},
        marker={"size": 6},
    ))

    # Forecast futuro (conecta ao último ponto observado)
    if not df_prev.empty:
        fut_x = df_prev["DATA"].tolist()
        fut_y = df_prev["SINISTRALIDADE_PREVISTA"].tolist()
        if not df_hist_sorted.empty:
            fut_x = [df_hist_sorted["DATA"].iloc[-1]] + fut_x
            fut_y = [float(df_hist_sorted["SINISTRALIDADE"].iloc[-1])] + fut_y

        fig.add_trace(go.Scatter(
            x=fut_x, y=fut_y,
            mode="lines+markers", name="Previsto",
            line={"color": "#ef5350", "width": 3},
            marker={"size": 6},
        ))

    fig.update_layout(
        title=f"Sinistralidade Climazon · {plano_filter} ({ver})",
        hovermode="x unified",
        legend_title_text="Período",
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        xaxis={
            "dtick": "M12",
            "tickformat": "%Y",
            "tickangle": 0,
        },
    )
    fig.update_yaxes(title="Sinistralidade")
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "O modelo acerta a direção da sinistralidade (se vai subir ou cair) com boa consistência. "
        "O erro médio é de ~18%, ou seja, se a sinistralidade real for 80%, o modelo pode prever entre 65% e 95%. "
        "Por isso, usamos o modelo para antecipar tendências e alertar para meses de risco, "
        "não para substituir o número final do fechamento."
    )

    summary = pd.DataFrame({
        "Série": ["Meses históricos", "Meses previstos", "Início previsão"],
        "Valor": [
            str(len(df_hist)),
            str(len(df_prev)),
            df_prev["DATA"].min().strftime("%Y-%m") if not df_prev.empty else "-",
        ],
    })
    st.dataframe(summary, hide_index=True, use_container_width=True)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    st.set_page_config(page_title="PIAM Analytics · Climazon", layout="wide")

    if not require_login():
        return

    st.title("Inteligência Analítica PIAM · Climazon")

    st.markdown(
        """
<style>
[data-testid="stSidebar"] .piam-footer {
    position: sticky; bottom: 0; left: 0; right: 0;
    margin-top: 1rem; padding: 0.65rem 0 0.25rem 0;
    border-top: 1px solid #b8bcc3; background: transparent;
}
[data-testid="stSidebar"] .piam-footer .piam-title {
    font-size: 1.05rem; font-weight: 700; color: #1f2a44; margin-bottom: 0.25rem;
}
[data-testid="stSidebar"] .piam-footer .piam-profile {
    font-size: 0.9rem; color: #6b7280;
}
[data-testid="stSidebar"][aria-expanded="false"] .piam-footer { display: none; }
</style>
""",
        unsafe_allow_html=True,
    )

    profile_name = st.session_state.get("climazon_authenticated_user", "")

    with st.sidebar:
        menu = option_menu(
            menu_title="Menu",
            options=["Correlação", "Predição", "Previsão", "Sair"],
            icons=["bar-chart-line", "activity", "calendar3", "box-arrow-right"],
            menu_icon="display",
            default_index=1,
            styles={
                "container": {"padding": "0!important", "background-color": "transparent"},
                "icon": {"color": "#5f6368", "font-size": "15px"},
                "nav-link": {
                    "font-size": "14px",
                    "text-align": "left",
                    "margin": "0px",
                    "padding": "10px 10px",
                    "--hover-color": "#f2f3f5",
                },
                "nav-link-selected": {"background-color": "#1565c0", "color": "white"},
                "menu-title": {"font-size": "20px", "font-weight": "700", "padding": "0px 6px 8px 6px"},
            },
        )
        st.markdown(
            f"""
<div class="piam-footer">
  <div class="piam-title">PIAM · Climazon</div>
  <div class="piam-profile">Perfil: {profile_name}</div>
</div>
""",
            unsafe_allow_html=True,
        )

    if menu == "Sair":
        st.session_state.pop("climazon_authenticated_user", None)
        st.rerun()
    elif menu == "Correlação":
        render_correlacao_tab()
    elif menu == "Predição":
        render_predicao_tab()
    else:
        render_previsao_tab()


if __name__ == "__main__":
    main()
