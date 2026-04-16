from __future__ import annotations

import contextlib
import json
import os
import re
import sys
import types
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_option_menu import option_menu

ROOT_DIR = Path(__file__).resolve().parents[1]
PREDICT_ROOT = ROOT_DIR / "data" / "processed" / "elgin" / "predict"
FEATURE_IMPACT_CSV = (
    ROOT_DIR
    / "data"
    / "processed"
    / "elgin"
    / "feature-impact"
    / "v3"
    / "feature_correlation_sinistralidade.csv"
)
FORECAST_CSV = (
    ROOT_DIR
    / "data"
    / "processed"
    / "elgin"
    / "forecast"
    / "v1"
    / "sinistralidade_forecast_completo.csv"
)
SIMULATION_SIZE = 50_000
DEFAULT_MODEL_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "elgin"
    / "predict"
    / "v12"
    / "models"
    / "model_MASTER_EMPRESARIAL.pkl"
)
AUTH_EMAIL = "marso@email.com"
AUTH_PASSWORD = "1234"


def require_login() -> bool:
    if st.session_state.get("authenticated_user"):
        return True

    st.title("Acesso a plataforma")
    st.caption("Informe login e senha para continuar.")

    with st.form("login_form", clear_on_submit=False):
        email = st.text_input("Login", value="")
        password = st.text_input("Senha", value="", type="password")
        submitted = st.form_submit_button("Entrar", type="primary")

    if submitted:
        if email.strip().lower() == AUTH_EMAIL and password == AUTH_PASSWORD:
            st.session_state["authenticated_user"] = AUTH_EMAIL
            st.success("Login realizado com sucesso.")
            st.rerun()
        else:
            st.error("Login ou senha invalidos.")

    return False

class TwoStageModel:
    """
    Classe de compatibilidade para desserializar modelos legados (.pkl/.joblib)
    treinados no pipeline de predição agregada.
    """

    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self._smearing = 1.0
        self.macro_scale = 1.0
        self.clf_ = None
        self.reg_ = None

    def _predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        p_pos = self.clf_.predict_proba(X)[:, 1]
        log_y = self.reg_.predict(X)
        return p_pos * (np.exp(log_y) * self._smearing)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._predict_raw(X) * float(getattr(self, "macro_scale", 1.0))

    def predict_stages(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        p_pos = self.clf_.predict_proba(X)[:, 1]
        scale = float(getattr(self, "macro_scale", 1.0))
        y_hat = self._predict_raw(X) * scale
        return p_pos, y_hat


@contextlib.contextmanager
def _main_module_for_legacy_joblib():
    """
    Modelos legados podem apontar para __main__.TwoStageModel ou
    predict_agregado_dyn.TwoStageModel. Criamos módulos shim em memória para
    desserializar sem importar (e executar) pipelines.
    """
    real_main = sys.modules.get("__main__")
    real_dyn = sys.modules.get("predict_agregado_dyn")

    shim_main = types.ModuleType("__main__")
    shim_main.TwoStageModel = TwoStageModel
    shim_dyn = types.ModuleType("predict_agregado_dyn")
    shim_dyn.TwoStageModel = TwoStageModel

    sys.modules["__main__"] = shim_main
    sys.modules["predict_agregado_dyn"] = shim_dyn
    try:
        yield
    finally:
        if real_main is not None:
            sys.modules["__main__"] = real_main
        else:
            sys.modules.pop("__main__", None)
        if real_dyn is not None:
            sys.modules["predict_agregado_dyn"] = real_dyn
        else:
            sys.modules.pop("predict_agregado_dyn", None)


@st.cache_data(show_spinner=False)
def _latest_catalog_path() -> Path:
    candidates = sorted(PREDICT_ROOT.glob("v*/catalogo_perfis_top100.json"))
    if not candidates:
        raise FileNotFoundError("Nenhum catalogo_perfis_top100.json encontrado em data/processed/elgin/predict")

    def _version_key(path: Path) -> int:
        m = re.search(r"v(\d+)", str(path.parent.name), flags=re.IGNORECASE)
        return int(m.group(1)) if m else -1

    return max(candidates, key=_version_key)


@st.cache_data(show_spinner=False)
def load_profiles() -> pd.DataFrame:
    catalog_path = _latest_catalog_path()
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    perfis = payload.get("perfis", [])
    if not perfis:
        raise ValueError(f"Catalogo vazio em {catalog_path}")

    rows: list[dict[str, Any]] = []
    for p in perfis:
        resumo = p.get("resumo", {})
        row = {
            "profile_id": str(p.get("label", f"#{p.get('rank', '?')}")),
            "rank": int(p.get("rank", -1)),
            "indice_parquet": int(p.get("indice_parquet", -1)),
            "plano": str(resumo.get("plano", "")),
            "sinistralidade_final": float(resumo.get("sinistralidade_final", 0.0) or 0.0),
            "valor_faturamento": float(resumo.get("valor_faturamento", 0.0) or 0.0),
            "n_meses_obs": int(resumo.get("n_meses_obs", 0) or 0),
            "payload": p.get("payload", {}),
        }
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(["rank", "profile_id"]).reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def load_feature_impact_csv() -> pd.DataFrame:
    if not FEATURE_IMPACT_CSV.is_file():
        raise FileNotFoundError(f"Arquivo nao encontrado: {FEATURE_IMPACT_CSV}")
    df = pd.read_csv(FEATURE_IMPACT_CSV)
    req = {"feature", "spearman"}
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"CSV sem colunas obrigatorias: {sorted(miss)}")
    df["spearman"] = pd.to_numeric(df["spearman"], errors="coerce")
    df = df.dropna(subset=["spearman"])
    df["abs_spearman"] = df["spearman"].abs()
    return df.sort_values("abs_spearman", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_forecast_csv() -> pd.DataFrame:
    if not FORECAST_CSV.is_file():
        raise FileNotFoundError(f"Arquivo nao encontrado: {FORECAST_CSV}")
    df = pd.read_csv(FORECAST_CSV)
    if "DATA" not in df.columns:
        raise ValueError("CSV de forecast sem coluna DATA")
    df["DATA"] = pd.to_datetime(df["DATA"], errors="coerce")
    return df.dropna(subset=["DATA"]).sort_values("DATA").reset_index(drop=True)


def _counts_from_percentages(percentages: dict[str, float], total: int) -> dict[str, int]:
    keys = list(percentages.keys())
    raw = np.array([max(0.0, percentages[k]) for k in keys], dtype=float)
    if raw.sum() == 0:
        raise ValueError("Defina pelo menos uma proporcao > 0%")
    normalized = raw / raw.sum()
    counts = np.floor(normalized * total).astype(int)
    remainder = int(total - counts.sum())
    if remainder > 0:
        frac = normalized * total - counts
        order = np.argsort(-frac)
        for i in order[:remainder]:
            counts[i] += 1
    return {k: int(c) for k, c in zip(keys, counts)}


def _prepare_input_for_two_stage_model(model: Any, raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Alinha colunas/tipos com o treino do TwoStageModel (LightGBM).
    """
    expected_cols = list(getattr(model, "feature_names_", []))
    if not expected_cols:
        return raw_df.copy()

    X = raw_df.copy()
    for c in expected_cols:
        if c not in X.columns:
            X[c] = np.nan
    X = X[expected_cols].copy()

    cat_values = []
    try:
        cat_values = list(getattr(model.clf_.booster_, "pandas_categorical", []) or [])
    except Exception:
        cat_values = []

    # Ordem esperada do treino agregado: faixa_etaria, sexo, tipo_cadastro.
    preferred_cat_cols = [c for c in ("faixa_etaria", "sexo", "tipo_cadastro") if c in X.columns]
    cat_cols = preferred_cat_cols[: len(cat_values)] if cat_values else []

    for idx, col in enumerate(cat_cols):
        allowed = list(cat_values[idx])
        fill_value = allowed[0] if allowed else "missing"
        s = X[col].astype(str).replace({"nan": fill_value, "None": fill_value}).fillna(fill_value)
        X[col] = pd.Categorical(s, categories=allowed if allowed else None)

    for c in X.columns:
        if c in cat_cols:
            continue
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)

    return X


def _maybe_predict_payloads(unique_payload_df: pd.DataFrame) -> tuple[pd.Series | None, str]:
    model_path = (os.getenv("STREAMLIT_MODEL_PATH") or "").strip()
    if not model_path and DEFAULT_MODEL_PATH.is_file():
        model_path = str(DEFAULT_MODEL_PATH)
    if not model_path:
        return None, "Modo estimativa: modelo nao encontrado (defina STREAMLIT_MODEL_PATH)."

    path = Path(model_path)
    try:
        if path.is_file() and path.suffix.lower() in {".pkl", ".joblib"}:
            import joblib

            with _main_module_for_legacy_joblib():
                model = joblib.load(path)
            X = _prepare_input_for_two_stage_model(model, unique_payload_df)
            preds = pd.Series(model.predict(X), index=unique_payload_df.index, dtype=float)
            return preds, f"Inferencia com modelo local: {path.name}"

        import mlflow.pyfunc

        pyfunc_model = mlflow.pyfunc.load_model(model_path)
        out = pyfunc_model.predict(unique_payload_df)
        if isinstance(out, pd.DataFrame):
            if "sinistralidade_prevista" in out.columns:
                preds = pd.to_numeric(out["sinistralidade_prevista"], errors="coerce")
            else:
                preds = pd.to_numeric(out.iloc[:, 0], errors="coerce")
        else:
            preds = pd.to_numeric(pd.Series(out), errors="coerce")
        preds = preds.astype(float)
        preds.index = unique_payload_df.index
        return preds, f"Inferencia com pyfunc: {model_path}"
    except Exception as e:
        return None, f"Modo estimativa: falha ao carregar modelo ({e})"


def _simulate_cohort(
    profiles_df: pd.DataFrame, percentages: dict[str, float]
) -> tuple[pd.DataFrame, str, dict[str, float]]:
    counts = _counts_from_percentages(percentages, SIMULATION_SIZE)
    selected = profiles_df[profiles_df["profile_id"].isin(percentages.keys())].copy()
    selected = selected.set_index("profile_id")

    cohort_rows: list[dict[str, Any]] = []
    expanded_payload_rows: list[dict[str, Any]] = []
    expanded_profile_ids: list[str] = []

    for pid, qty in counts.items():
        if qty <= 0:
            continue
        base = selected.loc[pid]
        payload = dict(base["payload"])
        expanded_payload_rows.extend([payload] * qty)
        expanded_profile_ids.extend([pid] * qty)
        cohort_rows.append(
            {
                "profile_id": pid,
                "qtd_usuarios": qty,
                "plano": base["plano"],
                "sinistralidade_historica_perfil": float(base["sinistralidade_final"]),
                "valor_faturamento_perfil": float(base["valor_faturamento"]),
                "percentual_entrada": float(percentages[pid]),
            }
        )

    cohort_df = pd.DataFrame(cohort_rows)
    payload_df = pd.DataFrame(expanded_payload_rows)

    pred_series, mode_msg = _maybe_predict_payloads(payload_df)
    if pred_series is None:
        fallback_map = selected["sinistralidade_final"].astype(float).to_dict()
        expanded_pred = pd.Series([fallback_map[pid] for pid in expanded_profile_ids], dtype=float)
    else:
        expanded_pred = pred_series.reset_index(drop=True).astype(float)

    expanded_df = pd.DataFrame(
        {
            "profile_id": expanded_profile_ids,
            "sinistralidade_prevista": expanded_pred.values,
        }
    )
    expanded_df["valor_faturamento"] = expanded_df["profile_id"].map(
        selected["valor_faturamento"].astype(float).to_dict()
    )
    expanded_df["produto"] = expanded_df["sinistralidade_prevista"] * expanded_df["valor_faturamento"]

    agg = (
        expanded_df.groupby("profile_id", as_index=False)
        .agg(
            sinistralidade_prevista_perfil=("sinistralidade_prevista", "mean"),
            valor_faturamento_total=("valor_faturamento", "sum"),
            produto_total=("produto", "sum"),
        )
    )
    cohort_df = cohort_df.merge(agg, on="profile_id", how="left")

    cohort_df["sinistralidade_prevista_total"] = (
        cohort_df["sinistralidade_prevista_perfil"] * cohort_df["qtd_usuarios"]
    )
    numerador = float(expanded_df["produto"].sum())
    denominador = float(expanded_df["valor_faturamento"].sum())
    media_ponderada = float(numerador / denominador) if denominador else float("nan")

    summary = {
        "sinistralidade_media_ponderada": media_ponderada,
        "faturamento_total_base": denominador,
        "numerador_ponderado": numerador,
        "volume_vidas_simuladas": float(len(expanded_df)),
    }
    return cohort_df.sort_values("qtd_usuarios", ascending=False).reset_index(drop=True), mode_msg, summary


def _sinistralidade_status(pct: float) -> tuple[str, str]:
    """
    Faixas de referência (operadora): aceitável ~70–75%.
    Retorna (rótulo, cor CSS).
    """
    if pct < 0:
        return "Valor negativo", "#c62828"
    if pct < 70.0:
        return "Boa sinistralidade", "#2e7d32"
    if pct <= 75.0:
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
        "qtd_conta_PRONTO SOCORRO": "Pronto socorro",
        "qtd_esp_outros": "Esp. outros",
        "qtd_conta_ATENDIMENTO AMBULATORIAL": "Atend. amb.",
        "qtd_servico_LABORATÓRIO": "Laboratório",
        "qtd_conta_EXTERNO": "Conta externa",
        "qtd_servico_ULTRA-SONOGRAFIA": "Ultrassom",
        "qtd_servico_RADIOLOGIA": "Radiologia",
        "qtd_esp_lab_imagem": "Lab. imagem",
        "qtd_servico___OUTROS__": "Serv. outros",
        "qtd_esp_cardio": "Esp. cardio",
        "qtd_esp_gine": "Esp. gineco",
        "qtd_esp_oftal": "Esp. oftalmo",
        "qtd_servico_TOMOGRAFIA": "Tomografia",
        "qtd_esp_orto": "Esp. ortopedia",
        "qtd_conta_INTERNADO": "Conta internação",
        "qtd_servico_CARDIOLOGIA": "Serv. cardio",
        "qtd_servico_CIRURGICO": "Serv. cirúrgico",
        "qtd_servico_FISIOTERAPIA": "Fisioterapia",
        "valor_faturamento": "Faturamento",
        "qtd_servico_CLÍNICO": "Serv. clínico",
    }
    if feature in mapping:
        return mapping[feature]
    label = feature.replace("tx_", "taxa ").replace("qtd_", "").replace("_", " ").strip()
    words = [part.capitalize() for part in label.split() if part]
    if not words:
        return feature
    return " ".join(words[:2])


def _format_feature_value(payload: dict[str, Any], feature: str) -> str:
    if feature == "sexo":
        value = payload.get("sexo")
        if value is None and "is_fem" in payload:
            return "F" if int(payload.get("is_fem", 0)) == 1 else "M"
        return str(value) if value is not None else "-"

    if feature == "idade":
        value = payload.get("idade")
        return f"{int(value)} anos" if value is not None and not pd.isna(value) else "-"

    value = payload.get(feature)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    if isinstance(value, float):
        if abs(value) >= 100:
            return f"{value:,.0f}".replace(",", ".")
        if abs(value) >= 1:
            return f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        return f"{value:.4f}".replace(".", ",")
    return str(value)


def _top_card_features(feature_impact_df: pd.DataFrame, payload: dict[str, Any]) -> list[str]:
    excluded = {"idade", "sexo", "valor_faturamento"}
    ranked = feature_impact_df["feature"].tolist()
    return [f for f in ranked if f in payload and f not in excluded][:2]


def _get_prediction_selection_state() -> tuple[dict[str, bool], dict[str, float]]:
    selected_map = st.session_state.setdefault("prediction_selected_map", {})
    percentage_map = st.session_state.setdefault("prediction_percentage_map", {})
    return selected_map, percentage_map


@st.dialog("Detalhes do perfil", width="large")
def _show_profile_modal(profile: dict[str, Any], highlight_features: list[str]) -> None:
    st.subheader(str(profile["profile_id"]))
    st.caption(f"Plano: {profile['plano']}")

    payload = dict(profile["payload"])
    shown_front = {"idade", "sexo", *highlight_features}
    details_rows = [
        {"Feature": _format_feature_name(k), "Valor": _format_feature_value(payload, k)}
        for k in payload.keys()
        if k not in shown_front
    ]
    if details_rows:
        st.dataframe(pd.DataFrame(details_rows), width="stretch", hide_index=True)
    else:
        st.info("Sem outras features para exibir.")


def _render_prediction_results() -> None:
    result_df = st.session_state.get("prediction_result_df")
    summary = st.session_state.get("prediction_summary")
    mode_msg = st.session_state.get("prediction_mode_msg")

    if result_df is None or summary is None:
        st.session_state["prediction_step"] = "selection"
        st.rerun()

    st.subheader("Etapa 2 de 2: Resultado da inferência")
    st.caption("Resultado da simulação com base nos perfis e proporções selecionados.")

    if st.button("Voltar para seleção de perfis"):
        st.session_state["prediction_step"] = "selection"
        st.rerun()

    st.success(mode_msg)
    total_users = int(summary["volume_vidas_simuladas"])
    weighted_pred = float(summary["sinistralidade_media_ponderada"])

    label, color = _sinistralidade_status(weighted_pred)
    num_color = "#c62828" if weighted_pred < 0 else color
    st.markdown(
        f"""
<div style="text-align:center;padding:1rem 0;">
  <div style="font-size:2.4rem;font-weight:700;color:{num_color};">{weighted_pred:.2f}%</div>
  <div style="font-size:1.15rem;font-weight:600;color:{color};margin-top:0.35rem;">{label}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    out = result_df.copy()
    out["peso_real_%"] = 100.0 * out["qtd_usuarios"] / total_users
    out["peso_real_%"] = out["peso_real_%"].round(2)
    out["sinistralidade_historica_%"] = out["sinistralidade_historica_perfil"].round(2).astype(str) + "%"
    out["sinistralidade_prevista_%"] = out["sinistralidade_prevista_perfil"].round(2).astype(str) + "%"
    st.dataframe(
        out[
            [
                "profile_id",
                "plano",
                "peso_real_%",
                "sinistralidade_historica_%",
                "sinistralidade_prevista_%",
            ]
        ],
        width="stretch",
        hide_index=True,
    )

    chart_df = out.sort_values("sinistralidade_prevista_perfil", ascending=False).head(15)
    st.bar_chart(
        chart_df.set_index("profile_id")["sinistralidade_prevista_perfil"],
        width="stretch",
    )


def render_prediction_tab() -> None:
    st.header("Predição agregada")
    st.caption("Etapa 1: selecione perfis do catalogo e defina a proporcao de cada um na carteira simulada.")

    profiles = load_profiles()
    feature_impact_df = load_feature_impact_csv()
    st.session_state.setdefault("prediction_step", "selection")
    selected_map, percentage_map = _get_prediction_selection_state()

    if st.session_state["prediction_step"] == "results":
        _render_prediction_results()
        return

    total_pages = 5
    st.session_state.setdefault("prediction_catalog_page", 1)
    selected_page = int(st.session_state["prediction_catalog_page"])

    st.caption("Página de perfis")
    nav_cols = st.columns([1.6, 0.7, 0.7, 0.7, 0.7, 0.7, 1.6])
    with nav_cols[0]:
        if st.button("◀ Anterior", key="pred_page_prev", disabled=selected_page <= 1):
            st.session_state["prediction_catalog_page"] = max(1, selected_page - 1)
            st.rerun()
    for i in range(1, total_pages + 1):
        with nav_cols[i]:
            if st.button(
                str(i),
                key=f"pred_page_{i}",
                type="primary" if selected_page == i else "secondary",
            ):
                st.session_state["prediction_catalog_page"] = i
                st.rerun()
    with nav_cols[6]:
        if st.button("Próximo ▶", key="pred_page_next", disabled=selected_page >= total_pages):
            st.session_state["prediction_catalog_page"] = min(total_pages, selected_page + 1)
            st.rerun()

    selected_page = int(st.session_state["prediction_catalog_page"])

    page_size = 20
    start_idx = (selected_page - 1) * page_size
    end_idx = start_idx + page_size
    profiles_page = profiles.iloc[start_idx:end_idx].copy()

    st.write(f"Exibindo perfis **{start_idx + 1} a {min(end_idx, len(profiles))}** do catálogo top 100.")

    card_cols = st.columns(4)
    for idx, (_, row) in enumerate(profiles_page.iterrows()):
        profile = row.to_dict()
        payload = dict(profile["payload"])
        pid = str(profile["profile_id"])
        highlight_features = _top_card_features(feature_impact_df, payload)
        current_selected = bool(selected_map.get(pid, False))
        current_pct = float(percentage_map.get(pid, 0.0))

        front_items = [
            ("Idade", _format_feature_value(payload, "idade")),
            ("Sexo", _format_feature_value(payload, "sexo")),
        ]
        front_items.extend(
            [(_format_feature_name(feature), _format_feature_value(payload, feature)) for feature in highlight_features]
        )

        with card_cols[idx % 4]:
            with st.container(border=True):
                st.markdown(f"#### {pid}")
                st.caption(f"Plano: {profile['plano']}")
                for label, value in front_items:
                    st.write(f"**{label}:** {value}")

                selected = st.checkbox("Selecionar", value=current_selected, key=f"ui_selected_{pid}")
                selected_map[pid] = bool(selected)
                if selected:
                    percentage_map[pid] = float(
                        st.number_input(
                            "Proporcao (%)",
                            min_value=0.0,
                            max_value=100.0,
                            value=current_pct,
                            step=0.1,
                            key=f"ui_pct_{pid}",
                        )
                    )
                else:
                    percentage_map[pid] = 0.0
                    st.number_input(
                        "Proporcao (%)",
                        min_value=0.0,
                        max_value=100.0,
                        value=0.0,
                        step=0.1,
                        key=f"ui_pct_{pid}",
                        disabled=True,
                    )

                if st.button("Ver detalhes", key=f"detail_{pid}"):
                    _show_profile_modal(profile, highlight_features)

    selected_profiles = [
        pid
        for pid in profiles["profile_id"].tolist()
        if bool(selected_map.get(pid, False))
    ]
    percentages = {pid: float(percentage_map.get(pid, 0.0)) for pid in selected_profiles}
    total_pct = float(sum(percentages.values()))

    st.divider()
    st.subheader("Resumo da seleção")
    if st.button("Limpar seleção", key="clear_prediction_selection"):
        selected_map.clear()
        percentage_map.clear()
        for pid in profiles["profile_id"].tolist():
            st.session_state.pop(f"ui_selected_{pid}", None)
            st.session_state.pop(f"ui_pct_{pid}", None)
        st.rerun()

    st.write(f"Perfis selecionados: **{len(selected_profiles)}**")
    st.write(f"Soma informada: **{total_pct:.2f}%** (normalizada internamente para 100%).")

    if selected_profiles:
        selection_df = profiles[profiles["profile_id"].isin(selected_profiles)][["profile_id", "plano"]].copy()
        selection_df["proporcao_%"] = selection_df["profile_id"].map(percentages).fillna(0.0)
        selection_df["proporcao_%"] = selection_df["proporcao_%"].round(2)
        st.dataframe(selection_df, width="stretch", hide_index=True)
    else:
        st.info("Selecione ao menos um perfil para habilitar a simulação.")

    if st.button("Executar simulacao de inferencia", type="primary", disabled=not selected_profiles):
        try:
            result_df, mode_msg, summary = _simulate_cohort(profiles, percentages)
        except Exception as e:
            st.error(f"Falha ao simular carteira: {e}")
            return

        st.session_state["prediction_result_df"] = result_df
        st.session_state["prediction_mode_msg"] = mode_msg
        st.session_state["prediction_summary"] = summary
        st.session_state["prediction_step"] = "results"
        st.rerun()


def render_feature_impact_tab() -> None:
    st.header("Correlação")
    st.caption("Analise estatica com base no CSV consolidado de correlacao Spearman.")
    df = load_feature_impact_csv()
    df = df.copy()
    df["feature_label"] = df["feature"].map(_format_feature_name)

    st.metric("Total de features", len(df))
    st.dataframe(
        df[["feature_label", "spearman", "abs_spearman"]].rename(
            columns={
                "feature_label": "Feature",
                "spearman": "Spearman",
                "abs_spearman": "|Spearman|",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    top_n = st.slider("Top N por |Spearman|", min_value=5, max_value=min(30, len(df)), value=15)
    top = df.head(top_n).sort_values("spearman")
    top["sentido"] = np.where(top["spearman"] >= 0, "Positivo", "Negativo")

    fig = px.bar(
        top,
        x="spearman",
        y="feature_label",
        orientation="h",
        color="sentido",
        color_discrete_map={"Positivo": "#2e7d32", "Negativo": "#c62828"},
        title=f"Top {top_n} features por |Spearman|",
        labels={
            "feature_label": "Feature",
            "spearman": "Spearman",
            "sentido": "Impacto",
        },
        hover_data={"feature": True, "feature_label": False, "spearman": ":.4f"},
    )
    fig.update_layout(
        height=max(420, top_n * 28),
        yaxis={"categoryorder": "array", "categoryarray": top["feature_label"].tolist()},
        legend_title_text="Impacto",
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
    )
    fig.update_traces(marker_line_width=0)
    st.plotly_chart(fig, width="stretch")


def render_forecast_tab() -> None:
    st.header("Previsão")
    st.caption("Serie mensal agregada a partir dos dados diarios, com destaque para o periodo previsto.")

    df = load_forecast_csv()
    df = df.copy()
    reg = df.get("REGISTO", pd.Series("", index=df.index)).astype(str).str.lower()
    competencia = df.get("COMPETENCIA", pd.Series("", index=df.index)).astype(str)
    df["competencia_mes"] = pd.to_datetime(competencia + "-01", errors="coerce")
    df["SINISTRALIDADE"] = pd.to_numeric(df.get("SINISTRALIDADE"), errors="coerce")
    df["SINISTRALIDADE_PREVISTA"] = pd.to_numeric(df.get("SINISTRALIDADE_PREVISTA"), errors="coerce")

    observed_daily = df[reg.isin(["historico", "teste"]) & df["SINISTRALIDADE"].notna()].copy()
    forecast_daily = df[reg.eq("previsao") & df["SINISTRALIDADE_PREVISTA"].notna()].copy()

    observed_monthly = (
        observed_daily.groupby("competencia_mes", as_index=False)
        .agg(sinistralidade_mensal=("SINISTRALIDADE", "mean"))
    )
    observed_monthly["periodo"] = "Histórico"

    forecast_monthly = (
        forecast_daily.groupby("competencia_mes", as_index=False)
        .agg(sinistralidade_mensal=("SINISTRALIDADE_PREVISTA", "mean"))
    )
    forecast_monthly["periodo"] = "Previsto"

    forecast_monthly = forecast_monthly.sort_values("competencia_mes").head(3).reset_index(drop=True)
    if not forecast_monthly.empty:
        st.subheader("Próximos 3 meses")
        metric_cols = st.columns(len(forecast_monthly))
        for col, (_, row) in zip(metric_cols, forecast_monthly.iterrows()):
            value = float(row["sinistralidade_mensal"])
            label, color = _sinistralidade_status(value)
            with col:
                st.markdown(
                    f"""
<div style="border:1px solid #e6e6e6;border-radius:12px;padding:1rem;text-align:center;">
  <div style="font-size:0.95rem;font-weight:600;color:#666;">{row["competencia_mes"].strftime("%m/%Y")}</div>
  <div style="font-size:2rem;font-weight:700;color:{color};margin-top:0.35rem;">{value:.2f}%</div>
  <div style="font-size:0.95rem;font-weight:600;color:{color};margin-top:0.35rem;">{label}</div>
</div>
""",
                    unsafe_allow_html=True,
                )

    observed_plot = observed_monthly.sort_values("competencia_mes").rename(
        columns={"competencia_mes": "Mês", "sinistralidade_mensal": "Sinistralidade"}
    )
    forecast_plot = forecast_monthly.sort_values("competencia_mes").rename(
        columns={"competencia_mes": "Mês", "sinistralidade_mensal": "Sinistralidade"}
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=observed_plot["Mês"],
            y=observed_plot["Sinistralidade"],
            mode="lines+markers",
            name="Histórico",
            line={"color": "#1f77b4", "width": 3},
            marker={"size": 6},
        )
    )

    if not forecast_plot.empty:
        forecast_x = forecast_plot["Mês"].tolist()
        forecast_y = forecast_plot["Sinistralidade"].tolist()
        if not observed_plot.empty:
            forecast_x = [observed_plot["Mês"].iloc[-1], *forecast_x]
            forecast_y = [observed_plot["Sinistralidade"].iloc[-1], *forecast_y]

        fig.add_trace(
            go.Scatter(
                x=forecast_x,
                y=forecast_y,
                mode="lines+markers",
                name="Previsto",
                line={"color": "#ef5350", "width": 3},
                marker={"size": 6},
            )
        )

    fig.update_layout(
        hovermode="x unified",
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        legend_title_text="Período",
        title="Sinistralidade mensal",
        xaxis={
            "dtick": "M12",
            "tickformat": "%Y",
            "tickangle": 0,
        },
    )
    fig.update_yaxes(title="Sinistralidade (%)")
    st.plotly_chart(fig, width="stretch")

    summary = pd.DataFrame(
        {
            "Série": ["Meses históricos", "Meses previstos", "Início previsão"],
            "Valor": [
                str(int(len(observed_monthly))),
                str(int(len(forecast_monthly))),
                forecast_monthly["competencia_mes"].min().strftime("%Y-%m") if not forecast_monthly.empty else "-",
            ],
        }
    )
    st.dataframe(summary, hide_index=True, width="stretch")


def main() -> None:
    st.set_page_config(page_title="Elgin - Analises", layout="wide")

    if not require_login():
        return

    st.title("PIAM - Inteligência Analítica")
    profile_name = st.session_state.get("authenticated_user", AUTH_EMAIL)
    st.markdown(
        """
<style>
[data-testid="stSidebar"] .piam-footer {
    position: sticky;
    bottom: 0;
    left: 0;
    right: 0;
    margin-top: 1rem;
    padding: 0.65rem 0 0.25rem 0;
    border-top: 1px solid #b8bcc3;
    background: transparent;
}
[data-testid="stSidebar"] .piam-footer .piam-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #1f2a44;
    margin-bottom: 0.25rem;
}
[data-testid="stSidebar"] .piam-footer .piam-profile {
    font-size: 0.9rem;
    color: #6b7280;
}
[data-testid="stSidebar"][aria-expanded="false"] .piam-footer {
    display: none;
}
</style>
""",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        menu = option_menu(
            menu_title="Menu",
            options=["Correlação", "Predição", "Previsão", "Sair"],
            icons=["bar-chart-line", "activity", "graph-up-arrow", "box-arrow-right"],
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
                "nav-link-selected": {"background-color": "#ff4757", "color": "white"},
                "menu-title": {"font-size": "20px", "font-weight": "700", "padding": "0px 6px 8px 6px"},
            },
        )
        st.markdown(
            f"""
<div class="piam-footer">
  <div class="piam-title">PIAM</div>
  <div class="piam-profile">Perfil: {profile_name}</div>
</div>
""",
            unsafe_allow_html=True,
        )

    if menu == "Sair":
        st.session_state.pop("authenticated_user", None)
        st.rerun()

    if menu == "Predição":
        render_prediction_tab()
    elif menu == "Correlação":
        render_feature_impact_tab()
    else:
        render_forecast_tab()


if __name__ == "__main__":
    main()
