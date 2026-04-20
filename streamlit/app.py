from __future__ import annotations

import contextlib
import json
import os
import re
import subprocess
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
FEATURE_IMPACT_ROOT = ROOT_DIR / "data" / "processed" / "elgin" / "feature-impact"
FORECAST_ROOT = ROOT_DIR / "data" / "processed" / "elgin" / "forecast"
RAW_PANEL_PARQUET = (
    ROOT_DIR / "data" / "raw" / "elgin" / "base_analitica" / "painel_sinistralidade_v1.parquet"
)
SIMULATION_SIZE = 50_000
AUTH_EMAIL = "marso@email.com"
AUTH_PASSWORD = "1234"
QUASI_LEAKAGE_HIDE: set[str] = {
    "qtd_eventos_sinistro",
    "qtd_carater_eletivo",
    "qtd_carater_urgencia",
}

# Modo de idade (what-if): valores enviados ao pipeline devem ser "simples" | "correlacionado"
MODO_IDADE_LABEL: dict[str, str] = {
    "simples": "Só idade (efeito isolado)",
    "correlacionado": "Idade + correlacionados (cenário plausível)",
}
MODO_IDADE_RADIO_HELP = (
    "**Só idade:** altera apenas a coluna `idade`; o restante do painel do mês permanece igual. "
    "Útil para **auditoria** e para ver o **efeito marginal** da idade, sem impor outras hipóteses.\n\n"
    "**Idade + correlacionados:** além da idade, o motor aplica **ajustes amortecidos** em variáveis "
    "associadas (regras heurísticas do what-if). Tende a ser **mais rico em narrativa operacional**, "
    "por imitar um perfil que se move em conjunto."
)


def _fmt_sinistralidade_index_pct(value: Any, *, nd: int = 2) -> str:
    """Índice em razão (ex.: 0,976 ou 1,071) → percentagem na UI (97,60% ou 107,10%)."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "-"
    if np.isnan(x):
        return "-"
    return f"{x * 100.0:.{nd}f}%"


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


def _latest_feature_impact_version_dir() -> Path:
    if not FEATURE_IMPACT_ROOT.is_dir():
        raise FileNotFoundError(f"Diretório não encontrado: {FEATURE_IMPACT_ROOT}")
    candidates = [p for p in FEATURE_IMPACT_ROOT.iterdir() if p.is_dir() and re.fullmatch(r"v\d+", p.name, re.IGNORECASE)]
    if not candidates:
        raise FileNotFoundError("Nenhuma versão vN encontrada em data/processed/elgin/feature-impact")
    return max(candidates, key=lambda p: _version_key(p.name))


def list_feature_impact_competencias() -> tuple[Path, list[str]]:
    ver_dir = _latest_feature_impact_version_dir()
    comps = [p.name for p in ver_dir.iterdir() if p.is_dir() and re.fullmatch(r"\d{4}-\d{2}", p.name)]
    comps = sorted(comps)
    return ver_dir, comps


def load_feature_impact_csv(competencia: str | None = None) -> tuple[pd.DataFrame, str]:
    # Novo formato: vN/<YYYY-MM>/feature_correlation_sinistralidade.csv
    ver_dir, comps = list_feature_impact_competencias()
    if comps:
        comp_sel = competencia if competencia in comps else comps[-1]
        csv_path = ver_dir / comp_sel / "feature_correlation_sinistralidade.csv"
    else:
        # Fallback legado (v3 único CSV)
        comp_sel = "legado"
        csv_path = FEATURE_IMPACT_CSV

    if not csv_path.is_file():
        raise FileNotFoundError(f"Arquivo nao encontrado: {csv_path}")

    df = pd.read_csv(csv_path)
    req = {"feature", "spearman"}
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"CSV sem colunas obrigatorias: {sorted(miss)}")
    df["spearman"] = pd.to_numeric(df["spearman"], errors="coerce")
    df = df.dropna(subset=["spearman"])
    df["abs_spearman"] = df["spearman"].abs()
    return df.sort_values("abs_spearman", ascending=False).reset_index(drop=True), comp_sel


def load_forecast_csv() -> pd.DataFrame:
    if not FORECAST_ROOT.is_dir():
        raise FileNotFoundError(f"Diretório não encontrado: {FORECAST_ROOT}")
    version_dirs = [p for p in FORECAST_ROOT.iterdir() if p.is_dir() and re.fullmatch(r"v\d+", p.name, re.IGNORECASE)]
    if not version_dirs:
        raise FileNotFoundError(f"Nenhuma versão vN encontrada em {FORECAST_ROOT}")
    latest_dir = max(version_dirs, key=lambda p: _version_key(p.name))
    forecast_csv = latest_dir / "sinistralidade_forecast_completo.csv"
    if not forecast_csv.is_file():
        raise FileNotFoundError(f"Arquivo nao encontrado: {forecast_csv}")
    df = pd.read_csv(forecast_csv)
    if "DATA" not in df.columns:
        raise ValueError("CSV de forecast sem coluna DATA")
    df["DATA"] = pd.to_datetime(df["DATA"], errors="coerce")
    return df.dropna(subset=["DATA"]).sort_values("DATA").reset_index(drop=True)


def _version_key(name: str) -> int:
    m = re.search(r"v(\d+)", str(name), flags=re.IGNORECASE)
    return int(m.group(1)) if m else -1


def list_predict_versions() -> list[str]:
    if not PREDICT_ROOT.is_dir():
        return []
    versions = [p.name for p in PREDICT_ROOT.iterdir() if p.is_dir() and re.fullmatch(r"v\d+", p.name, re.IGNORECASE)]
    return sorted(versions, key=_version_key, reverse=True)


@st.cache_data(show_spinner=False)
def latest_local_model_path() -> Path | None:
    versions = list_predict_versions()
    for ver in versions:
        models_dir = PREDICT_ROOT / ver / "models"
        if not models_dir.is_dir():
            continue
        files = sorted(models_dir.glob("model_*.pkl"))
        if files:
            return files[0]
    return None


@st.cache_data(show_spinner=False)
def load_features_catalog(version_label: str) -> dict[str, Any]:
    path = PREDICT_ROOT / version_label / "catalogo_features_intervencao.json"
    if not path.is_file():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "grupos" not in payload:
        raise ValueError(f"Catálogo inválido em {path}: campo 'grupos' ausente")
    return payload


def _run_predict_features(
    *,
    version_label: str,
    feature: str,
    delta_pct: float,
    modo_idade: str,
) -> tuple[bool, str]:
    cmd = [
        sys.executable,
        str(ROOT_DIR / "pipelines" / "elgin" / "predict-features-mensal.py"),
        "--versao",
        version_label,
        "--skip-mlflow",
        "--feature",
        feature,
        "--delta-pct",
        str(delta_pct),
        "--modo-idade",
        modo_idade,
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT_DIR),
        text=True,
        capture_output=True,
    )
    output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    return proc.returncode == 0, output.strip()


@st.cache_data(show_spinner=False)
def load_kpi_mes_atual() -> dict[str, Any]:
    if not RAW_PANEL_PARQUET.is_file():
        raise FileNotFoundError(f"Arquivo não encontrado: {RAW_PANEL_PARQUET}")
    cols = ["competencia", "valor_faturamento", "valor_sinistro_ajustado", "sinistralidade_final"]
    df = pd.read_parquet(RAW_PANEL_PARQUET, columns=cols)
    if "competencia" not in df.columns:
        raise ValueError("Base raw sem coluna competencia")
    comp = pd.to_datetime(df["competencia"], errors="coerce")
    if comp.isna().all():
        comp = pd.to_datetime(df["competencia"].astype(str), errors="coerce")
    comp_ref = comp.max().to_period("M")
    mask = comp.dt.to_period("M") == comp_ref
    d = df.loc[mask].copy()
    fat = pd.to_numeric(d["valor_faturamento"], errors="coerce").fillna(0.0)
    sin_adj = pd.to_numeric(d["valor_sinistro_ajustado"], errors="coerce").fillna(0.0)
    denom = float(fat.sum())
    if denom > 0 and not sin_adj.isna().all():
        sin_mes_ajust = float(sin_adj.sum() / denom)
    else:
        sin_final = pd.to_numeric(d["sinistralidade_final"], errors="coerce").fillna(0.0)
        sin_mes_ajust = float((sin_final * fat).sum() / denom) if denom > 0 else float("nan")
    return {
        "competencia_referencia": str(comp_ref),
        "sinistralidade_real_mes": sin_mes_ajust,
        "sinistralidade_ajustada_mes": sin_mes_ajust,
        "n_vidas_mes": int(len(d)),
        "faturamento_mes": denom,
    }


def load_latest_what_if_result(version_label: str) -> dict[str, Any] | None:
    path = PREDICT_ROOT / version_label / "what_if_mensal" / "resultado_what_if.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def render_calibracao_mes_section(
    *,
    kpi_mes: dict[str, Any] | None,
    baseline_predito: float | None,
) -> None:
    if not kpi_mes:
        return
    real_mes = float(kpi_mes.get("sinistralidade_real_mes", float("nan")))
    if np.isnan(real_mes):
        return

    st.markdown("### Calibração do mês")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Competência", str(kpi_mes.get("competencia_referencia", "-")))
    c2.metric(
        "Sinistralidade ajustada (mês)",
        _fmt_sinistralidade_index_pct(real_mes),
        help="Índice do mês (sinistros ajustados ÷ faturamento). Valor ×100 em %.",
    )

    if baseline_predito is None or np.isnan(float(baseline_predito)):
        c3.metric("Baseline predito (modelo)", "-")
        c4.metric("Gap calibração", "-")
        st.caption("Execute uma simulação para atualizar o baseline predito do mês.")
        return

    pred_mes = float(baseline_predito)
    gap_abs = pred_mes - real_mes
    gap_rel = (gap_abs / real_mes * 100.0) if real_mes != 0 else float("nan")
    c3.metric(
        "Baseline predito (modelo)",
        _fmt_sinistralidade_index_pct(pred_mes),
        help="Índice previsto pelo modelo (mesma escala; ×100 em %).",
    )
    c4.metric(
        "Gap calibração",
        f"{gap_rel:+.2f}%",
        help="Erro relativo do baseline face ao real: (predito − real) / real.",
    )
    st.caption(
        f"Diferença em pontos percentuais (predito − real): **{gap_abs * 100.0:+.2f} p.p.**"
    )


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
    if not model_path:
        latest_local = latest_local_model_path()
        if latest_local is not None and latest_local.is_file():
            model_path = str(latest_local)
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


def render_prediction_tab() -> None:
    st.header("Predição por features")
    st.caption("Selecione grupo e feature para executar intervenção no pipeline real de predição.")
    kpi: dict[str, Any] | None = None
    try:
        kpi = load_kpi_mes_atual()

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Competência de referência", kpi["competencia_referencia"])
        k2.metric(
            "Sinistralidade ajustada (mês)",
            _fmt_sinistralidade_index_pct(kpi.get("sinistralidade_ajustada_mes")),
            help="Índice do mês (sinistros ajustados ÷ faturamento). Valor ×100 em %.",
        )
        k3.metric("Vidas no mês", f"{int(kpi['n_vidas_mes']):,}")
        k4.metric("Faturamento do mês", f"{float(kpi['faturamento_mes']):,.2f}")
        st.caption(
            "**Sinistralidade ajustada:** índice sinistros ajustados ÷ faturamento do mês; "
            "os números acima estão em **%** (índice × 100)."
        )
    except Exception as e:
        st.warning(f"Não foi possível carregar KPI mensal da base raw: {e}")

    versions = list_predict_versions()
    if not versions:
        st.error("Nenhuma pasta vN encontrada em data/processed/elgin/predict.")
        return

    # UI sem escolha manual de versão: usa sempre a versão mais recente.
    version_label = versions[0]
    ultimo_resultado = load_latest_what_if_result(version_label)
    baseline_inicial = None
    if ultimo_resultado and "sinistralidade_antes" in ultimo_resultado:
        baseline_inicial = float(ultimo_resultado.get("sinistralidade_antes", float("nan")))
    if st.session_state.get("pred_features_hide_baseline_ui"):
        baseline_inicial = None
    render_calibracao_mes_section(kpi_mes=kpi, baseline_predito=baseline_inicial)

    try:
        catalog = load_features_catalog(version_label)
    except Exception as e:
        st.error(f"Falha ao carregar catálogo de features da versão {version_label}: {e}")
        return

    grupos = catalog.get("grupos", [])
    if not grupos:
        st.warning("Catálogo sem grupos/features disponíveis.")
        return

    grupo_nomes = [str(g.get("grupo", "sem_grupo")) for g in grupos]
    grupo_sel = st.selectbox("Grupo de características", options=grupo_nomes)
    grupo_obj = next((g for g in grupos if str(g.get("grupo")) == grupo_sel), grupos[0])

    features = grupo_obj.get("features", [])
    if not features:
        st.warning("Grupo sem features disponíveis.")
        return

    feat_labels = [
        f"{_format_feature_name(str(f.get('feature', '')))}  | elegíveis(+): {f.get('n_elegiveis_delta_positivo', 0):,}"
        for f in features
    ]
    feat_idx = st.selectbox("Feature para intervenção", options=list(range(len(features))), format_func=lambda i: feat_labels[i])
    feat = features[feat_idx]
    feat_name = str(feat.get("feature"))

    delta_pct = st.slider("Delta da intervenção (%)", min_value=-90.0, max_value=200.0, value=20.0, step=1.0)
    modo_idade = "simples"
    if feat_name == "idade":
        modo_idade = st.radio(
            "Modo de idade",
            options=["simples", "correlacionado"],
            format_func=lambda k: MODO_IDADE_LABEL.get(str(k), str(k)),
            horizontal=True,
            help=MODO_IDADE_RADIO_HELP,
        )

    b_exec, b_clear = st.columns(2)
    with b_exec:
        run_clicked = st.button("Executar inferência what-if", type="primary", use_container_width=True)
    with b_clear:
        clear_clicked = st.button(
            "Limpar última inferência",
            type="secondary",
            use_container_width=True,
            help="Remove da sessão o painel de resultados e o baseline predito na calibração até nova execução.",
        )

    if clear_clicked:
        for k in ("pred_features_last_output", "pred_features_last_ok", "pred_features_last_version"):
            st.session_state.pop(k, None)
        st.session_state["pred_features_hide_baseline_ui"] = True
        st.rerun()

    if run_clicked:
        ok, output = _run_predict_features(
            version_label=version_label,
            feature=feat_name,
            delta_pct=float(delta_pct),
            modo_idade=modo_idade,
        )
        st.session_state["pred_features_last_output"] = output
        st.session_state["pred_features_last_ok"] = ok
        st.session_state["pred_features_last_version"] = version_label
        if ok:
            st.session_state["pred_features_hide_baseline_ui"] = False
        st.rerun()

    if "pred_features_last_ok" in st.session_state:
        ok = bool(st.session_state.get("pred_features_last_ok"))
        out = str(st.session_state.get("pred_features_last_output", ""))
        last_ver = str(st.session_state.get("pred_features_last_version", version_label))
        if ok:
            st.success("Inferência executada com sucesso.")
        else:
            st.error("Falha ao executar inferência.")

        what_if_dir = PREDICT_ROOT / last_ver / "what_if_mensal"
        resultado_path = what_if_dir / "resultado_what_if.json"
        relatorio_path = what_if_dir / "relatorio_intervencao.json"
        resultado: dict[str, Any] = {}
        relatorio: dict[str, Any] = {}

        if resultado_path.is_file():
            try:
                resultado = json.loads(resultado_path.read_text(encoding="utf-8"))
            except Exception as e:
                st.warning(f"Não foi possível ler resultado_what_if.json: {e}")
        if relatorio_path.is_file():
            try:
                relatorio = json.loads(relatorio_path.read_text(encoding="utf-8"))
            except Exception as e:
                st.warning(f"Não foi possível ler relatorio_intervencao.json: {e}")

        if resultado:
            st.subheader("Resultado da Simulação")
            rr1, rr2, rr3, rr4 = st.columns(4)
            rr1.metric(
                "Sinistralidade antes",
                _fmt_sinistralidade_index_pct(resultado.get("sinistralidade_antes", 0.0)),
                help="Índice macro antes da intervenção (×100 em %).",
            )
            rr2.metric(
                "Sinistralidade depois",
                _fmt_sinistralidade_index_pct(resultado.get("sinistralidade_depois", 0.0)),
                help="Índice macro depois da intervenção (×100 em %).",
            )
            d_abs = float(resultado.get("delta_absoluto", 0.0))
            rr3.metric(
                "Delta (p.p.)",
                f"{d_abs * 100.0:+.2f} p.p.",
                help="Variação do índice em pontos percentuais (diferença das taxas × 100).",
            )
            rr4.metric("Delta relativo", f"{float(resultado.get('delta_relativo_pct', 0.0)):+.4f}%")

            ee1, ee2, ee3 = st.columns(3)
            ee1.metric("Usuários afetados", f"{int(resultado.get('n_individuos_afetados', 0)):,}")
            ee2.metric("Usuários não elegíveis", f"{int(resultado.get('n_individuos_nao_elegiveis', 0)):,}")
            ee3.metric("Versão do modelo", str(resultado.get("versao_modelo", last_ver)))

            st.markdown("**Explicabilidade da intervenção**")
            intervs = resultado.get("intervencoes", [])
            if isinstance(intervs, list) and intervs:
                exp_df = pd.DataFrame(intervs)
                if not exp_df.empty:
                    if "feature" in exp_df.columns:
                        exp_df["feature_label"] = exp_df["feature"].astype(str).map(_format_feature_name)
                    show_cols = [c for c in ["feature_label", "feature", "delta_pct"] if c in exp_df.columns]
                    st.dataframe(
                        exp_df[show_cols].rename(
                            columns={
                                "feature_label": "Feature",
                                "feature": "Coluna técnica",
                                "delta_pct": "Delta (%)",
                            }
                        ),
                        width="stretch",
                        hide_index=True,
                    )
            if feat_name == "idade":
                st.caption(
                    f"Modo de idade aplicado: **{MODO_IDADE_LABEL.get(modo_idade, modo_idade)}** (`{modo_idade}`)"
                )

        if relatorio:
            st.markdown("**Detalhamento de elegibilidade**")
            interv = relatorio.get("intervencoes", [])
            if isinstance(interv, list) and interv:
                rel_df = pd.DataFrame(interv)
                if not rel_df.empty:
                    if "feature" in rel_df.columns:
                        rel_df["feature_label"] = rel_df["feature"].astype(str).map(_format_feature_name)
                    cols_map = {
                        "feature_label": "Feature",
                        "feature": "Coluna técnica",
                        "delta_pct": "Delta (%)",
                        "n_elegiveis": "Elegíveis",
                        "n_nao_elegiveis": "Não elegíveis",
                    }
                    keep_cols = [
                        c
                        for c in [
                            "feature_label",
                            "feature",
                            "delta_pct",
                            "n_elegiveis",
                            "n_nao_elegiveis",
                        ]
                        if c in rel_df.columns
                    ]
                    st.dataframe(rel_df[keep_cols].rename(columns=cols_map), width="stretch", hide_index=True)
            pct_base = relatorio.get("pct_base_afetada")
            if pct_base is not None:
                st.caption(f"Percentual da base afetada: {float(pct_base):.2f}%")

        with st.expander("Detalhes técnicos da execução", expanded=not ok):
            st.code(out or "(sem saída)")


def render_feature_impact_tab() -> None:
    st.header("Correlação")
    st.caption("Análise de correlação Spearman por competência (mês).")

    try:
        ver_dir, comps = list_feature_impact_competencias()
    except Exception:
        ver_dir, comps = (Path("."), [])

    if comps:
        comp_default = comps[-1]
        comp_sel = st.selectbox("Competência", options=comps, index=len(comps) - 1)
        st.caption(f"Versão ativa: {ver_dir.name}")
        df, comp_loaded = load_feature_impact_csv(comp_sel)
    else:
        df, comp_loaded = load_feature_impact_csv(None)
        st.warning("Formato legado detectado: exibindo CSV único (sem separação por competência).")

    df = df.copy()
    # Oculta variáveis quasi-leakage por padrão na UI de correlação.
    df = df[~df["feature"].isin(QUASI_LEAKAGE_HIDE)].copy()
    df["feature_label"] = df["feature"].map(_format_feature_name)

    st.metric("Total de features", len(df))
    st.dataframe(
        df[["feature_label", "spearman"]].rename(
            columns={
                "feature_label": "Feature",
                "spearman": "Grau de impacto"
            }
        ),
        width="stretch",
        hide_index=True,
    )

    top_n = st.slider("Top N por Grau de Impacto", min_value=5, max_value=min(30, len(df)), value=15)
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
        labels={
            "feature_label": "Feature",
            "spearman": "Grau de impacto",
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
