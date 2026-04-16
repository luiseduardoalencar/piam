"""
Configuração correta e estável do MLflow.
Agora:
- NÃO força modelos antigos
- NÃO força experimento antigo
- Autentica Registry + Tracking
- Funciona com model registry `models:/<model_name>/latest`
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ===============================================
#  CARREGAR ENV **antes** de importar mlflow — senão o cliente usa ./mlruns por defeito.
# ===============================================
# Sem override (predefinição do python-dotenv): variáveis já definidas no SO / terminal
# prevalecem sobre o .env. Usa as mesmas chaves que o .env.example:
# MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD,
# AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION.
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

import mlflow
from mlflow.tracking import MlflowClient


def aplicar_tracking_uri_servidor() -> bool:
    """
    Aponta o cliente MLflow para ``MLFLOW_TRACKING_URI`` (REST) e replica auth no ambiente.
    Chamado ao importar este módulo e no início de ``configurar_mlflow`` — evita o backend
    local em ``./mlruns`` quando o URI está definido no ``.env``.
    """
    uri = (os.getenv("MLFLOW_TRACKING_URI") or "").strip()
    if not uri:
        return False
    mlflow.set_tracking_uri(uri)
    u = os.getenv("MLFLOW_TRACKING_USERNAME")
    p = os.getenv("MLFLOW_TRACKING_PASSWORD")
    if u is not None:
        os.environ["MLFLOW_TRACKING_USERNAME"] = u
    if p is not None:
        os.environ["MLFLOW_TRACKING_PASSWORD"] = p
    os.environ["MLFLOW_REGISTRY_URI"] = uri
    return True


# Tracking remoto logo após carregar o .env (import ``config.mlflow_config`` ≠ registar runs).
if not aplicar_tracking_uri_servidor():
    print(
        "[MLflow] MLFLOW_TRACKING_URI ausente — defina no .env para não cair no backend local ./mlruns."
    )


# ===============================================
#  FUNÇÃO PRINCIPAL
# ===============================================
def configurar_mlflow(experiment_name: str, *, preparar_experimento: bool = True):
    """
    Configura tracking + registro + credenciais AWS.

    ``preparar_experimento=False``: só define URI e credenciais (leitura / inferência).
    Não cria experimento nem chama ``set_experiment`` — evita efeitos no servidor.
    """

    # ----------------------------
    # VARIÁVEIS OBRIGATÓRIAS
    # ----------------------------
    required_vars = [
        "MLFLOW_TRACKING_URI",
        "MLFLOW_TRACKING_USERNAME",
        "MLFLOW_TRACKING_PASSWORD",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
    ]

    missing = [v for v in required_vars if not os.getenv(v)]
    if missing:
        raise EnvironmentError(
            "[ERRO] Variáveis de ambiente faltando:\n"
            + "\n".join(f" - {v}" for v in missing)
        )

    aplicar_tracking_uri_servidor()
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()

    # ----------------------------
    # CREDENCIAIS AWS (mesmos nomes que .env.example / boto3 / upload S3 de artefatos)
    # PutObject usa esta conta; tem de ter acesso ao bucket do artifact store do servidor MLflow.
    # InvalidAccessKeyId = chave inexistente ou revogada; credenciais temporárias: AWS_SESSION_TOKEN.
    # ----------------------------
    os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")
    os.environ["AWS_DEFAULT_REGION"] = os.getenv("AWS_DEFAULT_REGION")

    # ----------------------------
    # CONFIGURAR EXPERIMENTO (treino / logging — opcional)
    # Experimento em soft-delete: restaurar antes na CLI, p.ex.:
    #   mlflow experiments restore -x <experiment_id>
    # (defina MLFLOW_TRACKING_URI / auth como no .env)
    # ----------------------------
    experiment_id = None
    if preparar_experimento:
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
                print(f"[MLFLOW] Criando experimento: {experiment_name}")
            else:
                experiment_id = experiment.experiment_id
                print(f"[MLFLOW] Usando experimento existente: {experiment_name} (ID {experiment_id})")

            mlflow.set_experiment(experiment_name)

        except Exception as e:
            print(f"[ERRO] Falha ao criar/ver experimento: {e}")
            experiment_id = None
    else:
        print(
            "[MLflow] Modo leitura/inferência: URI e credenciais configurados; "
            "sem criar experimento nem definir experimento ativo."
        )

    # ----------------------------
    # RETORNA CONFIG
    # ----------------------------
    config = {
        "tracking_uri": tracking_uri,
        "experiment_name": experiment_name,
        "experiment_id": experiment_id,
        "aws_region": os.getenv("AWS_DEFAULT_REGION"),
        "preparar_experimento": preparar_experimento,
    }

    print("\n=================================================")
    print("[OK] MLflow configurado corretamente")
    print("Tracking URI :", config["tracking_uri"])
    if preparar_experimento:
        print("Experimento  :", config["experiment_name"])
        print("ExperimentID :", config["experiment_id"])
    else:
        print("Experimento  : (não definido — modo leitura)")
        print("ExperimentID : —")
    print("AWS Region   :", config["aws_region"])
    print("=================================================\n")

    return config


# ===============================================
# UTILITÁRIOS
# ===============================================
def get_mlflow_client():
    return MlflowClient()


def get_experiment_info(experiment_name: str):
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp:
        return {
            "name": exp.name,
            "experiment_id": exp.experiment_id,
            "artifact_location": exp.artifact_location,
            "lifecycle_stage": exp.lifecycle_stage,
        }
    return None
