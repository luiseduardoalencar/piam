"""
Configurações Gerais do Projeto
Define paths, constantes e parâmetros globais
"""

from pathlib import Path
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURAÇÕES DE DISPLAY E WARNINGS
# ============================================================================
warnings.filterwarnings("ignore")


# ============================================================================
# PATHS DO PROJETO (Pathlib)
# ============================================================================
# Se este arquivo está em <repo>/config/settings.py,
# então o BASE_DIR é o diretório raiz do projeto
BASE_DIR: Path = Path(__file__).resolve().parents[1]

# Diretórios de dados
DATA_DIR: Path = BASE_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
AUXILIAR_DATA_DIR: Path = DATA_DIR / "auxiliar"

# Base analítica principal
BASE_ANALITICA_PATH: Path = PROCESSED_DATA_DIR / "base_analitica.csv"
PRODUCTS_BASE_PATH: Path = PROCESSED_DATA_DIR / "products_base.csv"

# Base auxiliar
FEATURE_CATALOG_PATH: Path = AUXILIAR_DATA_DIR / "feature_catalog.csv"


# Diretórios de saída
OUTPUT_DIR: Path = BASE_DIR / "outputs"
MODELS_DIR: Path = OUTPUT_DIR / "models"
PLOTS_DIR: Path = OUTPUT_DIR / "plots"
REPORTS_DIR: Path = OUTPUT_DIR / "reports"

# Criar diretórios de saída (idempotente)
for d in (OUTPUT_DIR, MODELS_DIR, PLOTS_DIR, REPORTS_DIR, AUXILIAR_DATA_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CONFIGURAÇÕES DE DADOS
# ============================================================================
# Encoding padrão (ajustar apenas se necessário)
DEFAULT_ENCODING: str = "latin1"

# ============================================================================
# PARÂMETROS GLOBAIS DE MODELAGEM
# ============================================================================
TEST_SIZE: float = 0.2
RANDOM_STATE: int = 42


# ============================================================================
# FAIXAS ETÁRIAS PARA FEATURES
# ============================================================================
FAIXAS_ETARIAS = {
    "0-18": (0, 18),
    "19-23": (19, 23),
    "24-28": (24, 28),
    "29-33": (29, 33),
    "34-38": (34, 38),
    "39-43": (39, 43),
    "44-48": (44, 48),
    "49-53": (49, 53),
    "54-58": (54, 58),
    "59+": (59, 150),
}


# ============================================================================
# FUNÇÃO DE VALIDAÇÃO
# ============================================================================
def validar_paths() -> bool:
    """
    Valida se os paths principais do projeto existem.

    Returns:
        bool: True se tudo estiver correto, False caso contrário.
    """
    paths = {
        "Base Analítica": BASE_ANALITICA_PATH,
    }

    erros = []
    for nome, path in paths.items():
        if not path.exists():
            erros.append(f"[ERRO] {nome}: {path}")
        else:
            print(f"[OK] {nome}: {path}")

    if erros:
        print("\n[AVISO] Problemas encontrados:")
        for err in erros:
            print(err)
        print("\n[SOLUÇÃO]")
        print("   1. Verifique se o arquivo existe em data/processed/")
        print("   2. Ou ajuste o path em config/settings.py")
        return False

    return True

# ============================================================================
# TESTE DO MÓDULO
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("CONFIGURAÇÕES DO PROJETO")
    print("=" * 60)

    print("\n[DIRETÓRIOS]")
    print(f"   Base Dir        : {BASE_DIR}")
    print(f"   Data Dir        : {DATA_DIR}")
    print(f"   Raw Data Dir    : {RAW_DATA_DIR}")
    print(f"   Processed Dir   : {PROCESSED_DATA_DIR}")
    print(f"   Output Dir      : {OUTPUT_DIR}")

    print("\n[ARQUIVOS DE DADOS]")
    validar_paths()

    print("\n[CONFIGURAÇÕES GLOBAIS]")
    print(f"   Test Size    : {TEST_SIZE}")
    print(f"   Random State : {RANDOM_STATE}")
    print(f"   Encoding     : {DEFAULT_ENCODING}")

    print("\n[FAIXAS ETÁRIAS]")
    for faixa, (min_idade, max_idade) in FAIXAS_ETARIAS.items():
        print(f"   {faixa}: {min_idade}-{max_idade} anos")

    print("=" * 60)