from fastapi import FastAPI
from fastapi.responses import FileResponse

from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI()


# Modelo dos dados de uma casa
class DadosCasa(BaseModel):
    Zoneamento: str
    AreaTerreno: float
    Bairro: str
    TipoConstrucao: str
    EstiloCasa: str
    QualidadeGeral: float
    CondicaoGeral: float
    AnoConstrucao: float
    AnoReforma: float
    TipoFundacao: str
    QualidadeExterna: str
    CondicaoExterna: str
    AreaTotalPorao: float
    QualidadeAquecimento: str
    ArCentral: str
    AreaHabitavel: float
    BanheirosCompletos: float
    MeiosBanheiros: float
    QualidadeCozinha: str
    TotalComodos: float
    Lareiras: float
    TipoGaragem: str
    AcabamentoGaragem: str
    VagasGaragem: float
    PavimentacaoEntrada: str
    AreaDeckMadeira: float
    AreaVarandaAberta: float
    AreaPiscina: float
    QualidadePiscina: str
    CondicaoVenda: str


# Carregando o modelo e os tratamentos
modelo = joblib.load("modelo.pkl")
encoder = joblib.load("encoder.pkl")
scaler = joblib.load("scaler.pkl")


# Colunas usadas pelo modelo
colunas = [
    "Zoneamento",
    "AreaTerreno",
    "Bairro",
    "TipoConstrucao",
    "EstiloCasa",
    "QualidadeGeral",
    "CondicaoGeral",
    "AnoConstrucao",
    "AnoReforma",
    "TipoFundacao",
    "QualidadeExterna",
    "CondicaoExterna",
    "AreaTotalPorao",
    "QualidadeAquecimento",
    "ArCentral",
    "AreaHabitavel",
    "BanheirosCompletos",
    "MeiosBanheiros",
    "QualidadeCozinha",
    "TotalComodos",
    "Lareiras",
    "TipoGaragem",
    "AcabamentoGaragem",
    "VagasGaragem",
    "PavimentacaoEntrada",
    "AreaDeckMadeira",
    "AreaVarandaAberta",
    "AreaPiscina",
    "QualidadePiscina",
    "CondicaoVenda"
]


@app.get("/")
def inicio():
    return FileResponse("index.html")


@app.post("/prever")
def prever(dados: DadosCasa):

    # Transformar os dados recebidos em DataFrame
    dados = pd.DataFrame([dados.model_dump()])

    # Garantir a ordem das colunas
    dados = dados[colunas]

    # Separar colunas categóricas e numéricas
    dados_categoricos = dados.select_dtypes(include=["object"])
    dados_numericos = dados.select_dtypes(include=["number"])

    # Aplicar OneHotEncoder
    dados_codificados = encoder.transform(dados_categoricos)

    # Aplicar StandardScaler
    dados_numericos_padronizados = scaler.transform(dados_numericos)

    # Juntar os dados
    dados_finais = np.hstack([
        dados_numericos_padronizados,
        dados_codificados
    ])

    # Fazer a previsão
    previsao = modelo.predict(dados_finais)

    return {
        "preco_previsto": round(float(previsao[0]), 2)
    }