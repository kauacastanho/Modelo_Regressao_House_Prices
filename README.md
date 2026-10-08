# Previsão de Preços de Casas — Documentação do Projeto

Modelo de **regressão linear** para prever o preço de venda de casas (`SalePrice`), treinado com o dataset *House Prices – Advanced Regression Techniques* (Kaggle) e exposto por uma **API FastAPI** com uma interface web simples.

---

## Sumário

1. [Visão geral](#1-visão-geral)
2. [Estrutura do projeto](#2-estrutura-do-projeto)
3. [Dados](#3-dados)
4. [Modelagem passo a passo](#4-modelagem-passo-a-passo)
5. [Artefatos exportados](#5-artefatos-exportados)
6. [API](#6-api)
7. [Interface web](#7-interface-web)
8. [Como executar](#8-como-executar)
9. [Interpretação do modelo](#9-interpretação-do-modelo)
10. [Limitações e pontos de atenção](#10-limitações-e-pontos-de-atenção)
11. [Sugestões de melhoria](#11-sugestões-de-melhoria)
12. [Dicionário de variáveis](#12-dicionário-de-variáveis)

---

## 1. Visão geral

| Item | Descrição |
|---|---|
| **Problema** | Regressão supervisionada: prever o preço de venda de uma casa |
| **Variável alvo** | `SalePrice` (renomeada para `PrecoVenda`) |
| **Dataset** | `train.csv` do Kaggle (1460 linhas × 81 colunas) |
| **Algoritmo** | `LinearRegression` (scikit-learn) |
| **Pré-processamento** | `OneHotEncoder` (categóricas) + `StandardScaler` (numéricas) |
| **Métricas** | MAE, RMSE e R² em conjunto de teste (20%) |
| **Serviço** | FastAPI com endpoint `POST /prever` e página HTML em `GET /` |

### Fluxo geral

```
train.csv
   │
   ▼
Análise exploratória ──► Seleção de features ──► Renomeação (PT-BR)
   │
   ▼
Tratamento de nulos ──► Correlação / remoção de colunas ──► Remoção de outliers
   │
   ▼
Split treino/teste (80/20)
   │
   ├─► OneHotEncoder (categóricas)
   └─► StandardScaler (numéricas)
   │
   ▼
LinearRegression ──► Avaliação (MAE, RMSE, R²)
   │
   ▼
modelo.pkl + encoder.pkl + scaler.pkl ──► API FastAPI ──► index.html
```

---

## 2. Estrutura do projeto

```
API_house_prices/
├── api.py          # API FastAPI (validação, pré-processamento e previsão)
├── index.html      # Interface web para testar a API
├── modelo.pkl      # Regressão Linear treinada
├── encoder.pkl     # OneHotEncoder ajustado no treino
└── scaler.pkl      # StandardScaler ajustado no treino

g1ml_house_prices.py   # Script/notebook de análise, treino e avaliação
```

---

## 3. Dados

### 3.1 Fonte

Kaggle — *House Prices: Advanced Regression Techniques* (`train.csv`).
Descreve casas residenciais de Ames, Iowa (EUA). **Os preços estão em dólares americanos (USD)** e as categorias usam os códigos originais em inglês (ex.: `RL`, `TA`, `Gd`).

### 3.2 Análise inicial

O script faz a inspeção básica com:

- `df.info()`, `df.head()`, `df.tail()`, `df.describe()`
- Contagem de linhas e colunas (`df.shape`)
- Valores nulos por coluna (`df.isnull().sum()`)
- Linhas duplicadas (`df.duplicated().sum()`)

### 3.3 Seleção de features (1ª seleção)

Das 80 colunas preditoras originais, foram escolhidas **33 variáveis** com base em conhecimento de domínio, agrupadas por tema:

| Grupo | Variáveis originais |
|---|---|
| Terreno e localização | `MSZoning`, `LotArea`, `Neighborhood` |
| Tipo e estilo | `BldgType`, `HouseStyle` |
| Qualidade e idade | `OverallQual`, `OverallCond`, `YearBuilt`, `YearRemodAdd` |
| Estrutura externa | `Foundation`, `ExterQual`, `ExterCond` |
| Porão | `TotalBsmtSF` |
| Sistemas | `HeatingQC`, `CentralAir` |
| Área útil | `GrLivArea` |
| Cômodos | `FullBath`, `HalfBath`, `BedroomAbvGr`, `KitchenAbvGr`, `KitchenQual`, `TotRmsAbvGrd` |
| Lareira | `Fireplaces` |
| Garagem | `GarageType`, `GarageFinish`, `GarageCars`, `GarageArea`, `PavedDrive` |
| Áreas externas | `WoodDeckSF`, `OpenPorchSF` |
| Piscina | `PoolArea`, `PoolQC` |
| Venda | `SaleCondition` |

As colunas foram renomeadas para português (ex.: `GrLivArea` → `AreaHabitavel`, `OverallQual` → `QualidadeGeral`). O mapeamento completo está na [seção 12](#12-dicionário-de-variáveis).

### 3.4 Valores nulos

Após a seleção, restaram três colunas com nulos. Em todas, o nulo **não significa dado ausente**, e sim "o imóvel não possui esse item" (conforme a documentação do dataset):

| Coluna | Nulos | Tratamento |
|---|---:|---|
| `TipoGaragem` | 81 | `fillna("NoGarage")` |
| `AcabamentoGaragem` | 81 | `fillna("NoGarage")` |
| `QualidadePiscina` | 1453 | `fillna("NoPool")` |

Isso transforma o nulo em uma categoria própria, que o `OneHotEncoder` consegue usar.

### 3.5 Análise exploratória

- **Histogramas** (`plotly.express.histogram`) de todas as 19 colunas numéricas, incluindo o alvo.
- **Boxplots** das mesmas colunas, para identificar outliers.
- **Matriz de correlação** (`seaborn.heatmap`) entre as colunas numéricas.
- **Gráficos de dispersão** contra `PrecoVenda` para: `AreaHabitavel`, `AreaTerreno`, `QualidadeGeral`, `AnoConstrucao` e `AreaTotalPorao`.

### 3.6 Remoção de colunas por correlação

Com base na matriz de correlação, foram removidas três colunas:

| Coluna removida | Observação |
|---|---|
| `Cozinhas` | Quase sem variação (a imensa maioria das casas tem 1 cozinha) |
| `Quartos` | Tende a ser redundante com `TotalComodos` |
| `AreaGaragem` | Tende a ser redundante com `VagasGaragem` |

> A justificativa acima é a interpretação típica para esse tipo de remoção. Vale registrar no notebook os valores de correlação que embasaram cada decisão.

Resultado: **30 features** finais (15 numéricas + 15 categóricas), que são exatamente os 30 campos aceitos pela API.

### 3.7 Remoção de outliers

Foram removidos **2 registros**, identificados visualmente nos gráficos de dispersão:

1. A casa com **maior `AreaHabitavel`** (`max()`), que tem área muito grande e preço baixo, fugindo da tendência.
2. O registro de **índice 185**, uma casa construída antes de 1900 com preço acima de US$ 400.000 (`AnoConstrucao < 1900` e `PrecoVenda > 400000`).

Base final: **1458 linhas**.

---

## 4. Modelagem passo a passo

### 4.1 Separação X / y

```python
y = dados_limpos["PrecoVenda"]
X = dados_limpos.drop("PrecoVenda", axis=1)
```

### 4.2 Divisão treino / teste

```python
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

- 80% treino (≈ 1166 linhas) e 20% teste (≈ 292 linhas).
- `random_state=42` garante reprodutibilidade.
- A divisão é feita **antes** de qualquer ajuste de encoder/scaler, o que evita *data leakage*.

### 4.3 Codificação das variáveis categóricas

```python
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_treino_codificado = encoder.fit_transform(X_treino_categorico)  # fit só no treino
X_teste_codificado  = encoder.transform(X_teste_categorico)
```

- `handle_unknown="ignore"`: categorias nunca vistas no treino geram uma linha de zeros (não quebram a API).
- As 15 colunas categóricas geram **92 colunas** binárias.
- Não foi usado `drop="first"`; como o `LinearRegression` do scikit-learn lida com multicolinearidade (usa mínimos quadrados via SVD/pseudo-inversa), isso não impede o treino, mas afeta a interpretação individual dos coeficientes.

| Variável | Nº de categorias |
|---|---:|
| `Bairro` | 25 |
| `TipoGaragem` | 7 |
| `CondicaoVenda` | 6 |
| `TipoFundacao` | 6 |
| `Zoneamento` | 5 |
| `TipoConstrucao` | 5 |
| `CondicaoExterna` | 5 |
| `EstiloCasa` | 8 |
| `QualidadeExterna`, `QualidadeAquecimento`, `QualidadeCozinha`, `QualidadePiscina`, `AcabamentoGaragem` | 4 cada |
| `PavimentacaoEntrada` | 3 |
| `ArCentral` | 2 |
| **Total** | **92** |

### 4.4 Padronização das variáveis numéricas

```python
scaler = StandardScaler()
X_treino_numerico_padronizado = scaler.fit_transform(X_treino_numerico)
X_teste_numerico_padronizado  = scaler.transform(X_teste_numerico)
```

Cada coluna numérica passa a ter média 0 e desvio padrão 1 (parâmetros calculados **apenas no treino**). As 15 colunas numéricas, na ordem em que o scaler as espera:

```
AreaTerreno, QualidadeGeral, CondicaoGeral, AnoConstrucao, AnoReforma,
AreaTotalPorao, AreaHabitavel, BanheirosCompletos, MeiosBanheiros,
TotalComodos, Lareiras, VagasGaragem, AreaDeckMadeira,
AreaVarandaAberta, AreaPiscina
```

### 4.5 Montagem da matriz final

```python
X_treino_final = np.hstack([X_treino_numerico_padronizado, X_treino_codificado])
X_teste_final  = np.hstack([X_teste_numerico_padronizado,  X_teste_codificado])
```

**Ordem das colunas: primeiro as 15 numéricas padronizadas, depois as 92 dummies** — total de **107 colunas**. Essa ordem precisa ser idêntica na API.

> No script, existe um primeiro `np.hstack` com dados numéricos *não* padronizados, que é depois sobrescrito pelo segundo. Ele pode ser removido sem efeito no resultado.

### 4.6 Treinamento

```python
modelo = LinearRegression()
modelo.fit(X_treino_final, y_treino)
```

O modelo treinado possui **107 coeficientes** e **intercepto ≈ 174.110,61** (preço médio estimado para uma casa com todas as numéricas na média e todas as dummies zeradas).

### 4.7 Avaliação

```python
mae  = mean_absolute_error(y_teste, y_pred)
rmse = np.sqrt(mean_squared_error(y_teste, y_pred))
r2   = r2_score(y_teste, y_pred)
```

| Métrica | O que mede |
|---|---|
| **MAE** | Erro absoluto médio, em dólares. Mais fácil de interpretar |
| **RMSE** | Raiz do erro quadrático médio, em dólares. Penaliza mais os erros grandes |
| **R²** | Proporção da variância do preço explicada pelo modelo (1 = perfeito) |

Também é gerado um gráfico **Preço Real × Preço Previsto** com a linha de referência y = x; quanto mais próximos os pontos dessa linha, melhor o modelo.

> **Preencha aqui os resultados da sua execução** (o `train.csv` não estava no zip, então não foi possível recalcular as métricas):
>
> | Métrica | Valor |
> |---|---|
> | MAE | _a preencher_ |
> | RMSE | _a preencher_ |
> | R² | _a preencher_ |

---

## 5. Artefatos exportados

A API carrega três arquivos gerados com `joblib`:

| Arquivo | Objeto | Papel |
|---|---|---|
| `modelo.pkl` | `LinearRegression` | Faz a previsão |
| `encoder.pkl` | `OneHotEncoder` | Codifica as 15 colunas categóricas |
| `scaler.pkl` | `StandardScaler` | Padroniza as 15 colunas numéricas |

Os três foram serializados com **scikit-learn 1.6.1**.

> O script `g1ml_house_prices.py` **não contém** as linhas de exportação. Recomenda-se adicionar ao final:
>
> ```python
> import joblib
> joblib.dump(modelo,  "modelo.pkl")
> joblib.dump(encoder, "encoder.pkl")
> joblib.dump(scaler,  "scaler.pkl")
> ```

---

## 6. API

Implementada em **FastAPI** (`api.py`).

### 6.1 Endpoints

| Método | Rota | Descrição |
|---|---|---|
| `GET` | `/` | Retorna a página `index.html` |
| `POST` | `/prever` | Recebe os dados de uma casa e retorna o preço previsto |
| `GET` | `/docs` | Documentação interativa Swagger (gerada automaticamente pelo FastAPI) |

### 6.2 Fluxo interno de `/prever`

1. **Validação** — o corpo JSON é validado pelo modelo Pydantic `DadosCasa` (30 campos obrigatórios; strings para categóricas, `float` para numéricas).
2. **DataFrame** — `pd.DataFrame([dados.model_dump()])`.
3. **Ordem das colunas** — `dados[colunas]`, garantindo a mesma ordem do treino.
4. **Separação** — `select_dtypes("object")` para categóricas e `select_dtypes("number")` para numéricas.
5. **Transformação** — `encoder.transform(...)` e `scaler.transform(...)`.
6. **Junção** — `np.hstack([numéricas_padronizadas, categóricas_codificadas])` (mesma ordem do treino).
7. **Previsão** — `modelo.predict(...)`.
8. **Resposta** — `{"preco_previsto": <float com 2 casas decimais>}`.

### 6.3 Contrato da requisição

**`POST /prever`** — `Content-Type: application/json`

| Campo | Tipo | Valores / observações |
|---|---|---|
| `Zoneamento` | string | `C (all)`, `FV`, `RH`, `RL`, `RM` |
| `AreaTerreno` | número | Área do lote (pés²) |
| `Bairro` | string | Um dos 25 bairros (ex.: `NAmes`, `CollgCr`, `NridgHt`) |
| `TipoConstrucao` | string | `1Fam`, `2fmCon`, `Duplex`, `Twnhs`, `TwnhsE` |
| `EstiloCasa` | string | `1Story`, `1.5Fin`, `1.5Unf`, `2Story`, `2.5Fin`, `2.5Unf`, `SFoyer`, `SLvl` |
| `QualidadeGeral` | número | 1 a 10 |
| `CondicaoGeral` | número | 1 a 10 |
| `AnoConstrucao` | número | Ano |
| `AnoReforma` | número | Ano |
| `TipoFundacao` | string | `BrkTil`, `CBlock`, `PConc`, `Slab`, `Stone`, `Wood` |
| `QualidadeExterna` | string | `Ex`, `Gd`, `TA`, `Fa` |
| `CondicaoExterna` | string | `Ex`, `Gd`, `TA`, `Fa`, `Po` |
| `AreaTotalPorao` | número | pés² |
| `QualidadeAquecimento` | string | `Ex`, `Gd`, `TA`, `Fa` |
| `ArCentral` | string | `Y` ou `N` |
| `AreaHabitavel` | número | pés² acima do solo |
| `BanheirosCompletos` | número | |
| `MeiosBanheiros` | número | |
| `QualidadeCozinha` | string | `Ex`, `Gd`, `TA`, `Fa` |
| `TotalComodos` | número | Cômodos acima do solo (sem banheiros) |
| `Lareiras` | número | |
| `TipoGaragem` | string | `2Types`, `Attchd`, `Basment`, `BuiltIn`, `CarPort`, `Detchd`, `NoGarage` |
| `AcabamentoGaragem` | string | `Fin`, `RFn`, `Unf`, `NoGarage` |
| `VagasGaragem` | número | Capacidade em carros |
| `PavimentacaoEntrada` | string | `Y`, `P`, `N` |
| `AreaDeckMadeira` | número | pés² |
| `AreaVarandaAberta` | número | pés² |
| `AreaPiscina` | número | pés² |
| `QualidadePiscina` | string | `Ex`, `Gd`, `Fa`, `NoPool` |
| `CondicaoVenda` | string | `Normal`, `Abnorml`, `AdjLand`, `Alloca`, `Family`, `Partial` |

**Exemplo:**

```json
{
  "Zoneamento": "RL",
  "AreaTerreno": 8000,
  "Bairro": "NAmes",
  "TipoConstrucao": "1Fam",
  "EstiloCasa": "1Story",
  "QualidadeGeral": 6,
  "CondicaoGeral": 6,
  "AnoConstrucao": 1995,
  "AnoReforma": 2005,
  "TipoFundacao": "PConc",
  "QualidadeExterna": "Gd",
  "CondicaoExterna": "Gd",
  "AreaTotalPorao": 900,
  "QualidadeAquecimento": "Gd",
  "ArCentral": "Y",
  "AreaHabitavel": 1600,
  "BanheirosCompletos": 2,
  "MeiosBanheiros": 1,
  "QualidadeCozinha": "Gd",
  "TotalComodos": 7,
  "Lareiras": 1,
  "TipoGaragem": "Attchd",
  "AcabamentoGaragem": "RFn",
  "VagasGaragem": 2,
  "PavimentacaoEntrada": "Y",
  "AreaDeckMadeira": 100,
  "AreaVarandaAberta": 40,
  "AreaPiscina": 0,
  "QualidadePiscina": "NoPool",
  "CondicaoVenda": "Normal"
}
```

### 6.4 Resposta

**200 OK**

```json
{ "preco_previsto": 192628.85 }
```

**422 Unprocessable Entity** — campo ausente ou de tipo errado (validação do Pydantic).

### 6.5 Exemplo com `curl`

```bash
curl -X POST http://127.0.0.1:8000/prever \
  -H "Content-Type: application/json" \
  -d @casa.json
```

### 6.6 Exemplo com Python

```python
import requests

r = requests.post("http://127.0.0.1:8000/prever", json=casa)
print(r.json()["preco_previsto"])
```

### 6.7 Casas de exemplo e previsões obtidas

As quatro casas de teste do script foram executadas com os `.pkl` entregues:

| Perfil | Bairro | Qualidade geral | Área habitável (pés²) | Preço previsto (USD) |
|---|---|---:|---:|---:|
| Casa simples | Edwards | 4 | 1000 | **89.400,81** |
| Casa média | NAmes | 6 | 1600 | **192.628,85** |
| Casa grande | NridgHt | 8 | 2800 | **422.703,37** |
| Alto padrão | NridgHt | 10 | 3500 | **748.161,18** |

A progressão é coerente com o esperado (mais qualidade e área, maior preço). O resultado do alto padrão, porém, deve ser lido com cuidado (ver [seção 10](#10-limitações-e-pontos-de-atenção)).

---

## 7. Interface web

`index.html` é uma página estática, servida em `GET /`, com:

- Uma `<textarea>` pré-preenchida com um JSON de exemplo (casa de `CollgCr`).
- Um botão **"Prever preço"** que faz `fetch("/prever", { method: "POST", ... })`.
- Exibição do resultado formatado em dólares (`toLocaleString("en-US")`).
- Tratamento de erro: JSON inválido na caixa de texto ou resposta de erro da API (exibida em vermelho).

Como a página é servida pela própria API e chama `/prever` por caminho relativo, não há problemas de CORS.

---

## 8. Como executar

### 8.1 Dependências

```bash
pip install fastapi uvicorn pandas numpy scikit-learn==1.6.1 joblib
```

(Para o notebook de análise: `matplotlib`, `seaborn`, `plotly`.)

> Use a mesma versão do scikit-learn do treino (1.6.1) para evitar o `InconsistentVersionWarning` ao carregar os `.pkl`.

### 8.2 Subir a API

Dentro da pasta que contém `api.py` e os `.pkl` (os caminhos são relativos):

```bash
python -m uvicorn api:app --reload
# no Windows: py -m uvicorn api:app --reload
```

- Interface: <http://127.0.0.1:8000>
- Swagger: <http://127.0.0.1:8000/docs>

### 8.3 Reexecutar o treino

1. Baixe `train.csv` do Kaggle e coloque ao lado do script.
2. Execute `g1ml_house_prices.py` (ou o notebook original no Colab).
3. Exporte os três artefatos com `joblib.dump` (ver seção 5).
4. Copie os `.pkl` para a pasta da API.

---

## 9. Interpretação do modelo

Como as variáveis numéricas foram padronizadas, o coeficiente de cada uma indica **quanto o preço varia (em USD) para um aumento de 1 desvio padrão** da variável, mantendo as demais constantes. Isso torna as magnitudes comparáveis entre si.

### Numéricas (coeficientes do `modelo.pkl`)

| Variável | Coeficiente (USD / 1 desvio padrão) |
|---|---:|
| `AreaHabitavel` | +30.846 |
| `AnoConstrucao` | +12.515 |
| `AreaTotalPorao` | +11.739 |
| `QualidadeGeral` | +10.814 |
| `VagasGaragem` | +7.817 |
| `CondicaoGeral` | +7.238 |
| `AreaTerreno` | +6.886 |
| `AreaPiscina` | +5.185 |
| `Lareiras` | +2.590 |
| `AreaDeckMadeira` | +2.258 |
| `AreaVarandaAberta` | +1.298 |
| `MeiosBanheiros` | +1.187 |
| `AnoReforma` | +376 |
| `BanheirosCompletos` | +292 |
| `TotalComodos` | −2.428 |

**Leitura:** a área habitável é, com folga, o fator numérico mais influente, seguida por ano de construção, porão e qualidade geral.

### Categóricas (maiores efeitos, em USD, em relação à ausência da categoria)

| Efeito positivo | USD | Efeito negativo | USD |
|---|---:|---|---:|
| `QualidadePiscina_Ex` | +149.598 | `QualidadePiscina_Fa` | −79.978 |
| `Bairro_StoneBr` | +44.763 | `QualidadePiscina_Gd` | −68.827 |
| `Bairro_NoRidge` | +32.145 | `EstiloCasa_2.5Fin` | −29.386 |
| `QualidadeExterna_Ex` | +29.613 | `Zoneamento_C (all)` | −22.669 |
| `QualidadeCozinha_Ex` | +21.505 | `TipoGaragem_2Types` | −22.093 |
| `Bairro_Crawfor` | +20.962 | `Bairro_NWAmes` | −19.288 |
| `Bairro_NridgHt` | +19.541 | `Bairro_Gilbert` | −17.855 |

Como não foi usado `drop="first"`, cada dummy é lida em relação ao conjunto das demais categorias da mesma variável, e a interpretação individual é aproximada.

---

## 10. Limitações e pontos de atenção

**Sobre o modelo**

- **Regressão linear sem transformação do alvo.** Preços de imóveis costumam ser assimétricos à direita; sem `log(PrecoVenda)`, o modelo tende a errar proporcionalmente mais nas casas caras e pode, em teoria, prever valores negativos para casas muito simples.
- **Extrapolação linear.** Combinações de atributos acima do que existe na base (como a casa "alto padrão" dos exemplos, com 3500 pés², piscina de 500 e qualidade 10) produzem previsões pouco confiáveis.
- **Piscina com coeficientes instáveis.** Pouquíssimas casas da base têm piscina (a coluna tem 1453 nulos em 1460 linhas). Por isso `QualidadePiscina_Ex` (+149 mil) e `Gd`/`Fa` (negativos) refletem poucos exemplos e não devem ser interpretados como regra.
- **Multicolinearidade.** `TotalComodos` tem coeficiente negativo, contraintuitivo, provavelmente porque a informação já está em `AreaHabitavel`. Isso não prejudica muito a previsão, mas atrapalha a interpretação.
- **Validação única.** Só há um *hold-out* de 20%; não há validação cruzada, então a métrica pode variar conforme o `random_state`.
- **Escopo geográfico e temporal.** O modelo reflete o mercado de Ames, Iowa (2006–2010). Os bairros só existem para essa cidade, e o preço previsto está em USD daquela época.

**Sobre os dados**

- A remoção de outliers é manual (2 registros), e a de índice 185 depende da posição no `DataFrame`. Se o CSV for reordenado, a linha removida muda. Usar o `Id` do Kaggle é mais robusto.
- A variável `maior_valor` na etapa do `AnoConstrucao` é criada, mas não é usada; a remoção real é feita por `drop(185)`.

**Sobre a API**

- **Todos os 30 campos são obrigatórios.** Não há valores padrão nem tratamento de ausentes.
- **Categorias não validadas.** Um valor inexistente (ex.: `"Bairro": "Xyz"`) não gera erro: o encoder o ignora e a previsão sai sem a contribuição daquela variável, de forma silenciosa. Usar `Literal`/`Enum` no Pydantic evitaria isso.
- **Sem validação de faixa.** `QualidadeGeral: 50` ou `AreaHabitavel: -100` são aceitos.
- **Dependência da ordem.** A API assume que `select_dtypes` devolve as colunas numéricas na mesma ordem do treino. Isso funciona hoje porque `colunas` define a ordem, mas é frágil; o ideal é um único `Pipeline`/`ColumnTransformer`.
- **Caminhos relativos.** Os `.pkl` e o `index.html` são carregados a partir do diretório de trabalho; o `uvicorn` deve ser iniciado dentro da pasta da API.
- **Segurança.** Arquivos `.pkl` executam código ao serem carregados; carregue apenas arquivos de fonte confiável.
- **Versão do scikit-learn.** Carregar os `.pkl` em versão diferente da 1.6.1 emite aviso e, em casos extremos, pode causar resultados inválidos.

---

## 11. Sugestões de melhoria

1. **Aplicar `np.log1p` ao alvo** e reverter com `np.expm1` na previsão.
2. **Unificar o pré-processamento em um `Pipeline` + `ColumnTransformer`** e salvar um único `.pkl`, eliminando o risco de divergência entre treino e API.
3. **Validação cruzada** (`cross_val_score`, `KFold`) para uma estimativa mais estável do desempenho.
4. **Comparar com outros modelos**: `Ridge`, `Lasso`, `ElasticNet`, `RandomForest`, `GradientBoosting`/`XGBoost`.
5. **Validar entradas na API** com `Literal`/`Enum` e `Field(ge=..., le=...)` do Pydantic, e retornar erros claros.
6. **Engenharia de atributos**: idade da casa (`AnoVenda − AnoConstrucao`), área total (porão + habitável), indicador "foi reformada", agrupamento de bairros raros.
7. **Endpoint `/health`** e **testes automatizados** (`pytest` + `TestClient`).
8. **`requirements.txt`** com versões fixas e, opcionalmente, um `Dockerfile`.
9. **Conversão de unidades** (pés² → m², USD → BRL) na interface, se o público for brasileiro.

---

## 12. Dicionário de variáveis

| Nome no projeto | Coluna original | Tipo | Descrição |
|---|---|---|---|
| `Zoneamento` | `MSZoning` | Categórica | Classificação de zoneamento |
| `AreaTerreno` | `LotArea` | Numérica | Área do lote (pés²) |
| `Bairro` | `Neighborhood` | Categórica | Bairro dentro de Ames |
| `TipoConstrucao` | `BldgType` | Categórica | Tipo de moradia |
| `EstiloCasa` | `HouseStyle` | Categórica | Estilo (andares) |
| `QualidadeGeral` | `OverallQual` | Numérica | Qualidade geral de materiais e acabamento (1–10) |
| `CondicaoGeral` | `OverallCond` | Numérica | Condição geral (1–10) |
| `AnoConstrucao` | `YearBuilt` | Numérica | Ano de construção |
| `AnoReforma` | `YearRemodAdd` | Numérica | Ano da reforma (igual ao da construção se não houve) |
| `TipoFundacao` | `Foundation` | Categórica | Tipo de fundação |
| `QualidadeExterna` | `ExterQual` | Categórica | Qualidade do material externo |
| `CondicaoExterna` | `ExterCond` | Categórica | Estado atual do material externo |
| `AreaTotalPorao` | `TotalBsmtSF` | Numérica | Área total do porão (pés²) |
| `QualidadeAquecimento` | `HeatingQC` | Categórica | Qualidade do aquecimento |
| `ArCentral` | `CentralAir` | Categórica | Ar-condicionado central (Y/N) |
| `AreaHabitavel` | `GrLivArea` | Numérica | Área habitável acima do solo (pés²) |
| `BanheirosCompletos` | `FullBath` | Numérica | Banheiros completos acima do solo |
| `MeiosBanheiros` | `HalfBath` | Numérica | Lavabos acima do solo |
| `QualidadeCozinha` | `KitchenQual` | Categórica | Qualidade da cozinha |
| `TotalComodos` | `TotRmsAbvGrd` | Numérica | Cômodos acima do solo (sem banheiros) |
| `Lareiras` | `Fireplaces` | Numérica | Número de lareiras |
| `TipoGaragem` | `GarageType` | Categórica | Localização da garagem (`NoGarage` se não houver) |
| `AcabamentoGaragem` | `GarageFinish` | Categórica | Acabamento interno (`NoGarage` se não houver) |
| `VagasGaragem` | `GarageCars` | Numérica | Capacidade em carros |
| `PavimentacaoEntrada` | `PavedDrive` | Categórica | Pavimentação da entrada |
| `AreaDeckMadeira` | `WoodDeckSF` | Numérica | Área do deck (pés²) |
| `AreaVarandaAberta` | `OpenPorchSF` | Numérica | Área da varanda aberta (pés²) |
| `AreaPiscina` | `PoolArea` | Numérica | Área da piscina (pés²) |
| `QualidadePiscina` | `PoolQC` | Categórica | Qualidade da piscina (`NoPool` se não houver) |
| `CondicaoVenda` | `SaleCondition` | Categórica | Condição da venda |
| `PrecoVenda` | `SalePrice` | **Alvo** | Preço de venda (USD) |

**Removidas após a análise de correlação:** `Cozinhas` (`KitchenAbvGr`), `Quartos` (`BedroomAbvGr`), `AreaGaragem` (`GarageArea`).

**Legenda das escalas de qualidade/condição:** `Ex` = Excelente · `Gd` = Bom · `TA` = Típico/Médio · `Fa` = Regular · `Po` = Ruim.
