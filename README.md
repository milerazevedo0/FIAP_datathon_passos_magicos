# 📘 Datathon Passos Mágicos – Monitoramento de Risco de Defasagem Educacional
📌 Visão Geral

Este projeto foi desenvolvido no contexto do Datathon Passos Mágicos, com o objetivo de aplicar técnicas de Machine Learning para identificar alunos com risco de defasagem educacional, a partir de indicadores acadêmicos e pedagógicos.

A solução vai além do treinamento de um modelo preditivo, incorporando boas práticas de engenharia de Machine Learning (MLOps), como:

API para inferência

Testes unitários

Monitoramento contínuo

Detecção de drift

Painel visual

Containerização com Docker

# 🎯 Objetivo do Projeto

Construir um sistema capaz de:

Prever se um aluno apresenta risco de defasagem educacional

Fornecer a probabilidade associada à predição

Disponibilizar a predição via API

Monitorar a estabilidade dos dados ao longo do tempo

Detectar mudanças de distribuição (drift) nos dados de produção

# 🧠 Definição do Problema

O problema foi tratado como uma classificação binária, onde:

Valor	Significado
0	Aluno não possui risco de defasagem
1	Aluno possui risco de defasagem
🎯 Target

Coluna Defas ou Defasagem (dependendo da aba)

Regra:

Valor < 0 → risco de defasagem

Valor >= 0 → sem risco

# 📊 Dados Utilizados

Arquivo principal:

BASE DE DADOS PEDE 2024 - DATATHON.xlsx


Abas utilizadas:

PEDE2022

PEDE2023

PEDE2024

## 🔹 Features de Entrada

As features utilizadas no modelo foram padronizadas para lidar com inconsistências entre abas:

IAA
IEG
IPS
IPP
IDA
IPV
IAN
PORTUGUES (Por / Portug)
MATEMATICA (Mat / Matem)
INGLES (Ing / Inglês)


O pipeline trata automaticamente:

Diferenças de nomenclatura

Ordem variável das colunas

Valores ausentes

# ⚙️ Pipeline de Machine Learning
## 🔹 Pré-processamento

Padronização de nomes de colunas

Seleção das features relevantes

Criação do target

Tratamento de valores ausentes (imputação)

## 🔹 Feature Engineering

Consolidação de colunas equivalentes

Garantia de schema consistente

## 🔹 Treinamento

Validação temporal:

Treino: PEDE2022 + PEDE2023

Teste: PEDE2024

Modelo treinado sobre dados escalados

Métricas avaliadas:

Precision

Recall

F1-score

ROC AUC

## 🔹 Persistência

Artefatos salvos:

artifacts/
├── model.pkl
├── scaler.pkl
├── imputer.pkl
└── features_used.pkl

# 📡 API de Inferência

A solução expõe uma API FastAPI para consumo do modelo.

## 🔹 Endpoint de predição

POST /predict

Exemplo de input:

{
  "IAA": 6.5,
  "IEG": 6.8,
  "IPS": 6.3,
  "IPP": 6.6,
  "IDA": 6.7,
  "IPV": 6.4,
  "IAN": 6.5,
  "PORTUGUES": 6.2,
  "MATEMATICA": 6.9,
  "INGLES": 6.4
}


Exemplo de resposta:

{
  "prediction": 0,
  "prediction_label": "Aluno não possui risco de defasagem",
  "probability": 0.27
}

# 🧪 Testes Unitários

Foram implementados testes unitários cobrindo:

Pré-processamento

Feature engineering

Pipeline de treino

Predição

Cobertura:

100% dos módulos críticos

Ferramenta utilizada:

pytest

Execução:

pytest

# 📈 Monitoramento e Detecção de Drift
## 🔹 Logging de Produção

Cada predição é registrada com:

Features de entrada

Predição

Probabilidade

Timestamp (fuso São Paulo)

Arquivo:

data/predictions_log.csv

## 🔹 Detecção de Drift

Utilizado PSI (Population Stability Index)

Comparação entre:

Baseline (dados reais do treino, escala original)

Dados de produção

Classificação:

Sem drift → PSI < 0.1

Drift moderado → 0.1 ≤ PSI < 0.25

Drift severo → PSI ≥ 0.25

Endpoint:

GET /monitoring/drift

# 📊 Painel Visual de Monitoramento

Foi implementado um painel web simples, acessível via navegador, para acompanhamento visual do drift.

## 📍 URL:

/monitoring/dashboard


O painel exibe:

Feature

Valor do PSI

Status com cores indicativas

## 🏠 Página Inicial (README Web)

A aplicação disponibiliza uma página inicial (/) que funciona como um README interativo, contendo:

Contexto do projeto

Objetivo

Endpoints disponíveis

Acesso ao painel de drift

# 🚀 Instalação e Execução da Aplicação

## 🔹 Opção 1 – Execução Local (sem Docker)

#### Pré-requisitos
- Python 3.10+
- pip

#### 1️⃣ Clone o repositório
```bash
git clone <url-do-repositorio>
cd datathon_passos_magicos_V2
```

#### 2️⃣ Crie e ative o ambiente virtual
```bash
python -m venv venv
```
Windows:
```bash
venv\Scripts\activate
```
Linux / Mac:
```bash
source venv/bin/activate
```

#### 3️⃣ Instale as dependências
```bash
pip install -r requirements.txt
```

#### 4️⃣ Execute o treinamento
```bash
python train_model.py
```

#### 5️⃣ Suba a API
```bash
uvicorn app.main:app --reload
```

Acesse:
- http://localhost:8000/
- http://localhost:8000/docs
- http://localhost:8000/monitoring/dashboard

---

## 🔹 Opção 2 – Execução com Docker (Recomendado)

#### Pré-requisitos
- Docker
- Docker Compose

#### Build e execução
```bash
docker-compose up --build
```

Acesse:
- http://localhost:8000/
- http://localhost:8000/docs
- http://localhost:8000/monitoring/dashboard

---

# 🗂️ Estrutura do Projeto
datathon_passos_magicos/
│
├── app/
│   ├── main.py
│   ├── monitoring_routes.py
│   ├── routes.py
│   ├── schemas.py
│   └── templates/
│       ├── index.html
│       └── drift_dashboard.html
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── predict.py
│   ├── config.py
│   └── model.py
│
├── monitoring/
│   ├── logger.py
│   ├── run_drift_check.py
│   ├── drift_history.py
│   └── drift.py
│
├── artifacts/
│   ├── features_used.pkl
│   ├── imputer.pkl
│   ├── model.pkl
│   └── scaler.pkl
│
├── data/
│   ├── baseline_stats.json
│   └── predictions_log.csv
│
├── tests/
│   ├── conftest.py
│   ├── test_feature_engineering.py
│   ├── test_predict.py
│   ├── test_preprocessing.py
│   └── test_train_pipeline.py
│
├── Dockerfile
├── docker-compose.yml
├── BASE DE DADOS PEDE 2024 - DATATHON.xlsx
├── train_model.py
├── generate_control_production_data.py
├── requirements.txt
└── README.md

# 🧱 Stack Tecnológica Utilizada
- Python 3.10+
- Pandas, NumPy
- Scikit-learn
- FastAPI, Uvicorn
- Pydantic
- Pytest
- Jinja2
- Docker, Docker Compose
- OpenPyXL
- Git

# 🏁 Conclusão

O projeto atende integralmente aos requisitos do Datathon, entregando:

Modelo de Machine Learning funcional

API para inferência

Testes automatizados

Monitoramento contínuo

Detecção de drift

Painel visual

Containerização

Além disso, incorpora práticas de MLOps, elevando a solução para um nível próximo ao de ambiente produtivo real.

# 👨‍💻 Observação Final

Este projeto foi desenvolvido para fins educacionais e analíticos, demonstrando a aplicação prática de Machine Learning, engenharia de dados e monitoramento de modelos.