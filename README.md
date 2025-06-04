# Previsão de Sífilis Congênita a Partir de Registros do PMCP

Este projeto visa prever a sífilis congênita utilizando técnicas de aprendizado de máquina a partir de registros do PMCP (Programa de Prevenção da Transmissão Vertical). Esta versão foca em uma análise aprimorada com engenharia de atributos avançada, métodos de ensemble e aprendizado sensível a custos para otimizar a detecção e o custo-benefício do rastreamento.

## Destaques Técnicos da Análise Aprimorada

O núcleo desta análise, detalhado em `notebooks/final_notebook.ipynb`, inclui:

* **Engenharia de Atributos Avançada:** Criação de novas features como `RISK_SCORE`, `SOCIO_ECONOMIC_INDEX`, `HEALTH_AWARENESS` e `AGE_RISK_CATEGORY` para capturar relações complexas nos dados.
* **Pré-processamento e Balanceamento:**
    * Transformação de dados utilizando `SimpleImputer`, `StandardScaler` e `OneHotEncoder`.
    * Uso da técnica `SMOTETomek` para lidar com o desbalanceamento de classes.
* **Seleção e Importância de Atributos:** Análise de relevância das features utilizando Random Forest e Mutual Information.
* **Modelagem Avançada com Ensembles:**
    * Implementação e avaliação de modelos base otimizados: Regressão Logística, Random Forest, XGBoost e Gradient Boosting.
    * Construção e avaliação de modelos de ensemble: `VotingClassifier` (soft voting) e `StackingClassifier`.
* **Aprendizado Sensível a Custos:**
    * Definição de uma matriz de custos para simular o impacto financeiro de diferentes erros de classificação (Falsos Positivos, Falsos Negativos).
    * Otimização do threshold de decisão dos modelos para minimizar o custo total associado ao rastreamento.
* **Modelo Recomendado:** `Gradient Boosting`, com um threshold otimizado de **0.250**, demonstrou o melhor desempenho na análise de custo-benefício.
* **Artefato de Deployment:** O modelo final recomendado, juntamente com o pré-processador e metadados relevantes (threshold, nomes das features, matriz de custo), é salvo em `models/enhanced_deployment_model.joblib`.

## Uso

1.  Certifique-se de que o arquivo de dados brutos (ex: `data_set.csv`) esteja localizado no diretório `data/`.
2.  Para executar o pipeline completo de análise, engenharia de atributos, treinamento de modelos, avaliação e otimização de custos, execute o notebook Jupyter:
    `notebooks/final_notebook.ipynb`
3.  O modelo treinado e pronto para deployment (`enhanced_deployment_model.joblib`) estará disponível no diretório `models/` após a execução bem-sucedida do notebook.

## Reprodutibilidade

As seeds aleatórias são fixadas em `42` em todos os processos estocásticos (divisão de dados, modelos, etc.) para garantir a reprodutibilidade dos resultados. Consulte `src/utils.py` e os notebooks para detalhes específicos da implementação.

## Integrantes do Grupo

Lara Pessoa - 

Pedro Xavier - @PHxavier1

Rafael Menezes -

Théo Moura - @theomilll

## Disciplina

Aprendizado de Máquina - 2025.1

## Instituição de ensino

CESAR School