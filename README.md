# 📈 Python DCF Valuation Model

Este repositório contém um modelo de Fluxo de Caixa Descontado (DCF) automatizado em Python, desenvolvido para realizar o valuation de empresas de capital aberto com rigor técnico e eficiência analítica. O projeto une os fundamentos contábeis da UEFS com lógica de programação para suporte à tomada de decisão estratégica em FP&A e M&A.

## 🔍 Visão Geral
O modelo processa dados financeiros históricos para projetar fluxos de caixa futuros e determinar o valor intrínseco de um ativo.
* **Rigor Contábil**: Análise profunda de Demonstrações Financeiras (DRE e Balanço) para embasar as premissas.
* **Diferencial Analítico**: Foco em análise Marginal para capturar a real geração de valor em cenários de crescimento.
* **Automação**: Coleta de dados via API e geração automática de indicadores-chave como EBITDA, Margens e Fluxo de Caixa Livre.

## 🛠️ Tecnologias e Bibliotecas
* **Python**: Core da automação e lógica de modelagem.
* **Pandas & NumPy**: Manipulação de grandes volumes de dados e cálculos financeiros complexos.
* **Yfinance**: Extração de dados de mercado e cotações em tempo real.
* **Matplotlib**: Visualização de projeções e análise de sensibilidade.

## 📊 Metodologia Financeira
A estruturação do modelo segue as melhores práticas de mercado:
1. **Projeções de Fluxo de Caixa (FCFF)**: Horizonte de 5 a 10 anos baseado em taxas de crescimento histórico e setorial.
2. **Cálculo do WACC**: Estimativa do custo médio ponderado de capital com ajuste de prêmios de risco.
3. **Valor de Perpetuidade (Terminal Value)**: Aplicação do modelo de Gordon com taxas de crescimento conservadoras.
4. **Análise de Sensibilidade**: Matriz que cruza variações de WACC vs. g para definir um range de valor justo.

## 🚀 Como Executar
1. Clone este repositório: `git clone https://github.com/Giovanni075/dcf-python`
2. Instale as dependências necessárias: `pip install -r requirements.txt`
3. Execute o script principal ou o Jupyter Notebook para visualizar a análise completa.

## 👨‍💻 Sobre o Autor
**Giovanni Silva de Souza**
* Graduando em Ciências Contábeis (UEFS) e Análise e Desenvolvimento de Sistemas.
* Assistente Fiscal na Stratus, com sólida base em rotinas contábeis e fiscais.
* Certificado CPA-20 (ANBIMA).
* Objetivo: Atuar em FP&A, Valuation e M&A, gerando valor através de análise de dados e visão estratégica.

---
📫 **Vamos nos conectar?** [LinkedIn](https://www.linkedin.com/in/giovanni-silva-de-souza-b50198282)
