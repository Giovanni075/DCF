DCF Analyzer v2.1 — Modelo de Valuation Interativo
Link do App: 🚀 https://giovanni-valuation.streamlit.app/ Acessar Dashboard em Tempo Real

Este repositório contém um sistema avançado de Valuation por Fluxo de Caixa Descontado (DCF) automatizado em Python. O projeto foi desenhado para unir o rigor técnico das Ciências Contábeis com a eficiência da Análise e Desenvolvimento de Sistemas (ADS), focado em suporte à tomada de decisão estratégica em FP&A e M&A.

🎯 Proposta do Projeto
Diferente de modelos estáticos em planilhas, este Analyzer permite uma exploração dinâmica de teses de investimento. Ele automatiza a coleta de dados de empresas de capital aberto (B3) e permite o ajuste em tempo real de premissas macroeconômicas e operacionais.

🛠️ Stack Técnica e Ambiente Anaconda
Um diferencial fundamental deste projeto é a sua reprodutibilidade e estabilidade. O desenvolvimento e a validação dos cálculos financeiros foram realizados utilizando o Anaconda Navigator.

Por que Anaconda?
O uso do ecossistema Anaconda foi estratégico para garantir:

Isolamento de Ambiente: Prevenção de conflitos entre bibliotecas de ciência de dados (Pandas, Numpy) e interface (Streamlit).

Integridade dos Cálculos: Garantia de que as funções estatísticas e financeiras utilizem versões estáveis, evitando erros de precisão contábil.

Workflow Profissional: Demonstração de domínio de ferramentas padrão da indústria para Data Science e Engenharia de Dados.

🚀 Funcionalidades de Elite
Integração com B3: Busca automática de 5 anos de histórico financeiro via Yahoo Finance API.

Análise de Sensibilidade Interativa: Heatmap dinâmico que cruza WACC vs. Crescimento Perpetuidade (g) para definir o range de valor justo.

Decomposição Waterfall: Visualização clara do peso do Valor Terminal na composição do Equity Value.

DCF Reverso: Algoritmo de otimização numérica que calcula o crescimento implícito necessário para justificar o preço atual de mercado.

Projeção de Cenários: Alternância instantânea entre cenários Base, Otimista e Conservador com impactos automáticos em margens e payout.

📡 Resiliência de Sistemas (Contingência de API)
Como boa prática de ADS, o sistema conta com uma camada de proteção contra falhas externas. Caso a API do Yahoo Finance atinja o limite de requisições (Rate Limit) no servidor de nuvem, o App oferece o Modo de Demonstração.

Dados Mockados: Carregamento instantâneo de dados históricos auditados da WEG S.A., permitindo que a análise de funcionalidades e a navegação nunca sejam interrompidas.

💻 Como Executar (Ambiente Local)
Se você utiliza Anaconda, siga os passos abaixo para rodar o modelo:

Criar o ambiente via terminal (Conda Prompt):

Bash
conda create --name dcf-analyzer python=3.10
Ativar o ambiente:

Bash
conda activate dcf-analyzer
Instalar dependências (Padrão requirements.txt):

Bash
pip install -r requirements.txt
Rodar o App:

Bash
streamlit run modelo_valuation.py
🧠 Sobre o Autor
Giovanni Silva de Souza

🎓 Graduando em Ciências Contábeis (7º Semestre) — Universidade Estadual de Feira de Santana (UEFS).

💻 Graduando em Análise e Desenvolvimento de Sistemas.

🏦 Certificado CPA-20 (ANBIMA).

💼 Atuante em rotinas administrativas e organizacionais na Stratus.

🎯 Foco de carreira em Asset Management, Análise de Crédito e FP&A.

Desenvolvido com Python, Streamlit, Pandas e muito rigor contábil.
