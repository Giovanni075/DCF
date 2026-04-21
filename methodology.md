## 🧠 Metodologia e Fundamentação Contábil (Matemática do Modelo)

O **DCF Analyzer** não é uma "caixa-preta". Todos os cálculos seguem as melhores práticas de Finanças Corporativas, normas contábeis e a metodologia de avaliação de empresas de Aswath Damodaran e McKinsey.

Abaixo estão as fórmulas e a lógica step-by-step implementadas no motor de cálculo (`modelo_valuation.py`):

### 1. Reconstrução Histórica e Fluxo de Caixa Livre (FCFF)
O modelo processa o balanço e a DRE dos últimos 5 anos para extrair o Fluxo de Caixa Livre para a Firma (FCFF).

* **NOPAT (Lucro Operacional Líquido Após Impostos):** Calculado com base no EBIT ajustado pela alíquota efetiva histórica de Imposto de Renda.
  $$NOPAT = EBIT \times (1 - IR_{efetivo})$$

* **Variação da Necessidade de Capital de Giro ($\Delta NCG$):** Mede o impacto no caixa das contas operacionais.
  $$NCG_{t} = ContasAReceber_{t} + Estoques_{t} - Fornecedores_{t}$$
  $$\Delta NCG = -(NCG_{t} - NCG_{t-1})$$

* **Fluxo de Caixa Livre para a Firma (FCFF):**
  $$FCFF = NOPAT + Depreciação + \Delta NCG - CAPEX$$
  *(Nota: No código, as despesas operacionais e saídas de caixa são tratadas com sinal negativo para somatório direto).*

### 2. Projeções (Próximos 5 Anos)
A modelagem prospectiva (cenários Base, Otimista e Conservador) utiliza vetores de premissas dinâmicas ou definidas pelo usuário:
* **Receita:** Cresce a uma taxa $g$ anual.
* **Margens:** CPV (Custo do Produto Vendido) e SG&A são projetados como um percentual (%) fixo da Receita.
* **Capital de Giro:** Projetado via ciclo financeiro, utilizando os prazos médios de recebimento (PMR), estoque (PME) e pagamento (PMP).

### 3. Custo de Capital (WACC)
O modelo utiliza o WACC (Custo Médio Ponderado de Capital) fornecido nos parâmetros para descontar os fluxos de caixa futuros a valor presente, refletindo a estrutura de capital e o risco inerente do negócio.

$$VP_{FCFF} = \sum_{t=1}^{n} \frac{FCFF_{t}}{(1 + WACC)^{t}}$$

### 4. Valor Terminal (Abordagem Fundamentalista)
Diferente de modelos simplistas, este projeto calcula a Perpetuidade baseada nos fundamentos de retorno sobre o capital, garantindo que a empresa não cresça mais do que seus reinvestimentos permitem.

* **ROIC Terminal e Reinvestimento:**
  $$ROIC_{TV} = \frac{NOPAT_{n}}{CapitalEmpregado_{n}}$$
  $$TaxaReinvestimento = \frac{g_{perp}}{ROIC_{TV}}$$
  
* **FCFF na Perpetuidade:**
  $$FCFF_{perp} = NOPAT_{n} \times (1 - TaxaReinvestimento)$$

* **Cálculo do Valor Terminal (Modelo de Gordon):**
  $$TV = \frac{FCFF_{perp} \times (1 + g_{perp})}{WACC - g_{perp}}$$

* **Valor Presente do Valor Terminal:**
  $$VP_{TV} = \frac{TV}{(1 + WACC)^{n}}$$

### 5. Enterprise Value e Preço Justo (Equity Value)
Para chegar ao valor do acionista, o modelo consolida os fluxos e desconta o endividamento líquido.

* **Enterprise Value (Valor da Firma):**
  $$Enterprise Value = VP_{FCFF} + VP_{TV}$$

* **Equity Value (Valor Justo do Acionista):**
  $$Equity Value = Enterprise Value - DividaLiquida$$
  
* **Preço Justo por Ação:**
  $$Preço Justo = \frac{Equity Value}{Número de Ações em Circulação}$$

### 6. DCF Reverso (Reverse Valuation)
Utilizando o algoritmo de otimização `brentq` da biblioteca SciPy, o modelo calcula a taxa de crescimento ($g$) implícita que o mercado está precificando no momento. 

O algoritmo resolve iterativamente a seguinte equação para $g$:
$$PreçoJusto(g) - PreçoAtual de Mercado = 0$$
Isso permite responder rapidamente à pergunta: *"Quanto essa empresa precisa crescer por ano para justificar seu preço atual na B3?"*
