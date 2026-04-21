# 🧮 Metodologia e Fundamentação Técnica — DCF Analyzer v2.1

> **Modelo de Valuation por Fluxo de Caixa Descontado aplicado à B3**  
> Baseado nas metodologias de Aswath Damodaran (*Investment Valuation*) e McKinsey (*Valuation: Measuring and Managing the Value of Companies*)

---

## Sumário

1. [Visão Geral da Arquitetura](#1-visão-geral-da-arquitetura)
2. [Coleta e Tratamento de Dados](#2-coleta-e-tratamento-de-dados)
3. [Reconstrução Histórica — FCFF](#3-reconstrução-histórica--fcff)
4. [Defaults Dinâmicos (Piloto Automático)](#4-defaults-dinâmicos-piloto-automático)
5. [Alíquota Efetiva de IR — Por Que 34%?](#5-alíquota-efetiva-de-ir--por-que-34)
6. [Projeção de Demonstrativo de Resultados](#6-projeção-de-demonstrativo-de-resultados)
7. [Ciclo Financeiro e Capital de Giro](#7-ciclo-financeiro-e-capital-de-giro)
8. [FCFF e FCFE Projetados](#8-fcff-e-fcfe-projetados)
9. [Custo de Capital (WACC)](#9-custo-de-capital-wacc)
10. [Valor Terminal — Abordagem Fundamentalista](#10-valor-terminal--abordagem-fundamentalista)
11. [Enterprise Value → Equity Value → Preço Justo](#11-enterprise-value--equity-value--preço-justo)
12. [Método Alternativo: EV/EBITDA Setorial](#12-método-alternativo-evebitda-setorial)
13. [Cenários e Análise de Sensibilidade](#13-cenários-e-análise-de-sensibilidade)
14. [DCF Reverso (Reverse Valuation)](#14-dcf-reverso-reverse-valuation)
15. [Projeção de Dividendos](#15-projeção-de-dividendos)
16. [Indicadores Fundamentalistas](#16-indicadores-fundamentalistas)
17. [Limitações e Premissas do Modelo](#17-limitações-e-premissas-do-modelo)
18. [Referências](#18-referências)

---

## 1. Visão Geral da Arquitetura

O DCF Analyzer é um modelo quantitativo de valuation estruturado em três camadas:

```
┌──────────────────────────────────────────────────────────────┐
│  CAMADA 1 — DADOS                                            │
│  yfinance API → DRE + Balanço + Fluxo de Caixa (5 anos)     │
│  Fallback: Modo Demo com dados reais da WEG                  │
└───────────────────────┬──────────────────────────────────────┘
                        │
┌───────────────────────▼──────────────────────────────────────┐
│  CAMADA 2 — MOTOR DE CÁLCULO                                 │
│  Reconstrução FCFF histórico                                 │
│  Defaults dinâmicos via médias históricas                    │
│  Projeção de DRE + NCG + FCFF (3 cenários × 5 anos)         │
│  Valor Terminal fundamentalista (ROIC-based)                 │
│  DCF Reverso via scipy.optimize.brentq                       │
└───────────────────────┬──────────────────────────────────────┘
                        │
┌───────────────────────▼──────────────────────────────────────┐
│  CAMADA 3 — VISUALIZAÇÃO (Streamlit + Plotly)                │
│  Heatmap de sensibilidade WACC × g                           │
│  Waterfall de decomposição de valor                          │
│  Tabela de premissas histórico vs. projetado                 │
│  Indicadores fundamentalistas com barras de nível            │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. Coleta e Tratamento de Dados

Os dados são obtidos via `yfinance` para ações da B3 (sufixo `.SA`). O modelo busca automaticamente:

| Fonte yfinance | Dados Extraídos |
|---|---|
| `ticker.financials` | DRE: Receita, CPV, SG&A, Depreciação, Juros, IR |
| `ticker.balance_sheet` | Balanço: Caixa, Recebíveis, Estoques, Imobilizado, Fornecedores, Dívida, PL |
| `ticker.cashflow` | CAPEX |
| `ticker.dividends` | Histórico de dividendos por cota |
| `ticker.info` | Preço atual, ações em circulação, Market Cap |

### Normalização de Sinais

O `yfinance` retorna algumas linhas com sinais inconsistentes entre empresas. O modelo normaliza explicitamente:

```python
cpv          = [-abs(v) for v in cpv]          # sempre negativo (custo)
sga          = [-abs(v) for v in sga]          # sempre negativo (despesa)
depreciacao  = [-abs(v) for v in depreciacao]  # sempre negativo (não-caixa)
fornecedores = [-abs(v) for v in fornecedores] # sempre negativo (passivo)
capex        = [-abs(v) for v in capex]        # sempre negativo (saída de caixa)
```

### Ordem Cronológica

O `yfinance` retorna os dados do mais recente para o mais antigo. O modelo inverte todos os vetores para ordem cronológica (ex.: 2020 → 2021 → 2022 → 2023 → 2024) antes de qualquer cálculo.

---

## 3. Reconstrução Histórica — FCFF

O Fluxo de Caixa Livre para a Firma (FCFF — *Free Cash Flow to the Firm*) é a medida central do modelo. Representa o caixa disponível para **todos os provedores de capital** (acionistas e credores), antes do efeito da estrutura de financiamento.

### 3.1 EBIT

$$\text{EBIT} = \text{Receita} + \text{CPV} + \text{SG\&A} + \text{Depreciação}$$

> Os três últimos termos são negativos, portanto a soma equivale à subtração contábil convencional.

### 3.2 NOPAT (Net Operating Profit After Tax)

$$\text{NOPAT} = \text{EBIT} \times (1 - IR_{\text{efetivo}})$$

A alíquota efetiva é calculada historicamente como:

$$IR_{\text{efetivo}} = \min\left(\frac{|\text{Provisão IR}|}{\text{LAI}},\ 0{,}34\right)$$

O `min(..., 0.34)` garante que distorções pontuais (anos com créditos fiscais, diferimentos ou prejuízos compensados) não contaminem os múltiplos históricos com alíquotas irreais. **Veja a Seção 5** para a fundamentação do teto de 34%.

### 3.3 Variação da Necessidade de Capital de Giro (ΔNCG)

$$\text{NCG}_t = \text{Contas a Receber}_t + \text{Estoques}_t + \text{Fornecedores}_t$$

$$\Delta\text{NCG} = -(\text{NCG}_t - \text{NCG}_{t-1})$$

> O sinal negativo converte a variação contábil em impacto de caixa: um **aumento** da NCG **consome** caixa (saída), e uma **redução** da NCG **libera** caixa (entrada).

### 3.4 FCFF

$$\boxed{\text{FCFF} = \text{NOPAT} - \underbrace{(\text{Depreciação})}_{\text{negativo}} + \Delta\text{NCG} + \underbrace{\text{CAPEX}}_{\text{negativo}}}$$

Na implementação, como Depreciação e CAPEX já estão com sinal negativo:

```python
fcff_i = nopat - depreciacao[i] + var_ncg + capex[i]
# Equivale a: NOPAT + |Dep| - ΔNCG_positivo - |CAPEX|
```

---

## 4. Defaults Dinâmicos (Piloto Automático)

O diferencial do modelo é calcular os **parâmetros de projeção diretamente da história da empresa**, em vez de usar valores genéricos. Isso é feito na função `calc_historico_medio()`.

### Parâmetros Calculados Automaticamente

| Parâmetro | Fórmula | Janela |
|---|---|---|
| CMV (% Receita) | `|CPV| / Receita` | Média dos 3 últimos anos |
| SG&A (% Receita) | `|SG&A| / Receita` | Média dos 3 últimos anos |
| Depreciação (% Imobilizado) | `|Depr.| / Imobilizado` | Média dos 3 últimos anos |
| Juros (% Dívida) | `|Juros| / Dívida Total` | Média dos 3 últimos anos |
| Prazo de Recebimento (PMR) | `(Recebíveis / Receita) × 365` | Média dos 3 últimos anos |
| Prazo de Estoque (PME) | `(Estoques / |CPV|) × 365` | Média dos 3 últimos anos |
| Prazo de Pagamento (PMP) | `(|Fornecedores| / |CPV|) × 365` | Média dos 3 últimos anos |
| Multiplicador CAPEX | `|CAPEX| / Depreciação` | Média dos 3 últimos anos |
| IR efetivo | `|IR| / LAI` (cap 34%) | Média dos 3 últimos anos |

### CAGR Histórico de Crescimento

O crescimento-base é calculado como o CAGR de 3 anos da receita:

$$g_{\text{base}} = \left(\frac{\text{Receita}_{t}}{\text{Receita}_{t-3}}\right)^{1/3} - 1$$

Limitado ao intervalo $[-10\%, +20\%]$ para evitar que anos atípicos (IPOs, reestruturações, pandemia) distorçam as projeções.

Os cenários derivam desse CAGR:

| Cenário | Crescimento |
|---|---|
| Conservador | `g_base × 0,5` |
| Base | `g_base` |
| Otimista | `g_base × 1,5` |

---

## 5. Alíquota Efetiva de IR — Por Que 34%?

No Brasil, as empresas com **lucro real** (obrigatório para companhias abertas de grande porte) estão sujeitas a:

| Tributo | Base | Alíquota |
|---|---|---|
| IRPJ — Imposto de Renda Pessoa Jurídica | Lucro Tributável | 15% |
| IRPJ — Adicional | Lucro Tributável > R$ 20.000/mês | 10% |
| CSLL — Contribuição Social sobre o Lucro Líquido | Lucro Ajustado | 9% |
| **Total** | | **34%** |

$$IR_{\text{efetivo\_máx}} = 15\% + 10\% + 9\% = \mathbf{34\%}$$

O modelo utiliza **34% como teto** e não como valor fixo. Na prática, o IR efetivo realizado pelas empresas pode ser menor por incentivos fiscais (ex.: JCP — Juros sobre Capital Próprio, Zona Franca de Manaus, Lei do Bem), diferimentos ou prejuízos a compensar. O modelo detecta isso automaticamente nos históricos e aplica a alíquota real observada — mas **nunca permite que supere 34%**, pois isso indicaria distorção nos dados.

---

## 6. Projeção de Demonstrativo de Resultados

Para cada um dos 5 anos projetados (e em cada cenário), a DRE é construída linha a linha:

### 6.1 Receita

$$\text{Receita}_t = \text{Receita}_{t-1} \times (1 + g)$$

### 6.2 Custos Operacionais

$$\text{CPV}_t = -\text{Receita}_t \times \%\text{CMV}$$

$$\text{SG\&A}_t = -\text{Receita}_t \times \%\text{SGA}$$

### 6.3 Imobilizado, Depreciação e CAPEX

O imobilizado é atualizado dinamicamente a cada período:

$$\text{Depreciação}_t = -\text{Imobilizado}_{t-1} \times \%\text{dep}$$

$$\text{CAPEX}_t = -|\text{Mult\_CAPEX} \times \text{Depreciação}_t|$$

$$\text{Imobilizado}_t = \text{Imobilizado}_{t-1} + |\text{CAPEX}_t| - |\text{Depreciação}_t|$$

> O multiplicador CAPEX captura se a empresa está em modo de **expansão** (CAPEX > Depreciação, multiplicador > 1) ou de **manutenção** (multiplicador ≈ 1).

### 6.4 EBIT, Juros e Lucro Líquido

$$\text{EBIT}_t = \text{Receita}_t + \text{CPV}_t + \text{SG\&A}_t + \text{Depreciação}_t$$

$$\text{Juros}_t = -\text{Dívida}_{t} \times \%\text{juros}$$

$$\text{LAI}_t = \text{EBIT}_t + \text{Juros}_t$$

$$\text{IR}_t = \begin{cases} -\text{LAI}_t \times IR_{\%} & \text{se } \text{LAI}_t > 0 \\ 0 & \text{se } \text{LAI}_t \leq 0 \end{cases}$$

$$\text{Lucro Líquido}_t = \text{LAI}_t + \text{IR}_t$$

### 6.5 Dívida Projetada

A dívida é amortizada gradualmente a 5% ao ano:

$$\text{Dívida}_t = \max(\text{Dívida}_{t-1} \times 0{,}95,\ 0)$$

---

## 7. Ciclo Financeiro e Capital de Giro

O capital de giro é projetado via o **Ciclo de Conversão de Caixa**, derivado dos prazos médios operacionais:

$$\text{Contas a Receber}_t = \text{Receita}_t \times \frac{\text{PMR}}{365}$$

$$\text{Estoques}_t = |\text{CPV}_t| \times \frac{\text{PME}}{365}$$

$$\text{Fornecedores}_t = -|\text{CPV}_t| \times \frac{\text{PMP}}{365}$$

$$\text{NCG}_t = \text{Contas a Receber}_t + \text{Estoques}_t + \text{Fornecedores}_t$$

A variação da NCG impacta o caixa período a período:

$$\Delta\text{NCG}_t = -(\text{NCG}_t - \text{NCG}_{t-1})$$

> **Nota:** Para o primeiro ano projetado, `NCG_{t-1}` é o último valor histórico disponível, garantindo continuidade com o passado real da empresa.

---

## 8. FCFF e FCFE Projetados

### 8.1 FCFF

$$\text{NOPAT}_t = \text{EBIT}_t \times (1 - IR_{\%})$$

$$\text{FCFF}_t = \text{NOPAT}_t - \underbrace{(\text{Depreciação}_t)}_{\text{negativo}} + \Delta\text{NCG}_t + \underbrace{\text{CAPEX}_t}_{\text{negativo}}$$

### 8.2 FCFE (Free Cash Flow to Equity)

O FCFE desconta o custo da dívida e sua amortização:

$$\text{FCFE}_t = \text{FCFF}_t + \text{Juros}_t \times (1 - IR_{\%}) - \text{Amort. Dívida}_t$$

Onde a amortização de dívida é: $\text{Amort.}_t = \text{Dívida}_t \times 5\%$

### 8.3 FCF (Free Cash Flow — Perspectiva do Acionista)

$$\text{FCO}_t = \text{Lucro Líquido}_t - \text{Depreciação}_t + \Delta\text{NCG}_t$$

$$\text{FCF}_t = \text{FCO}_t + \text{CAPEX}_t$$

---

## 9. Custo de Capital (WACC)

O WACC é inserido pelo usuário como parâmetro. Para ações brasileiras, o modelo sugere como referência a composição:

$$\text{WACC} = K_e \times \frac{E}{E+D} + K_d \times (1-IR) \times \frac{D}{E+D}$$

Onde:
- $K_e$ = Custo do capital próprio (CAPM ajustado para Brasil: SELIC + prêmio de risco)
- $K_d$ = Custo da dívida (CDI + spread de crédito)
- $E/(E+D)$ = Proporção de capital próprio no capital total
- $D/(E+D)$ = Proporção de dívida no capital total

> **Referência para B3 (2024–2025):** WACCs entre 12% e 22% são comuns para empresas listadas, refletindo a taxa SELIC elevada e o risco-Brasil. O modelo permite qualquer valor entre 5% e 40%.

---

## 10. Valor Terminal — Abordagem Fundamentalista

O Valor Terminal (TV) representa o valor da empresa **além do horizonte de projeção explícita** (ano 5+). É o componente que geralmente representa 60–80% do Enterprise Value em DCFs de empresas maduras.

O modelo utiliza a **abordagem de reinvestimento fundamentalista de Damodaran**, que amarra o crescimento na perpetuidade ao Retorno sobre o Capital Investido (ROIC). Isso evita o erro clássico de assumir que uma empresa pode crescer para sempre sem reinvestir proporcionalmente.

### 10.1 ROIC Terminal

$$\text{ROIC}_{TV} = \frac{\text{NOPAT}_n}{\text{Capital Empregado}_n}$$

$$\text{Capital Empregado}_n = \text{Imobilizado}_n + \text{NCG}_n$$

### 10.2 Taxa de Reinvestimento na Perpetuidade

$$\text{Taxa de Reinvestimento} = \frac{g_{\text{perp}}}{\text{ROIC}_{TV}}$$

Limitada ao intervalo $[0\%, 95\%]$ para evitar valores economicamente incoerentes.

> **Intuição:** Se o ROIC = 20% e o crescimento na perpetuidade = 5%, a empresa precisa reinvestir 25% do NOPAT para financiar esse crescimento. Os outros 75% são distribuídos como FCFF livre.

### 10.3 FCFF na Perpetuidade

$$\text{FCFF}_{\text{perp}} = \text{NOPAT}_n \times (1 - \text{Taxa de Reinvestimento})$$

### 10.4 Valor Terminal (Modelo de Gordon)

$$\text{TV} = \frac{\text{FCFF}_{\text{perp}} \times (1 + g_{\text{perp}})}{\text{WACC} - g_{\text{perp}}}$$

> **Restrição:** O modelo exige $\text{WACC} > g_{\text{perp}}$, que é matematicamente necessário para que a série geométrica convirja. Na prática, o crescimento na perpetuidade não pode superar a taxa de crescimento da economia de longo prazo.

### 10.5 Valor Presente do TV

$$\text{VP}_{TV} = \frac{\text{TV}}{(1 + \text{WACC})^n}$$

---

## 11. Enterprise Value → Equity Value → Preço Justo

### 11.1 Valor Presente dos FCFFs Explícitos

$$\text{VP}_{\text{FCFF}} = \sum_{t=1}^{5} \frac{\text{FCFF}_t}{(1 + \text{WACC})^t}$$

### 11.2 Enterprise Value

$$\text{Enterprise Value} = \text{VP}_{\text{FCFF}} + \text{VP}_{TV}$$

### 11.3 Dívida Líquida

$$\text{Dívida Líquida} = \text{Dívida Total} - \text{Caixa e Equivalentes}$$

> Quando negativa (caixa > dívida), a dívida líquida **adiciona** valor ao acionista — a empresa tem posição credora líquida.

### 11.4 Equity Value e Preço Justo

$$\text{Equity Value} = \text{Enterprise Value} - \text{Dívida Líquida}$$

$$\boxed{\text{Preço Justo por Ação} = \frac{\text{Equity Value}}{\text{N° de Ações em Circulação (M)}}}$$

### 11.5 Upside / Downside

$$\text{Upside} = \frac{\text{Preço Justo} - \text{Preço Atual}}{\text{Preço Atual}} \times 100\%$$

### 11.6 Veredicto de Valuation (Mediana dos Cenários)

| Upside vs. Mediana | Veredicto |
|---|---|
| > +30% | 🟢 BARATA |
| +10% a +30% | 🟡 LEVEMENTE BARATA |
| -10% a +10% | 🔵 PREÇO JUSTO |
| -25% a -10% | 🟡 LEVEMENTE CARA |
| < -25% | 🔴 CARA |

---

## 12. Método Alternativo: EV/EBITDA Setorial

Como método de confirmação, o modelo calcula um preço justo baseado no múltiplo EV/EBITDA do setor:

$$\text{EV}_{\text{múltiplo}} = \text{EV/EBITDA}_{\text{setor}} \times \text{EBITDA}_{\text{forward}}$$

$$\text{Preço Justo (Múltiplo)} = \frac{\text{EV}_{\text{múltiplo}} - \text{Dívida Líquida}}{\text{Ações}}$$

Onde $\text{EBITDA}_{\text{forward}}$ é o EBITDA projetado para o Ano 1 do cenário Base.

> Este método é útil para **cross-check**: se o DCF e o múltiplo setorial convergirem, há maior convicção na estimativa. Divergências grandes indicam que a empresa negocia com prêmio ou desconto estrutural ao setor.

---

## 13. Cenários e Análise de Sensibilidade

### 13.1 Deltas de Cenário

Os cenários Otimista e Conservador são construídos como **deltas** sobre o cenário Base:

| Parâmetro | Otimista | Conservador |
|---|---|---|
| Crescimento | Base + 3 p.p. | Base − 3 p.p. |
| CMV | Base − 2 p.p. | Base + 2 p.p. |
| SG&A | Base − 2 p.p. | Base + 2 p.p. |
| PMR | Base − 5 dias | Base + 5 dias |
| PME | Base − 5 dias | Base + 5 dias |
| PMP | Base + 5 dias | Base − 5 dias |
| Payout | Base + 5 p.p. | Base − 5 p.p. |

### 13.2 Heatmap de Sensibilidade (WACC × g)

O modelo executa 49 combinações (7 WACCs × 7 taxas de crescimento) para construir a matriz de sensibilidade:

$$\text{WACC} \in \{10\%, 13\%, 15\%, 18\%, 20\%, 22\%, 25\%\}$$

$$g \in \{0\%, 3\%, 5\%, 7\%, 10\%, 13\%, 15\%\}$$

Para cada par $(\text{WACC}, g)$, recalcula-se o DCF completo e registra-se o preço justo resultante.

---

## 14. DCF Reverso (Reverse Valuation)

O DCF Reverso responde à pergunta: *"Qual taxa de crescimento está implícita no preço atual de mercado?"*

O problema é resolvido como uma equação de ponto fixo usando o algoritmo **Brent-Dekker** (`scipy.optimize.brentq`), que encontra a raiz de uma função contínua em um intervalo:

$$f(g) = \text{Preço Justo}(g) - \text{Preço Atual} = 0$$

Busca-se $g^*$ no intervalo $[-30\%, +150\%]$.

```python
from scipy.optimize import brentq

def diff(c):
    pp = dict(p)
    pp["cresc_base"] = c
    return calc_cenario("R", hist, preco, acoes, pp)["pj"] - preco

cresc_impl = brentq(diff, -0.30, 1.50)
```

> **Utilidade prática:** Se o crescimento implícito for muito acima do histórico da empresa, o mercado está precificando um cenário extremamente otimista — o que pode indicar sobrevalorização. Se for abaixo do histórico, o mercado pode estar sendo excessivamente pessimista.

---

## 15. Projeção de Dividendos

Os dividendos projetados derivam diretamente do lucro líquido e do payout definido nas premissas:

$$\text{Dividendos}_t = \max(\text{Lucro Líquido}_t \times \text{Payout}, \ 0)$$

$$\text{DPA}_t = \frac{\text{Dividendos}_t}{\text{Ações em Circulação}}$$

$$\text{Dividend Yield}_t = \frac{\text{DPA}_t}{\text{Preço Atual}} \times 100\%$$

> Não há distribuição de dividendos quando o lucro líquido é negativo (`max(..., 0)`), respeitando a legislação societária brasileira (Lei 6.404/76).

---

## 16. Indicadores Fundamentalistas

O modelo calcula automaticamente dois conjuntos de indicadores: históricos (último balanço) e projetados (Ano 1).

### Múltiplos de Preço

| Indicador | Fórmula |
|---|---|
| P/L (Price to Earnings) | `Preço / LPA` |
| P/VPA (Price to Book) | `Preço / (PL / Ações)` |
| EV/EBITDA | `(MC + Dívida Líquida) / EBITDA` |
| EV/Receita | `EV / Receita` |
| P/SR (Price to Sales) | `MC / Receita` |

### Margens

| Indicador | Fórmula |
|---|---|
| Margem Bruta | `(Receita + CPV) / Receita` |
| Margem EBITDA | `EBITDA / Receita` |
| Margem EBIT | `EBIT / Receita` |
| Margem Líquida | `Lucro Líquido / Receita` |

### Alavancagem e Solidez

| Indicador | Fórmula |
|---|---|
| Dívida / PL | `Dívida Total / Patrimônio Líquido` |
| Dívida Líquida / EBITDA | `(Dívida − Caixa) / EBITDA` |
| Cobertura de Juros | `EBIT / |Juros|` |

### Rentabilidade

| Indicador | Fórmula |
|---|---|
| ROE | `Lucro Líquido / PL × 100` |
| ROIC | `NOPAT / (PL + Dívida Líquida) × 100` |
| CAGR Receita 3 anos | `(Receita_t / Receita_{t-3})^{1/3} - 1` |
| CAGR Lucro 3 anos | `(LL_t / LL_{t-3})^{1/3} - 1` |

---

## 17. Limitações e Premissas do Modelo

Este modelo é uma ferramenta de suporte analítico, **não** uma recomendação de investimento.

### Premissas Simplificadoras

- **Crescimento constante** na fase explícita (5 anos). Modelos mais sofisticados podem usar crescimento decrescente (fade period).
- **Estrutura de capital estável**: o WACC não muda ao longo dos anos projetados.
- **Margens constantes**: CMV e SG&A são percentuais fixos da receita. Na realidade, há efeito de alavancagem operacional.
- **Dívida amortizada linearmente** a 5% ao ano. Cada empresa tem um cronograma real de vencimentos.
- **Alíquota IR uniforme**: não captura variações por regime tributário (Lucro Presumido, incentivos fiscais específicos).

### Fontes de Incerteza

- Dados do `yfinance` podem conter inconsistências, especialmente para empresas com demonstrações consolidadas complexas.
- O Valor Terminal representa tipicamente 60–80% do EV — pequenas mudanças em WACC e `g_perp` têm impacto desproporcional no preço justo.
- Para empresas com prejuízo recorrente, crescimento negativo recente ou alto endividamento, os resultados devem ser interpretados com cautela adicional.

### O Que o Modelo Não Calcula

- Risco específico por cenário (VaR, simulação de Monte Carlo)
- Impacto de M&A, desinvestimentos ou emissão de novas ações
- Ajustes por participações minoritárias e coligadas
- Efeito de opções reais (expansão, abandono, diferimento)

---

## 18. Referências

- **Damodaran, A.** (2012). *Investment Valuation: Tools and Techniques for Determining the Value of Any Asset* (3rd ed.). Wiley Finance.
- **Koller, T., Goedhart, M., & Wessels, D.** (2020). *Valuation: Measuring and Managing the Value of Companies* (7th ed.). McKinsey & Company / Wiley.
- **Gordon, M. J.** (1962). *The Investment, Financing, and Valuation of the Corporation*. Irwin.
- **Brent, R. P.** (1973). *Algorithms for Minimization without Derivatives*. Prentice-Hall. *(Base do algoritmo `brentq` usado no DCF Reverso)*
- **Lei 6.404/76** — Lei das Sociedades por Ações (Brasil).
- **RIR/2018 (Decreto 9.580/2018)** — Regulamento do Imposto de Renda (IRPJ + ADICIONAL + CSLL = 34%).

---

<div align="center">

**DCF Analyzer v2.1** · Desenvolvido com Python, Streamlit e Plotly  
Dados via `yfinance` · Otimização via `scipy.optimize`  

*"Price is what you pay. Value is what you get."* — Warren Buffett

</div>
