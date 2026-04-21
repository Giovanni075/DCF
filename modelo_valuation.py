"""
DCF ANALYZER v2.1 — Modelo de Valuation por Fluxo de Caixa Descontado
Evolução: "Piloto Automático" (Data-Driven Defaults) baseados no histórico real de cada empresa.
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import brentq

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DCF Analyzer v2.1",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

:root {
    --bg: #080c14; --surface: #0d1220; --card: #111827;
    --border: #1e2a40; --border-bright: #2a3d5c;
    --text: #e2e8f0; --muted: #64748b; --accent: #3b82f6;
    --green: #10b981; --red: #ef4444; --amber: #f59e0b;
    --purple: #8b5cf6; --cyan: #06b6d4;
}
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; color: var(--text); }
.stApp { background: var(--bg); }
[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border) !important; }
[data-testid="stSidebar"] * { color: var(--text) !important; }
.stTabs [data-baseweb="tab-list"] { background: var(--surface); border-bottom: 1px solid var(--border); gap: 0; }
.stTabs [data-baseweb="tab"] {
    background: transparent; color: var(--muted) !important;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; font-weight: 600;
    letter-spacing: 0.05em; border-bottom: 2px solid transparent;
    padding: 0.6rem 1.2rem; transition: all 0.2s;
}
.stTabs [aria-selected="true"] { color: var(--accent) !important; border-bottom: 2px solid var(--accent) !important; background: transparent !important; }
.stNumberInput input, .stTextInput input, .stSelectbox select {
    background: var(--card) !important; border: 1px solid var(--border) !important;
    color: var(--text) !important; border-radius: 6px !important;
    font-family: 'IBM Plex Mono', monospace !important; font-size: 0.85rem !important;
}
.stNumberInput input:focus, .stTextInput input:focus {
    border-color: var(--accent) !important; box-shadow: 0 0 0 2px rgba(59,130,246,0.15) !important;
}
[data-testid="metric-container"] {
    background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 1rem;
}
[data-testid="metric-container"] label {
    color: var(--muted) !important; font-size: 0.72rem !important;
    font-family: 'IBM Plex Mono', monospace !important;
    letter-spacing: 0.08em !important; text-transform: uppercase;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important; font-size: 1.4rem !important; font-weight: 600;
}
[data-testid="stMetricDelta"] { font-family: 'IBM Plex Mono', monospace !important; }
.stDataFrame { border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
[data-testid="stDataFrame"] th {
    background: var(--surface) !important; color: var(--muted) !important;
    font-family: 'IBM Plex Mono', monospace !important; font-size: 0.72rem !important;
    text-transform: uppercase; letter-spacing: 0.06em;
}
[data-testid="stDataFrame"] td { font-family: 'IBM Plex Mono', monospace !important; font-size: 0.82rem !important; color: var(--text) !important; }
.stButton button {
    background: var(--card) !important; border: 1px solid var(--border) !important;
    color: var(--text) !important; font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important; font-weight: 600 !important;
    border-radius: 6px !important; transition: all 0.2s !important; letter-spacing: 0.04em;
}
.stButton button:hover { border-color: var(--accent) !important; color: var(--accent) !important; background: rgba(59,130,246,0.08) !important; }
hr { border-color: var(--border) !important; margin: 1rem 0; }
.stInfo, .stWarning, .stSuccess, .stError { border-radius: 6px !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 0.82rem !important; }
.dcf-card { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 1.2rem 1.4rem; margin-bottom: 1rem; }
.badge { display: inline-block; padding: 0.35rem 1rem; border-radius: 100px; font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem; font-weight: 600; letter-spacing: 0.06em; }
.badge-green { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }
.badge-red   { background: rgba(239,68,68,0.15);  color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }
.badge-amber { background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }
.badge-blue  { background: rgba(59,130,246,0.15); color: #3b82f6; border: 1px solid rgba(59,130,246,0.3); }
.prem-table { width: 100%; border-collapse: collapse; font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; }
.prem-table th { background: #0d1220; color: #64748b; font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.08em; padding: 0.5rem 0.7rem; border-bottom: 1px solid #1e2a40; text-align: right; }
.prem-table th:first-child { text-align: left; }
.prem-table td { padding: 0.45rem 0.7rem; border-bottom: 1px solid #1a2235; text-align: right; color: #e2e8f0; }
.prem-table td:first-child { text-align: left; color: #94a3b8; }
.prem-table tr:hover td { background: rgba(59,130,246,0.04); }
.prem-table .hist-val { color: #94a3b8; }
.prem-table .proj-val { color: #60a5fa; font-weight: 600; }
.prem-table .section-header td { background: #0a0f1c; color: #475569; font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.1em; padding: 0.3rem 0.7rem; border-bottom: 1px solid #1e2a40; }
.ind-table { width: 100%; border-collapse: collapse; font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem; }
.ind-table th { background: #0d1220; color: #64748b; font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.08em; padding: 0.5rem 0.8rem; border-bottom: 1px solid #1e2a40; }
.ind-table td { padding: 0.55rem 0.8rem; border-bottom: 1px solid #141c2e; }
.ind-table .ind-name { color: #94a3b8; }
.ind-table .ind-val { font-weight: 600; font-size: 0.9rem; }
.ind-table .ind-bar-bg { background: #1e2a40; border-radius: 100px; height: 4px; min-width: 80px; }
.ind-table .ind-bar { border-radius: 100px; height: 4px; }
.ind-table .green { color: #10b981; }
.ind-table .amber { color: #f59e0b; }
.ind-table .red   { color: #ef4444; }
.ind-table .muted { color: #64748b; }
.ind-section { font-size: 0.65rem; color: #475569; text-transform: uppercase; letter-spacing: 0.1em; }
.ind-table .section-row td { background: #0a0f1c; padding: 0.25rem 0.8rem; border-bottom: 1px solid #1e2a40; }
.div-table { width: 100%; border-collapse: collapse; font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem; }
.div-table th { background: #0d1220; color: #64748b; font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.08em; padding: 0.5rem 0.8rem; border-bottom: 1px solid #1e2a40; text-align: right; }
.div-table th:first-child { text-align: left; }
.div-table td { padding: 0.5rem 0.8rem; border-bottom: 1px solid #141c2e; text-align: right; }
.div-table td:first-child { text-align: left; color: #94a3b8; }
.div-table .highlight { color: #f59e0b; font-weight: 600; }
.preco-container { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 1.5rem; text-align: center; }
.upside-bar-bg { background: #1e2a40; border-radius: 100px; height: 8px; position: relative; overflow: hidden; }
.upside-bar-fill { height: 8px; border-radius: 100px; transition: width 0.5s ease; }
.section-label { font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 0.5rem; margin-top: 1.2rem; }
.app-header { background: linear-gradient(135deg, #0d1220 0%, #111827 100%); border-bottom: 1px solid #1e2a40; padding: 1rem 0; margin-bottom: 1.5rem; }
.app-title { font-family: 'IBM Plex Mono', monospace; font-size: 1.1rem; font-weight: 600; color: #e2e8f0; letter-spacing: 0.04em; }
.app-subtitle { font-family: 'IBM Plex Sans', sans-serif; font-size: 0.78rem; color: #64748b; margin-top: 0.15rem; }
.sidebar-section { font-family: 'IBM Plex Mono', monospace; font-size: 0.62rem; font-weight: 600; color: #3b82f6; text-transform: uppercase; letter-spacing: 0.12em; padding: 0.4rem 0 0.2rem 0; border-bottom: 1px solid #1e2a40; margin-bottom: 0.3rem; margin-top: 0.8rem; }
[data-baseweb="select"] { background: var(--card) !important; }
[data-baseweb="select"] > div { background: var(--card) !important; border: 1px solid var(--border) !important; color: var(--text) !important; border-radius: 6px !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# AÇÕES
# ─────────────────────────────────────────────────────────────
ACOES_DISPONIVEIS = {
    "WEGE3 — WEG":              "WEGE3.SA",
    "KLBN11 — Klabin":          "KLBN11.SA",
    "LREN3 — Lojas Renner":     "LREN3.SA",
    "MGLU3 — Magazine Luiza":   "MGLU3.SA",
    "VALE3 — Vale":             "VALE3.SA",
    "PETR4 — Petrobras PN":     "PETR4.SA",
    "ITUB4 — Itaú Unibanco":    "ITUB4.SA",
    "BBDC4 — Bradesco PN":      "BBDC4.SA",
    "RENT3 — Localiza":         "RENT3.SA",
    "RADL3 — Raia Drogasil":    "RADL3.SA",
    "EGIE3 — Engie Brasil":     "EGIE3.SA",
    "TAEE11 — Taesa":           "TAEE11.SA",
    "VIVT3 — Telefônica/Vivo":  "VIVT3.SA",
    "SUZB3 — Suzano":           "SUZB3.SA",
    "EMBR3 — Embraer":          "EMBR3.SA",
    "HAPV3 — Hapvida":          "HAPV3.SA",
    "Custom":                   "__custom__",
}

# ─────────────────────────────────────────────────────────────
# DEFAULTS DE EMERGÊNCIA & DELTAS DOS CENÁRIOS
# ─────────────────────────────────────────────────────────────
DEFAULT = {
    "wacc": 0.18, "ir_pct": 0.34, "cresc_perp": 0.05, "ev_ebitda_set": 9.0,
    "cresc_base": 0.07, "cresc_cons": 0.00, "cresc_otim": 0.12,
    "cmv_pct": 0.60, "sga_pct": 0.20, "dep_pct_imob": 0.10, "juros_pct_div": 0.08,
    "prazo_rec": 60, "prazo_est": 45, "prazo_pag": 60,
    "mult_capex": 1.20, "payout": 0.35, "divida_proj": 0.0,
}

DELTA = {
    "Otimista": {
        "cresc": 0.03, "cmv": -0.02, "sga": -0.02,
        "prazo_r": -5, "prazo_e": -5, "prazo_p": 5, "payout": 0.05,
    },
    "Conservador": {
        "cresc": -0.03, "cmv": 0.02, "sga": 0.02,
        "prazo_r": 5, "prazo_e": 5, "prazo_p": -5, "payout": -0.05,
    },
}

# ─────────────────────────────────────────────────────────────
# BUSCA YFINANCE
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def buscar_dados(ticker: str):
    try:
        ativo = yf.Ticker(ticker)
        info  = ativo.info

        preco = info.get("currentPrice") or info.get("regularMarketPrice") or 10.0

        if info.get("sharesOutstanding"):
            acoes = info["sharesOutstanding"] / 1e6
        elif info.get("marketCap"):
            acoes = (info["marketCap"] / preco) / 1e6
        elif info.get("impliedSharesOutstanding"):
            acoes = info["impliedSharesOutstanding"] / 1e6
        else:
            acoes = 793.0

        fin = ativo.financials
        bs  = ativo.balance_sheet
        cf  = ativo.cashflow

        if fin is None or fin.empty:
            raise ValueError("Sem dados financeiros disponíveis")

        n    = min(5, fin.shape[1])
        cols = list(fin.columns[:n])
        anos = [int(c.year) for c in cols]

        def safe(df, keys, col_idx):
            if df is None or df.empty: return None
            for k in keys:
                m = [r for r in df.index if k.lower() in r.lower()]
                if m:
                    try:
                        v = float(df.loc[m[0], df.columns[col_idx]])
                        return v / 1e6 if not np.isnan(v) else None
                    except: pass
            return None

        def col_all(df, keys):
            return [safe(df, keys, i) for i in range(n)]

        def fill_none(lst, fallback=0.0):
            return [v if v is not None else fallback for v in lst]

        receita     = fill_none(col_all(fin, ["Total Revenue"]))
        cpv         = fill_none(col_all(fin, ["Cost Of Revenue", "COGS"]))
        sga         = fill_none(col_all(fin, ["Selling General Administrative", "Operating Expense"]))
        depreciacao = fill_none(col_all(fin, ["Reconciled Depreciation", "Depreciation Amortization"]))
        juros       = fill_none(col_all(fin, ["Interest Expense", "Net Interest Income"]))
        ir          = fill_none(col_all(fin, ["Tax Provision", "Income Tax"]))
        caixa       = fill_none(col_all(bs,  ["Cash And Cash Equivalents", "Cash Cash Equivalents"]))
        rec_rec     = fill_none(col_all(bs,  ["Receivables", "Net Receivables", "Accounts Receivable"]))
        estoques    = fill_none(col_all(bs,  ["Inventory"]))
        imobilizado = fill_none(col_all(bs,  ["Net PPE", "Property Plant Equipment"]))
        fornecedores= fill_none(col_all(bs,  ["Accounts Payable", "Payables"]))
        divida      = fill_none(col_all(bs,  ["Total Debt", "Long Term Debt"]))
        pl          = fill_none(col_all(bs,  ["Stockholders Equity", "Total Equity", "Common Stock Equity"]))
        capex_raw   = col_all(cf, ["Capital Expenditure", "Purchase Of PPE"])

        try:
            div_series = ativo.dividends
            dividendos = []
            if not div_series.empty:
                div_by_year = div_series.groupby(div_series.index.year).sum()
                for ano in anos: 
                    dpa_ano = div_by_year.get(ano, 0.0)
                    dividendos.append(dpa_ano * acoes) 
            else:
                dividendos = [0.0] * n
        except:
            dividendos = [0.0] * n

        # Normaliza sinais
        cpv          = [-abs(v) for v in cpv]
        sga          = [-abs(v) for v in sga]
        depreciacao  = [-abs(v) for v in depreciacao]
        fornecedores = [-abs(v) for v in fornecedores]
        capex        = [-abs(v) if v is not None else 0.0 for v in capex_raw]

        fcff_hist = []
        for i in range(n):
            ebit_i = receita[i] + cpv[i] + sga[i] + depreciacao[i]
            lai_hist = ebit_i + juros[i]
            ir_efetivo = abs(ir[i] / lai_hist) if lai_hist > 0 and ir[i] != 0 else 0.34
            ir_efetivo = min(ir_efetivo, 0.34) 
            nopat  = ebit_i * (1 - ir_efetivo)
            
            var_ncg = 0
            if i > 0:
                ncg_i   = rec_rec[i] + estoques[i] + fornecedores[i]
                ncg_ant = rec_rec[i-1] + estoques[i-1] + fornecedores[i-1]
                var_ncg = -(ncg_i - ncg_ant)
            
            fcff_i = nopat - depreciacao[i] + var_ncg + capex[i]
            fcff_hist.append(fcff_i)

        # Inverte para ordem cronológica
        anos         = list(reversed(anos))
        receita      = list(reversed(receita))
        cpv          = list(reversed(cpv))
        sga          = list(reversed(sga))
        depreciacao  = list(reversed(depreciacao))
        juros        = list(reversed(juros))
        ir           = list(reversed(ir))
        caixa        = list(reversed(caixa))
        rec_rec      = list(reversed(rec_rec))
        estoques     = list(reversed(estoques))
        imobilizado  = list(reversed(imobilizado))
        fornecedores = list(reversed(fornecedores))
        divida       = list(reversed(divida))
        pl           = list(reversed(pl))
        capex        = list(reversed(capex))
        dividendos   = list(reversed(dividendos))
        fcff_hist    = list(reversed(fcff_hist))

        cresc_hist = [None]
        for i in range(1, n):
            r0, r1 = receita[i-1], receita[i]
            cresc_hist.append((r1-r0)/r0 if r0 else None)

        dpa_hist = [d / acoes if acoes > 0 else 0.0 for d in dividendos]
        dy_hist  = [dpa / preco * 100 for dpa in dpa_hist]

        hist = dict(
            receita=receita, cpv=cpv, sga=sga, depreciacao=depreciacao,
            juros=juros, ir=ir, caixa=caixa, rec_receber=rec_rec,
            estoques=estoques, imobilizado=imobilizado,
            fornecedores=fornecedores, divida=divida, pl=pl,
            capex=capex, dividendos=dividendos, dpa_hist=dpa_hist,
            cresc_hist=cresc_hist, fcff_hist=fcff_hist, dy_hist=dy_hist,
        )

        nome  = info.get("longName", ticker)
        setor = info.get("sector", "")
        return hist, preco, acoes, anos, nome, setor, None

    except Exception as e:
        return None, None, None, None, None, None, str(e)


# ─────────────────────────────────────────────────────────────
# DEFAULTS DINÂMICOS (MÉDIAS HISTÓRICAS DE FATO)
# ─────────────────────────────────────────────────────────────
def calc_historico_medio(hist):
    if not hist or not hist.get("receita"):
        return DEFAULT

    def safe_mean(vals):
        v = [x for x in vals if x is not None and np.isfinite(x)]
        return float(np.mean(v)) if v else 0.0

    n = len(hist["receita"])
    cmv_l, sga_l, dep_l, jur_l, rec_l, est_l, pag_l, ir_l, mult_capex_l = [], [], [], [], [], [], [], [], []

    for i in range(n):
        r   = hist["receita"][i]
        c   = abs(hist["cpv"][i]) if hist["cpv"][i] else 0.0
        s   = abs(hist["sga"][i]) if hist["sga"][i] else 0.0
        d   = abs(hist["depreciacao"][i]) if hist["depreciacao"][i] else 0.0
        im  = hist["imobilizado"][i] if hist["imobilizado"][i] else 0.0
        div = hist["divida"][i] if hist["divida"][i] else 0.0
        j   = abs(hist["juros"][i]) if hist["juros"][i] else 0.0
        prec= hist["rec_receber"][i] if hist["rec_receber"][i] else 0.0
        pest= hist["estoques"][i] if hist["estoques"][i] else 0.0
        ppag= abs(hist["fornecedores"][i]) if hist["fornecedores"][i] else 0.0
        cap = abs(hist["capex"][i]) if hist["capex"][i] else 0.0
        
        lai = r - c - s - d - j
        tax = abs(hist["ir"][i]) if hist["ir"][i] else 0.0

        if r > 0:
            cmv_l.append(c / r)
            sga_l.append(s / r)
            rec_l.append((prec / r) * 365.0)
        if im > 0:
            dep_l.append(d / im)
        if div > 0:
            jur_l.append(j / div)
        if c > 0:
            est_l.append((pest / c) * 365.0)
            pag_l.append((ppag / c) * 365.0)
        if lai > 0 and tax > 0:
            ir_l.append(min(tax / lai, 0.34)) 
        if d > 0:
            mult_capex_l.append(cap / d)

    d_out = dict(DEFAULT)
    if cmv_l: d_out["cmv_pct"]        = safe_mean(cmv_l[-3:])
    if sga_l: d_out["sga_pct"]        = safe_mean(sga_l[-3:])
    if dep_l: d_out["dep_pct_imob"]   = safe_mean(dep_l[-3:])
    if jur_l: d_out["juros_pct_div"]  = safe_mean(jur_l[-3:])
    if rec_l: d_out["prazo_rec"]      = int(safe_mean(rec_l[-3:]))
    if est_l: d_out["prazo_est"]      = int(safe_mean(est_l[-3:]))
    if pag_l: d_out["prazo_pag"]      = int(safe_mean(pag_l[-3:]))
    if ir_l:  d_out["ir_pct"]         = safe_mean(ir_l[-3:])
    if mult_capex_l: d_out["mult_capex"] = safe_mean(mult_capex_l[-3:])

    cagr3 = (hist["receita"][-1] / hist["receita"][max(0, n-4)]) ** (1/3) - 1 if n >= 4 and hist["receita"][max(0, n-4)] > 0 else 0.05
    cagr_limpo = min(max(cagr3, -0.1), 0.20) 
    
    d_out["cresc_base"] = cagr_limpo
    d_out["cresc_cons"] = cagr_limpo * 0.5
    d_out["cresc_otim"] = cagr_limpo * 1.5

    return d_out


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
def render_sidebar_header():
    with st.sidebar:
        st.markdown("## 📊 DCF Analyzer v2.1")
        st.markdown("---")

        if "lista_acoes" not in st.session_state:
            st.session_state["lista_acoes"] = dict(ACOES_DISPONIVEIS)

        st.markdown('<div class="sidebar-section">ATIVO</div>', unsafe_allow_html=True)

        acao_nome = st.selectbox(
            "Ação",
            list(st.session_state["lista_acoes"].keys()),
            index=0,
            label_visibility="collapsed",
        )
        ticker = st.session_state["lista_acoes"][acao_nome]

        if ticker == "__custom__":
            novo_ticker = st.text_input("Ticker (ex: KLBN11.SA)").upper()
            if novo_ticker:
                if not novo_ticker.endswith(".SA"):
                    novo_ticker += ".SA"
                ticker = novo_ticker
                if st.button(f"💾 Salvar {ticker} na lista"):
                    nome_exibicao = f"{ticker.replace('.SA','')} — Customizado"
                    del st.session_state["lista_acoes"]["Custom"]
                    st.session_state["lista_acoes"][nome_exibicao] = ticker
                    st.session_state["lista_acoes"]["Custom"] = "__custom__"
                    st.rerun()

        if st.button("🔄 Atualizar Dados", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.markdown("---")
    return ticker


def render_sidebar_params(dyn_def):
    with st.sidebar:
        st.markdown('<div class="sidebar-section">MACRO</div>', unsafe_allow_html=True)
        wacc          = st.number_input("WACC (%)",              value=dyn_def["wacc"]*100,          min_value=5.0,   max_value=40.0,  step=0.5,  format="%.1f") / 100
        cresc_perp    = st.number_input("Cresc. Perpetuidade (%)", value=dyn_def["cresc_perp"]*100, min_value=1.0,   max_value=10.0,  step=0.25, format="%.2f") / 100
        ir_pct        = st.number_input("Alíquota IR (%)",       value=dyn_def["ir_pct"]*100,        min_value=0.0,   max_value=40.0,  step=1.0,  format="%.1f") / 100
        ev_ebitda_set = st.number_input("EV/EBITDA Setor (x)", value=dyn_def["ev_ebitda_set"],       min_value=2.0,   max_value=30.0,  step=0.5,  format="%.1f")

        st.markdown('<div class="sidebar-section">CRESCIMENTO (HISTÓRICO)</div>', unsafe_allow_html=True)
        cresc_base = st.number_input("Crescimento Base (%)",        value=dyn_def["cresc_base"]*100, min_value=-30.0, max_value=80.0, step=0.5, format="%.1f") / 100
        cresc_otim = st.number_input("Crescimento Otimista (%)",    value=dyn_def["cresc_otim"]*100, min_value=-30.0, max_value=100.0, step=0.5, format="%.1f") / 100
        cresc_cons = st.number_input("Crescimento Conservador (%)", value=dyn_def["cresc_cons"]*100, min_value=-50.0, max_value=50.0, step=0.5, format="%.1f") / 100

        st.markdown('<div class="sidebar-section">MARGENS & CUSTOS</div>', unsafe_allow_html=True)
        cmv_pct      = st.number_input("CMV (% Receita)",    value=min(dyn_def["cmv_pct"]*100, 500.0),     min_value=0.0, max_value=500.0, step=0.5, format="%.1f") / 100
        sga_pct      = st.number_input("SG&A (% Receita)",   value=min(dyn_def["sga_pct"]*100, 500.0),     min_value=0.0, max_value=500.0, step=0.5, format="%.1f") / 100
        dep_pct_imob = st.number_input("Deprec. (% Imob.)",  value=min(dyn_def["dep_pct_imob"]*100, 200.0),min_value=0.0, max_value=200.0, step=0.5, format="%.1f") / 100
        juros_pct    = st.number_input("Juros (% Dívida)",   value=min(dyn_def["juros_pct_div"]*100, 100.0),min_value=0.0, max_value=100.0, step=0.25,format="%.2f") / 100

        st.markdown('<div class="sidebar-section">CAPITAL DE GIRO</div>', unsafe_allow_html=True)
        prazo_rec = st.number_input("Prazo Receb. (dias)", value=int(min(dyn_def["prazo_rec"], 1000)), min_value=0, max_value=1000, step=5)
        prazo_est = st.number_input("Prazo Estoque (dias)", value=int(min(dyn_def["prazo_est"], 1000)), min_value=0, max_value=1000, step=5)
        prazo_pag = st.number_input("Prazo Pagam. (dias)", value=int(min(dyn_def["prazo_pag"], 1000)), min_value=0, max_value=1000, step=5)

        st.markdown('<div class="sidebar-section">CAPEX & DIVIDENDOS</div>', unsafe_allow_html=True)
        mult_capex = st.number_input("Mult. Capex (x Deprec.)", value=dyn_def["mult_capex"],   min_value=0.0, max_value=10.0,  step=0.05, format="%.2f")
        payout     = st.number_input("Payout Dividendos (%)",   value=dyn_def["payout"]*100,   min_value=0.0, max_value=100.0, step=5.0,  format="%.1f") / 100

        return dict(
            wacc=wacc, ir_pct=ir_pct, cresc_perp=cresc_perp,
            ev_ebitda_set=ev_ebitda_set,
            cresc_base=cresc_base, cresc_otim=cresc_otim, cresc_cons=cresc_cons,
            cmv_pct=cmv_pct, sga_pct=sga_pct, dep_pct_imob=dep_pct_imob,
            juros_pct_div=juros_pct,
            prazo_rec=prazo_rec, prazo_est=prazo_est, prazo_pag=prazo_pag,
            mult_capex=mult_capex, payout=payout,
            divida_proj=dyn_def["divida_proj"],
        )


# ─────────────────────────────────────────────────────────────
# MOTOR DE CÁLCULO
# ─────────────────────────────────────────────────────────────
def calc_cenario(nome, hist, preco, acoes, p, delta=None):
    d = dict(p)
    if delta:
        d["cresc_base"]  += delta.get("cresc",   0)
        d["cmv_pct"]     += delta.get("cmv",     0)
        d["sga_pct"]     += delta.get("sga",     0)
        d["prazo_rec"]   += delta.get("prazo_r", 0)
        d["prazo_est"]   += delta.get("prazo_e", 0)
        d["prazo_pag"]   += delta.get("prazo_p", 0)
        d["payout"]      += delta.get("payout",  0)

    n = 5
    receita, cpv_l, sga_l, depreciacao_l = [], [], [], []
    ebit_l, juros_l, ir_l, lucro_l       = [], [], [], []
    imob_l, rec_rec_l, estoq_l, forn_l   = [], [], [], []
    fcf_l, div_l, caixa_l                = [], [], []
    fcff_l, fcfe_l, divida_l             = [], [], []

    div_t = d["divida_proj"] 

    for i in range(n):
        rec_ant = hist["receita"][-1] if i == 0 else receita[-1]
        rec     = rec_ant * (1 + d["cresc_base"])
        receita.append(rec)

        cpv_l.append(-rec * d["cmv_pct"])
        sga_l.append(-rec * d["sga_pct"])

        imob_ant = hist["imobilizado"][-1] if i == 0 else imob_l[-1]
        dep      = -imob_ant * d["dep_pct_imob"]           
        capex_i  = -abs(d["mult_capex"] * dep)             
        depreciacao_l.append(dep)

        imob_l.append(imob_ant + abs(capex_i) - abs(dep))

        ebit_i = rec + cpv_l[-1] + sga_l[-1] + dep
        ebit_l.append(ebit_i)

        jur = -div_t * d["juros_pct_div"]
        juros_l.append(jur)
        divida_l.append(div_t)
        div_t = max(div_t * 0.95, 0.0)

        lai  = ebit_i + jur
        ir_i = -lai * d["ir_pct"] if lai > 0 else 0.0
        ir_l.append(ir_i)
        ll = lai + ir_i
        lucro_l.append(ll)

        rec_rec_l.append(rec            * d["prazo_rec"] / 365)
        estoq_l.append(abs(cpv_l[-1])  * d["prazo_est"] / 365)
        forn_l.append(-abs(cpv_l[-1])  * d["prazo_pag"] / 365)

        if i == 0:
            ncg_ant = (hist["rec_receber"][-1] + hist["estoques"][-1] + hist["fornecedores"][-1])
        else:
            ncg_ant = rec_rec_l[-2] + estoq_l[-2] + forn_l[-2]
        ncg_i   = rec_rec_l[-1] + estoq_l[-1] + forn_l[-1]
        var_cg  = -(ncg_i - ncg_ant)

        nopat_i = ebit_i * (1 - d["ir_pct"])
        fcff_i  = nopat_i - dep + var_cg + capex_i
        fcff_l.append(fcff_i)

        amort_div = divida_l[-1] * 0.05
        fcfe_i    = fcff_i + jur * (1 - d["ir_pct"]) - amort_div
        fcfe_l.append(fcfe_i)

        fco   = ll - dep + var_cg   
        fcf_i = fco + capex_i
        fcf_l.append(fcf_i)

        div_pagos = max(ll * d["payout"], 0.0)
        div_l.append(div_pagos)

        caixa_ant = hist["caixa"][-1] if i == 0 else caixa_l[-1]
        caixa_l.append(caixa_ant + fcf_i - div_pagos)

    wacc = d["wacc"]
    cp   = d["cresc_perp"]

    vp_fcff = sum(fcff_l[i] / (1 + wacc) ** (i + 1) for i in range(n))

    nopat_tv = ebit_l[-1] * (1 - d["ir_pct"])
    div_liq_tv = div_t - caixa_l[-1]                                 
    capital_employed_tv = max(imob_l[-1] + rec_rec_l[-1] + estoq_l[-1] + forn_l[-1], 1.0)
    roic_tv      = nopat_tv / capital_employed_tv
    reinv_rate_tv = min(max(cp / max(roic_tv, 0.0001), 0.0), 0.95)  
    fcff_perp    = nopat_tv * (1 - reinv_rate_tv)

    tv    = fcff_perp * (1 + cp) / (wacc - cp) if wacc > cp else 0.0
    vp_tv = tv / (1 + wacc) ** n

    div_liq = d["divida_proj"] - hist["caixa"][-1]   

    ve     = vp_fcff + vp_tv
    equity = ve - div_liq
    pj     = equity / acoes if acoes > 0 else 0.0
    upside = (pj - preco) / preco * 100 if preco else 0.0

    ebitda_fwd = ebit_l[0] - depreciacao_l[0]   

    mc    = preco * acoes
    ev    = mc + div_liq
    ev_eb = ev / ebitda_fwd if ebitda_fwd > 0 else None

    dpa = [v / acoes for v in div_l]
    dy  = [v / preco * 100 for v in dpa]

    return {
        "nome": nome,
        "receita": receita, "cpv": cpv_l, "sga": sga_l,
        "depreciacao": depreciacao_l, "ebit": ebit_l,
        "juros": juros_l, "ir": ir_l, "lucro": lucro_l,
        "fcf": fcf_l, "fcff": fcff_l, "fcfe": fcfe_l,
        "imob": imob_l, "divida": divida_l,
        "vp_fcff": vp_fcff, "vp_tv": vp_tv, "ve": ve,
        "equity": equity, "pj": pj, "upside": upside,
        "ebitda_fwd": ebitda_fwd, "ev_eb": ev_eb, "mc": mc,
        "ev": ev, "div_liq": div_liq,
        "dividendos": div_l, "dpa": dpa, "dy": dy,
        "roic_tv": roic_tv, "reinv_rate_tv": reinv_rate_tv,
    }


def calc_todos(hist, preco, acoes, p):
    return {
        "Base":        calc_cenario("Base",        hist, preco, acoes, p),
        "Otimista":    calc_cenario("Otimista",    hist, preco, acoes, p, DELTA["Otimista"]),
        "Conservador": calc_cenario("Conservador", hist, preco, acoes, p, DELTA["Conservador"]),
    }


def veredicto_valuation(res_todos, preco):
    pjs     = [r["pj"] for r in res_todos.values()]
    mediana = float(np.median(pjs))
    upside  = (mediana - preco) / preco * 100

    if upside > 30:    return "BARATA",             upside, "green"
    elif upside > 10:  return "LEVEMENTE BARATA",  upside, "amber"
    elif upside > -10: return "PREÇO JUSTO",        upside, "blue"
    elif upside > -25: return "LEVEMENTE CARA",    upside, "amber"
    else:              return "CARA",              upside, "red"


def calc_sens(hist, preco, acoes, p):
    waccs  = [0.10, 0.13, 0.15, 0.18, 0.20, 0.22, 0.25]
    crescs = [0.00, 0.03, 0.05, 0.07, 0.10, 0.13, 0.15]
    mat = np.zeros((len(waccs), len(crescs)))
    for i, w in enumerate(waccs):
        for j, c in enumerate(crescs):
            pp = dict(p)
            pp["wacc"] = w
            pp["cresc_base"] = c
            r = calc_cenario("S", hist, preco, acoes, pp)
            mat[i, j] = r["pj"]
    return pd.DataFrame(
        mat,
        index=[f"{w*100:.0f}%" for w in waccs],
        columns=[f"{c*100:.0f}%" for c in crescs],
    )


def calc_indicadores(hist, preco, acoes, res_base, p):
    rec_hist   = hist["receita"][-1]
    ebit_h     = hist["receita"][-1] + hist["cpv"][-1] + hist["sga"][-1] + hist["depreciacao"][-1]
    ebitda_h   = ebit_h - hist["depreciacao"][-1] 
    ll_h       = ebit_h + hist["juros"][-1] + hist["ir"][-1]
    pl_h       = hist["pl"][-1]
    div_liq_h  = hist["divida"][-1] - hist["caixa"][-1]
    ev_h       = (preco * acoes) + div_liq_h
    mc_h       = preco * acoes

    mg_bruta_h = (rec_hist + hist["cpv"][-1]) / rec_hist * 100 if rec_hist else 0.0
    mg_ebitda_h= ebitda_h / rec_hist * 100 if rec_hist else 0.0
    mg_ebit_h  = ebit_h / rec_hist * 100 if rec_hist else 0.0
    mg_liq_h   = ll_h / rec_hist * 100 if rec_hist else 0.0

    lpa_h      = ll_h / acoes if acoes else 0
    pl_rat_h   = preco / lpa_h if lpa_h > 0 else None
    pvpa_h     = preco / (pl_h / acoes) if pl_h > 0 and acoes > 0 else None
    ev_eb_h    = ev_h / ebitda_h if ebitda_h > 0 else None
    ev_rec_h   = ev_h / rec_hist if rec_hist > 0 else None
    psr_h      = mc_h / rec_hist if rec_hist > 0 else None

    div_pl_h   = hist["divida"][-1] / pl_h if pl_h > 0 else None
    div_eb_h   = div_liq_h / ebitda_h if ebitda_h > 0 else None
    cob_jur_h  = ebit_h / abs(hist["juros"][-1]) if hist["juros"][-1] != 0 else None

    roe_h      = ll_h / pl_h * 100 if pl_h > 0 else None
    ir_rate_h  = abs(hist["ir"][-1] / (ebit_h + hist["juros"][-1])) if (ebit_h + hist["juros"][-1]) > 0 else 0.34
    roic_h     = ebit_h * (1 - ir_rate_h) / (pl_h + div_liq_h) * 100 if (pl_h + div_liq_h) > 0 else None

    n_h = len(hist["receita"])
    cagr3 = (hist["receita"][-1] / hist["receita"][max(0, n_h-4)]) ** (1/3) - 1 if n_h >= 4 and hist["receita"][max(0, n_h-4)] > 0 else None
    lucros_hist = [hist["receita"][i] + hist["cpv"][i] + hist["sga"][i] + hist["depreciacao"][i] + hist["juros"][i] + hist["ir"][i] for i in range(n_h)]
    cagr_ll = (lucros_hist[-1] / lucros_hist[max(0, n_h-4)]) ** (1/3) - 1 if n_h >= 4 and lucros_hist[max(0,n_h-4)] > 0 else None

    ind_hist = {
        "Margens (Ano Anterior)": {
            "Margem Bruta":     (mg_bruta_h,  "%", 30,  50,  "maior melhor"),
            "Margem EBITDA":    (mg_ebitda_h, "%", 10,  25,  "maior melhor"),
            "Margem EBIT":      (mg_ebit_h,   "%", 5,   15,  "maior melhor"),
            "Margem Líquida":   (mg_liq_h,    "%", 3,   10,  "maior melhor"),
        },
        "Múltiplos": {
            "P/L (Trailing)":   (pl_rat_h,    "x", 8,   20,  "menor melhor"),
            "P/VPA":            (pvpa_h,      "x", 1,   3,   "menor melhor"),
            "EV/EBITDA":        (ev_eb_h,     "x", 4,   12,  "menor melhor"),
            "EV/Receita":       (ev_rec_h,    "x", 0.3, 2,   "menor melhor"),
            "P/SR":             (psr_h,       "x", 0.3, 2,   "menor melhor"),
        },
        "Solidez": {
            "Dívida/PL":        (div_pl_h,    "x", 0,   1.5, "menor melhor"),
            "Dívida Liq/EBITDA":(div_eb_h,    "x", 0,   2.5, "menor melhor"),
            "Cobertura Juros":  (cob_jur_h,   "x", 3,   8,   "maior melhor"),
        },
        "Rentabilidade & Crescimento": {
            "ROE":              (roe_h,       "%", 10,  25,  "maior melhor"),
            "ROIC":             (roic_h,      "%", 8,   20,  "maior melhor"),
            "CAGR Receita 3a":  (cagr3*100 if cagr3 else None,  "%", 5, 20, "maior melhor"),
            "CAGR Lucro 3a":    (cagr_ll*100 if cagr_ll else None, "%", 5, 20, "maior melhor"),
        },
    }

    rec_p    = res_base["receita"][0]
    ebit_p   = res_base["ebit"][0]
    ebitda_p = res_base["ebitda_fwd"]
    ll_p     = res_base["lucro"][0]

    mg_bruta_p = (rec_p + res_base["cpv"][0]) / rec_p * 100 if rec_p else 0
    mg_ebitda_p= ebitda_p / rec_p * 100 if rec_p else 0
    mg_ebit_p  = ebit_p / rec_p * 100 if rec_p else 0
    mg_liq_p   = ll_p / rec_p * 100 if rec_p else 0

    lpa_p      = ll_p / acoes if acoes else 0
    pl_rat_p   = preco / lpa_p if lpa_p > 0 else None
    ev_eb_p    = ev_h / ebitda_p if ebitda_p > 0 else None
    ev_rec_p   = ev_h / rec_p if rec_p > 0 else None
    psr_p      = mc_h / rec_p if rec_p > 0 else None
    
    cob_jur_p  = ebit_p / abs(res_base["juros"][0]) if res_base["juros"][0] != 0 else None
    roe_p      = ll_p / pl_h * 100 if pl_h > 0 else None
    roic_p     = ebit_p * (1 - p["ir_pct"]) / (pl_h + div_liq_h) * 100 if (pl_h + div_liq_h) > 0 else None

    ind_proj = {
        "Margens Projetadas (Ano 1)": {
            "Margem Bruta":     (mg_bruta_p,  "%", 30,  50,  "maior melhor"),
            "Margem EBITDA":    (mg_ebitda_p, "%", 10,  25,  "maior melhor"),
            "Margem EBIT":      (mg_ebit_p,   "%", 5,   15,  "maior melhor"),
            "Margem Líquida":   (mg_liq_p,    "%", 3,   10,  "maior melhor"),
        },
        "Múltiplos Forward (Ano 1)": {
            "P/L (Forward)":    (pl_rat_p,    "x", 8,   20,  "menor melhor"),
            "EV/EBITDA (Fwd)":  (ev_eb_p,     "x", 4,   12,  "menor melhor"),
            "EV/Receita (Fwd)": (ev_rec_p,    "x", 0.3, 2,   "menor melhor"),
            "P/SR (Fwd)":       (psr_p,       "x", 0.3, 2,   "menor melhor"),
        },
        "Rentabilidade & Solidez Projetada": {
            "ROE Projetado":    (roe_p,       "%", 10,  25,  "maior melhor"),
            "ROIC Projetado":   (roic_p,      "%", 8,   20,  "maior melhor"),
            "Cobertura Juros":  (cob_jur_p,   "x", 3,   8,   "maior melhor"),
        }
    }

    return ind_hist, ind_proj


# ─────────────────────────────────────────────────────────────
# HELPERS FORMATAÇÃO
# ─────────────────────────────────────────────────────────────
def fmt_rs(v):
    if v is None: return "—"
    return f"R$ {v:,.0f}M"

def fmt_num(v, dec=1, suffix=""):
    if v is None: return "—"
    return f"{v:.{dec}f}{suffix}"

def cor_ind(val, bom, otimo, direcao):
    if val is None: return "muted"
    if direcao == "maior melhor":
        return "green" if val >= otimo else ("amber" if val >= bom else "red")
    else:
        return "green" if val <= bom else ("amber" if val <= otimo else "red")

def frac_ind(val, bom, otimo, direcao):
    if val is None: return 0.0
    try:
        if direcao == "maior melhor":
            return min(max((val - bom) / (otimo - bom + 0.001), 0.0), 1.0)
        else:
            return min(max((otimo - val) / (otimo - bom + 0.001), 0.0), 1.0)
    except:
        return 0.5

def cor_hex(nome):
    return {"green": "#10b981", "amber": "#f59e0b", "red": "#ef4444",
            "muted": "#64748b", "blue": "#3b82f6"}.get(nome, "#64748b")


# ─────────────────────────────────────────────────────────────
# GRÁFICOS PLOTLY
# ─────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    plot_bgcolor="#0d1220",
    paper_bgcolor="#080c14",
    font=dict(family="monospace", color="#94a3b8", size=11),
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(orientation="h", x=0, y=-0.18, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
)
PLOTLY_AXIS = dict(gridcolor="#1e2a40", tickfont=dict(size=9), zeroline=False)

CORES_CENARIO = {
    "Base":        "#f59e0b",
    "Otimista":    "#10b981",
    "Conservador": "#3b82f6",
    "Histórico":   "#475569",
}

def chart_projecao(hist, anos_hist, anos_proj, res):
    n_h = len(anos_hist)
    ebit_h = [hist["receita"][i] + hist["cpv"][i] + hist["sga"][i] + hist["depreciacao"][i]
              for i in range(n_h)]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Receita (R$M)", "EBIT (R$M)", "FCFF (R$M)"),
        horizontal_spacing=0.06,
    )

    for col_idx, (metric_hist, metric_key) in enumerate(
        [(hist["receita"], "receita"), (ebit_h, "ebit"), (hist.get("fcff_hist", [0]*n_h), "fcff")], 1
    ):
        fig.add_trace(go.Bar(
            x=[str(a) for a in anos_hist], y=metric_hist,
            name="Histórico", marker_color=CORES_CENARIO["Histórico"],
            showlegend=(col_idx == 1), legendgroup="hist",
        ), row=1, col=col_idx)

        for cen_nome, r in res.items():
            vals = r[metric_key] if metric_key != "ebit" else r["ebit"]
            fig.add_trace(go.Bar(
                x=[str(a) for a in anos_proj], y=vals,
                name=cen_nome, marker_color=CORES_CENARIO[cen_nome], opacity=0.85,
                showlegend=(col_idx == 1), legendgroup=cen_nome,
            ), row=1, col=col_idx)

    fig.update_layout(**PLOTLY_LAYOUT, height=340, barmode="group")
    for i in range(1, 4):
        fig.update_xaxes(PLOTLY_AXIS, row=1, col=i)
        fig.update_yaxes(PLOTLY_AXIS, row=1, col=i)
    return fig

def chart_waterfall_dcf(rb, acoes):
    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["relative", "relative", "relative", "total"],
        x=["VP FCFFs", "Valor Terminal", "(-) Dívida Líquida", "Equity"],
        y=[rb["vp_fcff"], rb["vp_tv"], -rb["div_liq"], 0],
        text=[f"R$ {rb['vp_fcff']:,.0f}M", f"R$ {rb['vp_tv']:,.0f}M",
              f"(R$ {abs(rb['div_liq']):,.0f}M)", f"R$ {rb['equity']:,.0f}M"],
        textposition="outside",
        increasing_marker_color="#10b981",
        decreasing_marker_color="#ef4444",
        totals_marker_color="#3b82f6",
        connector=dict(line=dict(color="#1e2a40", width=1)),
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=320, showlegend=False,
                      yaxis=dict(**PLOTLY_AXIS, title="R$M"),
                      xaxis=dict(**PLOTLY_AXIS))
    return fig

def chart_heatmap(sens, preco):
    z    = sens.values
    text = [[f"R$ {v:.2f}" for v in row] for row in z]

    zmin = z.min()
    zmax = z.max()

    colorscale = [
        [0.0,  "#450a0a"],
        [0.25, "#7c2d12"],
        [0.45, "#1e3a5f"],
        [0.55, "#1e4a3f"],
        [0.75, "#065f46"],
        [1.0,  "#052e16"],
    ]

    fig = go.Figure(go.Heatmap(
        z=z, x=list(sens.columns), y=list(sens.index),
        text=text, texttemplate="%{text}",
        textfont=dict(family="monospace", size=11),
        colorscale=colorscale, zmin=zmin, zmax=zmax,
        colorbar=dict(
            title=dict(text="Preço Justo (R$)", font=dict(family="monospace", size=10, color="#94a3b8")),
            tickfont=dict(family="monospace", size=9, color="#94a3b8"),
            bgcolor="#080c14", bordercolor="#1e2a40", borderwidth=1,
        ),
        hovertemplate="WACC: %{y}<br>g: %{x}<br>Preço Justo: %{text}<extra></extra>",
    ))

    fig.add_shape(
        type="line", x0=-0.5, x1=len(sens.columns)-0.5,
        y0=-0.5, y1=-0.5, line=dict(color="#f59e0b", width=1, dash="dot"),
    )

    fig.update_layout(
        **PLOTLY_LAYOUT, height=380,
        xaxis=dict(**PLOTLY_AXIS, title="Crescimento Base (g)"),
        yaxis=dict(**PLOTLY_AXIS, title="WACC", autorange="reversed"),
        annotations=[dict(
            text=f"Preço atual: R$ {preco:.2f}",
            x=0, y=1.04, xref="paper", yref="paper",
            font=dict(family="monospace", size=10, color="#f59e0b"),
            showarrow=False,
        )],
    )
    return fig

def chart_historico(hist, anos_hist):
    n_h = len(anos_hist)
    ebit_h = [hist["receita"][i] + hist["cpv"][i] + hist["sga"][i] + hist["depreciacao"][i]
              for i in range(n_h)]
    ll_h   = [ebit_h[i] + hist["juros"][i] + hist["ir"][i] for i in range(n_h)]
    anos_s = [str(a) for a in anos_hist]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=anos_s, y=hist["receita"], name="Receita",   mode="lines+markers", line=dict(color="#3b82f6", width=2)))
    fig.add_trace(go.Scatter(x=anos_s, y=ebit_h,          name="EBIT",      mode="lines+markers", line=dict(color="#f59e0b", width=2)))
    fig.add_trace(go.Scatter(x=anos_s, y=ll_h,            name="Lucro Líq", mode="lines+markers", line=dict(color="#10b981", width=2)))
    fig.add_trace(go.Scatter(x=anos_s, y=hist.get("fcff_hist", [0]*n_h), name="FCFF", mode="lines+markers", line=dict(color="#8b5cf6", width=2)))

    fig.update_layout(**PLOTLY_LAYOUT, height=300, yaxis=dict(**PLOTLY_AXIS, title="R$M"), xaxis=dict(**PLOTLY_AXIS))
    return fig


# ─────────────────────────────────────────────────────────────
# ABA 0 — DCF / CENÁRIOS
# ─────────────────────────────────────────────────────────────
def tab_dcf(hist, preco, acoes, anos_hist, anos_proj, res, p):
    rb = res["Base"]
    ro = res["Otimista"]
    rc = res["Conservador"]

    verd_label, verd_upside, verd_cor = veredicto_valuation(res, preco)
    badge_class = f"badge-{verd_cor}"

    col1, col2, col3, col4, col5 = st.columns(5)
    for col, (n_cen, r, emoji) in zip(
        [col1, col2, col3],
        [("Conservador", rc, "🔵"), ("Base", rb, "🟡"), ("Otimista", ro, "🟢")]
    ):
        col.metric(
            f"{emoji} {n_cen}", f"R$ {r['pj']:.2f}", f"{r['upside']:+.1f}%",
            delta_color="normal" if r["upside"] >= 0 else "inverse",
        )

    with col4:
        ev_eb_set  = p.get("ev_ebitda_set", 9.0)
        preco_ev   = (ev_eb_set * rb["ebitda_fwd"] - rb["div_liq"]) / acoes
        ups_ev     = (preco_ev - preco) / preco * 100
        col4.metric("📊 EV/EBITDA Setor", f"R$ {preco_ev:.2f}", f"{ups_ev:+.1f}%",
                    delta_color="normal" if ups_ev >= 0 else "inverse")

    with col5:
        st.markdown(f"""
        <div style="background:#111827;border:1px solid #1e2a40;border-radius:8px;padding:1rem;text-align:center;">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.62rem;color:#64748b;text-transform:uppercase;letter-spacing:0.1em;">PREÇO ATUAL</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:1.6rem;font-weight:600;color:#e2e8f0;margin:0.2rem 0;">R$ {preco:.2f}</div>
            <span class="badge {badge_class}">{verd_label}</span>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.72rem;color:#94a3b8;margin-top:0.4rem;">{verd_upside:+.1f}% vs mediana</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="section-label">📈 Projeção — Histórico & Cenários</p>', unsafe_allow_html=True)
    st.plotly_chart(chart_projecao(hist, anos_hist, anos_proj, res), use_container_width=True)

    st.markdown('<p class="section-label">🧮 Decomposição do Valor — Cenário Base</p>', unsafe_allow_html=True)
    col_wf, col_tv = st.columns([2, 1])
    with col_wf:
        st.plotly_chart(chart_waterfall_dcf(rb, acoes), use_container_width=True)
    with col_tv:
        st.markdown(f"""
        <div class="dcf-card" style="border-left: 3px solid #8b5cf6; margin-top: 0.5rem;">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.62rem;color:#64748b;text-transform:uppercase;margin-bottom:0.5rem;">Terminal Value</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.78rem;color:#94a3b8;line-height:1.6;">
                ROIC Terminal: <span style="color:#e2e8f0;">{rb['roic_tv']*100:.1f}%</span><br>
                Taxa Reinvestimento: <span style="color:#e2e8f0;">{rb['reinv_rate_tv']*100:.1f}%</span><br>
                FCFF Perpetuidade: <span style="color:#60a5fa;">R$ {rb['vp_fcff']:.0f}M VP</span><br>
                TV (VP): <span style="color:#8b5cf6;">R$ {rb['vp_tv']:,.0f}M</span><br>
                TV / EV: <span style="color:#f59e0b;">{rb['vp_tv']/rb['ve']*100:.1f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="section-label">📋 Tabela de Premissas — Histórico & Projeção</p>', unsafe_allow_html=True)

    anos_labels_h = [str(a) for a in anos_hist]
    anos_labels_p = [str(a) for a in anos_proj]
    n_h = len(anos_hist)
    n_p = len(anos_proj)

    def pct(v): return "—" if v is None else f"{v*100:.0f}%"
    def dias(v): return "—" if v is None else f"{v:.0f}"
    def rsm(v): return "—" if v is None else (f"({abs(v):,.0f})" if v < 0 else f"{v:,.0f}")

    cresc_h = hist["cresc_hist"]
    cmv_h   = [abs(hist["cpv"][i])/hist["receita"][i]   if hist["receita"][i] else None for i in range(n_h)]
    sga_h   = [abs(hist["sga"][i])/hist["receita"][i]   if hist["receita"][i] else None for i in range(n_h)]
    dep_h   = [abs(hist["depreciacao"][i])/hist["imobilizado"][i] if hist["imobilizado"][i] else None for i in range(n_h)]
    jur_h   = [abs(hist["juros"][i])/hist["divida"][i]  if hist["divida"][i] else None for i in range(n_h)]
    prec_h  = [hist["rec_receber"][i]/hist["receita"][i]*365 if hist["receita"][i] else None for i in range(n_h)]
    pest_h  = [hist["estoques"][i]/abs(hist["cpv"][i])*365   if hist["cpv"][i]  else None for i in range(n_h)]
    ppag_h  = [abs(hist["fornecedores"][i])/abs(hist["cpv"][i])*365 if hist["cpv"][i] else None for i in range(n_h)]

    header = "".join(f"<th>{a}</th>" for a in anos_labels_h) + \
             "".join(f"<th style='color:#60a5fa;'>{a}</th>" for a in anos_labels_p)

    def row(label, hist_vals, proj_vals, fmt_fn, is_section=False):
        if is_section:
            n_cols = n_h + n_p + 1
            return f'<tr class="section-header"><td colspan="{n_cols}">{label}</td></tr>'
        hist_tds = "".join(f'<td class="hist-val">{fmt_fn(v)}</td>' for v in hist_vals)
        proj_tds = "".join(f'<td class="proj-val">{fmt_fn(v)}</td>' for v in [p[k] for k in (proj_vals if isinstance(proj_vals, list) else [])])
        if not isinstance(proj_vals, list):
            proj_tds = "".join(f'<td class="proj-val">{fmt_fn(proj_vals)}</td>' * n_p)
        return f"<tr><td>{label}</td>{hist_tds}{proj_tds}</tr>"

    t = f"""
    <table class="prem-table">
    <thead><tr><th>Premissa</th>{header}</tr></thead>
    <tbody>
    {row("CRESCIMENTO & RECEITA", [], [], None, True)}
    <tr><td>Crescimento Receita</td>{"".join(f'<td class="hist-val">{pct(v)}</td>' for v in cresc_h)}{"".join(f'<td class="proj-val">{pct(p["cresc_base"])}</td>' for _ in range(n_p))}</tr>
    {row("CUSTOS", [], [], None, True)}
    <tr><td>CMV (% Receita)</td>{"".join(f'<td class="hist-val">{pct(v)}</td>' for v in cmv_h)}{"".join(f'<td class="proj-val">{pct(p["cmv_pct"])}</td>' for _ in range(n_p))}</tr>
    <tr><td>SG&A (% Receita)</td>{"".join(f'<td class="hist-val">{pct(v)}</td>' for v in sga_h)}{"".join(f'<td class="proj-val">{pct(p["sga_pct"])}</td>' for _ in range(n_p))}</tr>
    <tr><td>Deprec. (% Imob.)</td>{"".join(f'<td class="hist-val">{pct(v)}</td>' for v in dep_h)}{"".join(f'<td class="proj-val">{pct(p["dep_pct_imob"])}</td>' for _ in range(n_p))}</tr>
    <tr><td>Juros (% Dívida)</td>{"".join(f'<td class="hist-val">{pct(v)}</td>' for v in jur_h)}{"".join(f'<td class="proj-val">{pct(p["juros_pct_div"])}</td>' for _ in range(n_p))}</tr>
    <tr><td>IR Alíquota</td>{"".join(f'<td class="hist-val">—</td>' for _ in range(n_h))}{"".join(f'<td class="proj-val">{pct(p["ir_pct"])}</td>' for _ in range(n_p))}</tr>
    {row("CAPITAL DE GIRO", [], [], None, True)}
    <tr><td>Prazo Receb. (dias)</td>{"".join(f'<td class="hist-val">{dias(v)}</td>' for v in prec_h)}{"".join(f'<td class="proj-val">{dias(p["prazo_rec"])}</td>' for _ in range(n_p))}</tr>
    <tr><td>Prazo Estoque (dias)</td>{"".join(f'<td class="hist-val">{dias(v)}</td>' for v in pest_h)}{"".join(f'<td class="proj-val">{dias(p["prazo_est"])}</td>' for _ in range(n_p))}</tr>
    <tr><td>Prazo Pagam. (dias)</td>{"".join(f'<td class="hist-val">{dias(v)}</td>' for v in ppag_h)}{"".join(f'<td class="proj-val">{dias(p["prazo_pag"])}</td>' for _ in range(n_p))}</tr>
    {row("DÍVIDA PROJETADA", [], [], None, True)}
    <tr><td>Saldo Dívida (R$M)</td>{"".join(f'<td class="hist-val">{rsm(hist["divida"][i])}</td>' for i in range(n_h))}{"".join(f'<td class="proj-val">{rsm(v)}</td>' for v in rb["divida"])}</tr>
    </tbody>
    </table>
    """
    st.markdown(t, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="section-label">📊 DRE Projetada — Cenários</p>', unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns(3)
    for col, (n_cen, r) in zip([col_a, col_b, col_c], res.items()):
        with col:
            st.markdown(f"**{n_cen}**")
            df_cen = pd.DataFrame({
                "Ano":       [str(a) for a in anos_proj],
                "Receita":   [f"{v:,.0f}" for v in r["receita"]],
                "EBIT":      [f"{v:,.0f}" for v in r["ebit"]],
                "Lucro Liq": [f"{v:,.0f}" for v in r["lucro"]],
                "FCFF":      [f"{v:,.0f}" for v in r["fcff"]],
                "FCFE":      [f"{v:,.0f}" for v in r["fcfe"]],
            }).set_index("Ano")
            st.dataframe(df_cen, use_container_width=True)

    st.markdown("---")
    st.markdown('<p class="section-label">🔄 Crescimento Implícito (DCF Reverse)</p>', unsafe_allow_html=True)
    try:
        def diff(c):
            pp = dict(p)
            pp["cresc_base"] = c
            return calc_cenario("R", hist, preco, acoes, pp)["pj"] - preco
        cresc_impl = brentq(diff, -0.30, 1.50)
        cor_impl = "#10b981" if cresc_impl < p["cresc_base"] else "#f59e0b"
        st.markdown(f"""
        <div class="dcf-card" style="border-left: 3px solid {cor_impl};">
            <span style="font-family:'IBM Plex Mono',monospace;color:#94a3b8;font-size:0.78rem;">
            Para justificar R$ {preco:.2f}, a empresa precisa crescer
            <strong style="color:{cor_impl};font-size:1.1rem;"> {cresc_impl*100:.1f}% a.a.</strong>
            — você projeta <strong style="color:#60a5fa;">{p['cresc_base']*100:.1f}%</strong>
            </span>
        </div>
        """, unsafe_allow_html=True)
    except:
        st.info("Não foi possível calcular o crescimento implícito com os parâmetros atuais.")


# ─────────────────────────────────────────────────────────────
# ABA 1 — DIVIDENDOS
# ─────────────────────────────────────────────────────────────
def tab_dividendos(hist, preco, acoes, anos_hist, anos_proj, res):
    st.markdown('<p class="section-label">🏦 Dividendos por Cota — Histórico</p>', unsafe_allow_html=True)
    n_h = len(anos_hist)
    dpa_hist = hist.get("dpa_hist", [0.0]*n_h)
    dy_hist  = hist.get("dy_hist",  [0.0]*n_h)
    div_tot  = hist.get("dividendos", [0.0]*n_h)

    table_html = "<table class='div-table'><thead><tr><th style='text-align:left'>Ano</th><th>Dividendos Totais</th><th>DPA (R$)</th><th>Yield (%)</th></tr></thead><tbody>"
    for i, ano in enumerate(anos_hist):
        table_html += f"<tr><td>{ano}</td><td>{fmt_rs(div_tot[i])}</td><td class='highlight'>{fmt_num(dpa_hist[i],2)}</td><td>{fmt_num(dy_hist[i],2,'%')}</td></tr>"
    table_html += "</tbody></table>"
    st.markdown(table_html, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="section-label">📬 Dividendos por Cota — Projeção</p>', unsafe_allow_html=True)
    proj_html = """<table class="div-table"><thead><tr>
        <th style="text-align:left">Ano</th>
        <th>DPA Cons. (R$)</th><th>Yield Cons. (%)</th>
        <th>DPA Base (R$)</th><th>Yield Base (%)</th>
        <th>DPA Otim. (R$)</th><th>Yield Otim. (%)</th>
    </tr></thead><tbody>"""
    for i, ano in enumerate(anos_proj):
        proj_html += f"""<tr>
            <td>{ano}</td>
            <td style="color:#3b82f6;">{fmt_num(res['Conservador']['dpa'][i],2)}</td>
            <td style="color:#3b82f6;">{fmt_num(res['Conservador']['dy'][i],2,'%')}</td>
            <td style="color:#f59e0b;" class="highlight">{fmt_num(res['Base']['dpa'][i],2)}</td>
            <td style="color:#f59e0b;">{fmt_num(res['Base']['dy'][i],2,'%')}</td>
            <td style="color:#10b981;">{fmt_num(res['Otimista']['dpa'][i],2)}</td>
            <td style="color:#10b981;">{fmt_num(res['Otimista']['dy'][i],2,'%')}</td>
        </tr>"""
    proj_html += "</tbody></table>"
    st.markdown(proj_html, unsafe_allow_html=True)

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    for col, (n_cen, r) in zip([col1, col2, col3], res.items()):
        with col:
            st.metric(f"DPA Acum. 5a — {n_cen}", f"R$ {sum(r['dpa']):.2f}/cota",
                      f"Yield médio: {sum(r['dy'])/5:.1f}%/a")

    st.markdown("---")
    st.markdown('<p class="section-label">📊 Payout — Projeção Base vs Otimista</p>', unsafe_allow_html=True)
    pw_html = """<table class="div-table"><thead><tr>
        <th style="text-align:left">Ano</th>
        <th>Lucro Base</th><th>Div. Base</th><th>Payout Base</th>
        <th>Lucro Otim.</th><th>Div. Otim.</th><th>Payout Otim.</th>
    </tr></thead><tbody>"""
    for i, ano in enumerate(anos_proj):
        rb = res["Base"]; ro = res["Otimista"]
        po_b = rb["dividendos"][i]/rb["lucro"][i]*100 if rb["lucro"][i] > 0 else 0
        po_o = ro["dividendos"][i]/ro["lucro"][i]*100 if ro["lucro"][i] > 0 else 0
        pw_html += f"""<tr>
            <td>{ano}</td>
            <td>{rb['lucro'][i]:,.0f}</td><td>{rb['dividendos'][i]:,.0f}</td>
            <td style="color:#f59e0b;">{po_b:.1f}%</td>
            <td>{ro['lucro'][i]:,.0f}</td><td>{ro['dividendos'][i]:,.0f}</td>
            <td style="color:#10b981;">{po_o:.1f}%</td>
        </tr>"""
    pw_html += "</tbody></table>"
    st.markdown(pw_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# ABA 2 — INDICADORES
# ─────────────────────────────────────────────────────────────
def tab_indicadores(hist, preco, acoes, anos_hist, res, p):
    rb = res["Base"]
    ind_hist, ind_proj = calc_indicadores(hist, preco, acoes, rb, p)

    verd_label, verd_upside, verd_cor = veredicto_valuation(res, preco)
    badge_class = f"badge-{verd_cor}"

    col_preco, col_ind = st.columns([1, 3])

    with col_preco:
        gauge_val   = min(max((verd_upside + 50) / 100, 0.0), 1.0)
        gauge_color = cor_hex(verd_cor)
        gauge_pct   = int(gauge_val * 100)

        st.markdown(f"""
        <div class="preco-container">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;color:#64748b;text-transform:uppercase;letter-spacing:0.1em;">PREÇO ATUAL</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:2.2rem;font-weight:600;color:#e2e8f0;margin:0.3rem 0;">R$ {preco:.2f}</div>
            <span class="badge {badge_class}">{verd_label}</span>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.78rem;color:#94a3b8;margin-top:0.5rem;">{verd_upside:+.1f}% vs preço justo mediano</div>
            <div style="margin-top:1.2rem;">
                <div style="display:flex;justify-content:space-between;font-family:'IBM Plex Mono',monospace;font-size:0.58rem;color:#475569;margin-bottom:0.3rem;">
                    <span>CARA</span><span>JUSTA</span><span>BARATA</span>
                </div>
                <div class="upside-bar-bg">
                    <div class="upside-bar-fill" style="width:{gauge_pct}%;background:{gauge_color};"></div>
                </div>
            </div>
            <div style="margin-top:1.2rem;border-top:1px solid #1e2a40;padding-top:0.8rem;">
                <div style="font-family:'IBM Plex Mono',monospace;font-size:0.62rem;color:#64748b;margin-bottom:0.5rem;">PREÇO JUSTO (R$)</div>
                <div style="display:flex;justify-content:space-between;">
                    <div style="text-align:center;">
                        <div style="color:#3b82f6;font-family:'IBM Plex Mono',monospace;font-size:1rem;font-weight:600;">R$ {res['Conservador']['pj']:.2f}</div>
                        <div style="color:#64748b;font-size:0.6rem;font-family:'IBM Plex Mono',monospace;">CONS.</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="color:#f59e0b;font-family:'IBM Plex Mono',monospace;font-size:1.1rem;font-weight:600;">R$ {res['Base']['pj']:.2f}</div>
                        <div style="color:#64748b;font-size:0.6rem;font-family:'IBM Plex Mono',monospace;">BASE</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="color:#10b981;font-family:'IBM Plex Mono',monospace;font-size:1rem;font-weight:600;">R$ {res['Otimista']['pj']:.2f}</div>
                        <div style="color:#64748b;font-size:0.6rem;font-family:'IBM Plex Mono',monospace;">OTIM.</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_ind:
        st.markdown('<p class="section-label">📊 Indicadores Fundamentalistas</p>', unsafe_allow_html=True)
        sub_tab1, sub_tab2 = st.tabs(["🏛️ Histórico (Último Balanço)", "🔮 Projetado (Premissas Ano 1)"])
        
        def render_ind_table(indicadores_dict):
            ind_html = """<table class="ind-table">
            <thead><tr><th style="text-align:left">Indicador</th><th>Valor</th><th>Status</th><th style="min-width:100px">Nível</th><th>Referência</th></tr></thead><tbody>"""
            for cat, items in indicadores_dict.items():
                ind_html += f'<tr class="section-row"><td colspan="5"><span class="ind-section">{cat}</span></td></tr>'
                for nome_i, (val, unit, bom, otimo, direcao) in items.items():
                    cor    = cor_ind(val, bom, otimo, direcao)
                    frac   = frac_ind(val, bom, otimo, direcao)
                    cor_h  = cor_hex(cor)
                    bar_w  = int(frac * 100)
                    val_str = "—" if val is None else (f"{val:.1f}%" if unit == "%" else f"{val:.1f}x")
                    status  = {"green": "✅ Ótimo", "amber": "⚠️ Médio", "red": "❌ Fraco", "muted": "—"}.get(cor, "—")
                    ref_txt = f"Ref: {bom:.0f}–{otimo:.0f}{unit}" if unit == "%" else f"Ref: {bom:.1f}–{otimo:.1f}x"
                    ind_html += f"""<tr>
                        <td class="ind-name">{nome_i}</td>
                        <td class="ind-val {cor}">{val_str}</td>
                        <td style="font-size:0.72rem;color:{cor_h};">{status}</td>
                        <td><div class="ind-bar-bg"><div class="ind-bar" style="width:{bar_w}%;background:{cor_h};"></div></div></td>
                        <td style="font-size:0.68rem;color:#475569;">{ref_txt}</td>
                    </tr>"""
            ind_html += "</tbody></table>"
            st.markdown(ind_html, unsafe_allow_html=True)

        with sub_tab1: render_ind_table(ind_hist)
        with sub_tab2: render_ind_table(ind_proj)

    st.markdown("---")
    st.markdown('<p class="section-label">📉 Evolução Histórica (R$M)</p>', unsafe_allow_html=True)
    st.plotly_chart(chart_historico(hist, anos_hist), use_container_width=True)

    n_h = len(anos_hist)
    ebit_h = [hist["receita"][i]+hist["cpv"][i]+hist["sga"][i]+hist["depreciacao"][i] for i in range(n_h)]
    ll_h   = [ebit_h[i]+hist["juros"][i]+hist["ir"][i] for i in range(n_h)]
    mg_b_h = [(hist["receita"][i]+hist["cpv"][i])/hist["receita"][i]*100 if hist["receita"][i] else 0 for i in range(n_h)]
    mg_e_h = [ebit_h[i]/hist["receita"][i]*100 if hist["receita"][i] else 0 for i in range(n_h)]
    mg_l_h = [ll_h[i]/hist["receita"][i]*100    if hist["receita"][i] else 0 for i in range(n_h)]

    evo_html = "<table class='prem-table'><thead><tr><th>Métrica</th>" + "".join(f"<th>{a}</th>" for a in anos_hist) + "</tr></thead><tbody>"
    rows_evo = [
        ("Receita (R$M)",       hist["receita"],      lambda v: f"{v:,.0f}"),
        ("EBIT (R$M)",          ebit_h,               lambda v: f"{v:,.0f}"),
        ("Lucro Líquido (R$M)", ll_h,                 lambda v: f"{v:,.0f}"),
        ("Margem Bruta (%)",    mg_b_h,               lambda v: f"{v:.1f}%"),
        ("Margem EBIT (%)",     mg_e_h,               lambda v: f"{v:.1f}%"),
        ("Margem Líquida (%)",  mg_l_h,               lambda v: f"{v:.1f}%"),
        ("Dívida Total (R$M)",  hist["divida"],        lambda v: f"{v:,.0f}"),
        ("Caixa (R$M)",         hist["caixa"],         lambda v: f"{v:,.0f}"),
        ("PL (R$M)",            hist["pl"],            lambda v: f"{v:,.0f}"),
        ("FCFF (R$M)",          hist.get("fcff_hist",[0]*n_h), lambda v: f"{v:,.0f}"),
    ]
    for label, vals, fmt_fn in rows_evo:
        evo_html += f"<tr><td>{label}</td>" + "".join(f"<td class='hist-val'>{fmt_fn(v)}</td>" for v in vals) + "</tr>"
    evo_html += "</tbody></table>"
    st.markdown(evo_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# ABA 3 — SENSIBILIDADE
# ─────────────────────────────────────────────────────────────
def tab_sensibilidade(hist, preco, acoes, res, p):
    st.markdown('<p class="section-label">🌡️ Heatmap — WACC × Crescimento → Preço Justo</p>', unsafe_allow_html=True)
    sens = calc_sens(hist, preco, acoes, p)
    st.plotly_chart(chart_heatmap(sens, preco), use_container_width=True)

    st.markdown("""
    <div style="display:flex;gap:1.2rem;font-family:'IBM Plex Mono',monospace;font-size:0.68rem;margin-top:0.4rem;">
        <span style="color:#10b981;">■ &gt;+30% upside</span>
        <span style="color:#065f46;">■ +10% a +30%</span>
        <span style="color:#93c5fd;">■ ±10% (justo)</span>
        <span style="color:#fb923c;">■ -10% a -30%</span>
        <span style="color:#f87171;">■ &lt;-30%</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="section-label">📦 Decomposição do Valor Detalhada — Base</p>', unsafe_allow_html=True)
    rb = res["Base"]
    df_decomp = pd.DataFrame({
        "Componente":     ["VP FCFFs (5a)", "Valor Terminal (VP)", "(-) Dívida Líquida", "= Equity (Valor Justo)"],
        "Valor (R$M)":    [f"{rb['vp_fcff']:,.0f}", f"{rb['vp_tv']:,.0f}", f"({abs(rb['div_liq']):,.0f})", f"{rb['equity']:,.0f}"],
        "% do EV":        [f"{rb['vp_fcff']/rb['ve']*100:.1f}%" if rb['ve'] else "—",
                           f"{rb['vp_tv']/rb['ve']*100:.1f}%"   if rb['ve'] else "—", "—", "100%"],
        "R$/Ação":        [f"{rb['vp_fcff']/acoes:.2f}", f"{rb['vp_tv']/acoes:.2f}",
                           f"({abs(rb['div_liq'])/acoes:.2f})", f"{rb['pj']:.2f}"],
    })
    st.dataframe(df_decomp, use_container_width=True, hide_index=True)

    st.info(
        f"**ROIC Terminal (Base):** {rb['roic_tv']*100:.1f}%  |  "
        f"**Taxa Reinvestimento Perpetuidade:** {rb['reinv_rate_tv']*100:.1f}%  |  "
        f"TV representa {rb['vp_tv']/rb['ve']*100:.1f}% do EV"
    )

    st.markdown("---")
    st.markdown('<p class="section-label">📡 Sensibilidade WACC & Crescimento Perpetuidade</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        waccs = np.linspace(0.08, 0.30, 20)
        rows  = [{"WACC (%)": f"{w*100:.1f}%",
                  "Preço Justo (R$)": f"R$ {calc_cenario('S', hist, preco, acoes, {**p, 'wacc': w})['pj']:.2f}",
                  "Upside (%)": f"{(calc_cenario('S', hist, preco, acoes, {**p, 'wacc': w})['pj'] - preco)/preco*100:+.1f}%"}
                 for w in waccs]
        st.markdown("**Preço Justo × WACC**")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=350)

    with col2:
        cps  = np.linspace(0.01, 0.09, 17)
        rows = [{"g Perpetuidade (%)": f"{c*100:.1f}%",
                 "Preço Justo (R$)": f"R$ {calc_cenario('S', hist, preco, acoes, {**p, 'cresc_perp': c})['pj']:.2f}",
                 "Upside (%)": f"{(calc_cenario('S', hist, preco, acoes, {**p, 'cresc_perp': c})['pj'] - preco)/preco*100:+.1f}%"}
                for c in cps]
        st.markdown("**Preço Justo × Cresc. Perpetuidade**")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=350)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    ticker = render_sidebar_header()

    with st.spinner(f"Buscando dados para {ticker}…"):
        hist, preco, acoes, anos_hist, nome_empresa, setor, erro = buscar_dados(ticker)

    if erro or hist is None:
        st.error(f"❌ Erro ao buscar dados para {ticker}: {erro}")
        st.info("Verifique se o ticker está correto (ex: LREN3.SA, VALE3.SA)")
        return

    dyn_defaults = calc_historico_medio(hist)
    dyn_defaults["divida_proj"] = hist["divida"][-1]

    p = render_sidebar_params(dyn_defaults)

    anos_proj = list(range(anos_hist[-1] + 1, anos_hist[-1] + 6))
    res       = calc_todos(hist, preco, acoes, p)

    col_h1, col_h2, col_h3 = st.columns([3, 1, 1])
    with col_h1:
        st.markdown(f"""
        <div class="app-header">
            <div class="app-title">📊 {nome_empresa} ({ticker.replace('.SA','')})</div>
            <div class="app-subtitle">{setor} · yfinance · Histórico {anos_hist[0]}–{anos_hist[-1]} · Projeção {anos_proj[0]}–{anos_proj[-1]} · IR={p['ir_pct']*100:.0f}% · WACC={p['wacc']*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with col_h2:
        st.metric("Ações em Circulação", f"{acoes:.0f}M")
    with col_h3:
        st.metric("Market Cap", f"R$ {preco*acoes:,.0f}M")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊  DCF / Cenários",
        "💰  Dividendos",
        "📈  Indicadores",
        "🌡️  Sensibilidade",
    ])

    with tab1: tab_dcf(hist, preco, acoes, anos_hist, anos_proj, res, p)
    with tab2: tab_dividendos(hist, preco, acoes, anos_hist, anos_proj, res)
    with tab3: tab_indicadores(hist, preco, acoes, anos_hist, res, p)
    with tab4: tab_sensibilidade(hist, preco, acoes, res, p)


if __name__ == "__main__":
    main()
