
# -*- coding: utf-8 -*-
"""
TAREFA – REGRESSÃO LINEAR (Brou)
Variável dependente: Rp_ew_fi - Rf
Período: Não Crise

Ordem da tarefa:
1) Exploração inicial (correlações, heatmap, scatterplots)
2) Ajuste do modelo inicial (OLS com fatores de Fama-French)
3) Teste de Box-Cox e eventual reajuste
4) Seleção do modelo (AIC/BIC + Forward/Backward/Stepwise)
5) Multicolinearidade (VIF e Número de Condição)
6) Correção de heterocedasticidade (Newey-West com 6 lags)
7) Diagnósticos (resíduos, alavancagem, DFFITS, DFBETAS, COVRATIO)
8) Teste de Ljung-Box nos resíduos do modelo final
9) Importância relativa dos regressores (ΔR²)
10) Exportação de tabelas e gráficos

Arquivos esperados no diretório:
- "FFF_ret (1).csv"

Observações:
- Fatores usados: Rm-Rf, SMB, HML, RMW, CMA, Mom, ST_Rev, LT_Rev (se existir).
- Os períodos de crise (bear markets) são baseados na lista do PDF/Hartford Funds e são excluídos para formar o "Não Crise".
"""

# ============================
# Imports
# ============================
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from scipy.stats import boxcox
import statsmodels.api as sm
from statsmodels.api import OLS, add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor, OLSInfluence
from statsmodels.stats.diagnostic import acorr_ljungbox

warnings.simplefilter("ignore")
sns.set(style="whitegrid")
plt.rcParams['figure.dpi'] = 150

# ============================
# 0) Carregar dados e definir período "Não Crise"
# ============================
# Lê o CSV principal e renomeia a coluna de data
fff = pd.read_csv("FFF_ret (1).csv").rename(columns={"Unnamed: 0": "Date"})
fff["Date"] = pd.to_datetime(fff["Date"], dayfirst=True, errors="coerce")
fff = fff.sort_values("Date").reset_index(drop=True)

# Define fatores disponíveis no arquivo (inclui LT_Rev se existir)
base_factors = ["Rm-Rf","SMB","HML","RMW","CMA","Mom","ST_Rev","LT_Rev"]
factors = [c for c in base_factors if c in fff.columns]

# Variável dependente (excesso de retorno EW-FI): Y = Rp_ew_fi - Rf
fff["Y"] = fff["Rp_ew_fi"] - fff["Rf"]

# Períodos de crise (bear markets) conforme o PDF/Hartford Funds (datas aproximadas)
# Estes períodos são excluídos para formar o conjunto NÃO CRISE
crisis_ranges = [
    ("1929-08-01", "1932-06-30"),
    ("1937-03-01", "1942-04-30"),
    ("1946-05-01", "1949-06-30"),
    ("1956-08-01", "1957-10-31"),
    ("1961-12-01", "1962-06-30"),
    ("1966-02-01", "1966-10-31"),
    ("1968-11-01", "1970-05-31"),
    ("1973-01-01", "1974-10-31"),
    ("1980-11-01", "1982-08-31"),
    ("1987-08-01", "1987-12-31"),
    ("2000-03-01", "2002-10-31"),
    ("2007-10-01", "2009-03-31"),
    ("2020-02-01", "2020-03-31"),
]

mask_non_crisis = pd.Series(True, index=fff.index)
for start, end in crisis_ranges:
    mask_non_crisis &= ~fff["Date"].between(pd.Timestamp(start), pd.Timestamp(end))

data_nc = fff.loc[mask_non_crisis].copy()

# Remove linhas com NaN em Y ou nos fatores efetivamente disponíveis
cols_needed = ["Y"] + factors
data_nc = data_nc.dropna(subset=cols_needed).copy()

# ============================
# 1) Exploração Inicial
# ============================
# Matriz de correlação e heatmap
corr_mat = data_nc[["Y"] + factors].corr()

os.makedirs("figs", exist_ok=True)
plt.figure(figsize=(8,6))
sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Matriz de correlação – Não Crise (Brou)")
plt.tight_layout()
plt.savefig("figs/01_heatmap_correlacao.png")
plt.close()

# Scatterplots: Y vs cada fator (com linha de tendência)
for fac in factors:
    plt.figure(figsize=(4,3))
    sns.scatterplot(x=data_nc[fac], y=data_nc["Y"], s=10, alpha=0.6)
    sns.regplot(x=data_nc[fac], y=data_nc["Y"], scatter=False, color="red", ci=None)
    plt.xlabel(fac)
    plt.ylabel("Y = Rp_ew_fi - Rf")
    plt.title(f"Dispersão: Y vs {fac}")
    plt.tight_layout()
    plt.savefig(f"figs/01_scatter_{fac}.png")
    plt.close()

# ============================
# 2) Ajuste do Modelo Inicial (OLS com fatores)
# ============================
X0 = add_constant(data_nc[factors])
y0 = data_nc["Y"]
model0 = OLS(y0, X0).fit()

# ============================
# 3) Teste de Box-Cox (na variável dependente)
# ============================
# Box-Cox exige Y>0. Para viabilizar, deslocamos Y para ser positivo (se necessário)
min_y = y0.min()
shift = 1e-6 - min_y if min_y <= 0 else 0.0
y_pos = y0 + shift

lambda_bc = None
y_bc = None
try:
    y_bc, lambda_bc = boxcox(y_pos)
except ValueError:
    # Se não for possível, mantém sem transformação
    y_bc, lambda_bc = None, None

# Critério simples: se |lambda-1| > 0.10, usar Y transformado; senão, manter Y original
if lambda_bc is not None and abs(lambda_bc - 1.0) > 0.10:
    y_for_fit = y_bc
    y_label = f"Y_boxcox(lambda={lambda_bc:.2f})"
else:
    y_for_fit = y0
    y_label = "Y (sem transformação)"

X_bc = add_constant(data_nc[factors])
model_bc = OLS(y_for_fit, X_bc).fit()

# ============================
# 4) Seleção do Modelo
# ============================
# 4.1) Busca por subconjuntos (All-subsets) e ranking por AIC/BIC
results_subset = []
for k in range(1, len(factors)+1):
    for subset in combinations(factors, k):
        Xs = add_constant(data_nc[list(subset)])
        m = OLS(y_for_fit, Xs).fit()
        results_subset.append({
            "vars": subset,
            "k": k,
            "AIC": m.aic,
            "BIC": m.bic,
            "R2": m.rsquared
        })
res_df = pd.DataFrame(results_subset).sort_values(["AIC","BIC"]).reset_index(drop=True)

best_aic_vars = list(res_df.loc[0,"vars"])
best_bic_vars = list(res_df.sort_values("BIC").loc[0,"vars"])

# 4.2) Forward stepwise (AIC)
remaining = set(factors)
selected_fwd = []
current_aic = np.inf
while remaining:
    aic_candidates = []
    for var in list(remaining):
        Xs = add_constant(data_nc[selected_fwd + [var]])
        m = OLS(y_for_fit, Xs).fit()
        aic_candidates.append((m.aic, var))
    best_aic, best_var = min(aic_candidates, key=lambda x: x[0])
    if best_aic < current_aic - 1e-8:
        selected_fwd.append(best_var)
        remaining.remove(best_var)
        current_aic = best_aic
    else:
        break

# 4.3) Backward stepwise (AIC) – começa com todos e remove
selected_bwd = factors.copy()
while True:
    aic_full = OLS(y_for_fit, add_constant(data_nc[selected_bwd])).fit().aic
    aic_drop = []
    for var in selected_bwd:
        candidate = [v for v in selected_bwd if v != var]
        m = OLS(y_for_fit, add_constant(data_nc[candidate])).fit()
        aic_drop.append((m.aic, var))
    best_aic, drop_var = min(aic_drop, key=lambda x: x[0])
    if best_aic < aic_full - 1e-8:
        selected_bwd.remove(drop_var)
    else:
        break

# 4.4) Stepwise (aditivo-removido, AIC)
selected_sw = []
remaining = set(factors)
current_aic = np.inf
changed = True
while changed:
    changed = False
    # tentativas de inclusão
    add_aic = []
    for var in list(remaining):
        Xs = add_constant(data_nc[selected_sw + [var]])
        m = OLS(y_for_fit, Xs).fit()
        add_aic.append((m.aic, var))
    if add_aic:
        best_aic, best_var = min(add_aic, key=lambda x: x[0])
        if best_aic < current_aic - 1e-8:
            selected_sw.append(best_var)
            remaining.remove(best_var)
            current_aic = best_aic
            changed = True
    # tentativas de remoção
    rem_aic = []
    for var in list(selected_sw):
        candidate = [v for v in selected_sw if v != var]
        Xs = add_constant(data_nc[candidate]) if candidate else add_constant(pd.DataFrame(index=data_nc.index))
        m = OLS(y_for_fit, Xs).fit()
        rem_aic.append((m.aic, var, candidate))
    if rem_aic:
        best_aic_rem, var_rem, cand = min(rem_aic, key=lambda x: x[0])
        full_aic = OLS(y_for_fit, add_constant(data_nc[selected_sw])).fit().aic
        if best_aic_rem < full_aic - 1e-8:
            selected_sw.remove(var_rem)
            remaining.add(var_rem)
            current_aic = best_aic_rem
            changed = True

# Escolha final: modelo que minimiza BIC
final_vars = best_bic_vars if len(best_bic_vars)>0 else factors
X_final = add_constant(data_nc[final_vars])
model_final_ols = OLS(y_for_fit, X_final).fit()

# ============================
# 5) Multicolinearidade: VIF e Número de Condição
# ============================
# VIF (padroniza X para evitar escalas diferentes)
X_for_vif = data_nc[final_vars].copy()
X_for_vif = (X_for_vif - X_for_vif.mean())/X_for_vif.std(ddof=0)
X_for_vif = add_constant(X_for_vif)

vifs = []
for i, col in enumerate(X_for_vif.columns):
    if col == "const":
        continue
    vifs.append({"variavel": col, "VIF": variance_inflation_factor(X_for_vif.values, i)})
vif_df = pd.DataFrame(vifs)

# Número de condição (cond. number) do X padronizado (sem constante)
Xcn = data_nc[final_vars].copy()
Xcn = (Xcn - Xcn.mean())/Xcn.std(ddof=0)
_, svals, _ = np.linalg.svd(Xcn.values, full_matrices=False)
cond_number = svals.max()/svals.min()

# ============================
# 6) Correção de Heterocedasticidade: Newey-West (6 lags)
# ============================
model_final_hac = OLS(y_for_fit, X_final).fit(cov_type="HAC", cov_kwds={"maxlags":6})

# ============================
# 7) Diagnósticos do Modelo
# ============================
inf = OLSInfluence(model_final_ols)
resid_student = inf.resid_studentized_internal
leverage_h = inf.hat_matrix_diag

influence_df = pd.DataFrame({
    "Date": data_nc["Date"].values,
    "resid_padronizado": resid_student,
    "leverage_h": leverage_h,
    "dffits": inf.dffits[0],
    "covratio": inf.cov_ratio,
})

# DFBETAS (uma coluna por parâmetro)
dfbetas = pd.DataFrame(inf.dfbetas, columns=["const"] + final_vars)
dfbetas.index = data_nc.index
influence_full = influence_df.join(dfbetas, how="left")

# ============================
# 8) Teste de Ljung-Box nos resíduos do modelo final
# ============================
ljung_box = acorr_ljungbox(model_final_ols.resid, lags=[1,5,10,20], return_df=True)

# ============================
# 9) Importância Relativa dos Regressors (ΔR²)
# ============================
X_all = add_constant(data_nc[final_vars])
full = OLS(y_for_fit, X_all).fit()
full_r2 = full.rsquared

imp_rows = []
for var in final_vars:
    vars_minus = [v for v in final_vars if v != var]
    m_minus = OLS(y_for_fit, add_constant(data_nc[vars_minus])).fit()
    delta_r2 = full_r2 - m_minus.rsquared
    imp_rows.append({"variavel": var, "delta_R2": delta_r2})
imp_df = pd.DataFrame(imp_rows).sort_values("delta_R2", ascending=False).reset_index(drop=True)

# ============================
# 10) Exportar Tabelas e Gráficos
# ============================
os.makedirs("tabelas", exist_ok=True)

# Correlações
corr_mat.to_csv("tabelas/01_correlacao.csv", index=True)

# Lista de modelos (AIC/BIC)
res_df.to_csv("tabelas/04_modelos_all_subsets.csv", index=False)

# VIF e número de condição
vif_df.to_csv("tabelas/05_vif.csv", index=False)
with open("tabelas/05_cond_number.txt","w") as f:
    f.write(f"Numero de Condicao (X padronizado): {cond_number:.2f}\n")

# Sumários dos modelos
with open("tabelas/02_modelo_inicial_OLS.txt","w") as f:
    f.write(model0.summary().as_text())
with open("tabelas/03_modelo_boxcox_OLS.txt","w") as f:
    f.write(model_bc.summary().as_text())
with open("tabelas/06_modelo_final_OLS.txt","w") as f:
    f.write(model_final_ols.summary().as_text())
with open("tabelas/06_modelo_final_NeweyWest.txt","w") as f:
    f.write(model_final_hac.summary().as_text())

# Diagnósticos e Ljung-Box
influence_full.to_csv("tabelas/07_diagnosticos_influencia.csv", index=False)
ljung_box.to_csv("tabelas/08_ljung_box.csv", index=True)

# Importância relativa
imp_df.to_csv("tabelas/09_importancia_relativa.csv", index=False)

# Relatório resumido em Excel
with pd.ExcelWriter("tabelas/Resultados_Brou_NaoCrise.xlsx", engine="openpyxl") as xlw:
    corr_mat.to_excel(xlw, sheet_name="correlacao")
    res_df.head(50).to_excel(xlw, sheet_name="modelos_top50", index=False)
    pd.DataFrame({"final_vars": final_vars}).to_excel(xlw, sheet_name="final_vars", index=False)
    pd.read_csv("tabelas/05_vif.csv").to_excel(xlw, sheet_name="vif", index=False)
    pd.DataFrame({"cond_number":[cond_number]}).to_excel(xlw, sheet_name="cond_number", index=False)
    ljung_box.to_excel(xlw, sheet_name="ljung_box")
    imp_df.to_excel(xlw, sheet_name="importancia_relativa", index=False)

print("Pipeline concluído. Arquivos gerados:")
print("- figs/01_heatmap_correlacao.png")
print("- figs/01_scatter_<fator>.png para cada fator")
print("- tabelas/*.csv, *.txt e Resultados_Brou_NaoCrise.xlsx")
