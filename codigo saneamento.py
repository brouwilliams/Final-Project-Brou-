
# ============================================
# ANÁLISE EXPLORATÓRIA DE DADOS - SNIS NORDESTE 1
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import f_regression
from linearmodels.panel import PanelOLS, RandomEffects, compare
import statsmodels.api as sm
import os

# ============================================
# 1. Carregamento e Filtragem dos Dados
# ============================================
df = pd.read_csv("snis_nordeste_1_filtrado (1).csv")
df = df[df['sigla_uf'].isin(['PE', 'AL', 'SE'])].copy()

# ============================================
# 2. Análise Exploratória
# ============================================
print("Dimensões do DataFrame:", df.shape)
print("\nTipos de variáveis:")
print(df.dtypes)
print("\nValores faltantes por coluna:")
print(df.isnull().sum())

# ============================================
# 3. Estatísticas Descritivas
# ============================================
numeric_df = df.select_dtypes(include=[np.number])
estatisticas = numeric_df.describe()
estatisticas.to_csv("estatisticas_descritivas.csv")

# ============================================
# 4. Matriz de Correlação + Heatmap
# ============================================
correlation_matrix = numeric_df.corr()
correlation_matrix.to_csv("matriz_correlacao.csv")

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, cmap='coolwarm', center=0)
plt.title('Matriz de Correlação')
plt.tight_layout()
plt.savefig('heatmap_correlacao.png')
plt.close()

# Correlações com AG001 e ES001
top_ag001_corr = correlation_matrix['populacao_atendida_agua'].drop('populacao_atendida_agua').abs().sort_values(ascending=False).head(10)
top_es001_corr = correlation_matrix['populacao_atendida_esgoto'].drop('populacao_atendida_esgoto').abs().sort_values(ascending=False).head(10)

# ============================================
# 5. Regressão Linear e Rankings
# ============================================
X_ag001 = numeric_df[top_ag001_corr.index].dropna()
y_ag001 = numeric_df.loc[X_ag001.index, 'populacao_atendida_agua']
X_es001 = numeric_df[top_es001_corr.index].dropna()
y_es001 = numeric_df.loc[X_es001.index, 'populacao_atendida_esgoto']

f_ag001, p_ag001 = f_regression(X_ag001, y_ag001)
f_es001, p_es001 = f_regression(X_es001, y_es001)

ranking_ag001 = pd.DataFrame({
    'variavel': top_ag001_corr.index,
    'correlacao': top_ag001_corr.values,
    'f_statistic': f_ag001,
    'p_value': p_ag001
}).sort_values(by='f_statistic', ascending=False)

ranking_es001 = pd.DataFrame({
    'variavel': top_es001_corr.index,
    'correlacao': top_es001_corr.values,
    'f_statistic': f_es001,
    'p_value': p_es001
}).sort_values(by='f_statistic', ascending=False)

ranking_ag001.to_csv("ranking_ag001.csv", index=False)
ranking_es001.to_csv("ranking_es001.csv", index=False)

# ============================================
# 6. Gráficos de Dispersão
# ============================================
os.makedirs("graficos", exist_ok=True)

def gerar_graficos(target, X, nome_target):
    for coluna in X.columns:
        plt.figure(figsize=(8, 6))
        sns.regplot(x=X[coluna], y=target, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
        plt.xlabel(coluna)
        plt.ylabel(nome_target)
        plt.title(f'{nome_target} vs {coluna}')
        plt.tight_layout()
        plt.savefig(f'graficos/{nome_target}_vs_{coluna}.png')
        plt.close()

gerar_graficos(y_ag001, X_ag001, 'AG001')
gerar_graficos(y_es001, X_es001, 'ES001')

# ============================================
# 7. Teste de Correlação (Pearson)
# ============================================
corr_ag001 = numeric_df.corrwith(numeric_df['populacao_atendida_agua']).sort_values(ascending=False)
corr_es001 = numeric_df.corrwith(numeric_df['populacao_atendida_esgoto']).sort_values(ascending=False)

corr_ag001.to_csv('correlacao_ag001.csv')
corr_es001.to_csv('correlacao_es001.csv')

# ============================================
# 8. Teste de Hausman (Painel)
# ============================================
panel_df = df.set_index(['id_municipio', 'ano'])
panel_numeric = panel_df[['populacao_atendida_agua', 'populacao_atendida_esgoto'] + list(top_ag001_corr.index)].dropna()
panel_numeric = sm.add_constant(panel_numeric)

# Modelos FE e RE para AG001
fe_model_ag001 = PanelOLS(panel_numeric['populacao_atendida_agua'], panel_numeric.drop('populacao_atendida_agua', axis=1), entity_effects=True)
re_model_ag001 = RandomEffects(panel_numeric['populacao_atendida_agua'], panel_numeric.drop('populacao_atendida_agua', axis=1))

fe_res_ag001 = fe_model_ag001.fit()
re_res_ag001 = re_model_ag001.fit()

hausman_ag001 = compare({'FE': fe_res_ag001, 'RE': re_res_ag001})

# Modelos FE e RE para ES001
fe_model_es001 = PanelOLS(panel_numeric['populacao_atendida_esgoto'], panel_numeric.drop('populacao_atendida_esgoto', axis=1), entity_effects=True)
re_model_es001 = RandomEffects(panel_numeric['populacao_atendida_esgoto'], panel_numeric.drop('populacao_atendida_esgoto', axis=1))

fe_res_es001 = fe_model_es001.fit()
re_res_es001 = re_model_es001.fit()

hausman_es001 = compare({'FE': fe_res_es001, 'RE': re_res_es001})

with open('hausman_ag001.txt', 'w') as f:
    f.write(str(hausman_ag001))
with open('hausman_es001.txt', 'w') as f:
    f.write(str(hausman_es001))

# Gráfico comparativo R²
hausman_stats = pd.DataFrame({
    'Modelo': ['AG001_FE', 'AG001_RE', 'ES001_FE', 'ES001_RE'],
    'R2': [fe_res_ag001.rsquared, re_res_ag001.rsquared, fe_res_es001.rsquared, re_res_es001.rsquared]
})

plt.figure(figsize=(8, 6))
sns.barplot(x='Modelo', y='R2', data=hausman_stats)
plt.title('Comparação R² - Modelos FE vs RE')
plt.tight_layout()
plt.savefig('grafico_hausman.png')
plt.close()

print(\"Processo concluído com sucesso!\")
