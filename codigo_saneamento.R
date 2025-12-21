############################################
# 0) PACOTES
############################################
library(tidyverse)
library(plm)
library(lmtest)
library(sandwich)
library(MASS)

############################################
# 1) BASE DE DADOS
############################################
# Base enviada na primeira entrega
dados <- read_csv("snis_nordeste_1_filtrado.csv")

# Conferência inicial
glimpse(dados)

############################################
# 2) PREPARAÇÃO DO PAINEL
############################################
# Justificativa:
# Dados municipais possuem heterogeneidade não observada (estrutura, geografia,
# governança). Logo, é necessário usar modelos de dados em painel.

dados <- dados %>%
  drop_na()

painel <- pdata.frame(
  dados,
  index = c("municipio", "ano")
)

############################################
# 3) JUSTIFICATIVA DAS VARIÁVEIS EXPLICATIVAS
############################################
# As variáveis abaixo foram escolhidas com base em:
# (i) literatura de saneamento
# (ii) correlação com AG001 e ES001
# (iii) interpretação econômica clara
#
# - receitas e despesas: capacidade financeira
# - empregados: capacidade operacional
# - investimentos: expansão de rede
# - energia e despesas: custo operacional

vars_exp <- c(
  "quantidade_empregado",
  "receita_operacional",
  "receita_operacional_direta_agua",
  "receita_operacional_direta_esgoto",
  "despesa_total_servico",
  "despesa_energia",
  "investimento_recurso_proprio_prestador",
  "investimento_recurso_oneroso_estado"
)

############################################
# 4) MODELOS DE PAINEL – AG001
############################################
# Modelo POLS (necessário para comparação)
mod_pols_ag <- plm(
  AG001 ~ ., 
  data = painel[, c("AG001", vars_exp)],
  model = "pooling"
)

# Modelo de Efeitos Fixos
mod_fe_ag <- plm(
  AG001 ~ ., 
  data = painel[, c("AG001", vars_exp)],
  model = "within"
)

# Modelo de Efeitos Aleatórios
mod_re_ag <- plm(
  AG001 ~ ., 
  data = painel[, c("AG001", vars_exp)],
  model = "random"
)

############################################
# 5) TESTE DE HAUSMAN – AG001
############################################
hausman_ag <- phtest(mod_fe_ag, mod_re_ag)
print(hausman_ag)

# Interpretação:
# p-valor < 0.05 → rejeita RE → FE é o modelo adequado

############################################
# 6) DIAGNÓSTICOS – AG001
############################################
# Heterocedasticidade
bp_ag <- pbpTest(mod_fe_ag)
print(bp_ag)

# Autocorrelação (Wooldridge)
wool_ag <- pwartest(mod_fe_ag)
print(wool_ag)

# Dependência seccional (Pesaran CD)
cd_ag <- pcdtest(mod_fe_ag, test = "cd")
print(cd_ag)

############################################
# 7) CORREÇÃO ROBUSTA – DRISCOLL-KRAAY (AG001)
############################################
coeftest(
  mod_fe_ag,
  vcov = vcovSCC(mod_fe_ag, type = "HC1")
)

############################################
# 8) SELEÇÃO DE VARIÁVEIS (AIC) DENTRO DO PAINEL
############################################
# Justificativa:
# A seleção não pode ser feita via regressão simples.
# Aqui usamos FE via dummies municipais (equivalente ao within).

lm_fe_ag <- lm(
  AG001 ~ . + factor(municipio),
  data = dados[, c("AG001", vars_exp, "municipio")]
)

step_ag <- stepAIC(
  lm_fe_ag,
  direction = "both",
  trace = FALSE
)

summary(step_ag)

############################################
# 9) MODELO FINAL FE – AG001
############################################
vars_ag_final <- names(coef(step_ag)) %>%
  str_remove("factor\\(municipio\\)") %>%
  setdiff("(Intercept)")

mod_fe_ag_final <- plm(
  as.formula(paste("AG001 ~", paste(vars_ag_final, collapse = "+"))),
  data = painel,
  model = "within"
)

coeftest(
  mod_fe_ag_final,
  vcov = vcovSCC(mod_fe_ag_final, type = "HC1")
)

############################################
# 10) REPETIÇÃO PARA ES001
############################################
mod_fe_es <- plm(
  ES001 ~ ., 
  data = painel[, c("ES001", vars_exp)],
  model = "within"
)

mod_re_es <- plm(
  ES001 ~ ., 
  data = painel[, c("ES001", vars_exp)],
  model = "random"
)

# Hausman
phtest(mod_fe_es, mod_re_es)

# Diagnósticos
pbpTest(mod_fe_es)
pwartest(mod_fe_es)
pcdtest(mod_fe_es, test = "cd")

# Driscoll-Kraay
coeftest(
  mod_fe_es,
  vcov = vcovSCC(mod_fe_es, type = "HC1")
)

############################################
# 11) SELEÇÃO DE VARIÁVEIS – ES001
############################################
lm_fe_es <- lm(
  ES001 ~ . + factor(municipio),
  data = dados[, c("ES001", vars_exp, "municipio")]
)

step_es <- stepAIC(
  lm_fe_es,
  direction = "both",
  trace = FALSE
)

summary(step_es)

vars_es_final <- names(coef(step_es)) %>%
  str_remove("factor\\(municipio\\)") %>%
  setdiff("(Intercept)")

mod_fe_es_final <- plm(
  as.formula(paste("ES001 ~", paste(vars_es_final, collapse = "+"))),
  data = painel,
  model = "within"
)

coeftest(
  mod_fe_es_final,
  vcov = vcovSCC(mod_fe_es_final, type = "HC1")
)

############################################
# FIM DO SCRIPT
############################################
