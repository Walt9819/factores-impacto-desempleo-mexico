## Main file ##

## Load libraries
suppressWarnings(library(mongolite)) # get data from MongoDB
suppressWarnings(library(dplyr)) # data manipulation
require(ISLR)
require(aod)

#### Lectura de datos ####
## Conexión a MongoDB
covidData.DB <- mongo(db="bedu18", collection="data_covid", url = "mongodb+srv://Henry:3eXoszlAIBpQzGGA@proyectobedu.jr6fz.mongodb.net/test")
logitData.DB <- mongo(db="bedu18", collection="data_logit", url = "mongodb+srv://Henry:3eXoszlAIBpQzGGA@proyectobedu.jr6fz.mongodb.net/test")

## Lectura datos (DENUE, ENOE y COVID)
# Lectura DENUE y COVID
allLogitData <- logitData.DB$find('{}')
allCovidData <- covidData.DB$find('{}')

# Lectura ENOE (opcional)

## Limpieza de datos
# Seleccionar columnas deseadas
# Transformar datos
# Limpieza general

#### Exploración de datos ####
## Gráfico `pairs` pairs(iris[,1:4], pch = 19, lower.panel = NULL)

## Heatmap correlaciones atributo vs atributo

#### Análisis ####
## Modelo "lineal" ENOE para predecir: probabilidad desempleo ~ c(sexo, edad, nivel educativo, sector económico, estado de residencia)

## Modelo "Logit" ENOE para predecir: probabilidad desempleo ~ c(sexo, edad, nivel educativo, sector económico, estado de residencia)
## check results
summary(allLogitData)
sapply(allLogitData, sd)
xtabs(~eda + clase2, data = allLogitData)

allLogitData$clase2 <- factor(allLogitData$clase2)
mylogit <- glm(clase2 ~ sex + eda + niv_ins + rama, data = allLogitData, family = "binomial")
summary(mylogit)
# Call:
#   glm(formula = clase2 ~ sex + eda + niv_ins + rama, family = "binomial", 
#       data = allLogitData)
# 
# Deviance Residuals: 
#   Min      1Q  Median      3Q     Max  
# -8.490   0.000   0.000   0.000   1.392  
# 
# Coefficients:
#   Estimate Std. Error  z value Pr(>|z|)    
# (Intercept) -15.02512    0.11163 -134.600   <2e-16 ***
#   sex          -0.03882    0.02620   -1.481    0.138    
# eda           0.97283    0.01019   95.492   <2e-16 ***
#   niv_ins       2.93707    0.03676   79.891   <2e-16 ***
#   rama         13.73648  129.67436    0.106    0.916    
# ---
#   Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 660434  on 689029  degrees of freedom
# Residual deviance:  37047  on 689025  degrees of freedom
# (13155 observations deleted due to missingness)
# AIC: 37057
# 
# Number of Fisher Scoring iterations: 25

confint.default(mylogit)
# 2.5 %      97.5 %
#   (Intercept)  -15.24391042 -14.8063378
# sex           -0.09017607   0.0125373
# eda            0.95285923   0.9927936
# niv_ins        2.86501159   3.0091220
# rama        -240.42059136 267.8935560
wald.test(b = coef(mylogit), Sigma = vcov(mylogit), Terms = 1:4)
# Wald test:
#   ----------
#   
#   Chi-squared test:
#   X2 = 29022.7, df = 4, P(> X2) = 0.0

exp(coef(mylogit))

##### Status: In process. Comment: Check other ways (should be 70% train 30% test)
newdata1 <- with(data_rl, data.frame(sex = mean(sex), eda = mean(eda), niv_ins = mean(niv_ins), rama = mean(rama), clase2 = factor(0:4)))
newdata1$rankP <- predict(mylogit, newdata = newdata1, type = "response")
newdata1

# Primer trimestre (sin restricciones)
# Segundo y tercer trimestre (con restricciones por COVID)

## Random forest
## Modelo Bayesiano (opcional)

## (Pensamiento al aire) Comparación entre variables antes del COVID (restricciones) y con COVID

#### Dashboard ####
## Indicadores propuestos por municipio (en mapa) de forma descriptiva

## Gráficos modelo "logit"

## Ingresar datos para generar predicción
