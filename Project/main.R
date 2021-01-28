## Main file ##

## Load libraries
suppressWarnings(library(mongolite)) # get data from MongoDB
suppressWarnings(library(dplyr)) # data manipulation
suppressWarnings(library(aod)) # logistic regression odds
suppressWarnings(library(ggplot2)) # graphs
require(ISLR)


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

# Análisis exploratorio

names(allLogitData)

str(allLogitData)

head(allLogitData)

summary(allLogitData)

# Omision de variables 
allLogitData <- na.omit(allLogitData)

allLogitData <- allLogitData[allLogitData$clase2 <= 3 & 
                               allLogitData$clase2 !=0 & 
                               allLogitData$eda >= 15 & 
                               allLogitData$eda <=65 &
                               allLogitData$niv_ins <=4, ]

summary(allLogitData)

# Varible dicotomica de desempleo 

allLogitData$clase2[allLogitData$clase2 == 1] <- 0 # No desempleados
allLogitData$clase2[allLogitData$clase2 == 2 | allLogitData$clase2 == 3] <- 1 # Desempleados abiertos

summary(allLogitData)

# Logistic regression

allLogitData$clase2 <- factor(allLogitData$clase2)

mylogit <- glm(clase2 ~ sex + eda + niv_ins + cve_ent + cve_mun, data = allLogitData, family = binomial)

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

# MODELOS ENOE 2020.1, 2020.2, 2020.3

# Conexión con MongoDB

ENOEData.DB <- mongo(db="bedu18", collection="data_enoe", url = "mongodb+srv://Henry:3eXoszlAIBpQzGGA@proyectobedu.jr6fz.mongodb.net/test")

AllDataENOE <- ENOEData.DB$find('{}')



# # PRIMER TRIMESTRE 2020

DataENOE120 <- AllDataENOE[AllDataENOE$per == 120, ]

# Características iniciales

names(DataENOE120)

str(DataENOE120)

head(DataENOE120)

summary(DataENOE120)


# Omision de variables 

DataENOE120 <- DataENOE120[DataENOE120$clase2 <= 3 & 
                             DataENOE120$clase2 != 0 & 
                             DataENOE120$eda >= 15 & 
                             DataENOE120$eda <= 65 &
                             DataENOE120$niv_ins <= 4, ]

summary(DataENOE120)

# Varible dicotomica de desempleo 

DataENOE120$clase2[DataENOE120$clase2 == 1] <- 0 # No desempleados

DataENOE120$clase2[DataENOE120$clase2 == 2 | DataENOE120$clase2 == 3] <- 1 # Desempleados abiertos

# Variable dicotomica sexo

DataENOE120$sex[DataENOE120$sex == 1] <- 0 # Hombre

DataENOE120$sex[DataENOE120$sex == 2] <- 1 # Mujer


# Variable categórica

DataENOE120$niv_ins <- factor(DataENOE120$niv_ins)


# Logistic regression

mylogit120 <- glm(clase2 ~ sex + eda + niv_ins, data = DataENOE120, family = "binomial")

summary(mylogit120)

# Prueba de Wald: Para saber el efecto de la variable categórica

wald.test(b = coef(mylogit120), Sigma = vcov(mylogit120), Terms = 4:6)

    # H0: El efecto de la variable categórica no es estadísticamente significativo
    # Resultado: Pvalue< 0.05, por tanto, se rechaza H0.

# Radios de probabilidad e intervalos de confianza al 95%

exp(cbind(OR = coef(mylogit120), confint(mylogit120)))

# Calculo de probabilidades

probmean120 <- with(DataENOE120, data.frame(sex = mean(sex), eda = mean(eda), niv_ins = factor(1:4)))

probmean120$niv_insP <- predict(mylogit120, newdata = probmean120, type = "response")

probmean120

mean(probmean120$niv_insP)
    # Probabilidad de estar desempleado a nivel nacional: 0.1117514

probdec120 <- with(DataENOE120, data.frame(sex = mean(sex), eda = rep(seq(from = 15, to = 65, length.out = 10),
                                              4), niv_ins = factor(rep(1:4, each = 10))))

probdec120n <- cbind(probdec120, predict(mylogit120, newdata = probdec120, type = "link",
                                    se = TRUE))
probdec120n<- within(probdec120n, {
  PredictedProb <- plogis(fit)
  LL <- plogis(fit - (1.96 * se.fit))
  UL <- plogis(fit + (1.96 * se.fit))
})

probdec120n

# Gráfica de probabilidades 

ggplot(probdec120n, aes(x = eda, y = PredictedProb))+ ggtitle("Desempleo abierto 2020.1") + geom_ribbon(aes(ymin = LL, 
      ymax = UL, fill = niv_ins), alpha = 0.2) + geom_line(aes(colour = niv_ins), size = 1)

# Prueba de ajuste del modelo

with(mylogit120, null.deviance - deviance)

with(mylogit120, df.null - df.residual)

with(mylogit120, pchisq(null.deviance - deviance, df.null - df.residual, lower.tail = FALSE))
    # Ho: Linear regression better than logistic regression
    # pvalue: 0, se recha la hipótesis nula


# # SEGUNDO TRIMESTRE 2020

DataENOE220 <- AllDataENOE[AllDataENOE$per == 220, ]

# Características iniciales

names(DataENOE220)

str(DataENOE220)

head(DataENOE220)

summary(DataENOE220)


# Omision de variables 

DataENOE220 <- DataENOE220[DataENOE220$clase2 <= 3 & 
                             DataENOE220$clase2 != 0 & 
                             DataENOE220$eda >= 15 & 
                             DataENOE220$eda <= 65 &
                             DataENOE220$niv_ins <= 4, ]

summary(DataENOE220)

# Varible dicotomica de desempleo 

DataENOE220$clase2[DataENOE220$clase2 == 1] <- 0 # No desempleados

DataENOE220$clase2[DataENOE220$clase2 == 2 | DataENOE220$clase2 == 3] <- 1 # Desempleados abiertos

# Variable dicotomica sexo

DataENOE220$sex[DataENOE220$sex == 1] <- 0 # Hombre

DataENOE220$sex[DataENOE220$sex == 2] <- 1 # Mujer


# Variable categórica

DataENOE220$niv_ins <- factor(DataENOE220$niv_ins)


# Logistic regression

mylogit220 <- glm(clase2 ~ sex + eda + niv_ins, data = DataENOE220, family = "binomial")

summary(mylogit220)

# Prueba de Wald: Para saber el efecto de la variable categórica

wald.test(b = coef(mylogit220), Sigma = vcov(mylogit120), Terms = 4:6)

    # H0: El efecto de la variable categórica no es estadísticamente significativo
    # Resultado: Pvalue< 0.05, por tanto, se rechaza H0.

# Radios de probabilidad e intervalos de confianza al 95%

exp(cbind(OR = coef(mylogit220), confint(mylogit220)))

# Calculo de probabilidades

probmean220 <- with(DataENOE220, data.frame(sex = mean(sex), eda = mean(eda), niv_ins = factor(1:4)))

probmean220$niv_insP <- predict(mylogit220, newdata = probmean220, type = "response")

probmean220

mean(probmean220$niv_insP)
    # Probabilidad de estar desempleado a nivel nacional: 0.3803087

probdec220 <- with(DataENOE220, data.frame(sex = mean(sex), eda = rep(seq(from = 15, to = 65, length.out = 10),
                                                                      4), niv_ins = factor(rep(1:4, each = 10))))

probdec220n <- cbind(probdec220, predict(mylogit220, newdata = probdec220, type = "link",
                                         se = TRUE))
probdec220n<- within(probdec220n, {
  PredictedProb <- plogis(fit)
  LL <- plogis(fit - (1.96 * se.fit))
  UL <- plogis(fit + (1.96 * se.fit))
})

probdec220n

# Gráfica de probabilidades 

ggplot(probdec220n, aes(x = eda, y = PredictedProb)) + ggtitle("Desempleo abierto 2020.2") + geom_ribbon(aes(ymin = LL, 
                                                                       ymax = UL, fill = niv_ins), alpha = 0.2) + geom_line(aes(colour = niv_ins), size = 1)

# Prueba de ajuste del modelo

with(mylogit220, null.deviance - deviance)

with(mylogit220, df.null - df.residual)

with(mylogit220, pchisq(null.deviance - deviance, df.null - df.residual, lower.tail = FALSE))

    # Ho: Linear regression better than logistic regression
    # pvalue: 0, se recha la hipótesis nula


# # TERCER TRIMESTRE 2020

DataENOE320 <- AllDataENOE[AllDataENOE$per == 320, ]

# Características iniciales

names(DataENOE320)

str(DataENOE320)

head(DataENOE320)

summary(DataENOE320)


# Omision de variables 

DataENOE320 <- DataENOE320[DataENOE320$clase2 <= 3 & 
                             DataENOE320$clase2 != 0 & 
                             DataENOE320$eda >= 15 & 
                             DataENOE320$eda <= 65 &
                             DataENOE320$niv_ins <= 4, ]

summary(DataENOE320)

# Varible dicotomica de desempleo 

DataENOE320$clase2[DataENOE320$clase2 == 1] <- 0 # No desempleados

DataENOE320$clase2[DataENOE320$clase2 == 2 | DataENOE320$clase2 == 3] <- 1 # Desempleados abiertos

# Variable dicotomica sexo

DataENOE320$sex[DataENOE320$sex == 1] <- 0 # Hombre

DataENOE320$sex[DataENOE320$sex == 2] <- 1 # Mujer


# Variable categórica

DataENOE320$niv_ins <- factor(DataENOE320$niv_ins)


# Logistic regression

mylogit320 <- glm(clase2 ~ sex + eda + niv_ins, data = DataENOE320, family = "binomial")

summary(mylogit320)

# Prueba de Wald: Para saber el efecto de la variable categórica

wald.test(b = coef(mylogit320), Sigma = vcov(mylogit320), Terms = 4:6)

  # H0: El efecto de la variable categórica no es estadísticamente significativo
  # Resultado: Pvalue< 0.05, por tanto, se rechaza H0.

# Radios de probabilidad e intervalos de confianza al 95%

exp(cbind(OR = coef(mylogit320), confint(mylogit320)))

# Calculo de probabilidades

probmean320 <- with(DataENOE320, data.frame(sex = mean(sex), eda = mean(eda), niv_ins = factor(1:4)))

probmean320$niv_insP <- predict(mylogit320, newdata = probmean320, type = "response")

probmean320

mean(probmean320$niv_insP)
    # Probabilidad de estar desempleado a nivel nacional: 0.2075618

probdec320 <- with(DataENOE320, data.frame(sex = mean(sex), eda = rep(seq(from = 15, to = 65, length.out = 10),
                                                                      4), niv_ins = factor(rep(1:4, each = 10))))

probdec320n <- cbind(probdec320, predict(mylogit320, newdata = probdec320, type = "link",
                                         se = TRUE))
probdec320n<- within(probdec320n, {
  PredictedProb <- plogis(fit)
  LL <- plogis(fit - (1.96 * se.fit))
  UL <- plogis(fit + (1.96 * se.fit))
})

probdec320n

# Gráfica de probabilidades 

ggplot(probdec320n, aes(x = eda, y = PredictedProb)) + ggtitle("Desempleo abierto 2020.3") + geom_ribbon(aes(ymin = LL, 
                                                                       ymax = UL, fill = niv_ins), alpha = 0.2) + geom_line(aes(colour = niv_ins), size = 1)

# Prueba de ajuste del modelo

with(mylogit320, null.deviance - deviance)

with(mylogit320, df.null - df.residual)

with(mylogit320, pchisq(null.deviance - deviance, df.null - df.residual, lower.tail = FALSE))

    # Ho: Linear regression better than logistic regression
    # Pvalue: 0, se recha la hipótesis nula



# LO QUE FALTA

# Primer trimestre (sin restricciones)
# Segundo y tercer trimestre (con restricciones por COVID)

## Random forest
## Modelo Bayesiano (opcional)

## (Pensamiento al aire) Comparación entre variables antes del COVID (restricciones) y con COVID

#### Dashboard ####
## Indicadores propuestos por municipio (en mapa) de forma descriptiva

## Gráficos modelo "logit"

## Ingresar datos para generar predicción
