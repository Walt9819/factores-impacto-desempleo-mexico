## Main file ##

## Load libraries
suppressWarnings(library(mongolite)) # get data from MongoDB
suppressWarnings(library(dplyr)) # data manipulation
suppressWarnings(library(aod)) # logistic regression odds
suppressWarnings(library(ggplot2)) # graphs
suppressWarnings(library(foreign))
suppressWarnings(library(rjson))
suppressWarnings(library(reshape2))
suppressWarnings(library(tidyr))
suppressWarnings(library(plotly))
suppressWarnings(library(zoo))
suppressWarnings(library(ISLR))

# Datos de conexi?n a MongoDB
url_path = 'mongodb+srv://Henry:3eXoszlAIBpQzGGA@proyectobedu.jr6fz.mongodb.net/test'

# Definici?n de carpeta de trabajo y conexi?n a base de datos MongoDB
path <- "C:/Users/BALAMLAPTOP2/Documents/GitHub/factores-impacto-desempleo-mexico/Project"
setwd(path)

################################# Carga Inicial: Inicio #################################
## ADVERTENCIA: Este carga se debe ejecutar una sola vez, debido a que la informaci?n 
## ya se encuentra disponible en la base de datos MongoDB para fines de este proyecto a 
# fecha corte Diciembre 2020.En caso de hacer pruebas, cambiar url path de conexi?n.
## Ninguna de las variables definidas en esta secci?n se deben considerar para an?lisis 
## posteriores (especialmente data frames).

## Encuesta Nacional de Ocupaci?n y Empleo (ENOE)
# Descargar bases de datos para los trimestres disponibles del a?o 2019 - 2020 en formato DBF.
# Disponibles en: https://www.inegi.org.mx/programas/enoe/15ymas/#Microdatos
# Como primer paso se recomienda extraer todos los conjuntos de datos de los 5 archivos ZIP.
# Lectura de Datos de sociodemogr?fico <SDEM>

# Definici?n de directorio de salida durante el proceso de descomprimir archivos se debe respetar 'enoe_sdem'
outDir <- "C:\\Users\\BALAMLAPTOP2\\Documents\\GitHub\\factores-impacto-desempleo-mexico\\Project\\enoe_sdem"

# Extrae los archivos .dbf de los archivos comprimidos ZIP
for (zfile in list.files(pattern = "*.zip$", recursive = TRUE)) {
  unzip(zfile, exdir = outDir)
}

# Lectura de todos los archivos .dbf en la carpeta del proyecto
rawdata <- lapply(list.files(pattern = "*.dbf$", recursive = TRUE), read.dbf)

# Extracci?n de atributos considerados para el modelo de regresi?n l?gistica y lineal. 
selecteddata <- lapply(rawdata, select, c("ENT", "MUN", "SEX", "EDA", "NIV_INS", "RAMA", "CLASE2", "PER"))

# Construcci?n de data frame y cambio de nombres
data_enoe <- do.call(rbind, selecteddata)
colnames(data_enoe) <- c("cve_ent", "cve_mun", "sex", "eda", "niv_ins", "rama", "clase2", "per")

# Se omiten valores NaN dentro de la base de datos.
data_enoe <- na.omit(data_enoe)

# Se establece una conexi?n a MongoDB y se cargan todos los datos en la colecci?n 'data_enoe'
mongo <- mongo(collection = "data_enoe", db = "bedu18", url = url_path, verbose = TRUE)
mongo$insert(data_enoe)

## Empleados asegurados en el IMSS (API Data M?xico)
# Conexi?n v?a API al modelo de datos Data M?xico de la Secretar?a de Econom?a

# Se construye URL de conexi?n para extraer datos del cubo IMSS con drilldown hasta Municipios y Meses para la m?trica Empleados asegurados
# Solicitud de datos con salida JSON
url_imss <- "https://dev-api.datamexico.org/tesseract/data.jsonrecords?cube=imss&drilldowns=Municipality%2CMonth&measures=Insured+Employment&parents=false&sparse=false"
json_imss <- fromJSON(paste(readLines(url_imss, warn=FALSE), collapse=""))

# Antes de aplicar un do.call con metodo rbind se deben anular las listas (unlist) para extraer los datos del JSON
json_imss <- lapply(json_imss$data, function(x) {
  x[sapply(x, is.null)] <- NA
  unlist(x)
})

# Construcci?n de dataframe y cambio de nombre a las columnas.
data_imss <-as.data.frame(do.call("rbind", json_imss))
colnames(data_imss) <- c("imun", "mun", "idmes", "mes","asegurados")

# Convertir de character a n?merico valores sobre asegurados en el IMSS
data_imss[,5] <- sapply(data_imss[, 5], as.numeric)

# Carga de datos IMSS a MongoDB (Comment: Crear funci?n para subir dataframes a MongoDB, entrada nombre de la colecci?n y dataframe)
mongo <- mongo(collection = "datamx_imss", db = "bedu18", 
               url = url_path, 
               verbose = TRUE)
mongo$insert(data_imss[,c(1:2,4:5)])

## COVID19 (API Data M?xico)
# Se construye URL de conexi?n para extraer datos del cubo gobmx_covid_stats_mun con drilldown 
# hasta municipios y mes para las m?tricas Casos diarios, Muertes diarias y hospitalizados diarios 
# (el agregado aplicado por el cubo OLAP es por defecto SUMA).
url_covid <- "https://api.datamexico.org/tesseract/cubes/gobmx_covid_stats_mun/aggregate.jsonrecords?drilldowns%5B%5D=Geography.Geography.Municipality&drilldowns%5B%5D=Reported+Date.Time.Month&measures%5B%5D=Daily+Cases&measures%5B%5D=Daily+Deaths&measures%5B%5D=Daily+Hospitalized&parents=false&sparse=false"
json_covid <- fromJSON(paste(readLines(url_covid, warn=FALSE), collapse=""))

# Antes de aplicar un do.call con metodo rbind se deben anular las listas (unlist) para extraer los datos del JSON
json_covid <- lapply(json_covid$data, function(x) {
  x[sapply(x, is.null)] <- NA
  unlist(x)
})

# Construcci?n de dataframe y cambio de nombre a las columnas.
data_covid <-as.data.frame(do.call("rbind", json_covid))
colnames(data_covid) <- c("imun", "mun", "idmes", "mes", "casos_diarios", "muertos_diarios", "hospitalizados_diarios")

# Convertir de character a n?merico valores 
data_covid[,5:7] <- sapply(data_covid[, 5:7], as.numeric)

# Carga de datos COVID a MongoDB 
# (Comment: Crear funci?n para subir dataframes a MongoDB, entrada nombre de la colecci?n y dataframe)
mongo <- mongo(collection = "datamx_covid", db = "bedu18", url = url_path, verbose = TRUE)
mongo$insert(data_covid[,c(1:2,4:7)])


## Integraci?n de datos para IMSS y COVID por municipio y fecha
# Para no perder resoluci?n en la calidad de los datos se ejecuta un inner join a la izquierda.
# Se mantienen los valores sin municipio definido con claves en entidad = 9 y municipio = 999. 
# (Omitir del an?lisis en caso de ser requerido aquellos registros con clave del municipio a 999, clave definida desde origen)
data_imss_covid <- merge(data_imss, 
                         data_covid, 
                         by = c("imun", "mes"), all.x = TRUE) 

# Se separa el atributo identificador del mes en anio y mes por convenci?n. Al igual que, el identificador del municipio.
# Se requiere de esa manera para gr?ficos y mapas a incorporar.
data_imss_covid <- data_imss_covid %>% separate(mes, into = c('anio', 'mes'), sep = '-')
data_imss_covid <- data_imss_covid %>% separate(imun, into = c('cve_ent', 'cve_mun'), sep = -3)

# Se filtra el a?o 2020
data_imss_covid <- data_imss_covid[data_imss_covid$anio == '2020', ]

# Se ejecuta un subset para mantener aquellos atributos no repetidos despu?s del merge.
data_imss_covid <- as.data.frame(subset(data_imss_covid, select=c("cve_ent", "cve_mun", "anio", "mes", "asegurados", "casos_diarios", "muertos_diarios", "hospitalizados_diarios")))

# Se asigna cero a aquellas m?tricas que no tienen concidencia despu?s del left join.
# Principalmente para aquellos municipios que empezaron a registrar casos de COVID en mese posteriores
data_imss_covid[is.na(data_imss_covid)] <- 0

# Convertir de character a n?merico valores 
data_imss_covid[,3:8] <- sapply(data_imss_covid[, 3:8], as.numeric)

# Carga de datos COVID a MongoDB 
# (Comment: Crear funci?n para subir dataframes a MongoDB, entrada nombre de la colecci?n y dataframe)
mongo <- mongo(collection = "datamx_imss_covid", db = "bedu18", url = url_path, verbose = TRUE)
mongo$insert(data_imss_covid)

################################# Carga Inicial: Fin #################################

#### Lectura de datos ####
## Conexi?n a MongoDB
covidData.DB <- mongo(db="bedu18", collection="data_covid", url = "mongodb+srv://Henry:3eXoszlAIBpQzGGA@proyectobedu.jr6fz.mongodb.net/test")
logitData.DB <- mongo(db="bedu18", collection="data_logit", url = "mongodb+srv://Henry:3eXoszlAIBpQzGGA@proyectobedu.jr6fz.mongodb.net/test")
imssDatamx.DB <- mongo(db="bedu18", collection="datamx_imss", url = "mongodb+srv://Henry:3eXoszlAIBpQzGGA@proyectobedu.jr6fz.mongodb.net/test")
imssCovid.DB <- mongo(db="bedu18", collection="datamx_imss_covid", url = "mongodb+srv://Henry:3eXoszlAIBpQzGGA@proyectobedu.jr6fz.mongodb.net/test")

## Lectura datos (DENUE, ENOE y COVID)
# Lectura DENUE y COVID
allLogitData <- logitData.DB$find('{}')
allCovidData <- covidData.DB$find('{}')
imssData <- imssDatamx.DB$find('{}')
imssCovid <- imssCovid.DB$find('{}')

# Lectura ENOE (opcional)

## Limpieza de datos
# Seleccionar columnas deseadas
# Transformar datos
# Limpieza general


#### Exploraci?n de datos ####

## Gr?fico `pairs` pairs(iris[,1:4], pch = 19, lower.panel = NULL)

## Heatmap correlaciones atributo vs atributo

#### An?lisis ####
## Modelo "lineal" ENOE para predecir: probabilidad desempleo ~ c(sexo, edad, nivel educativo, sector econ?mico, estado de residencia)

## Modelo "Logit" ENOE para predecir: probabilidad desempleo ~ c(sexo, edad, nivel educativo, sector econ?mico, estado de residencia)
## check results
summary(allLogitData)
sapply(allLogitData, sd)
xtabs(~eda + clase2, data = allLogitData)

# An?lisis exploratorio

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

# Conexi?n con MongoDB

ENOEData.DB <- mongo(db="bedu18", collection="data_enoe", url = "mongodb+srv://Henry:3eXoszlAIBpQzGGA@proyectobedu.jr6fz.mongodb.net/test")

AllDataENOE <- ENOEData.DB$find('{}')



# # PRIMER TRIMESTRE 2020

DataENOE120 <- AllDataENOE[AllDataENOE$per == 120, ]

# Caracter?sticas iniciales

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


# Variable categ?rica

DataENOE120$niv_ins <- factor(DataENOE120$niv_ins)


# Logistic regression

mylogit120 <- glm(clase2 ~ sex + eda + niv_ins, data = DataENOE120, family = "binomial")

summary(mylogit120)

# Prueba de Wald: Para saber el efecto de la variable categ?rica

wald.test(b = coef(mylogit120), Sigma = vcov(mylogit120), Terms = 4:6)

    # H0: El efecto de la variable categ?rica no es estad?sticamente significativo
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

# Gr?fica de probabilidades 

ggplotly(ggplot(probdec120n, aes(x = eda, y = PredictedProb))+ ggtitle("Desempleo abierto 2020.1") + geom_ribbon(aes(ymin = LL, 
      ymax = UL, fill = niv_ins), alpha = 0.2) + geom_line(aes(colour = niv_ins), size = 1))

# Prueba de ajuste del modelo

with(mylogit120, null.deviance - deviance)

with(mylogit120, df.null - df.residual)

with(mylogit120, pchisq(null.deviance - deviance, df.null - df.residual, lower.tail = FALSE))
    # Ho: Linear regression better than logistic regression
    # pvalue: 0, se recha la hip?tesis nula


# # SEGUNDO TRIMESTRE 2020

DataENOE220 <- AllDataENOE[AllDataENOE$per == 220, ]

# Caracter?sticas iniciales

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


# Variable categ?rica

DataENOE220$niv_ins <- factor(DataENOE220$niv_ins)


# Logistic regression

mylogit220 <- glm(clase2 ~ sex + eda + niv_ins, data = DataENOE220, family = "binomial")

summary(mylogit220)

# Prueba de Wald: Para saber el efecto de la variable categ?rica

wald.test(b = coef(mylogit220), Sigma = vcov(mylogit120), Terms = 4:6)

    # H0: El efecto de la variable categ?rica no es estad?sticamente significativo
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

# Gr?fica de probabilidades 

ggplotly(ggplot(probdec220n, aes(x = eda, y = PredictedProb)) + ggtitle("Desempleo abierto 2020.2") + geom_ribbon(aes(ymin = LL, 
                                                                       ymax = UL, fill = niv_ins), alpha = 0.2) + geom_line(aes(colour = niv_ins), size = 1))

# Prueba de ajuste del modelo

with(mylogit220, null.deviance - deviance)

with(mylogit220, df.null - df.residual)

with(mylogit220, pchisq(null.deviance - deviance, df.null - df.residual, lower.tail = FALSE))

    # Ho: Linear regression better than logistic regression
    # pvalue: 0, se recha la hip?tesis nula


# # TERCER TRIMESTRE 2020

DataENOE320 <- AllDataENOE[AllDataENOE$per == 320, ]

# Caracter?sticas iniciales

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


# Variable categ?rica

DataENOE320$niv_ins <- factor(DataENOE320$niv_ins)


# Logistic regression

mylogit320 <- glm(clase2 ~ sex + eda + niv_ins, data = DataENOE320, family = "binomial")

summary(mylogit320)

# Prueba de Wald: Para saber el efecto de la variable categ?rica

wald.test(b = coef(mylogit320), Sigma = vcov(mylogit320), Terms = 4:6)

  # H0: El efecto de la variable categ?rica no es estad?sticamente significativo
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

# Gr?fica de probabilidades 

ggplotly(ggplot(probdec320n, aes(x = eda, y = PredictedProb)) + ggtitle("Desempleo abierto 2020.3") + geom_ribbon(aes(ymin = LL, 
                                                                       ymax = UL, fill = niv_ins), alpha = 0.2) + geom_line(aes(colour = niv_ins), size = 1))

# Prueba de ajuste del modelo

with(mylogit320, null.deviance - deviance)

with(mylogit320, df.null - df.residual)

with(mylogit320, pchisq(null.deviance - deviance, df.null - df.residual, lower.tail = FALSE))

    # Ho: Linear regression better than logistic regression
    # Pvalue: 0, se recha la hip?tesis nula



# LO QUE FALTA

# Primer trimestre (sin restricciones)
# Segundo y tercer trimestre (con restricciones por COVID)

## Random forest
## Modelo Bayesiano (opcional)

## (Pensamiento al aire) Comparaci?n entre variables antes del COVID (restricciones) y con COVID

#### Dashboard ####
## Indicadores propuestos por municipio (en mapa) de forma descriptiva

## Gr?ficos modelo "logit"

## Ingresar datos para generar predicci?n

####  Visualizaciones sobre resultados del modelo y justificar la importancia del proyecto.
# Empleo en México 2019 - 2020
# Asignar formato a la fecha del conjunto de datos IMSS
imssData <- imssData %>% separate(mes, into = c('anio', 'mes'), sep = '-')
imssData$date_month <-as.Date(as.yearmon(paste(imssData$anio, "/", imssData$mes, sep=""), format="%Y/%m"))

# CUIDADO!!! esta librería causa problemas con `dplyr`
#detach(package:plyr) # o llamar explícitamente las funciones de `dplyr`

# Agrupado de los datos por el atributo fecha
data_chart1 <- imssData %>% group_by(date_month) %>% dplyr::summarise(asegurados = sum(asegurados))

# Visualización del empleo en México y su evolución mensual
# Se resalta la mayor caída de empleos registrada en México, ocasionada principalmente por la pandemia COVID-19. 
# Donde la tasa de ocupación entre Febrero y Julio del 2020 cayó % perdiendo mas de X millones de puestos formales como informales.

plot_ly(data = data_chart1, x = ~date_month, y = ~asegurados, mode = 'lines', line = list(color = 'rgb(205, 12, 24)', width = 4)) %>% 
  layout(title = "Empleo en México 2019 - 2020", xaxis = list(title = ""), yaxis = list (title = "Empleados"))


# Daily COVID cases
url_covid_daily <- "https://api.datamexico.org/tesseract/cubes/gobmx_covid_stats_mun/aggregate.jsonrecords?drilldowns%5B%5D=Geography.Geography.Municipality&drilldowns%5B%5D=Reported+Date.Time.Time&measures%5B%5D=Daily+Cases&measures%5B%5D=Daily+Deaths&measures%5B%5D=Daily+Hospitalized&parents=false&sparse=false"
json_covid_daily <- fromJSON(paste(readLines(url_covid_daily, warn=FALSE), collapse=""))

# Antes de aplicar un do.call con metodo rbind se deben anular las listas (unlist) para extraer los datos del JSON
json_covid_daily <- lapply(json_covid_daily$data, function(x) {
  x[sapply(x, is.null)] <- NA
  unlist(x)
})

# Construcci?n de dataframe y cambio de nombre a las columnas.
data_covid_daily <-as.data.frame(do.call("rbind", json_covid))
colnames(data_covid_daily) <- c("imun", "mun", "idmes", "mes", "casos_diarios", "muertos_diarios", "hospitalizados_diarios")

# Convertir de character a n?merico valores 
data_covid_daily[,5:7] <- sapply(data_covid_daily[, 5:7], as.numeric)
data_covid_daily$mes <- as.Date(data_covid_daily$mes) # Convertimos a fechas

summary(data_covid_daily) # Observamos los datos

# Separar fecha
monthCovid.data <- data_covid_daily %>% separate(mes, into = c('year', 'month', 'day'), sep = '-')
monthCovid.data$idmes <- paste(monthCovid.data$year, monthCovid.data$month, sep="")

# Obtener promedio por mes
monthCovid.data <- monthCovid.data %>% dplyr::group_by(imun, idmes) %>% 
                    dplyr::summarise(casos_diarios_prom = mean(casos_diarios), muertos_diarios_prom = mean(muertos_diarios), hospitalizados_diarios_prom = mean(hospitalizados_diarios))

# Juntar con los datos de asegurados
monthCovidImss.data <- merge(imssData %>% select(asegurados, imun, idmes), 
                         monthCovid.data,
                         by = c("imun", "idmes"), all.x = TRUE)

monthCovidImss.data <- monthCovidImss.data %>% dplyr::filter(substr(stri_reverse(imun), 1, 3) != "999") # eliminar municipios desconocidos (terminan con 999)

# Obtener tasa de cambio en el número de empleos
monthCovidImss.data <- monthCovidImss.data %>% dplyr::group_by(imun) %>% dplyr::mutate(tasa_empleabilidad = c(0, diff(asegurados)))
monthCovidImss.data[is.na(monthCovidImss.data)] <- 0 # municipios sin casos reportados hasta ese momento, marcar como 0 casos

monthCovImss.nacional <- monthCovidImss.data %>% dplyr::group_by(idmes) %>% 
          dplyr::summarise(casos = mean(casos_diarios_prom), 
                           tasa = mean(tasa_empleabilidad), 
                           muertes = mean(muertos_diarios_prom),
                           hospitalizados = mean(hospitalizados_diarios_prom)
                           )

monthCovImss.nacional <- monthCovImss.nacional %>% mutate(monYear = as.Date(as.yearmon(idmes, format="%Y%m")))

# Gráfico con tasa de empleabilidad 
plot_ly(data = monthCovImss.nacional, x = ~monYear, y = ~tasa, mode = 'lines', line = list(color = 'rgb(205, 12, 24)', width = 4)) %>% 
  layout(title = "Tasa de empleabilidad en México 2019 - 2020", xaxis = list(title = ""), yaxis = list (title = "Tasa empleabilidad"))


names(monthCovidImss.data)

# Create data for fitting linear model
trainCovImss.data <- monthCovidImss.data %>% dplyr::mutate(mun = as.factor(imun), 
                                                    casos = casos_diarios_prom, 
                                                    muertes = muertos_diarios_prom, 
                                                    hosp = hospitalizados_diarios_prom,
                                                    tasa = tasa_empleabilidad
                                                    ) %>% 
                                            dplyr::select(mun, casos, muertes, hosp, tasa)


pairs(trainCovImss.data[, 2:6], pch = 19, lower.panel = NULL) # data plot

# Working on this
covImss.lm <- lm(tasa ~ . - imun, trainCovImss.data)
