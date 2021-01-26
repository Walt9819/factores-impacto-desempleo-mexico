## Main file ##

## Load libraries
suppressWarnings(library(mongolite)) # get data from MongoDB
suppressWarnings(library(dplyr)) # data manipulation

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
## Modelo "lineal" ENUE para predecir: probabilidad desempleo ~ c(sexo, edad, nivel educativo, sector económico, estado de residencia)

## Modelo "Logit" ENUE para predecir: probabilidad desempleo ~ c(sexo, edad, nivel educativo, sector económico, estado de residencia)

# Primer trimestre (sin restricciones)
# Segundo y tercer trimestre (con restricciones por COVID)

## Random forest
## Modelo Bayesiano (opcional)

## (Pensamiento al aire) Comparación entre variables antes del COVID (restricciones) y con COVID

#### Dashboard ####
## Indicadores propuestos por municipio (en mapa) de forma descriptiva

## Gráficos modelo "logit"

## Ingresar datos para generar predicción
