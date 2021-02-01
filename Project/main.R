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
suppressWarnings(library(stringi))

# Datos de conexi?n a MongoDB
url_path = 'mongodb+srv://Henry:3eXoszlAIBpQzGGA@proyectobedu.jr6fz.mongodb.net/test'

# Definici?n de carpeta de trabajo y conexi?n a base de datos MongoDB
path <- "C:/Users/BALAMLAPTOP2/Documents/GitHub/factores-impacto-desempleo-mexico/Project"
setwd(path)

################################# Carga Inicial: Inicio #################################
## ADVERTENCIA: Este carga se debe ejecutar una sola vez, debido a que la informaci?n 
## ya se encuentra disponible en la base de datos MongoDB para fines de este proyecto a 
## fecha corte Diciembre 2020.En caso de hacer pruebas, cambiar url path de conexi?n.
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
data_enoe <- data_enoe[data_enoe$clase2 <= 3 & 
                         data_enoe$clase2 != 0 & 
                         data_enoe$eda >= 15 & 
                         data_enoe$eda <= 65 &
                         data_enoe$niv_ins <= 4, ]

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

write.csv(data_enoe, "data_enoe.csv", row.names = FALSE)
write.csv(data_imss, "data_imss.csv", row.names = FALSE)
write.csv(data_covid, "data_covid.csv", row.names = FALSE)


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

write.csv(data_imss_covid, "data_imss_covid.csv", row.names = FALSE)
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


##################################### INICIO : ENOE #####################################

# Conexion con MongoDB

ENOEData.DB <- mongo(db="bedu18", collection="data_enoe", url = "mongodb+srv://Henry:3eXoszlAIBpQzGGA@proyectobedu.jr6fz.mongodb.net/test")

AllDataENOE <- ENOEData.DB$find('{"$and": [{ "clase2" : {"$ne": 0}}, {"clase2": {"$lte": 3}}, {"eda": {"$gte": 15}}, {"eda": {"$lte": 65}}, {"niv_ins": {"$lte": 4}}]}')

# Omision y tratamiento de variables

AllDataENOE <- AllDataENOE[AllDataENOE$clase2 <= 3 & 
                             AllDataENOE$clase2 != 0 & 
                             AllDataENOE$eda >= 15 & 
                             AllDataENOE$eda <= 65 &
                             AllDataENOE$niv_ins <= 4, ]

# Varible dicotomica de desempleo 

AllDataENOE$clase2[AllDataENOE$clase2 == 1] <- 0 # No desempleados

AllDataENOE$clase2[AllDataENOE$clase2 == 2 | AllDataENOE$clase2 == 3] <- 1 # Desempleados abiertos

# Variable dicotomica sexo

AllDataENOE$sex[AllDataENOE$sex == 1] <- 0 # Hombre

AllDataENOE$sex[AllDataENOE$sex == 2] <- 1 # Mujer


# Variable categorica

AllDataENOE$niv_ins <- factor(AllDataENOE$niv_ins)



# # # # # # PRIMER TRIMESTRE 2020 # # # # # # 

DataENOE120 <- AllDataENOE[AllDataENOE$per == 120, ]

# Caracteristicas iniciales

names(DataENOE120)

str(DataENOE120)

head(DataENOE120)

summary(DataENOE120)

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

# Grafica de probabilidades 

ggplotly(ggplot(probdec120n, aes(x = eda, y = PredictedProb))+ ggtitle("Desempleo abierto 2020.1") + geom_ribbon(aes(ymin = LL, 
      ymax = UL, fill = niv_ins), alpha = 0.2) + geom_line(aes(colour = niv_ins), size = 1))

# Prueba de ajuste del modelo

with(mylogit120, null.deviance - deviance)

with(mylogit120, df.null - df.residual)

with(mylogit120, pchisq(null.deviance - deviance, df.null - df.residual, lower.tail = FALSE))
    # Ho: Linear regression better than logistic regression
    # pvalue: 0, se recha la hip?tesis nula


# # # # # # SEGUNDO TRIMESTRE 2020 # # # # # # 

DataENOE220 <- AllDataENOE[AllDataENOE$per == 220, ]

# Caracter?sticas iniciales

names(DataENOE220)

str(DataENOE220)

head(DataENOE220)

summary(DataENOE220)

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

# Grafica de probabilidades 

ggplotly(ggplot(probdec220n, aes(x = eda, y = PredictedProb)) + ggtitle("Desempleo abierto 2020.2") + geom_ribbon(aes(ymin = LL, 
                                                                       ymax = UL, fill = niv_ins), alpha = 0.2) + geom_line(aes(colour = niv_ins), size = 1))

# Prueba de ajuste del modelo

with(mylogit220, null.deviance - deviance)

with(mylogit220, df.null - df.residual)

with(mylogit220, pchisq(null.deviance - deviance, df.null - df.residual, lower.tail = FALSE))

    # Ho: Linear regression better than logistic regression
    # pvalue: 0, se recha la hip?tesis nula


# # # # # # TERCER TRIMESTRE 2020 # # # # # # 

DataENOE320 <- AllDataENOE[AllDataENOE$per == 320, ]

# Caracter?sticas iniciales

names(DataENOE320)

str(DataENOE320)

head(DataENOE320)

summary(DataENOE320)

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

# Grafica de probabilidades 

ggplotly(ggplot(probdec320n, aes(x = eda, y = PredictedProb)) + ggtitle("Desempleo abierto 2020.3") + geom_ribbon(aes(ymin = LL, 
                                                                       ymax = UL, fill = niv_ins), alpha = 0.2) + geom_line(aes(colour = niv_ins), size = 1))

# Prueba de ajuste del modelo

with(mylogit320, null.deviance - deviance)

with(mylogit320, df.null - df.residual)

with(mylogit320, pchisq(null.deviance - deviance, df.null - df.residual, lower.tail = FALSE))

    # Ho: Linear regression better than logistic regression
    # Pvalue: 0, se recha la hip?tesis nula

##################################### FIN : ENOE #####################################


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
imssData <- read.csv("data/data_imss.csv", header = TRUE)
monthCovidImss.data <- merge(imssData %>% select(asegurados, imun, idmes), 
                         monthCovid.data,
                         by = c("imun", "idmes"), all.x = TRUE)

# Falto agregar la libreria stringi 
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
# Marca error con yearmon, se tuvo agregar as.character
# Error: Problem with `mutate()` input `monYear`.
# x character string is not in a standard unambiguous format
# i Input `monYear` is `as.Date(as.yearmon(idmes, format = "%Y%m"))`.
# Run `rlang::last_error()` to see where the error occurred.
monthCovImss.nacional <- monthCovImss.nacional %>% dplyr::mutate(monYear = as.Date(as.yearmon(as.character(idmes), format="%Y%m")))
write.csv(monthCovImss.nacional, "monthcovimss_nacional.csv", row.names = FALSE)
write.csv(monthCovidImss.data, "monthcovidimss_data.csv", row.names = FALSE)

### Este gráfico CONSIDERO que puede ir dentro del dashboard (lástima que acabe en Oct-2020, pero los datos del IMSS no dan más :( )
# Gráfico con tasa de empleabilidad 
ay1 <- list(
  tickfont = list(color = "red"),
  overlaying = "y",
  side = "right",
  title = "Tasa empleabilidad"
)

ay2 <- list(
  tickfont = list(color = "red"),
  overlaying = "y",
  side = "right",
  title = "Promedio casos diarios confirmados COVID-19"
)

fig_imsscovid <- monthCovImss.nacional %>% plot_ly() %>% add_lines(x = ~monYear, y = ~tasa, name='') %>% add_lines(x = ~monYear, y = ~casos, name='', yaxis = "y2") %>% layout(title = "Tasa de empleabilidad por mes", yaxis1 = ay1, yaxis2 = ay2,xaxis = list(title=""))
fig_imsscovid



names(monthCovidImss.data)

# Create data for fitting linear model
trainCovImss.data <- monthCovidImss.data %>% dplyr::mutate(mun = as.factor(imun), 
                                                    casos = casos_diarios_prom, 
                                                    muertes = muertos_diarios_prom, 
                                                    hosp = hospitalizados_diarios_prom,
                                                    tasa = tasa_empleabilidad
                                                    ) %>% 
                                            dplyr::select(mun, casos, muertes, hosp, tasa)


### Ya está como png en `/Project/Pairs_Covid_Imss.png`
#pairs(trainCovImss.data[, 2:6], pch = 19, lower.panel = NULL) # data plot

# Working on this
covImss.lm <- lm(tasa ~ . - imun - mun, trainCovImss.data) # bad results
summary(covImss.lm)
# covImss.lm2 <- lm(tasa ~ cases, trainCovImss.data) # bad R^2

#=============================================================================================

#========================= Datos IMSS =========================

#### Lectura de datos ####
## Conexión a MongoDB
imssData.DB <- mongo(db="bedu18", collection="datamx_imss", url = "mongodb+srv://Henry:3eXoszlAIBpQzGGA@proyectobedu.jr6fz.mongodb.net/test")

## Lectura datos
allImssData <- imssData.DB$find('{}')

head(allImssData)

allImssData <- allImssData %>% separate(mes, into = c('anio', 'mes'), sep = '-')
allImssData$date_month <-as.Date(as.yearmon(paste(allImssData$anio, "/", allImssData$mes, sep=""), format="%Y/%m"))

head(allImssData)

# Plots
allImssData_chart1 <- allImssData %>% group_by(date_month) %>% summarise(asegurados = sum(asegurados))
plot_ly(allImssData_chart1, x = ~date_month, y = ~asegurados, mode = 'lines') %>% layout(title = "Empleo en México", xaxis = list(title = ""), yaxis = list (title = "Empleados"))

#========================= Datos ENOE =========================

#### Lectura de datos ####
## Conexión a MongoDB
enoeData.DB <- mongo(db="bedu18", collection="data_enoe", url = "mongodb+srv://Henry:3eXoszlAIBpQzGGA@proyectobedu.jr6fz.mongodb.net/test")

## Lectura datos
allEnoeData <- enoeData.DB$find('{}')

# Omision de variables 
allEnoeData <- na.omit(allEnoeData)

allEnoeData <- allEnoeData[allEnoeData$clase2 <= 3 & 
                               allEnoeData$clase2 !=0 & 
                               allEnoeData$eda >= 15 & 
                               allEnoeData$eda <=65 &
                               allEnoeData$niv_ins <=4, ]

head(allEnoeData)

#========================= Datos IMSS-COVID =========================

#### Lectura de datos ####
## Conexión a MongoDB
imssData.DB <- mongo(db="bedu18", collection="datamx_imss_covid", url = "mongodb+srv://Henry:3eXoszlAIBpQzGGA@proyectobedu.jr6fz.mongodb.net/test")

## Lectura datos
allImssCovidData <- imssData.DB$find('{}')

head(allImssCovidData)

#========================= Datos COVID =========================

#### Lectura de datos ####
## Conexión a MongoDB
covidData.DB <- mongo(db="bedu18", collection="datamx_covid", url = "mongodb+srv://Henry:3eXoszlAIBpQzGGA@proyectobedu.jr6fz.mongodb.net/test")

## Lectura datos
allCovidData <- covidData.DB$find('{}')

head(allCovidData)

#========================= Graficas =========================

## Enoe

enoe_chart1 <- allEnoeData %>% filter(niv_ins == 4, clase2 == 2 | clase2 == 3, per != 319) %>% group_by(per,sex) %>% count(sex) %>% mutate(per = as.character(per))
enoe_chart1 <- enoe_chart1 %>% mutate(sex = replace(sex,sex==1,'Hombre')) %>% mutate(sex = replace(sex,sex==2,'Mujer'))
enoe_chart1
fig_enoe1 <- enoe_chart1 %>% plot_ly(x = ~per,y = ~n,type = 'bar', split = ~sex)
fig_enoe1

enoe_chart2 <- allEnoeData %>% filter(per != 319) %>% group_by(niv_ins) %>% count(niv_ins)
enoe_chart2 <- enoe_chart2 %>% mutate(niv_ins = replace(niv_ins,niv_ins == '1','Primaria incompleta'))
enoe_chart2 <- enoe_chart2 %>% mutate(niv_ins = replace(niv_ins,niv_ins == '2','Primaria completa'))
enoe_chart2 <- enoe_chart2 %>% mutate(niv_ins = replace(niv_ins,niv_ins == '3','Secundaria completa'))
enoe_chart2 <- enoe_chart2 %>% mutate(niv_ins = replace(niv_ins,niv_ins == '4','Medio superior y superior'))
enoe_chart2
fig_enoe2 <- enoe_chart2 %>% plot_ly(labels = ~niv_ins, values = ~n, type = 'pie')
fig_enoe2

# Checar si se pueden utilizar

enoe_chart3 <- allEnoeData %>% filter(per != 319) %>% mutate(clase2 = replace(clase2,clase2 == 1,'Con empleo')) %>% mutate(clase2 = replace(clase2,clase2 == 2 | clase2 ==3,'Sin empleo'))
enoe_chart3 <- enoe_chart3 %>% group_by(per,niv_ins,clase2) %>% count(clase2) %>% mutate(per = as.character(per))
enoe_chart3
fig_enoe3 <- enoe_chart3 %>% plot_ly(x = ~per,y = ~n,type = 'bar',split = ~clase2)
fig_enoe3
fig_enoe4 <- enoe_chart3 %>% plot_ly(x = ~per,y = ~n,type = 'bar',split = ~clase2, color = ~niv_ins)
fig_enoe4

enoe_chart4 <- allEnoeData %>% select(eda,niv_ins) %>% group_by(niv_ins)
enoe_chart4
fig_enoe5 <- enoe_chart4 %>% plot_ly(x = ~niv_ins, y = ~eda, type = 'box')
fig_enoe5

## ImssCovid

# Usar esta grafica

imsscovid_chart1 <- allImssCovidData %>% group_by(mes) %>% summarise(casos = sum(casos_diarios), asegurados = sum(asegurados))
imsscovid_chart1

ay1 <- list(
  tickfont = list(color = "red"),
  overlaying = "y",
  side = "right",
  title = "casos positivos Covid-19"
)

ay2 <- list(
  tickfont = list(color = "red"),
  overlaying = "y",
  side = "right",
  title = "casos positivos Covid-19"
)

fig_imsscovid1 <- imsscovid_chart1 %>% plot_ly() %>% add_lines(x = ~mes, y = ~asegurados, name='') %>% add_lines(x = ~mes, y = ~casos,name='', yaxis = "y2") %>% layout(title = "Double Y Axis", yaxis1 = ay1, yaxis2 = ay2,xaxis = list(title="x"))
fig_imsscovid1

# imsscovid_chart2 <- allImssCovidData %>% group_by(cve_ent) %>% summarise(casos = sum(casos_diarios))
# imsscovid_chart2

# fig_imsscovid2 <- imsscovid_chart2 %>% plot_ly(type = 'pie', labels = ~cve_ent, values = ~casos)
# fig_imsscovid2

## Covid

# Usar unicamente el data frame

# allCovidData$cve_ent <- substr(allCovidData$imun,1,nchar(allCovidData$imun)-3)

# covid_chart1 <- allCovidData %>% group_by(cve_ent) %>% summarise(casos = sum(casos_diarios))
# covid_chart1 <- na.omit(covid_chart1)
# covid_chart1
# fig_covid1 <- covid_chart1 %>% plot_ly(type = 'pie', labels = ~cve_ent, values = ~casos)
# fig_covid1

