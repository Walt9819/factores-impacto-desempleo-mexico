# Postwork Sesion 03
library(ggplot2)
# Cambiar directorio de trabajo haciendo uso de ?setwd
setwd("C:/Users/BALAMLAPTOP2/Documents/GitHub/data-analysis-santander/Sesion04/Postworks/")

# Actividad 1
# Importa los datos de soccer de la temporada 2019/2020 de la primera división de la liga española a R, 
# los datos los puedes encontrar en el siguiente enlace: https://www.football-data.co.uk/spainm.php
# football <- read.csv("SP1.csv")
u1718 <- "https://www.football-data.co.uk/mmz4281/1718/SP1.csv"
u1819 <- "https://www.football-data.co.uk/mmz4281/1819/SP1.csv"
u1920 <- "https://www.football-data.co.uk/mmz4281/1920/SP1.csv"

download.file(url = u1718, destfile = "SP1-1718.csv", mode = "wb")
download.file(url = u1819, destfile = "SP1-1819.csv", mode = "wb")
download.file(url = u1920, destfile = "SP1-1920.csv", mode = "wb")

# Importamos los datos a R

rawdata <- lapply(list.files(pattern = "*.csv"), read.csv)


# Con la función select del paquete dplyr selecciona únicamente las columnas Date, HomeTeam, AwayTeam, FTHG, FTAG y FTR; 
# esto para cada uno de los data frames. (Hint: también puedes usar lapply).

selecteddata <- lapply(rawdata, select, Date, HomeTeam:FTR)


# Asegúrate de que los elementos de las columnas correspondientes de los nuevos data frames sean del mismo tipo (Hint 1: usa as.Date 
# y mutate para arreglar las fechas). Con ayuda de la función rbind forma un único data frame que contenga las seis columnas mencionadas 
# en el punto 3 (Hint 2: la función do.call podría ser utilizada).
mutateddata <- lapply(selecteddata, mutate, 
                      Date = as.Date(Date, "%d/%m/%y"),
                      HomeTeam = as.factor(HomeTeam),
                      AwayTeam = as.factor(AwayTeam),
                      FTHG = as.numeric(FTHG), 
                      FTAG = as.numeric(FTAG), 
                      FTR = as.factor(FTR))

football <- do.call(rbind, mutateddata)


# Actividad 2
# Del data frame que resulta de importar los datos a R, extrae las columnas que contienen los números de goles 
# anotados por los equipos que jugaron en casa (FTHG) y los goles anotados por los equipos que jugaron como visitante (FTAG)

# FTHG = Full Time Home Team Goals
# FTAG = Full Time Away Team Goals

# Actividad 3
# Consulta cómo funciona la función table en R al ejecutar en la consola ?table

# Posteriormente elabora tablas de frecuencias relativas para estimar las siguientes probabilidades:

#La probabilidad (marginal) de que el equipo que juega en casa anote x goles (x = 0, 1, 2, ...)
#La probabilidad (marginal) de que el equipo que juega como visitante anote y goles (y = 0, 1, 2, ...)
#La probabilidad (conjunta) de que el equipo que juega en casa anote x goles y el equipo que juega como visitante anote y goles (x = 0, 1, 2, ..., y = 0, 1, 2, ...)

# Calculo de frecuencia relativa por variable
# La frecuencia relativa es una medida estadística que se calcula como el cociente de la frecuencia absoluta de algún valor de la población/muestra (fi) 
# entre el total de valores que componen la población/muestra (N).

# Frecuencia relativa del equipo local
pm.fthg <- table(football$FTHG)/length(football$FTHG)
for (i in 1:length(pm.fthg)){
  print(paste("Frecuencia relativa de que el equipo local que juega en casa anote ", i-1, " goles es igual a ", pm.fthg[i]))
}

# Frecuencia relativa del equipo visitante
pm.ftag <- table(football$FTAG)/length(football$FTAG)
for (i in 1:length(pm.ftag)){
  print(paste("Frecuencia relativa de que el equipo visitante anote ", i-1, " goles es igual a ", pm.ftag[i]))
}

# Se calcula una tabla de probabilidad conjunta (?table), la cual se usa para calcular las probabilidades marginales y condicionales.
data <- football[c('FTHG','FTAG')]

rft <- table(data)/nrow(data)

for (i in 1:dim(rft)[1]){
  for (j in 1:dim(rft)[2]){
    print(paste("Probabilidad conjunta de que el equipo que juega en casa anote ", i-1, " y el equipo que juega como visitante anote ", j-1, " es igual a ", rft[i,j]))
  }
}

# Suma de las probabilidades debe ser igual a 1
sum(rft)

# Para calcular la probabilidad marginal de X = 0 esta dada por
# P(X=0)=P(X=0???Y=0)+P(X=0???Y=1)+P(X=0???Y=2)+P(X=0???Y=3)+...+P(X=0???Y=N)
# Esta es la suma de todos los elementos en la primera fila de tabla de probabilidades y se repite para cada de las filas en la matriz. Lo mismo se repite para Y sumando las columnas. 

# rft <- cbind(rft, px = rowSums(rft))
# rft <- rbind(rft, py = colSums(rft))
# 
# for (i in 1:length(pm.fthg)){
#   print(paste("Probabilidad marginal de que el equipo local que juega en casa anote ", i-1, " goles es igual a ", rft[i,7]))
# }
# 
# for (i in 1:length(pm.ftag)){
#   print(paste("Probabilidad marginal de que el equipo visitante anote ", i-1, " goles es igual a ", rft[8,i]))
# }

pmarg.gc <- rowSums(rft)
pmarg.gv <- colSums(rft)

for (i in 1:length(pm.fthg)){
  print(paste("Probabilidad marginal de que el equipo local que juega en casa anote ", i-1, " goles es igual a ", pmarg.gc[i]))
}

for (i in 1:length(pm.ftag)){
  print(paste("Probabilidad marginal de que el equipo visitante anote ", i-1, " goles es igual a ", pmarg.gv[i]))
}

# Postwork sesion 4
# Ahora investigarás la dependencia o independencia del número de goles anotados por el equipo de casa 
# y el número de goles anotados por el equipo visitante mediante un procedimiento denominado bootstrap, 
# revisa bibliografía en internet para que tengas nociones de este desarrollo.

# Ya hemos estimado las probabilidades conjuntas de que el equipo de casa anote X=x goles (x=0,1,... ,8), 
# y el equipo visitante anote Y=y goles (y=0,1,... ,6), en un partido. Obtén una tabla de cocientes al 
# dividir estas probabilidades conjuntas por el producto de las probabilidades marginales correspondientes.
df_cocientes <- table(data)/nrow(data)
for (i in 1:dim(df_cocientes)[1]){
  for (j in 1:dim(df_cocientes)[2]){
    df_cocientes[i,j] <- df_cocientes[i,j]/(pmarg.gc[i]*pmarg.gv[j])
    # print(paste("val ",val," i ",i-1," j ", j-1))
  }
}

# Mediante un procedimiento de boostrap, obtén más cocientes similares a los obtenidos en la tabla del punto 
# anterior. Esto para tener una idea de las distribuciones de la cual vienen los cocientes en la tabla anterior. 
# Menciona en cuáles casos le parece razonable suponer que los cocientes de la tabla en el punto 1, son iguales 
# a 1 (en tal caso tendríamos independencia de las variables aleatorias X y Y).

B = 999
football.boot = vector(mode="double")
for (i in 1:B+1){
  indices <- sample(dim(football)[1], size = 380, replace = TRUE)
  shfdata <- data[indices, ]
  shfpc <- table(shfdata)/nrow(shfdata)
  shfpm.gc <- rowSums(shfpc)
  shfpm.gv <- colSums(shfpc)
  shfcocientes <- table(shfdata)/nrow(shfdata)
  for (i in 1:dim(shfcocientes)[1]){
    for (j in 1:dim(shfcocientes)[2]){
      shfcocientes[i,j] <- shfcocientes[i,j]/(shfpm.gc[i]*shfpm.gv[j])
    }
  }
  football.boot <- append(football.boot, mean(shfcocientes))
}

par(mfrow=c(1,2))
hist(football.boot, col="slateblue1")
qqnorm(football.boot)
qqline(football.boot)
par(mfrow=c(1,1))
