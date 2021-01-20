# Postwork Sesion 02
library(dplyr)
# Cambiar directorio de trabajo haciendo uso de ?setwd
setwd("C:/Users/BALAMLAPTOP2/Documents/GitHub/data-analysis-santander/Sesion02/Postworks/")

# Importa los datos de soccer de las temporadas 2017/2018, 2018/2019 y 2019/2020 de la primera división de la liga española a R, 
# los datos los puedes encontrar en el siguiente enlace: https://www.football-data.co.uk/spainm.php

u1718 <- "https://www.football-data.co.uk/mmz4281/1718/SP1.csv"
u1819 <- "https://www.football-data.co.uk/mmz4281/1819/SP1.csv"
u1920 <- "https://www.football-data.co.uk/mmz4281/1920/SP1.csv"

download.file(url = u1718, destfile = "SP1-1718.csv", mode = "wb")
download.file(url = u1819, destfile = "SP1-1819.csv", mode = "wb")
download.file(url = u1920, destfile = "SP1-1920.csv", mode = "wb")

# Importamos los datos a R

rawdata <- lapply(list.files(pattern = "*.csv"), read.csv)

# Obten una mejor idea de las características de los data frames al usar las funciones: str, head, View y summary
# Probar con [[1]] hasta [[3]]

str(rawdata[[1]])
summary(rawdata[[1]])
head(rawdata[[1]])
View(rawdata[[1]])

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

data <- do.call(rbind, mutateddata)

head(data); tail(data)

# Obtener una mejor idea para todos los dataframes integrados con las funciones str, head, view and summary
str(data)
summary(data)
head(data)
View(data)
