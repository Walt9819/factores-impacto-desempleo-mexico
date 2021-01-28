
# Postwork Sesion 05
library(dplyr)

# Cambiar directorio de trabajo haciendo uso de ?setwd
setwd("C:/Users/data-analysis-santander/Sesion02/Postworks/")

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
tail(rawdata[[1]])
View(rawdata[[1]])

# Con la función select del paquete dplyr selecciona únicamente las columnas Date, HomeTeam, AwayTeam, FTHG, FTAG y FTR; 
# esto para cada uno de los data frames. (Hint: también puedes usar lapply).

selecteddata <- lapply(rawdata, select, Date, HomeTeam:FTR) #Ya que sólo el file u1920 tiene la variable "Time"


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
min(data$Date); max(data$Date)

# Obtener una mejor idea para todos los dataframes integrados con las funciones str, head, view and summary
str(data)
summary(data)
head(data)
View(data)

# A partir del conjunto de datos de soccer de la liga española de las temporadas 2017/2018, 2018/2019 y 2019/2020, crea el data frame 
# SmallData, que contenga las columnas date, home.team, home.score, away.team y away.score; esto lo puede hacer con ayuda de la función 
# select del paquete dplyr. Luego establece un directorio de trabajo y con ayuda de la función write.csv guarda el data frame como un 
# archivo csv con nombre soccer.csv. Puedes colocar como argumento row.names = FALSE en write.csv.

smallData <- select(data, date = Date, home.team = HomeTeam, 
                    home.score = FTHG, away.team = AwayTeam, 
                    away.score = FTAG)
head(smallData); tail(smallData)
write.csv(x = smallData, file = "soccer.csv", row.names = FALSE)

# Con la función create.fbRanks.dataframes del paquete fbRanks importe el archivo soccer.csv a R y al mismo tiempo asignelo a una 
# variable llamada listasoccer. Se creará una lista con los elementos scores y teams que son data frames listos para la función 
# rank.teams. Asigna estos data frames a variables llamadas anotaciones y equipos.

library(fbRanks)

listasoccer <- create.fbRanks.dataframes(scores.file = "soccer.csv")
anotaciones <- listasoccer$scores
equipos     <- listasoccer$teams

# Con ayuda de la función unique crea un vector de fechas (fecha) que no se repitan y que correspondan a las fechas en las que se jugaron
# partidos. Crea una variable llamada n que contenga el número de fechas diferentes. Posteriormente, con la función rank.teams y usando 
# como argumentos los data frames anotaciones y equipos, crea un ranking de equipos usando unicamente datos desde la fecha inicial y 
# hasta la penúltima fecha en la que se jugaron partidos, estas fechas las deberá especificar en max.date y min.date. Guarda los 
# resultados con el nombre ranking.

fecha   <- unique(anotaciones$date)
n       <- length(fecha)
ranking <- rank.teams(scores = anotaciones, teams = equipos, max.date = fecha[n-1], min.date = fecha[1])

# Finalmente estima las probabilidades de los eventos, el equipo de casa gana, el equipo visitante gana o el resultado es un empate para 
# los partidos que se jugaron en la última fecha del vector de fechas fecha. Esto lo puedes hacer con ayuda de la función predict y 
# usando como argumentos ranking y fecha[n] que deberá especificar en date.

pred <- predict(ranking, date = fecha[n])
## Model based on data from 2017-08-18 to 2020-12-22
## ---------------------------------------------
## 2020-12-23 Leganes vs Sevilla, HW 23%, AW 50%, T 27%, pred score 0.8-1.4  actual: T (1-1)
## 2020-12-23 Valencia vs Huesca, HW 57%, AW 20%, T 23%, pred score 1.8-1  actual: HW (2-1)
## 2020-12-23 Vallecano vs Levante, HW 26%, AW 51%, T 22%, pred score 1.3-1.9  actual: HW (2-1)
