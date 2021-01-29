# Postwork

# Importa el conjunto de datos match.data.csv a R y realiza lo siguiente:
theURL <- "https://raw.githubusercontent.com/beduExpert/Programacion-con-R-Santander/master/Sesion-06/Postwork/match.data.csv"
  
allData <- read.csv(theURL)
str(allData)
summary(allData)

# Agrega una nueva columna sumagoles que contenga la suma de goles por partido.

sumData <- allData %>% mutate(sumagoles = home.score + away.score)

# Obtén el promedio por mes de la suma de goles.
sumData <- sumData %>% mutate(date = as.Date(date, format="%Y-%m-%d"), monthYear = as.Date(date, format="%Y-%m-%d") %>% strftime("%Y-%m"))
str(sumData)
summary(sumData)

monthMean <- sumData %>% group_by(monthYear) %>% summarise(meanGoals = mean(sumagoles))

# Crea la serie de tiempo del promedio por mes de la suma de goles hasta diciembre de 2019.

View(monthMean) # check data

startDate <- c(strftime(minDate, format="%Y"), strftime(minDate, format = "%m")) %>% as.numeric() # get first date

completeSeries <- seq(min(sumData$date), max(sumData$date) + 10, by="month") # create a series with all months since beginning through end
completeSeries <- data.frame(Date = strftime(completeSeries, format="%Y-%m"), goals=rep(0, length(completeSeries))) # make df with all dates and initialise with 0 goals

# insert goal values for each date with data
for (i in 1:dim(monthMean)[1]) {
  index <- which(completeSeries$Date == monthMean$monthYear[i])
  completeSeries$goals[index] = monthMean$meanGoals[i]
}

# convert to time series
data.withMonths <- ts(completeSeries$goals, st = startDate, fr = 12) # convert series to ts with dates
summary(data.withMonths)

data.asIndex <- ts(monthMean$meanGoals, start = 1, frequency = 10) # convert goal means just with value != 0
summary(data.asIndex)

# Grafica la serie de tiempo.
ts.plot(data.withMonths) # with dates
ts.plot(data.asIndex) # without dates
