# # # POSTWORK 7

# Utilizando el manejador de BDD Mongodb Compass (previamente instalado), deberás de realizar 
# las siguientes acciones:
  
# 1. Alojar el fichero data.csv en una base de datos llamada match_games, nombrando al 
# collection como match

# Llamar biblioteca mongolite

suppressWarnings(library(mongolite))

# Datos de conexi?n a MongoDB
url_path = 'mongodb+srv://Henry:3eXoszlAIBpQzGGA@proyectobedu.jr6fz.mongodb.net/test'

# Definicion de carpeta de trabajo y conexion a base de datos MongoDB
path <- "D:/Data Science/R/Sesion7/"
setwd(path)

# Lectura de archivo .csv

data_p7 <- read.csv("data.csv")

# Carga del archivo .csv a MongoDB

mongo <- mongo(collection = "match", db = "match_games", url = url_path, verbose = TRUE)
mongo$insert(data_p7)


# 2. Una vez hecho esto, realizar un count para conocer el número de registros que se 
# tiene en la base

# Llamado de base de datos

GamesData.DB <- mongo(db="match_games", collection="match", url = "mongodb+srv://Henry:3eXoszlAIBpQzGGA@proyectobedu.jr6fz.mongodb.net/test")

allGamesData <- GamesData.DB$find('{}')

str(allGamesData)

# Numero de registros

nrow(allGamesData)

    # Resultado: 1140

# 3. Realiza una consulta utilizando la sintaxis de Mongodb, en la base de datos para conocer 
# el número de goles que metió el Real Madrid el 20 de diciembre de 2015 y contra que equipo 
# jugó, ¿perdió ó fue goleada?

real_madrid <- GamesData.DB$find('{ "Date" : "2015-12-20", "$or": [ {"HomeTeam" : "Real Madrid"}, {"AwayTeam" : "Real Madrid"}  ]}')

print(real_madrid)

    # No existen registros de juegos del Real Madrid para esa fecha

# 4. Por último, no olvides cerrar la conexión con la BDD

dbClearResult(GamesData.DB)
dbDisconnect(GamesData.DB)
