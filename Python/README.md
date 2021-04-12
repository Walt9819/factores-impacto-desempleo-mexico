# Módulo 2. Procesamiento de Datos con Python <br /> *Impacto del confinamineto en el sector restaurantero y de alojamiento en México*

**Equipo 18**

Integrantes:
- Irene López [ver perfil](https://www.linkedin.com/in/irene-lopez-rodriguez/)
- Laura Lozano [ver perfil](https://www.linkedin.com/in/laura-lozano-bautista/)
- Enrique Rodríguez [ver perfil](https://www.linkedin.com/in/enrique-rodriguez97/)
- Walter Rosales [ver perfil]()

## Video

[Video del proyecto](https://youtu.be/n-_0N_6R-FY)

## Identificación del problema
La pandemia de COVID-19 en México trajo consigo desempleo e inactividad, ya sea por incompatibilidad de los horarios personales con el trabajo en línea, recortes administrativos o falta de herramientas tecnológicas. Asimismo, las personas que se mantuvieron ocupadas durante este periodo vieron un cambio en la calidad de su empleo, pues los costos económicos de las medidas sanitarias que las autoridades tomaron se vieron reflejados en la reducción de la oferta laboral (implicando así una mayor carga laboral para los empleados que permanecieron en las empresas), una menor capacidad utilizada y el funcionamiento de las cadenas de valor.

En los Pre-Criterios 2021 de la SHCP se menciona que los sectores que se vieron afectados de manera más inmediata y persistente son los servicios de alojamiento, esparcimiento, comercio, transporte, y aquellos sectores mayormente dependientes del turismo local y foráneo. Por otra parte, el comercio electrónico y los servicios de telecomunicaciones y tecnologías de la información, así como la venta de productos farmacéuticos se vieron favorecidos por el aumento de la demanda de sus productos y servicios. 

Según estimaciones de la Cámara Nacional de la Industrial Restaurantera y de Alimentos Condimentados (CANIRAC), para finales de 2020 se acumuló una pérdida  de 450 mil empleos en la industria restaurantera, de los 2.1 millones que mantenía dicho sector a inicio de año. También prevé que el 50% del sector restaurantero tendrá dificultad para negociar deudas por crédito o pagos de nóminas, debido al impacto que ha provocado el cierre de establecimientos, reducción de horarios y aforo. Entre los estados más afectados por la pérdida de empleo se encuentran: Ciudad de México, Estado de México, Baja California, Chihuahua y Sonora, además, no se descarta que la cifra se incremente en caso de endurecerse las medidas de confinamiento.


## Objetivo
> Analizar la situación del empleo en México durante el 2020 en los sectores de la actividad económica, haciendo especial énfasis en las condiciones laborales del sector restaurantero y de alojamiento.

> Indagar si el tema referente al cierre de restaurantes y hoteles, realmente fue percibido como un problema por la población mexicana, así como el tipo de sentimientos experimentados en la difusión de las noticias.

## Preguntas de investigación
*	¿Cómo son las condiciones laborales de las personas que permanecieron en la ocupación durante la pandemia de COVID-19? 
*	¿Qué pasó con los trabajadores de los sectores más afectados? 
*	¿Cuál fue la condición laboral más precarizada en el sector restaurantero y de alojamiento?
*	¿Qué parte de la estructura productiva fue la más golpeada: micros, pequeñas, medianas o grandes empresas?
*   ¿En qué meses o fechas exactas existieron un mayor número de noticias referentes al cierre de restaurantes y su efecto en la economía mexicana?
*   ¿Cuál fue el sentimiento que predomino en la redacción de las notas periodisticas y si fueron sustentadas con hechos o meramente opiones?

## Bases de datos
Para este proyecto se ocuparon, en primera instancia, se continuó con el uso de la base de datos datos de la Encuesta Nacional de Ocupación y Empleo (ENOE) y la Encuesta Telefónica de Ocupación y Empleo, los casos diarios de COVID-19 consultados con la API DataMéxico correspondientes del 2020, y posteriormente notas periodísticas del periodico de circulación nacional La Jornada.

## Procedimiento

Existen tres fases durante el desarrollo del proyecto para esta fase: 
1. Planteamiento del problema sustentada con datos 
(ENOE) e identificación del sector económico más desprotegido durante la pandemia.
2. Notas periodísticas con esta información se indaga si el tema referente al cierre de restaurantes y diferentes comercios identificados anteriormente, realmente fue percibido como un problema por la población mexicana y que tipo de sentimientos fueron asociados en las noticias. 
3. Merge Casos diarios COVID-19 - Métricas sobre sentimientos en Notas periodísticas, este permite corroborar como el aumento de casos diarios y cambios en el semaforo COVID, aumentaban el número de noticias en periodicos nacionales.

## Problemas identificados en las bases de datos
> **NaN**: Se encontraron más de 14 mil datos con NaN en las variables de edad y sexo, por lo cual, esos registros fueron eliminados.

> **Tipo de dato**: Solamente dos variables estaban en un formato erróneo, sexo, la cual es una variable categórica, y edad, que está en números enteros, así que pasaron de float a integer.

> **Filtros**: Dentro de las variables se encuentran disponibles diversas categorías que llegan a no ser de interés para el fin del proyecto, por lo que se realizan filtros para obtener los datos con los que finalmente se trabajará.

> **Diccionarios**: Dentro de los datos con los que contábamos, el formato con el que se representaban las categorías es de tipo numérico, por lo que con base en el diccionario de las categorías encontrado en el sitio donde se descargaron los datos, se añadieron al dataframe la equivalencia de las categorías de cada variable.

> **Fechas**: Las fechas utilizadas en las notas periodísticas no fueron reconocidas durante el formateo del datetime, se optó por la opción rápida que fue cambiar los textos de los meses al inglés para que se transformarán las fechas al formato deseado.

> **Stopwords**: En este caso, se tuvo que alimentar varias veces el vector de stopwords para omitir palabras en las nubes que no generaban un significado al análisis. 

# **Trabajo a futuro**
* Nuestro equipo busca generar un algoritmo que determine la ubicación más estratégica para cualquier tipo de establecimiento, principalmente restaurantes y comercios con el fin de maximizar las ganancias comerciales y restablecer la economía de México post-pandemia. 

* Comúnmente los algoritmos con objetivos similares dependen de factores como: demanda, calidad, competencia, ingresos de la zona, comercios colindantes y movilidad urbana. Sin tomar en cuenta información que se puede extraer de las notas periodísticas como son las entidades mencionadas: lugares, personas, instituciones, y en el mejor de los casos contestar *¿qué vecindario parecería ser la ubicación óptima y más estratégica para las operaciones comerciales de restaurantes y hoteles dadas las noticias publicadas en años o meses anteriores?*
 
* Nuestro fundamento de razonamiento se basa en el poder adquisitivo, las regiones más afectadas durante la pandemia debido al cierre de restaurantes y las empresas con mayor éxito reportadas en los periódicos a nivel nacional. **`Sin embargo cabe mencionar que se utilizaran más APIs como Foursquare o Google Places y los extensos datos geográficos y censales de INEGI para robustecer nuestro algoritmo de predicción y apoyo en la toma de decisiones. `**

## Procedimiento adicional: Extracción de notas periodísticas con Scrapy
Scrapy fue utilizado para rastrear el sitio web de La Jornada (se puede actualizar la programación hacia otro periodico) y extraer las noticias de manera eficiente sobre las palabras clave *'restaurantes'* y *'covid'*. Con el objetivo de analizar la conversación sobre unos de los sectores económicos más afectados durante de la pandemia, con este análisis se logró corroborar que efectivamente el cierre y la reapertura fueron temas latentes durante durante el año 2020.  Los pasos que se siguieron para la extracción de las noticias fueron los siguientes:
1. Instalar los módulos scrapy y virtualenv:

> `pip install scrapy`

> `pip install virtualenv`

2. Se crea un ambiente virtual con Python y se activa:

```
$ virtualenv scrapyvenv
$ cd scrapyvenv
$ .\Scripts\activate
```
3. Se crea un proyecto Scrapy:

```
$ scrapy startproject webscrapy
```
4. Se crea una araña o spider:

```
$ cd webscrapy
$ scrapy genspider jornada www.jornada.com.mx
```
5. Al terminar de codificar la araña o en su defecto copiar los códigos establecidos en el siguiente enlace [News Web Scraper](https://github.com/Walt9819/factores-impacto-desempleo-mexico/tree/main/Python/newsscrapper), se ejecuta el siguiente comando para iniciar el proceso de extracción de información:

```
$ scrapy crawl jornada -a Keywords='restaurantes,covid' -o jornada_restaurantes.csv
```

6. El archivo con los datos sin ningún proceso de limpieza (raw) y las noticias traducidas al idioma inglés se encuentran en sus respectivos enlaces: [Raw Data Web Scraper](https://raw.githubusercontent.com/Walt9819/factores-impacto-desempleo-mexico/main/Python/data/jornada_restaurantes.csv) y [Noticias en Inglés](https://raw.githubusercontent.com/Walt9819/factores-impacto-desempleo-mexico/main/Python/data/translated_news.csv)

7. Los principales archivos que requirieron modificaciones o ajustes o agregar código nuevo en el scraper fueron: `jornada.py` y `items.py`.
