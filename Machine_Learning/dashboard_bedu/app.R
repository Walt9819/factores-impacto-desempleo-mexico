## app.R ##

library(shiny)
library(shinyjs)
library(shinydashboard)
library(shinythemes)
library(dplyr)
library(reshape2)
library(ggplot2)
library(plotly)
library(leaflet)
library(rgdal)

setwd("C:/Users/BALAMLAPTOP2/Documents/GitHub/factores-impacto-desempleo-mexico/Machine_Learning/dashboard_bedu")
mexico <- rgdal::readOGR("data/mun2019gw.shp")

data_km_pca <- read.csv("data/results_kmeans_pca.csv", header = T,encoding = "UTF-8")

linebreaks <- function(n){HTML(strrep(br(), n))}

ui <- fluidPage(
  
  useShinyjs(),
  
  dashboardPage(
#===============================dashboardPage=========================
    
    skin = "purple",
    
    dashboardHeader(title = "Inicio"),

    #===============================
    
    dashboardSidebar(
      
      sidebarMenu(
        menuItem("Inicio", tabName = "insights", icon = icon("area-chart")),
        menuItem("Pruebas", tabName = "pruebas", icon = icon("area-chart"))
      )
      
    ),

    #===============================
    
    dashboardBody(
      #===============================tabItems===============================
      tabItems(
        #===============================tabItem===============================
        tabItem(tabName = "insights",
                fluidRow(
                  box(
                    width = "100%",
                    titlePanel(h3("Recuperación económica post-COVID: Localización óptima de restaurantes en México")),
                    "La pandemia de COVID-19 en México trajo consigo desempleo, precarización de las condiciones laborales y cierre de comercios, fábricas y locales. En los Pre-Criterios 2021 de la SHCP se menciona que los sectores que se vieron afectados de manera más inmediata y persistente son los servicios de alojamiento, esparcimiento, comercio, transporte, y aquellos sectores mayormente dependientes del turismo local y foráneo. Por otra parte, el comercio electrónico y los servicios de telecomunicaciones y tecnologías de la información, así como la venta de productos farmacéuticos se vieron favorecidos por el aumento de la demanda de sus productos y servicios.",linebreaks(2),
                    "Según estimaciones de la Cámara Nacional de la Industrial Restaurantera y de Alimentos Condimentados (CANIRAC), para finales de 2020 se acumuló una pérdida de 450 mil empleos en la industria restaurantera, de los 2.1 millones que mantenía dicho sector a inicio de año. También prevé que el 50% del sector restaurantero tendrá dificultad para negociar deudas por crédito o pagos de nóminas, debido al impacto que ha provocado el cierre de establecimientos, reducción de horarios y aforo. Entre los estados más afectados por la pérdida de empleo se encuentran: Ciudad de México, Estado de México, Baja California, Chihuahua y Sonora, además, no se descarta que la cifra se incremente en caso de endurecerse las medidas de confinamiento.",linebreaks(2),
                    "Dadas las dificultades por las que ha atravesado el sector restaurantero en todo México, es viable pensar en las acciones que se llevarán a cabo durante la recuperación económica. Aquellos empresarios que retrasaron inversiones por la pandemia, o que ya ven un panorama favorable para expandir su negocio, necesitarán una guía para potenciar su alcance y asegurar el éxito de su nuevo restaurante. Por tanto, esta investigación se centra en calcular la localización óptima de un restaurante en México, a partir de datos a nivel municipal del Directorio Estadístico Nacional de Unidades Económicas (DENUE) y la Encuesta Nacional de Ocupación y Empleo (ENOE) del INEGI."
                  ),
                  
                  box(
                    width = "100%",
                    titlePanel(h3("Base de Datos")),
                    dataTableOutput ("data_table")
                  )
                  
                ),
                fluidRow(
                  box(
                    titlePanel(h3("Histograma")),
                    plotOutput("histograma", height = 900),
                    height = 1050
                  ),
                  
                  box(
                    titlePanel(h3("Ingreso laboral, población y restaurantes")),
                    selectInput(inputId = "opciones",label = "Seleccionar Opción",choices = c("Ingreso Laboral"=1,"Población Total"=2,"Restaurantes"=3),selected = "1"),
                    disabled(selectInput(inputId = "opciones2",label = "Seleccionar Opción",choices = c("Comida a la Carta y Corrida"=1,"Antojitos"=2,"Pescados y Mariscos"=3,"Comida Rapida"=4,"Tacos y Tortas"=5,"Autoservicio"=6,"Otros"=7))),
                    plotOutput("graf1", height = 800),
                    height = 1050
                  )
                  
                ),
                fluidRow(
                  box(
                    titlePanel(h3("Índices de Criminalidad")),
                    width = "100%",
                    selectInput(inputId = "opciones3",label = "Seleccionar Opción",choices = c("Amenazas"=1,"Daño de Propiedad"=2,"Extorsión"=3,"Robo de Negocio"=4),selected = "1"),
                    plotOutput("graf2", height = 800),
                    height = 1050
                  )
                  
                ),
                fluidRow(
                  box(
                    title = "Número de casos positivos con COVID-19 por entidad",
                    "Visualización del mapa estatico.", linebreaks(2),
                    "Grupo 0(Color rojo): el número de municipios contenidos corresponde a 2,357, los cuales presentan un bajo número del total de población, por lo mismo tienen baja población económicamente activa y bajo número de empleados. Esto conlleva que el total de establecimientos también sea bajo. Lo único favorable de este grupo es que hay ciertas regiones con altos salarios, pero este grupo dado sus características se convierte en la opción menos deseable para iniciar un negocio.",linebreaks(2),
                    "Grupo 1(Color amarillo): está conformado por 82 municipios con población total, ingreso laboral, población económicamente activa, número de ocupados y total de establecimientos en rangos promedios. Es probable que se exijan alimentos baratos o con precios promedio, y su número de establecimientos también indica que existe mayor competencia culinaria.",linebreaks(2),
                    "Grupo 2(No representados): están agrupados 24 municipios con altos valores en todos sus indicadores sociodemográficos, excluyendo únicamente el ingreso laboral debido a el salario de estas regiones son promedio. Aunque por la gran afluencia de la zona y el número de personas económicamente activas se puede determinar que estos municipios representan las mejores opciones para iniciar un negocio, lo único en contra es la cantidad de establecimientos que representan una competencia.",linebreaks(2),
                    status = "primary",
                    leafletOutput("mapMexico"),
                    #height = 700,
                    width = "100%"
                  )
                ),
        ),
        
        #===============================tabItem===============================
        tabItem(tabName = "pruebas",
                fluidRow(
                  
                )
        )
        
        #===============================tabItem===============================
        
      )
      #===============================tabItems===============================
    )
    
    #===============================
    #===============================dashboardPage=========================
  )
)

server <- function(input, output) {
  #===============================Server=========================
  
  url <- "https://raw.githubusercontent.com/Walt9819/factores-impacto-desempleo-mexico/main/Machine_Learning/data/agrupaciones.csv"
  data <- read.csv(url, header = T,encoding = "UTF-8")
  df = subset(data, select = -c(X) )
  df
  
  #===============================Outputs=========================
  
  output$data_table <- renderDataTable(data,options = list(lengthMenu = c(5,10,15),pageLength = 5))
  
  output$histograma <- renderPlot({
    
    d <- melt(df)
    ggplot(d,aes(x = value)) + 
      facet_wrap(~variable,scales = "free_x", ncol=4) + 
      geom_histogram(bins=10)+
      labs(x = "",y = "")
    
  })
  
  output$graf1 <- renderPlot({
    
    if(input$opciones == 3){
      enable("opciones2")
    }
    else{
      disable("opciones2")
    }
    
    if(input$opciones == 1){
      ggplot(data=df, aes(x=nom_ent,y=ing_lab, group=1)) +
        geom_bar(stat="identity", fill="blue", size=1)+
        labs(x = "Entidades",y = "Ingreso Laboral")+
        theme(text = element_text(size=16),axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
    }
    else if(input$opciones == 2){
      ggplot(data=df, aes(x=nom_ent, y=pobtot, group=1)) +
        geom_bar(stat="identity", fill="blue", size=1)+
        labs(x = "Entidades",y = "Población Total")+
        theme(text = element_text(size=16),axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
    }
    else if(input$opciones == 3){
      
      if(input$opciones2 == 1){
        ggplot(data=df, aes(x=nom_ent, group=1))+
          geom_bar(aes(y = comida_carta_corrida),stat="identity", fill = "red", size=1)+
          labs(x = "Entidades",y = "Número de restaurantes")+
          coord_cartesian(ylim = c(0, 15000))+
          theme(text = element_text(size=16),axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
      }
      else if(input$opciones2 == 2){
        ggplot(data=df, aes(x=nom_ent, group=1))+
          geom_bar(aes(y = antojitos),stat="identity", fill="steelblue", size=1)+
          coord_cartesian(ylim = c(0, 15000))+
          theme(text = element_text(size=16),axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
      }
      else if(input$opciones2 == 3){
        ggplot(data=df, aes(x=nom_ent, group=1))+
          geom_bar(aes(y = pescados_mariscos),stat="identity", fill = "green", size=1)+
          labs(x = "Entidades",y = "Número de restaurantes")+
          coord_cartesian(ylim = c(0, 15000))+
          theme(text = element_text(size=16),axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
      }
      else if(input$opciones2 == 4){
        ggplot(data=df, aes(x=nom_ent, group=1))+
          geom_bar(aes(y = comida_rapida),stat="identity", fill="purple", size=1)+
          labs(x = "Entidades",y = "Número de restaurantes")+
          coord_cartesian(ylim = c(0, 15000))+
          theme(text = element_text(size=16),axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
      }
      else if(input$opciones2 == 5){
        ggplot(data=df, aes(x=nom_ent, group=1))+
          geom_bar(aes(y = tacos_tortas),stat="identity", fill = "orange4", size=1)+
          labs(x = "Entidades",y = "Número de restaurantes")+
          coord_cartesian(ylim = c(0, 15000))+
          theme(text = element_text(size=16),axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
      }
      else if(input$opciones2 == 6){
        ggplot(data=df, aes(x=nom_ent, group=1))+
          geom_bar(aes(y = autoservicio),stat="identity", fill="deeppink4", size=1)+
          labs(x = "Entidades",y = "Número de restaurantes")+
          coord_cartesian(ylim = c(0, 15000))+
          theme(text = element_text(size=16),axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
      }
      else if(input$opciones2 == 7){
        ggplot(data=df, aes(x=nom_ent, group=1))+
          geom_bar(aes(y = otro_tipo_alimentos),stat="identity", fill="goldenrod1", size=1)+
          labs(x = "Entidades",y = "Número de restaurantes")+
          coord_cartesian(ylim = c(0, 15000))+
          theme(text = element_text(size=16),axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
      }
    }
    
  })
  
  output$graf2 <-renderPlot({
    
    if(input$opciones3==1){
      ggplot(data=df, aes(x=nom_ent, group=1))+
        geom_bar(aes(y = amenazas),stat="identity", fill = "red", size=1)+
        labs(x = "Entidades",y = "Número de delitos")+
        coord_cartesian(ylim = c(0, 25000))+
        theme(text = element_text(size=16),axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
    }
    else if(input$opciones3==2){
      ggplot(data=df, aes(x=nom_ent, group=1))+
        geom_bar(aes(y = danio_propiedad),stat="identity", fill = "goldenrod1", size=1)+
        labs(x = "Entidades",y = "Número de delitos")+
        coord_cartesian(ylim = c(0, 25000))+
        theme(text = element_text(size=16),axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
    }
    else if(input$opciones3==3){
      ggplot(data=df, aes(x=nom_ent, group=1))+
        geom_bar(aes(y = extorsion),stat="identity", fill = "orange4", size=1)+
        labs(x = "Entidades",y = "Número de delitos")+
        coord_cartesian(ylim = c(0, 25000))+
        theme(text = element_text(size=16),axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
    }
    else if(input$opciones3==4){
      ggplot(data=df, aes(x=nom_ent, group=1))+
        geom_bar(aes(y = robo_negocio),stat="identity", fill = "steelblue", size=1)+
        labs(x = "Entidades",y = "Número de delitos")+
        coord_cartesian(ylim = c(0, 25000))+
        theme(text = element_text(size=16),axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
    }
    
  })
  
  output$mapMexico <- renderLeaflet({
    #allCovidData$cve_ent <- substr(allCovidData$imun,1,nchar(allCovidData$imun)-3)
    
    #covid_chart1 <- allCovidData %>% group_by(cve_ent) %>% summarise(casos = sum(casos_diarios))
    #covid_chart1 <- na.omit(covid_chart1)
    #covid_chart1$fips <- sprintf("%02d", as.numeric(covid_chart1$cve_ent))
    #covid_chart1$fips <- paste0("MX", covid_chart1$fips)
    data_km_pca$CVEGEO <- as.character(data_km_pca$CVEGEO)
    data_km_pca$CVEGEO <- sprintf("%05d", as.numeric(data_km_pca$CVEGEO))
    
    mapamexico <- merge(mexico, data_km_pca, by = "CVEGEO", all.x = TRUE, duplicateGeoms = TRUE)
    
    mapamexico$label <- as.factor(mapamexico$label)
    
    factpal <- colorFactor(c("#FF4C4C", "#E9E946"), mapamexico$label)
    
    #pal <- colorNumeric("viridis", NULL)
    
    leaflet(mapamexico) %>%
      addTiles() %>%
      addPolygons(stroke = FALSE, smoothFactor = 0.3, fillOpacity = 1, color = ~factpal(label),
                  label = ~paste0(NOM_MUN, ": ", formatC(pobtot, big.mark = ",")))
  })
  
  #===============================Outputs=========================
  
  #===============================Server========================= 
}


shinyApp(ui, server)
