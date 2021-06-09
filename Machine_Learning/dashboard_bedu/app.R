## app.R ##

library(shiny)
library(shinyjs)
library(shinydashboard)
library(shinythemes)
library(dplyr)
library(reshape2)
library(ggplot2)
library(plotly)

ui <- fluidPage(
  
  useShinyjs(),
  
  dashboardPage(
#===============================dashboardPage=========================
    
    skin = "purple",
    
    dashboardHeader(title = "Inicio"),

    #===============================
    
    dashboardSidebar(
      
      sidebarMenu(
        menuItem("Insights", tabName = "insights", icon = icon("area-chart")),
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
                    titlePanel(h3("Análisis Exploratorio de la Base de Datos"))
                  ),
                  
                  box(
                    width = "100%",
                    titlePanel(h3("Datos")),
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
                    titlePanel(h3("Gráfica")),
                    selectInput(inputId = "opciones",label = "Seleccionar Opción",choices = c("Ingreso Laboral"=1,"Población Total"=2,"Restaurantes"=3,"Indice de delitos"),selected = "1"),
                    disabled(selectInput(inputId = "opciones2",label = "Seleccionar Opción",choices = c("Comida a la Carta y Corrida"=1,"Antojitos"=2,"Pescados y Mariscos"=3,"Comida Rapida"=4,"Tacos y Tortas"=5,"Autoservicio"=6,"Otros"=7))),
                    plotOutput("graf1", height = 800),
                    height = 1050
                  )
                  
                )
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
      ggplot(data=df, aes(x=nom_ent, y=ing_lab, group=1)) +
        geom_line(linetype = "dashed", color="blue", size=1)+
        geom_point(color="blue", size=5)+
        labs(x = "Entidades",y = "Ingreso Laboral")+
        theme(text = element_text(size=16),axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
    }
    else if(input$opciones == 2){
      ggplot(data=df, aes(x=nom_ent, y=pobtot, group=1)) +
        geom_line(linetype = "dashed", color="blue", size=1)+
        geom_point(color="blue", size=5)+
        labs(x = "Entidades",y = "Población Total")+
        theme(text = element_text(size=16),axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
    }
    else if(input$opciones == 3){
      
      if(input$opciones2 == 1){
        ggplot(data=df, aes(x=nom_ent, group=1))+
          geom_line(aes(y = comida_carta_corrida), color = "red", size=1)+
          labs(x = "Entidades",y = "Número de restaurantes")+
          coord_cartesian(ylim = c(0, 15000))+
          theme(text = element_text(size=16),axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
      }
      else if(input$opciones2 == 2){
        ggplot(data=df, aes(x=nom_ent, group=1))+
          geom_line(aes(y = antojitos), color="blue", size=1)+
          coord_cartesian(ylim = c(0, 15000))+
          theme(text = element_text(size=16),axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
      }
      else if(input$opciones2 == 3){
        ggplot(data=df, aes(x=nom_ent, group=1))+
          geom_line(aes(y = pescados_mariscos), color = "green", size=1)+
          labs(x = "Entidades",y = "Número de restaurantes")+
          coord_cartesian(ylim = c(0, 15000))+
          theme(text = element_text(size=16),axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
      }
      else if(input$opciones2 == 4){
        ggplot(data=df, aes(x=nom_ent, group=1))+
          geom_line(aes(y = comida_rapida), color="purple", size=1)+
          labs(x = "Entidades",y = "Número de restaurantes")+
          coord_cartesian(ylim = c(0, 15000))+
          theme(text = element_text(size=16),axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
      }
      else if(input$opciones2 == 5){
        ggplot(data=df, aes(x=nom_ent, group=1))+
          geom_line(aes(y = tacos_tortas), color = "orange4", size=1)+
          labs(x = "Entidades",y = "Número de restaurantes")+
          coord_cartesian(ylim = c(0, 15000))+
          theme(text = element_text(size=16),axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
      }
      else if(input$opciones2 == 6){
        ggplot(data=df, aes(x=nom_ent, group=1))+
          geom_line(aes(y = autoservicio), color="deeppink4", size=1)+
          labs(x = "Entidades",y = "Número de restaurantes")+
          coord_cartesian(ylim = c(0, 15000))+
          theme(text = element_text(size=16),axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
      }
      else if(input$opciones2 == 7){
        ggplot(data=df, aes(x=nom_ent, group=1))+
          geom_line(aes(y = otro_tipo_alimentos), color="goldenrod1", size=1)+
          labs(x = "Entidades",y = "Número de restaurantes")+
          coord_cartesian(ylim = c(0, 15000))+
          theme(text = element_text(size=16),axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
      }
    }
    
  })
  
  #===============================Outputs=========================
 
#===============================Server========================= 
}


shinyApp(ui, server)
