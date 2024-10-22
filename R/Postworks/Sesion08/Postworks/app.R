## app.R ##

library(ggplot2)
library(shiny)
library(shinydashboard)
library(shinythemes)
library(dplyr)

ui <- fluidPage(
    
    dashboardPage(
      
      skin = "purple",
      
      dashboardHeader(title = "Predicción de Resultados"),
      
      dashboardSidebar(
        
        sidebarMenu(
          menuItem("Histograma goles", tabName = "dashboard", icon = icon("dashboard")),
          menuItem("Goles casa - visitante", tabName = "goles", icon = icon("area-chart")),
          menuItem("Datos soccer", tabName = "datos-tabla", icon = icon("table")),
          menuItem("Factores de ganancia", tabName = "momios", icon = icon("file-picture-o"))
          
        )
        
      ),
      
      dashboardBody(
        
        tabItems(
          # Histograma
          tabItem(tabName = "dashboard",
                  fluidRow(
                    titlePanel("Goles por equipo"), 
                    selectInput("x", "Seleccione el valor de X",
                                choices = c("home.score", "away.score")),
                    
                    plotOutput("plot", height = 800, width = 1200)
                  )
          ),
          
          # Goles local - visitante
          tabItem(tabName = "goles", 
                  fluidRow(
                    titlePanel(h3("Probabilidad de goles casa - visitante")),
                    
                    img(src = "figura_1.png",height = 600,width = 800),
                    img(src = "figura_2.png",height = 600,width = 800),
                    img(src = "figura_3.png",height = 600,width = 800)
                  )
          ),

          # Tabla csv
          tabItem(tabName = "datos-tabla",
                  fluidRow(        
                    titlePanel(h3("Datos tabla soccer")),
                    dataTableOutput ("data_table")
                  )
          ), 
          
          # Momios
          tabItem(tabName = "momios",
                  fluidRow(
                    h3("Factor de ganancia Máximo"),
                    img( src = "img1.png",height = 350,width = 600),
                    h3("Factor de ganancia Promedio"),
                    img( src = "img2.png",height = 350,width = 600)
                  )
                  
          )
          
        )
      )
    )
  )

server <- function(input, output) {
  
  # Goles local - visitante
  output$plot <- renderPlot({
    
    data <-  read.csv("soccer.csv", header = T)
    data <- mutate(data, FTR = ifelse(home.score > away.score, "Home Score", ifelse(home.score < away.score, "Away Score", "Draw Score")))
    
    x <- data[,input$x]
    
    data %>% ggplot(aes(x, fill = FTR)) + geom_bar() + facet_wrap("away.team") + labs(x =input$x, y = "Número de Goles") + ylim(0,30)
    
  }) 
  
  #Data
  output$data_table <- renderDataTable(read.csv("soccer.csv", header = T),options = list(lengthMenu = c(15,30,50),pageLength = 15))
  
}


shinyApp(ui, server)
