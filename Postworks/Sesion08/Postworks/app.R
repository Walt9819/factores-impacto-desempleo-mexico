## app.R ##

library(ggplot2)
library(shiny)
library(shinydashboard)
library(shinythemes)

ui <- 
  
  fluidPage(
    
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
                    
                    plotOutput("plot1", height = 600, width = 1000)
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
  output$plot1 <- renderPlot({
    
    data <-  read.csv("soccer.csv", header = T)
    data <- mutate(data, FTR = ifelse(home.score > away.score, "Home Score", ifelse(home.score < away.score, "Away Score", "Draw Score")))
    
    x <- data[,input$x]
    
    data %>% ggplot(aes(x, fill = FTR)) + geom_bar() + facet_wrap("away.team") + labs(x =input$x, y = "Número de Goles") + ylim(0,30)
    
  })
  
  # Gráficas
  output$output_momios <- renderPlot({ 
    
    ggplot(mtcars, aes(x =  mtcars[,input$a] , y = mtcars[,input$y],colour = mtcars[,input$z] )) + 
      geom_point() +
      ylab(input$y) +
      xlab(input$x) + 
      theme_linedraw() + 
      facet_grid(input$z)
    
  })   
  
  #Data
  output$data_table <- renderDataTable({data},options = list(aLengthMenu = c(15,30,50),iDisplayLength = 15))
  
}


shinyApp(ui, server)
