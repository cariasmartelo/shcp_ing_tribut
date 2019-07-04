library(seasonal)
setwd('/Users/camiloariasm/Google Drive/laboral/shcp_ing_tribut')

iva = read.csv('inputs/iva_neto_r.csv')
iva = iva$iva_neto_.mdp._r
iva = ts(iva, start=c(2014, 1), end=c(2019,4), frequency = 12)
install.packages("ggfortify")
library(ggplot2)
library(ggfortify)
m <- seas(iva)
autoplot(final(m))
plot(m)
data.frame(m$series)
for (serie in m$series){
  print(length(serie))
  
}
class(AirPassengers)

decomposedRes <- decompose(iva, type="add")
class(iva)