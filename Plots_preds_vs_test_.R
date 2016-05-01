
#library(GGally)
#library(lattice)
all_test <- cbind.data.frame( preds, test)

#ggpairs(all_test)
#splom(~all_test, data = all_test, layout = c(5,5))

#png(file = "Corr_preds_test%d.png", bg = "transparent")
par(mfrow = c(2,2))
for (i in 2:ncol(all_test)) {
#for (i in 2:25) {
  print(i)
  plot(
       all_test[, 1], all_test[,i], 
       main = paste("1 vs ", i , sep = ""),
       pch = 21, cex = 0.5, col = "blue",
       xlab = "1", ylab = "y"
       )
}
