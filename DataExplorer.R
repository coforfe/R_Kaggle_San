

library(DataExplorer)
GenerateReport(train[,3:ncol(train)],
               output_file = "San_report.html",
               output_dir = getwd(),
               html_document(toc = TRUE, toc_depth = 6, theme = "flatly"))

tmp <- rnorm(100)
tmp[sample(1:100, 10)] <- NA
library(forecast)
# to find optimal lambda
lambda = BoxCox.lambda( tmp )
# now to transform vector
trans.vector = BoxCox( tmp, lambda)

library(MASS)
out <- boxcox(lm(datIn$v6 ~ 1))

