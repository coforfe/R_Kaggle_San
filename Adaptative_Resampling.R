
setwd("/Volumes/TOSHIBA EXT/Verbatin64/R-cosas/2016-01 - Kaggle/03_Santander")

library(caret)
library(doMC)
registerDoMC(3)
library(tictoc)

library(QSARdata)
data(Mutagen)

set.seed(4567)
inTraining <- createDataPartition(Mutagen_Outcome, p = .75, list = FALSE)
training_x <- Mutagen_Dragon[ inTraining,]
training_y <- Mutagen_Outcome[ inTraining]
testing_x  <- Mutagen_Dragon[-inTraining,]
testing_y  <- Mutagen_Outcome[-inTraining]

## Get rid of predictors that are very sparse
nzv <- nearZeroVar(training_x)
training_x <- training_x[, -nzv]
testing_x  <-  testing_x[, -nzv]

tic()
fitControl2 <- trainControl(method = "adaptive_cv",
                            verboseIter = TRUE,
                            allowParallel = TRUE,
                            number = 5,
                            repeats = 5,
                            ## Estimate class probabilities
                            classProbs = TRUE,
                            ## Evaluate performance using 
                            ## the following function
                            summaryFunction = twoClassSummary,
                            ## Adaptive resampling information:
                            adaptive = list(min = 10,
                                            alpha = 0.05,
                                            method = "BT",
                                            complete = TRUE))

set.seed(825)
svmFit2 <- train(x = training_x,
                 y = training_y,
                 method = "xgbTree",
                 trControl = fitControl2,
                 preProc = c("center", "scale"),
                 tuneLength = 4,
                 metric = "ROC")
toc()
