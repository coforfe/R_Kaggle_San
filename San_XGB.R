#------------------------------------------------------------
# SANTANDER - Customer Satisfaction - Start: 2016-03-21
#------------------------------------------------------------

setwd("/Volumes/TOSHIBA EXT/Verbatin64/R-cosas/2016-01 - Kaggle/03_Santander")

#Library loading
library(ggplot2) # Data visualization
library(readr) # CSV file I/O, e.g. the read_csv function
library(xgboost)
library(stringr)
library(caret)
library(data.table)
library(bit64)


cat("Read the train and test data\n")
train <- as.data.frame(fread("train.csv", na.string = c('NA', 9999999999)))
test  <- as.data.frame(fread("test.csv" , na.string = c('NA', 9999999999)))
#Reorder column in "train" TARGET is the last column
train <- train[, c(ncol(train), 1:(ncol(train) - 1))]

# train: 76020 x 371
# test: 75818 x 370
# TARGET: 0 - 1 

# #------------------------------------
# # "test" exploration
# library(DataExplorer)
# library(rmarkdown)
# GenerateReport(train[,3:ncol(train)],
#                output_file = "San_report.html",
#                output_dir = getwd(),
#                html_document(toc = TRUE, toc_depth = 6, theme = "flatly"))
# #------------------------------------


cat("Recode NAs to -997\n")
train[is.na(train)]   <- -997
test[is.na(test)]   <- -997


#All numeric, integer and some of them integer64?
cat("Class of each column\n")
cl_col <- as.data.frame(mapply(class, train))
cl_col <- data.frame(var = rownames(cl_col), classe = cl_col[,1])
unique(cl_col$classe)
tes_col <- as.data.frame(mapply(class, test))
tes_col <- data.frame(var = rownames(tes_col), classe = tes_col[,1])
unique(tes_col$classe)


cat("Get feature names\n")
all_cols <- names(train)[c(3:ncol(train))]

cat("Remove highly correlated features or near zero variation\n")
zer_set <- nearZeroVar( test[, 3:ncol(test)], names = TRUE, allowParallel = TRUE )
# 313 columns with nearZeroVar !!
god_col <- setdiff(all_cols, zer_set)

# Correlation among god_col
cor_set <- cor(test[, god_col])
high_cor <- findCorrelation(cor_set, cutoff = 0.95, names = TRUE) 
# 11 columns...
end_col <- setdiff(god_col, high_cor)
# Valid columns ... just 45!!.
end_col <- all_cols


# Clean DataSets
tra <- train[, end_col]
tra <- cbind.data.frame( train[, 1:2], tra)
tes <- test[, end_col]


datIn <- tra; datIn <- datIn[, 3:ncol(datIn)]
datIn$target <- as.factor(tra$TARGET)
datIn$target <- as.factor(ifelse(datIn$target == 1, 'yes', 'no'))
datIn <- datIn[, c(ncol(datIn), 1:(ncol(datIn) - 1))]
datTestpre <- tes

#--------------------------------------------------------
#-------------- READY TO MODEL
#--------------------------------------------------------
#Change set's size just to get something.
sizMod <- 1 * nrow(datIn)
datSamp <- datIn[sample(1:nrow(datIn), sizMod) , ]
#rm(datIn);gc()

inTrain <- createDataPartition(datSamp$target, p = 0.70 , list = FALSE)
trainDat <- datSamp[ inTrain, ]
testDat <- datSamp[ -inTrain, ]

library(doMC)
numCor <- parallel::detectCores() - 2; numCor
#numCor <- 2
registerDoMC(cores = numCor)


#---------------------------------
#---------------------- XGB (new version CARET)
#---------------------------------
#setwd("~/Downloads")

set.seed(6879)
a <- Sys.time();a
bootControl <- trainControl(number = 15,
                            summaryFunction = twoClassSummary,
                            classProbs = TRUE,
                            verboseIter = FALSE)

# bootControl <- trainControl(number=10)

xgbGrid <- expand.grid(
  eta = 0.4,
  max_depth = 11,
  nrounds = 150,
  gamma = 0,
  colsample_bytree = 0.45,
  min_child_weight = 1
)

#nrounds, max_depth, eta, gamma, colsample_bytree, min_child_weight
#eta (0,1) - default: 0.3
#max_depth (1-Inf) - default: 6
#gamma (0-Inf) - default: 0
#min_child_weight (0-Inf) - default: 1
#colsample_bytree (0-1) - default:1

modFitxgb <-  train(
  target ~ .,
  data = trainDat,
  trControl = bootControl,
  tuneGrid = xgbGrid,
  #tuneLength = 3,
  metric = "ROC",
  method = "xgbTree",
  verbose = 1,
  num_class = 2
)

modFitxgb

predxgb <- predict( modFitxgb, newdata = testDat[,2:ncol(testDat)] )
#ConfusionMatrix
conMatxgb <- confusionMatrix(testDat$target, predxgb); conMatxgb
conMatxgbdf <- as.data.frame(conMatxgb$overall); xgbAcc <- conMatxgbdf[1,1]; xgbAcc <- as.character(round(xgbAcc*100,2))
b <- Sys.time();b; b - a

if(nrow(xgbGrid) < 2  )  { resampleHist(modFitxgb) } else
{plot(modFitxgb, as.table = T) }
# resampleHist(modFitxgb) 
plot(modFitxgb, as.tablei = T) 

# #Variable Importance
# Impxgb <- varImp( modFitxgb, scale=F)
# plot(Impxgb, top=20)

#Best iteration
modBest <- modFitxgb$bestTune; modBest
modBestc <- paste(modBest[1],modBest[2],modBest[3], sep = "_")
#Execution time:
modFitxgb$times$final[3]
#Samples
samp <- dim(modFitxgb$resample)[1]
numvars <- ncol(trainDat)
xgbAcc <- round(100*max(modFitxgb$results$Accuracy),2); xgbAcc

#Save trainDat, testDat and Model objects.
format(object.size(modFitxgb), units = "Gb")

save(
  trainDat, testDat, modFitxgb,
  file = paste("XGB_",numvars,"vars_n",samp,"_grid",modBestc,"_",xgbAcc,"__.RData", sep="")
)




#------------------------------------------------------------
# RESULTS
#------------------------------------------------------------

#------------------------------------------------------------
# TEST
#------------------------------------------------------------


#------------------------------------------------------------
# PREDICT
#------------------------------------------------------------
modFit <- modFitxgb
in_err <- xgbAcc
#in_err <- 77.92
modtype <- modFit$method
samptmp <- modFit$resample; samp <- length(unique(samptmp$Resample))
numvars <- length(modFit$coefnames)
timval <- str_replace_all(Sys.time(), " |:", "_")

#Is it "prob" of 'yes' or 'no'....?
pred_SAN <- predict(modFit, newdata = datTestpre, type = "prob")
toSubmit <- data.frame(ID = test$ID, TARGET = pred_SAN$yes)

write.table(toSubmit, file = paste("Res_xxxx_noCorZer_", modtype,"_",numvars,"_n",samp,"_Acc_",in_err,"_", timval,".csv",sep=""),sep=","
            , row.names = FALSE,col.names = TRUE, quote = FALSE)


