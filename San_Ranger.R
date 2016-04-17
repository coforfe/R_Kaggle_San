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
#end_col <- all_cols


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
#---------------------- RANGER (randomForest)
#---------------------------------
#setwd("~/Downloads")

a <- Sys.time();a
set.seed(5789)

bootControl <- trainControl(number = 50,
                            summaryFunction = twoClassSummary,
                            classProbs = TRUE,
                            verboseIter = FALSE)

#rfGrid <- expand.grid(mtry = seq(29,33,1))
rfGrid <- expand.grid(mtry = 5)

modFitrf <-  train(
  target ~ .,
  data = trainDat,
  trControl = bootControl,
  tuneGrid = rfGrid,
  metric = "ROC",
  method = "ranger",
  num.trees = 500,
  importance = 'impurity',
  respect.unordered.factors = TRUE,
  verbose = TRUE,
  classification = TRUE
)

modFitrf

predrf <- predict( modFitrf, newdata = testDat[,2:ncol(testDat)] )
#ConfusionMatrix
conMatrf <- confusionMatrix(testDat$target, predrf); conMatrf
conMatrfdf <- as.data.frame(conMatrf$overall); 
rfAcc <- conMatrfdf[1,1]; 
rfAcc <- as.character(round(rfAcc*100,2))
# Result should be ROC
rfAcc <- modFitrf$results$ROC
b <- Sys.time();b; b - a

if (nrow(rfGrid) < 2  )  { resampleHist(modFitrf) } else
{plot(modFitrf, as.table = T) }

#Best iteration
modBest <- modFitrf$bestTune; modBest
modBestc <- as.character(modBest)
#Execution time:
modFitrf$times$final[3]
#Samples
samp <- dim(modFitrf$resample)[1] ; samp
numvars <- ncol(trainDat); numvars

#Variable Importance
Imprf <- varImp( modFitrf, scale = F)
plot(Imprf, top = 20)

# #Save trainDat, testDat and Model objects.
# save(
#   trainDat, testDat, modFitrf,
#   file = paste("ranger_",numvars,"vars_rf_n",samp,"_grid",modBestc,"_",rfAcc,"__.RData", sep="")
# )


#------------------------------------------------------------
# RESULTS
#------------------------------------------------------------
# mtry 5 -> 0.8226330

#------------------------------------------------------------
# TEST
#------------------------------------------------------------

#------------------------------------------------------------
# PREDICT
#------------------------------------------------------------
modFit <- modFitrf 
in_err <- rfAcc
modtype <- modFit$method
samptmp <- modFit$resample; samp <- length(unique(samptmp$Resample))
numvars <- length(modFit$coefnames)
timval <- str_replace_all(Sys.time(), " |:", "_")

#Is it "prob" of 'yes' or 'no'....?
pred_SAN <- predict(modFit, newdata = datTestpre, type = "prob")
toSubmit <- data.frame(ID = test$ID, TARGET = pred_SAN$yes)

write.table(toSubmit, file = paste("Res_xxxx_noCorZer_", modtype,"_",numvars,"_n",samp,"_Acc_",in_err,"_", timval,".csv",sep = ""),sep=","
            , row.names = FALSE,col.names = TRUE, quote = FALSE)


#------------------------------------------------------------
# IMPORTANCE
#------------------------------------------------------------
var_imp <- data.frame(var = rownames(Imprf$importance), impor = Imprf$importance$Overall)
var_imp_s <- var_imp[order(var_imp$impor, decreasing = TRUE), ]
top_imp <- as.vector(var_imp_s$var[1:10])

# > top_imp
# [1] "var15"                  "var38"                 
# [3] "saldo_var30"            "saldo_medio_var5_ult3" 
# [5] "saldo_medio_var5_hace2" "saldo_var42"           
# [7] "saldo_medio_var5_hace3" "saldo_medio_var5_ult1" 
# [9] "saldo_var5"             "num_var45_hace3"   
