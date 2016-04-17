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


#Feature Engineering....
#High important variables - 10 most improtant - Numeric.
high_imp_var <- c(
   "var15",                  "var38",                 
   "saldo_var30",            "saldo_medio_var5_ult3", 
   "saldo_medio_var5_hace2", "saldo_var42",
   "saldo_medio_var5_hace3", "saldo_medio_var5_ult1",
   "saldo_var5"            , "num_var45_hace3"    
)
# Create combinations of two and calculate ratios
# They will be new columns in the
#comb_high <- as.data.frame(t(combn(high_imp_var, 2)))
how_many <- 4
comb_high_3 <- as.data.frame(t(combn(high_imp_var[1:how_many], 2)))

#----- Train--------
# With just the three best one high importance
ratio_df <- 0
for (j in 1:nrow(comb_high_3)) {
  print(j)
  nam_a  <- as.character(comb_high_3$V1[j])
  nam_b  <- as.character(comb_high_3$V2[j])
  col_a  <- tra[,nam_a]
  col_b  <- tra[,nam_b]
  col_a[col_a == -997] <- NA
  col_b[col_b == -997] <- NA
  col_ab <- col_a / col_b  
  col_ab[is.na(col_ab)] <- -997
  ratio_df <- cbind.data.frame( ratio_df, col_ab) 
}
ratio_df <- ratio_df[, 2:ncol(ratio_df)]
names(ratio_df) <- paste(comb_high_3$V1,comb_high_3$V2, sep = "_y_")

#Add these new "ratio" columns to "tra".
tra_ex <- cbind.data.frame(tra, ratio_df)

#----- Test----------
# With just the three best one high importance
ratio_df <- 0
for (j in 1:nrow(comb_high_3)) {
  print(j)
  nam_a  <- as.character(comb_high_3$V1[j])
  nam_b  <- as.character(comb_high_3$V2[j])
  col_a  <- tes[,nam_a]
  col_b  <- tes[,nam_b]
  col_a[col_a == -997] <- NA
  col_b[col_b == -997] <- NA
  col_ab <- col_a / col_b  
  col_ab[is.na(col_ab)] <- -997
  ratio_df <- cbind.data.frame( ratio_df, col_ab) 
}
ratio_df <- ratio_df[, 2:ncol(ratio_df)]
names(ratio_df) <- paste(comb_high_3$V1,comb_high_3$V2, sep = "_y_")

#Add these new "ratio" columns to "tes".
tes_ex <- cbind.data.frame(tes, ratio_df)


datIn <- tra_ex; datIn <- datIn[, 3:ncol(datIn)]
datIn$target <- as.factor(tra$TARGET)
datIn$target <- as.factor(ifelse(datIn$target == 1, 'yes', 'no'))
datIn <- datIn[, c(ncol(datIn), 1:(ncol(datIn) - 1))]
datTestpre <- tes_ex


# ###Datos de Pawlus (no mejora: ROC: 0.79)
# load("cleanData.RData")
# datIn <- cbind.data.frame(train.y, train)
# names(datIn)[1] <- c('target')
# datIn$target <- as.factor(ifelse(datIn$target == 1, 'yes', 'no'))

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

bootControl <- trainControl(number = 5,
                            summaryFunction = twoClassSummary,
                            classProbs = TRUE,
                            verboseIter = FALSE)

#rfGrid <- expand.grid(mtry = seq(1,4,1))
rfGrid <- expand.grid(mtry = 4)

modFitrf <-  train(
  target ~ .,
  data = trainDat,
  trControl = bootControl,
  tuneGrid = rfGrid,
  metric = "ROC",
  method = "ranger",
  num.trees = 250,
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

file_out <- paste("Res_xxxx_noCorZer_Extended_", how_many,"_", modtype,"_",numvars,"_n",samp,"_Acc_",in_err,"_", timval,".csv",sep = "")
write.table(toSubmit, file = file_out, sep = "," , 
            row.names = FALSE,col.names = TRUE, quote = FALSE)


#------------------------------------------------------------
# IMPORTANCE
#------------------------------------------------------------

