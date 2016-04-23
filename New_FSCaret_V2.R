
#------------------------------------------------------------
# SANTANDER - Customer Satisfaction - New: 2016_04_20
#------------------------------------------------------------

#setwd("/Volumes/TOSHIBA EXT/Verbatin64/R-cosas/2016-01 - Kaggle/03_Santander")
setwd("/Volumes/PNY64BLUE/Kaggle_San")


library(readr) 
library(fscaret)
library(tictoc)

# Reading the data
dat_train <- read.csv("train.csv", stringsAsFactors = F)
dat_test <- read.csv("test.csv", stringsAsFactors = F)
# dat_train <- read_csv("train.csv")
# dat_test <- read_csv("test.csv")

#save(dat_test, dat_train, file = "dat_Original.RData")
load("dat_Original.RData")

# Let's use fscaret capabilities to select important columns.
# Possible NA ~ -999999
# 1. with -999999 -> works.
# 2. substituting -999999 by median...
# 3. With and without preprocessData (with ->OK / without -> no Metric)

dat_train[dat_train == -999999] <- NA
dat_test[dat_test == -999999] <- NA

set.seed(1234)
splitIndex <- createDataPartition(dat_train$TARGET, p = .75, list = FALSE, times = 1)
trainDF <- dat_train[ splitIndex,]
testDF  <- dat_train[-splitIndex,]

# limit models to use in ensemble and run fscaret
fsModels <- c("adaboost", "gbm", "treebag", "ridge", "lasso", 'C5.0' )
myFS <- fscaret(
                trainDF, testDF, 
                myTimeLimit      = 1800,     preprocessData = TRUE,
                Used.funcRegPred = fsModels, with.labels    = TRUE,
                supress.output   = FALSE,    no.cores       = 3, 
                installReqPckg   = TRUE,     impCalcMet     = "MSE",
                missData         = "meanCol"
               )

# analyze results
print(myFS$VarImp)
print(myFS$PPlabels)

# Best variables
# PreProcess has eliminated a lot of variables leaving just 38 columns
var_num <- rownames(myFS$VarImp$matrixVarImp.MSE)
nam_var <- myFS$PPlabel$Labels[as.numeric(var_num)]

# preprocessData = TRUE - with -999999
# [1] var15                    ind_var30                saldo_var30             
# [4] num_var42                num_var30                ind_var5                
# [7] var36                    var38                    num_var4                
# [10] imp_op_var39_ult1        num_var30_0              ind_var13_0             
# [13] saldo_medio_var5_ult1    num_var22_ult1           ind_var12_0             
# [16] saldo_medio_var5_hace2   imp_op_var39_comer_ult3  num_med_var22_ult3      
# [19] ind_var39_0              num_meses_var39_vig_ult3 saldo_var5              
# [22] num_var12_0              num_var41_0              ind_var43_recib_ult1    
# [25] saldo_medio_var5_hace3   num_var43_recib_ult1     num_var45_hace2         
# [28] saldo_medio_var5_ult3    num_var22_hace2          ind_var43_emit_ult1     
# [31] num_var45_ult1           ind_var37                num_var45_hace3         
# [34] imp_op_var41_comer_ult1  ind_var10_ult1           saldo_var42             
# [37] ID                       num_var22_hace3      

# With preprocessData = FALSE, no importance, due to outliers?...
# Use variables of TRUE

# preprocessData = TRUE - missData = "meanCol"
# [1] var15                    ind_var30                saldo_var30             
# [4] num_var42                num_var30                ind_var5                
# [7] var36                    var38                    num_var4                
# [10] imp_op_var39_ult1        num_var30_0              ind_var13_0             
# [13] saldo_medio_var5_ult1    num_var22_ult1           ind_var12_0             
# [16] imp_op_var39_comer_ult3  ind_var39_0              saldo_medio_var5_hace2  
# [19] num_med_var22_ult3       saldo_var5               num_meses_var39_vig_ult3
# [22] saldo_medio_var5_ult3    num_var41_0              num_var12_0             
# [25] num_var43_recib_ult1     num_var22_hace2          imp_op_var41_comer_ult1 
# [28] ind_var43_recib_ult1     saldo_medio_var5_hace3   ind_var43_emit_ult1     
# [31] num_var45_ult1           ind_var37                ID                      
# [34] num_var45_hace2          num_var45_hace3          ind_var10_ult1          
# [37] saldo_var42              num_var22_hace3         
# Results are equivalent to the previous case with -999999


#--------------------------------------------------------------------------------
#------------------- SECOND and THIRD WAY TO CHECK IMPORTANCE (Amunategui)
# create caret trainControl object to control the number of cross-validations performed
#---- Step (1)
outcomeName <- 'TARGET'
predictorNames <- setdiff(names(trainDF),outcomeName)

# transform outcome variable to text as this is required in caret for classification 
trainDF[,outcomeName] <- ifelse(trainDF[,outcomeName]==1,'yes','nope')
objControl <- trainControl(method='cv', number=2, returnResamp='none', summaryFunction = twoClassSummary, classProbs = TRUE)

objGBM <- train(trainDF[,predictorNames],  as.factor(trainDF[,outcomeName]),
                method    = 'gbm',
                trControl = objControl,
                metric    = "ROC",
                tuneGrid = expand.grid(n.trees = 5, interaction.depth = 3, shrinkage = 0.1, n.minobsinnode = 1)
)

predictions <- predict(object = objGBM, testDF[,predictorNames], type = 'prob')

#---- Step (2)
GetROC_AUC = function(probs, true_Y){
  # AUC approximation
  # http://stackoverflow.com/questions/4903092/calculate-auc-in-r
  # ty AGS
  probsSort = sort(probs, decreasing = TRUE, index.return = TRUE)
  val = unlist(probsSort$x)
  idx = unlist(probsSort$ix) 
  
  roc_y = true_Y[idx];
  stack_x = cumsum(roc_y == 0)/sum(roc_y == 0)
  stack_y = cumsum(roc_y == 1)/sum(roc_y == 1)   
  
  auc = sum((stack_x[2:length(roc_y)] - stack_x[1:length(roc_y)-1])*stack_y[2:length(roc_y)])
  return(auc)
}

refAUC <- GetROC_AUC(predictions[[2]],testDF[, outcomeName])
print(paste('AUC score:', refAUC))

#---- Step (3)
# Shuffle predictions for variable importance
AUCShuffle <- NULL
shuffletimes <- 500

featuresMeanAUCs <- c()
for (feature in predictorNames) {
  print(feature) 
  featureAUCs <- c()
  shuffledData <- testDF[,predictorNames]
  for (iter in 1:shuffletimes) {
    shuffledData[,feature] <- sample(shuffledData[,feature], length(shuffledData[,feature]))
    predictions <- predict(object = objGBM, shuffledData[,predictorNames], type = 'prob')
    featureAUCs <- c(featureAUCs,GetROC_AUC(predictions[[2]], testDF[,outcomeName]))
  }
  featuresMeanAUCs <- c(featuresMeanAUCs, mean(featureAUCs < refAUC))
}
AUCShuffle <- data.frame('feature' = predictorNames, 'importance' = featuresMeanAUCs)
AUCShuffle <- AUCShuffle[order(AUCShuffle$importance, decreasing = TRUE),]
print(AUCShuffle)

#---- Third Way....
# bonus - great package for fast variable importance
# remove NAs in dat_train
dat_trainNA <- dat_train
my_fun <- function(x) {
  idx_na <- is.na(x)
  x[idx_na] <- mean(x, na.rm = TRUE)
  return(x)
}
# dat_trainMean <- 0
# for (i in 1:ncol(dat_trainNA)) {
#   dat_trainNA[, i] <- my_fun(dat_trainNA[,i])
# }

dat_trainMean <- as.data.frame(apply(dat_trainNA, 2, my_fun  ))

library(mRMRe)
ind <- sapply(dat_trainMean, is.integer)
dat_trainMean[ind] <- lapply(dat_trainMean[ind], as.numeric)
dd <- mRMR.data(data = dat_trainMean)
feats <- mRMR.classic(data = dd, target_indices = c(ncol(dat_trainMean)), feature_count = 100)
variableImportance <- data.frame('importance' = feats@mi_matrix[nrow(feats@mi_matrix),])
variableImportance$feature <- rownames(variableImportance)
row.names(variableImportance) <- NULL
variableImportance <- na.omit(variableImportance)
variableImportance <- variableImportance[order(variableImportance$importance, decreasing = TRUE),]
print(variableImportance)
var_imp_mrmre <- variableImportance$feature[1:50]; var_imp_mrmre
#Very, very fast....!. The Best...

# [1] "var36"                   "var15"                   "ind_var8_0"             
# [4] "num_var8_0"              "imp_op_var39_efect_ult1" "imp_op_var41_efect_ult1"
# [7] "num_var8"                "ind_var8"                "imp_op_var41_ult1"      
# [10] "imp_op_var39_ult1"       "num_meses_var8_ult3"     "num_var22_ult1"         
# [13] "ind_var26_cte"           "ind_var25_cte"           "imp_op_var39_efect_ult3"
# [16] "num_op_var39_efect_ult1" "imp_op_var41_efect_ult3" "num_op_var41_efect_ult1"
# [19] "num_op_var39_efect_ult3" "imp_op_var40_efect_ult3" "num_op_var41_efect_ult3"
# [22] "ind_var25_0"             "ind_var25"               "imp_op_var40_efect_ult1"
# [25] "ind_var26_0"             "ind_var26"               "num_var25_0"            
# [28] "num_var25"               "num_op_var40_efect_ult1" "num_op_var40_efect_ult3"
# [31] "num_var26_0"             "num_var26"               "num_med_var22_ult3"     
# [34] "num_var22_ult3"          "num_reemb_var17_ult1"    "saldo_var40"            
# [37] "num_op_var39_ult1"       "num_op_var41_ult1"       "num_op_var41_ult3"      
# [40] "imp_op_var39_comer_ult1" "ind_var10_ult1"          "num_op_var39_ult3"      
# [43] "imp_op_var41_comer_ult1" "num_var22_hace2"         "ind_var40"              
# [46] "ind_var39"               "num_var40"               "num_var39"              
# [49] "ind_var30_0"             "ind_var1"     
#-------------------  END ---- SECOND and THIRD WAY TO CHECK IMPORTANCE (Amunategui)


#--------------------------------------------------------------------------------
#------------------- USING rfe (CARET)......
# set.seed(1)
# siz_val <- seq(5, 100, 5)
# x <- trainDF[, 1:(ncol(trainDF) - 1)]
# y <- trainDF$TARGET
# testX <- testDF[, 1:(ncol(trainDF) - 1)]
# testY <- testDF$TARGET
# 
# set.seed(1)
# lmProfile <- rfe(x = x, y = y, testX = testX, testY = testY,
#                  sizes = siz_val, 
#                  rfeControl = rfeControl(functions = lmFuncs,  number = 200))
# plot(lmProfile, type = c("o", "g"))
# 
# rfProfile <- rfe(x = x, y = y, testX = testX, testY = testY,
#                  sizes = siz_val, 
#                  rfeControl = rfeControl(functions = rfFuncs))
# plot(rfProfile, type = c("o", "g"))
# 
# bagProfile <- rfe(x = x, y = y, testX = testX, testY = testY,
#                   sizes = siz_val, 
#                   rfeControl = rfeControl(functions = treebagFuncs))
# plot(bagProfile, type = c("o", "g"))
# 
# ldaProfile <- rfe(x = x, y = y, testX = testX, testY = testY,
#                   sizes = siz_val, 
#                   rfeControl = rfeControl(functions = ldaFuncs, method = "cv"))
# plot(ldaProfile, type = c("o", "g"))



# # Mergin the test and train data
# dat_test$TARGET <- NA
# all_dat <- rbind(dat_train, dat_test)
# names_ori <- names(all_dat)

#-------- VARSET A ------------
A_set <- c(
  "var36","var15","ind_var8_0",
  "num_var8_0","imp_op_var39_efect_ult1","imp_op_var41_efect_ult1",
  "num_var8","ind_var8","imp_op_var41_ult1",
  "imp_op_var39_ult1","num_meses_var8_ult3","num_var22_ult1",
  "ind_var26_cte","ind_var25_cte","imp_op_var39_efect_ult3",
  "num_op_var39_efect_ult1","imp_op_var41_efect_ult3","num_op_var41_efect_ult1",
  "num_op_var39_efect_ult3","imp_op_var40_efect_ult3","num_op_var41_efect_ult3",
  "ind_var25_0","ind_var25","imp_op_var40_efect_ult1",
  "ind_var26_0","ind_var26","num_var25_0",
  "num_var25","num_op_var40_efect_ult1","num_op_var40_efect_ult3",
  "num_var26_0","num_var26","num_med_var22_ult3",
  "num_var22_ult3","num_reemb_var17_ult1","saldo_var40",
  "num_op_var39_ult1","num_op_var41_ult1","num_op_var41_ult3",
  "imp_op_var39_comer_ult1","ind_var10_ult1","num_op_var39_ult3",
  "imp_op_var41_comer_ult1","num_var22_hace2","ind_var40",
  "ind_var39","num_var40","num_var39",
  "ind_var30_0","ind_var1" 
)

B_set <- c(
  "var15","ind_var30","saldo_var30",
  "num_var42","num_var30","ind_var5",
  "var36" ,"var38","num_var4",
  "imp_op_var39_ult1","num_var30_0","ind_var13_0",
  "saldo_medio_var5_ult1","num_var22_ult1","ind_var12_0",
  "saldo_medio_var5_hace2","imp_op_var39_comer_ult3","num_med_var22_ult3",
  "ind_var39_0","num_meses_var39_vig_ult3","saldo_var5",
  "num_var12_0","num_var41_0","ind_var43_recib_ult1",
  "saldo_medio_var5_ult1","num_var22_ult1","ind_var12_0",
  "saldo_medio_var5_hace2","imp_op_var39_comer_ult3","num_med_var22_ult3",
  "ind_var39_0","num_meses_var39_vig_ult3","saldo_var5",
  "num_var12_0","num_var41_0","ind_var43_recib_ult1",
  "saldo_medio_var5_hace3","num_var43_recib_ult1,num_var45_hace2",
  "saldo_medio_var5_ult3","num_var22_hace2","ind_var43_emit_ult1",
  "num_var45_ult1","ind_var37","num_var45_hace3",
  "imp_op_var41_comer_ult1","ind_var10_ult1","saldo_var42",
  "num_var22_hace3"
)



#--------------------------------------------------------
#-------------- READY TO MODEL
#--------------------------------------------------------
library(caret)
datIn   <- dat_train
datIn$TARGET <- as.factor(datIn$TARGET)
datIn$TARGET <- ifelse(datIn$TARGET == 1 , "yes", "no")

sizMod  <- 1 * nrow(datIn)
datSamp <- datIn[sample(1:nrow(datIn), sizMod) , ]
#rm(datIn);gc()

inTrain  <- createDataPartition(datIn$TARGET, p = 0.70 , list = FALSE)
trainDat <- datSamp[ inTrain, ]
testDat  <- datSamp[ -inTrain, ]

library(doMC)
numCor <- parallel::detectCores() - 2; numCor
#numCor <- 2
registerDoMC(cores = numCor)


#-------------------------------------
#--------------------  MODELS ----------------->
#-------------------------------------
#----------------------  XGB
# Adaptative Resampling..
tic()
fitControl <- trainControl(method = "adaptive_cv",
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
modXgboost <- train(
                 x = trainDat[, 1:(ncol(trainDat) - 1)],
                 y = trainDat[, ncol(trainDat)],
                 method = "xgbTree",
                 trControl = fitControl,
                 preProc = c("center", "scale"),
                 tuneLength = 3,
                 metric = "ROC")
toc()

modFitxgb <- modXgboost

predxgb <- predict( modFitxgb, newdata = testDat[,1:(ncol(testDat) - 1)] )
#ConfusionMatrix
conMatxgb <- confusionMatrix(testDat$TARGET, predxgb); conMatxgb
conMatxgbdf <- as.data.frame(conMatxgb$overall); 
xgbAcc <- conMatxgbdf[1,1]; 
xgbAcc <- as.character(round(xgbAcc*100,2))
# Result should be ROC
xgbAcc <- max(modFitxgb$results$ROC)

# if (nrow(xgbGrid) < 2  )  { resampleHist(modFitxgb) } else
# {plot(modFitxgb, as.table = T) }
plot(modFitxgb, as.table = T) 

#Best iteration
modBest <- modFitxgb$bestTune; modBest
modBestc <- as.character(modBest)
#Execution time:
modFitxgb$times$final[3]
#Samples
samp <- dim(modFitxgb$resample)[1] ; samp
numvars <- ncol(trainDat); numvars

#Variable Importance
Impxgb <- varImp( modFitxgb, scale = F)
plot(Impxgb, top = 20)

#------------------------------------------------------------
# Predict
#------------------------------------------------------------
library(stringr)
modFit <- modFitxgb 
in_err <- xgbAcc
modtype <- modFit$method
samptmp <- modFit$resample; samp <- length(unique(samptmp$Resample))
numvars <- length(modFit$coefnames)
timval <- str_replace_all(Sys.time(), " |:", "_")

#Is it "prob" of 'yes' or 'no'....?
pred_SAN <- predict(modFit, newdata = dat_test, type = "prob")
toSubmit <- data.frame(ID = dat_test$ID, TARGET = pred_SAN$yes)

file_out <- paste("Res_xxxx_", modtype,"_",numvars,"_n",samp,"_Acc_",in_err,"_", timval,".csv",sep = "")
write.table(toSubmit, file = file_out, sep = "," , 
            row.names = FALSE,col.names = TRUE, quote = FALSE)


#-------------------------------------
#----------------------  GBM
# Adaptative Resampling..
tic()
fitControl <- trainControl(method = "adaptive_cv",
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
modFitgbm <- train(
  x = trainDat[, 1:(ncol(trainDat) - 1)],
  y = trainDat[, ncol(trainDat)],
  method = "gbm",
  trControl = fitControl,
  preProc = c("center", "scale"),
  tuneLength = 3,
  metric = "ROC")
toc()

modFitgbm

predgbm <- predict( modFitgbm, newdata = testDat[,1:(ncol(testDat) - 1)] )
#ConfusionMatrix
conMatgbm <- confusionMatrix(testDat$TARGET, predgbm); conMatgbm
conMatgbmdf <- as.data.frame(conMatgbm$overall); 
gbmAcc <- conMatgbmdf[1,1]; 
gbmAcc <- as.character(round(gbmAcc*100,2))
# Result should be ROC
gbmAcc <- max(modFitgbm$results$ROC)

# if (nrow(gbmGrid) < 2  )  { resampleHist(modFitgbm) } else
# {plot(modFitgbm, as.table = T) }
plot(modFitgbm, as.table = T) 

#Best iteration
modBest <- modFitgbm$bestTune; modBest
modBestc <- as.character(modBest)
#Execution time:
modFitgbm$times$final[3]
#Samples
samp <- dim(modFitgbm$resample)[1] ; samp
numvars <- ncol(trainDat); numvars

#Variable Importance
Impgbm <- varImp( modFitgbm, scale = F)
plot(Impgbm, top = 20)

#------------------------------------------------------------
# Predict
#------------------------------------------------------------
library(stringr)
modFit <- modFitgbm 
in_err <- gbmAcc
modtype <- modFit$method
samptmp <- modFit$resample; samp <- length(unique(samptmp$Resample))
numvars <- length(modFit$coefnames)
timval <- str_replace_all(Sys.time(), " |:", "_")

#Is it "prob" of 'yes' or 'no'....?
pred_SAN <- predict(modFit, newdata = dat_test, type = "prob")
toSubmit <- data.frame(ID = dat_test$ID, TARGET = pred_SAN$yes)

file_out <- paste("Res_xxxx_", modtype,"_",numvars,"_n",samp,"_Acc_",in_err,"_", timval,".csv",sep = "")
write.table(toSubmit, file = file_out, sep = "," , 
            row.names = FALSE,col.names = TRUE, quote = FALSE)



#-------------------------------------
#---------------------- ADABOOST
# Adaptative Resampling..
tic()
fitControl <- trainControl(method = "adaptive_cv",
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
modFitadaboost <- train(
  x = trainDat[, 1:(ncol(trainDat) - 1)],
  y = trainDat[, ncol(trainDat)],
  method = "adaboost",
  trControl = fitControl,
  preProc = c("center", "scale"),
  tuneLength = 3,
  metric = "ROC")
toc()

modFitadaboost

predadaboost <- predict( modFitadaboost, newdata = testDat[,1:(ncol(testDat) - 1)] )
#ConfusionMatrix
conMatadaboost <- confusionMatrix(testDat$TARGET, predadaboost); conMatadaboost
conMatadaboostdf <- as.data.frame(conMatadaboost$overall); 
adaboostAcc <- conMatadaboostdf[1,1]; 
adaboostAcc <- as.character(round(adaboostAcc*100,2))
# Result should be ROC
adaboostAcc <- max(modFitadaboost$results$ROC)

# if (nrow(adaboostGrid) < 2  )  { resampleHist(modFitadaboost) } else
# {plot(modFitadaboost, as.table = T) }
plot(modFitadaboost, as.table = T) 

#Best iteration
modBest <- modFitadaboost$bestTune; modBest
modBestc <- as.character(modBest)
#Execution time:
modFitadaboost$times$final[3]
#Samples
samp <- dim(modFitadaboost$resample)[1] ; samp
numvars <- ncol(trainDat); numvars

#Variable Importance
Impadaboost <- varImp( modFitadaboost, scale = F)
plot(Impadaboost, top = 20)

#------------------------------------------------------------
# Predict
#------------------------------------------------------------
library(stringr)
modFit <- modFitadaboost 
in_err <- adaboostAcc
modtype <- modFit$method
samptmp <- modFit$resample; samp <- length(unique(samptmp$Resample))
numvars <- length(modFit$coefnames)
timval <- str_replace_all(Sys.time(), " |:", "_")

#Is it "prob" of 'yes' or 'no'....?
pred_SAN <- predict(modFit, newdata = dat_test, type = "prob")
toSubmit <- data.frame(ID = dat_test$ID, TARGET = pred_SAN$yes)

file_out <- paste("Res_xxxx_", modtype,"_",numvars,"_n",samp,"_Acc_",in_err,"_", timval,".csv",sep = "")
write.table(toSubmit, file = file_out, sep = "," , 
            row.names = FALSE,col.names = TRUE, quote = FALSE)


#-------------------------------------
#---------------------- C5.0
# Adaptative Resampling..
tic()
fitControl <- trainControl(method = "adaptive_cv",
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
modFitc50 <- train(
  x = trainDat[, 1:(ncol(trainDat) - 1)],
  y = trainDat[, ncol(trainDat)],
  method = "C5.0",
  trControl = fitControl,
  preProc = c("center", "scale"),
  tuneLength = 3,
  metric = "ROC")
toc()

modFitc50

predc50 <- predict( modFitc50, newdata = testDat[,1:(ncol(testDat) - 1)] )
#ConfusionMatrix
conMatc50 <- confusionMatrix(testDat$TARGET, predc50); conMatc50
conMatc50df <- as.data.frame(conMatc50$overall); 
c50Acc <- conMatc50df[1,1]; 
c50Acc <- as.character(round(c50Acc*100,2))
# Result should be ROC
c50Acc <- max(modFitc50$results$ROC)

# if (nrow(c50Grid) < 2  )  { resampleHist(modFitc50) } else
# {plot(modFitc50, as.table = T) }
plot(modFitc50, as.table = T) 

#Best iteration
modBest <- modFitc50$bestTune; modBest
modBestc <- as.character(modBest)
#Execution time:
modFitc50$times$final[3]
#Samples
samp <- dim(modFitc50$resample)[1] ; samp
numvars <- ncol(trainDat); numvars

#Variable Importance
Impc50 <- varImp( modFitc50, scale = F)
plot(Impc50, top = 20)

#------------------------------------------------------------
# Predict
#------------------------------------------------------------
library(stringr)
modFit <- modFitc50 
in_err <- c50Acc
modtype <- modFit$method
samptmp <- modFit$resample; samp <- length(unique(samptmp$Resample))
numvars <- length(modFit$coefnames)
timval <- str_replace_all(Sys.time(), " |:", "_")

#Is it "prob" of 'yes' or 'no'....?
pred_SAN <- predict(modFit, newdata = dat_test, type = "prob")
toSubmit <- data.frame(ID = dat_test$ID, TARGET = pred_SAN$yes)

file_out <- paste("Res_xxxx_", modtype,"_",numvars,"_n",samp,"_Acc_",in_err,"_", timval,".csv",sep = "")
write.table(toSubmit, file = file_out, sep = "," , 
            row.names = FALSE,col.names = TRUE, quote = FALSE)





#-------------------------------------
#----------------------  RRF
# Adaptative Resampling..
tic()
fitControl <- trainControl(method = "adaptive_cv",
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
modFitrrf <- train(
  x = trainDat[, 1:(ncol(trainDat) - 1)],
  y = trainDat[, ncol(trainDat)],
  method = "RRF",
  trControl = fitControl,
  preProc = c("center", "scale"),
  tuneLength = 3,
  metric = "ROC")
toc()

modFitrrf

predrrf <- predict( modFitrrf, newdata = testDat[,1:(ncol(testDat) - 1)] )
#ConfusionMatrix
conMatrrf <- confusionMatrix(testDat$TARGET, predrrf); conMatrrf
conMatrrfdf <- as.data.frame(conMatrrf$overall); 
rrfAcc <- conMatrrfdf[1,1]; 
rrfAcc <- as.character(round(rrfAcc*100,2))
# Result should be ROC
rrfAcc <- max(modFitrrf$results$ROC)

# if (nrow(rrfGrid) < 2  )  { resampleHist(modFitrrf) } else
# {plot(modFitrrf, as.table = T) }
plot(modFitrrf, as.table = T) 

#Best iteration
modBest <- modFitrrf$bestTune; modBest
modBestc <- as.character(modBest)
#Execution time:
modFitrrf$times$final[3]
#Samples
samp <- dim(modFitrrf$resample)[1] ; samp
numvars <- ncol(trainDat); numvars

#Variable Importance
Imprrf <- varImp( modFitrrf, scale = F)
plot(Imprrf, top = 20)

#------------------------------------------------------------
# Predict
#------------------------------------------------------------
library(stringr)
modFit <- modFitrrf 
in_err <- rrfAcc
modtype <- modFit$method
samptmp <- modFit$resample; samp <- length(unique(samptmp$Resample))
numvars <- length(modFit$coefnames)
timval <- str_replace_all(Sys.time(), " |:", "_")

#Is it "prob" of 'yes' or 'no'....?
pred_SAN <- predict(modFit, newdata = dat_test, type = "prob")
toSubmit <- data.frame(ID = dat_test$ID, TARGET = pred_SAN$yes)

file_out <- paste("Res_xxxx_", modtype,"_",numvars,"_n",samp,"_Acc_",in_err,"_", timval,".csv",sep = "")
write.table(toSubmit, file = file_out, sep = "," , 
            row.names = FALSE,col.names = TRUE, quote = FALSE)



#-------------------------------------
#----------------------  blackboost
# Adaptative Resampling..
tic()
fitControl <- trainControl(method = "adaptive_cv",
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
modFitblackboost <- train(
  x = trainDat[, 1:(ncol(trainDat) - 1)],
  y = trainDat[, ncol(trainDat)],
  method = "blackboost",
  trControl = fitControl,
  preProc = c("center", "scale"),
  tuneLength = 3,
  metric = "ROC")
toc()

modFitblackboost

predblackboost <- predict( modFitblackboost, newdata = testDat[,1:(ncol(testDat) - 1)] )
#ConfusionMatrix
conMatblackboost <- confusionMatrix(testDat$TARGET, predblackboost); conMatblackboost
conMatblackboostdf <- as.data.frame(conMatblackboost$overall); 
blackboostAcc <- conMatblackboostdf[1,1]; 
blackboostAcc <- as.character(round(blackboostAcc*100,2))
# Result should be ROC
blackboostAcc <- max(modFitblackboost$results$ROC)

# if (nrow(blackboostGrid) < 2  )  { resampleHist(modFitblackboost) } else
# {plot(modFitblackboost, as.table = T) }
plot(modFitblackboost, as.table = T) 

#Best iteration
modBest <- modFitblackboost$bestTune; modBest
modBestc <- as.character(modBest)
#Execution time:
modFitblackboost$times$final[3]
#Samples
samp <- dim(modFitblackboost$resample)[1] ; samp
numvars <- ncol(trainDat); numvars

#Variable Importance
Impblackboost <- varImp( modFitblackboost, scale = F)
plot(Impblackboost, top = 20)

#------------------------------------------------------------
# Predict
#------------------------------------------------------------
library(stringr)
modFit <- modFitblackboost 
in_err <- blackboostAcc
modtype <- modFit$method
samptmp <- modFit$resample; samp <- length(unique(samptmp$Resample))
numvars <- length(modFit$coefnames)
timval <- str_replace_all(Sys.time(), " |:", "_")

#Is it "prob" of 'yes' or 'no'....?
pred_SAN <- predict(modFit, newdata = dat_test, type = "prob")
toSubmit <- data.frame(ID = dat_test$ID, TARGET = pred_SAN$yes)

file_out <- paste("Res_xxxx_", modtype,"_",numvars,"_n",samp,"_Acc_",in_err,"_", timval,".csv",sep = "")
write.table(toSubmit, file = file_out, sep = "," , 
            row.names = FALSE,col.names = TRUE, quote = FALSE)

