var_setA <- c(
  "var36",
  "var15",
  "ind_var8_0",
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
  "ind_var30_0","ind_var1", "TARGET"
)

#--------------------------------------------------------
#-------------- READY TO MODEL
#--------------------------------------------------------
library(tictoc)
library(caret)
#datIn   <- dat_train[, var_setA]
datIn   <- dat_train
datIn$TARGET <- ifelse(datIn$TARGET == 1 , "yes", "no")
datIn$TARGET <- as.factor(datIn$TARGET)

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

#----------------------  XGB
# Adaptative Resampling..
tic()
fitControl <- trainControl(method = "adaptive_cv",
                           verboseIter = TRUE,
                           allowParallel = TRUE,
                           number = 3,
                           repeats = 3,
                           ## Estimate class probabilities
                           classProbs = TRUE,
                           ## Evaluate performance using
                           ## the following function
                           summaryFunction = twoClassSummary,
                           ## Adaptive resampling information:
                           adaptive = list(min = 5,
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
# PREDICT
#------------------------------------------------------------
library(stringr)
modFit <- modFitxgb
in_err <- xgbAcc
modtype <- modFit$method
samptmp <- modFit$resample; samp <- length(unique(samptmp$Resample))
numvars <- length(modFit$coefnames)
numcols <- ncol(trainDat)
timval <- str_replace_all(Sys.time(), " |:", "_")

#Is it "prob" of 'yes' or 'no'....?
pred_SAN <- predict(modFit, newdata = dat_test, type = "prob")
toSubmit <- data.frame(ID = dat_test$ID, TARGET = pred_SAN$yes)

file_out <- paste("Res_xxxx_", modtype,"_",numcols,"_n",samp,"_Acc_",in_err,"_", timval,".csv",sep = "")
write.table(toSubmit, file = file_out, sep = "," ,
            row.names = FALSE,col.names = TRUE, quote = FALSE)
