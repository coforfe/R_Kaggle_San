
#------------------------------------------------------------
# SANTANDER - Customer Satisfaction - New: 2016_04_20
#------------------------------------------------------------

setwd("/Volumes/TOSHIBA EXT/Verbatin64/R-cosas/2016-01 - Kaggle/03_Santander")

library(xgboost)
library(Matrix)

#load("dataBest_train_dtrain_test_testid_2016_04_25.RData")

set.seed(1234)

train <- read.csv("train.csv")
test  <- read.csv("test.csv")

##### Removing IDs
train$ID <- NULL
test.id <- test$ID
test$ID <- NULL

##### Extracting TARGET
train.y <- train$TARGET
train$TARGET <- NULL

var15 = test['var15']
saldo_medio_var5_hace2 = test['saldo_medio_var5_hace2']
saldo_var33 = test['saldo_var33']
var38 = test['var38']
V21 = test['var21']
NV=test['num_var33']+test['saldo_medio_var33_ult3']+test['saldo_medio_var44_hace2']+test['saldo_medio_var44_hace3']+
  test['saldo_medio_var33_ult1']+test['saldo_medio_var44_ult1']
NV30 = test['num_var30']


##### 0 count per line
count0 <- function(x) {
  return( sum(x == 0) )
}
train$n0 <- apply(train, 1, FUN = count0)
test$n0 <- apply(test, 1, FUN = count0)

##### Removing constant features
cat("\n## Removing the constants features.\n")
for (f in names(train)) {
  if (length(unique(train[[f]])) == 1) {
    cat(f, "is constant in train. We delete it.\n")
    train[[f]] <- NULL
    test[[f]] <- NULL
  }
}

##### Removing identical features
features_pair <- combn(names(train), 2, simplify = F)
toRemove <- c()
for (pair in features_pair) {
  f1 <- pair[1]
  f2 <- pair[2]
  
  if (!(f1 %in% toRemove) & !(f2 %in% toRemove)) {
    if (all(train[[f1]] == train[[f2]])) {
      cat(f1, "and", f2, "are equals.\n")
      toRemove <- c(toRemove, f2)
    }
  }
}

feature.names <- setdiff(names(train), toRemove)

train$var38 <- log(train$var38)
test$var38 <- log(test$var38)

train <- train[, feature.names]
test <- test[, feature.names]

#---limit vars in test based on min and max vals of train
print('Setting min-max lims on test data')
for (f in colnames(train)) {
  lim <- min(train[,f])
  test[test[,f] < lim,f] <- lim
  
  lim <- max(train[,f])
  test[test[,f] > lim,f] <- lim  
}
#---

#save(train, test, file = "Best_clean_train_test_2016_04_25.RData")
#load("Best_clean_train_test_2016_04_25.RData")


#---------------------------------
#---------------------- XGBOOST
#---------------------------------

train$TARGET <- train.y

train <- sparse.model.matrix(TARGET ~ ., data = train)

dtrain <- xgb.DMatrix(data = train, label = train.y)
watchlist <- list(train = dtrain)

#save(train, dtrain, test, test.id, file = "dataBest_train_dtrain_test_testid_2016_04_25.RData")
#load("dataBest_train_dtrain_test_testid_2016_04_25.RData")

# Original - 0.841997
# param <- list(  objective           = "binary:logistic", 
#                 booster             = "gbtree",
#                 eval_metric         = "auc",
#                 eta                 = 0.0202048,
#                 max_depth           = 5,
#                 subsample           = 0.6815,
#                 colsample_bytree    = 0.701
# )  


# V1
# xgbGrid <- expand.grid(
#   et = seq(0.0201000, 0.0203000, length.out = 5),
#   md = 5,
#   ss = seq( 0.6800, 0.6900, length.out = 5),
#   cs = seq( 0.690, 0.710, length.out = 5),
#   mc = 1,
#   np = 1,
#   nr = 1500,
#   es = 15,
#   rn = 22598
# )

# V2
xgbGrid <- expand.grid(
  et = c( 0.0202048, 0.02008, 0.0201733333333333,0.02022,0.0204533333333333),
  md = 5,
  ss = c(0.6815,0.686666666666667,0.683333333333333),
  cs = c(0.701, 0.69, 0.71),
  mc = 1,
  np = 1,
  nr = 1500,
  es = 15,
  rn = 22598
)



# "","xgbAcc","xgbIdx","et","md","ss","cs","mc","np","nr","es","rn","ex_t"
# "21",0.842585,382,0.02008,5,0.686666666666667,0.69,1,1,1500,15,59529,2.61376696427663
# "133",0.842257,383,0.0201733333333333,5,0.683333333333333,0.71,1,1,1500,15,73640,2.70235291719437
# "23",0.84222,341,0.0201733333333333,5,0.686666666666667,0.69,1,1,1500,15,29128,2.36045801639557
# "134",0.842162,360,0.02022,5,0.683333333333333,0.71,1,1,1500,15,52227,2.55826103289922
# "149",0.841995,343,0.0204533333333333,5,0.686666666666667,0.71,1,1,1500,15,67736,2.43699963490168


#rn <- c(21219, sample(21219:(21219*2) , 5))
res_df <- data.frame(
  xgbAcc = 0, xgbIdx = 0, et = 0, md = 0,
  ss = 0, cs = 0, mc = 0, np = 0,
  nr = 0, es = 0, rn = 0,
  ex_t = 0
)

# rn_v <- sample(1e4:1e5, size = nrow(xgbGrid) ); rn_v
rn_v <- rep(0, nrow(xgbGrid) ); rn_v # zero
# rn_v <- 22598 #Cof - Example.

for (i in 1:nrow(xgbGrid)) {
  print(i)
  
  param <- list(  objective           = "binary:logistic", 
                  booster             = "gbtree",
                  eval_metric         = "auc",
                  eta                 = xgbGrid$et[i],
                  max_depth           = xgbGrid$md[i],
                  subsample           = xgbGrid$ss[i],
                  colsample_bytree    = xgbGrid$cs[i],
                  min_child_weight    = xgbGrid$mc[i],
                  num_paralallel_tree = xgbGrid$np[i],
                  set.seed            = rn_v[i]
  )
  
  ex_a <- Sys.time(); 
  
  n_folds <- 5
  cv_out <- xgb.cv(
    params = param, data = dtrain, nrounds = 1500, 
    nfold = n_folds, prediction = TRUE, stratified = TRUE, 
    verbose = TRUE, early.stop.round = 15, maximize = TRUE,
    print.every.n = 100 
    #metrics = 'auc'
  )
  
  model_perf <- max(cv_out$dt$test.auc.mean); model_perf
  best_iter <- which.max(cv_out$dt$test.auc.mean); best_iter
  
  ex_b <- Sys.time();  ex_t <- ex_b - ex_a; (ex_t)
  cat(paste('\n', ex_t,'\n'))
  
  #Store results
  res_df[i,1]  <- model_perf
  res_df[i,2]  <- best_iter
  res_df[i,3]  <- xgbGrid$et[i]
  res_df[i,4]  <- xgbGrid$md[i]
  res_df[i,5]  <- xgbGrid$ss[i]
  res_df[i,6]  <- xgbGrid$cs[i]
  res_df[i,7]  <- xgbGrid$mc[i]
  res_df[i,8]  <- xgbGrid$np[i]
  res_df[i,9]  <- xgbGrid$nr[i]
  res_df[i,10] <- xgbGrid$es[i]
  res_df[i,11] <- rn_v[i]
  res_df[i,12] <- ex_t
  
  cat("\n")
  print(res_df)
  cat("\n")
  
} #for (i in 1:...)...end of loop


#--------------------------------------------------------
#-------------- GRAPHICAL ANALYSIS
#--------------------------------------------------------
#
library(lattice)
xy_gr <- xyplot(
  xgbAcc ~ et | ss * cs
  #,data = res_df[res_df$xgbAcc > 0.8415, ]
  ,data = res_df
  ,type = "b"
  ,strip = strip.custom(strip.names = TRUE, strip.levels = TRUE)
)
print(xy_gr)

library(stringr)
dat_tim  <- str_replace_all(Sys.time()," |:","_")
Bestxgb <- max(res_df$xgbAcc)
res_csv <- paste("res_df_xgb_BestAcc_", Bestxgb ,"_",dat_tim,"_.csv", sep = "")
write.csv(res_df, file = res_csv)

#----------------------------
# Now lets get the best result and get a prediction
#res_df <- read.csv("res_df_Best_0.842539_time_2016-04-25_18_43_12_.csv")
res_df <- res_df[ order(res_df$xgbAcc, decreasing = TRUE),]
best_v <- 2
in_err <- res_df$xgbAcc[best_v]; in_err

#load("dataBest_train_dtrain_test_testid_2016_04_25.RData")

param_best <- list(
  objective           = "binary:logistic", 
  booster             = "gbtree",
  eval_metric         = "auc",
  eta                 = res_df$et[best_v],
  max_depth           = res_df$md[best_v],
  subsample           = res_df$ss[best_v],
  colsample_bytree    = res_df$cs[best_v],
  min_child_weight    = res_df$mc[best_v],
  num_paralallel_tree = res_df$np[best_v],
  set.seed            = res_df$rn[best_v]
)

pl_us <- res_df$xgbIdx[best_v] * 0.20
#pl_us <- 0
clf <- xgb.train(   
  params              = param_best, 
  data                = dtrain, 
  nrounds             = (res_df$xgbIdx[best_v] + pl_us ), 
  verbose             = 1,
  watchlist           = watchlist,
  #maximize            = FALSE,
  maximize            = TRUE,
  early.stop.round    = 50,
  print.every.n       = 100,
  nthread             = 4
)


#--------------------------------------------------------
#-------------- PREDICTION
#--------------------------------------------------------
test_pred <- test
test_pred$TARGET <- -1
test_pred <- sparse.model.matrix(TARGET ~ ., data = test_pred)

#--------------------------------------------------------
#-------------- FILE UPLOAD
#--------------------------------------------------------
#feature.names

preds <- predict(clf, test_pred)
pred  <- predict(clf,train)
AUC <- function(actual,predicted) {
  library(pROC)
  auc <- auc(as.numeric(actual),as.numeric(predicted))
  auc 
}
AUC(train.y,pred) ##AUC
library(MLmetrics)
AUC(train.y,pred) ##AUC

# Under 23 year olds are always happy
#Cof
sum(var15 < 23)
sum(var38 > 3988596)
sum(saldo_var33  > 0)
sum(NV > 0)
sum(V21 > 7500)
sum(saldo_medio_var5_hace2 > 160000)
#----

preds[var15 < 23] = 0
preds[saldo_medio_var5_hace2 > 160000] = 0
preds[saldo_var33 > 0] = 0
preds[var38 > 3988596] = 0
preds[NV > 0] = 0
preds[V21 > 7500] = 0
preds[NV30 > 9]=0

submission <- data.frame(ID = test.id, TARGET = preds)
cat("saving the submission file\n")
library(stringr)
dat_tim  <- str_replace_all(Sys.time()," |:","_")
n_folds <- 3
file_out <- paste("Res_XXXX_XGB_Pawlus_cv_", n_folds, "_plus_", pl_us, "_Acc_", in_err,"_time_", dat_tim, "_.csv", sep = "" )
write.csv(submission, file = file_out, row.names = F)


#*********************************************
#-------------- END OF PROGRAM
#*********************************************

