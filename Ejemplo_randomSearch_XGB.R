
#------------------------------------------------------------
# SANTANDER - Customer Satisfaction - New: 2016_04_20
#------------------------------------------------------------

setwd("/Volumes/TOSHIBA EXT/Verbatin64/R-cosas/2016-01 - Kaggle/03_Santander")


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

#---------------------------------
#---------------------- RandomSearch
# http://stackoverflow.com/questions/35050846/xgboost-in-r-how-does-xgb-cv-pass-the-optimal-parameters-into-xgb-train

best_seednumber = 1234
best_auc = 0
best_auc_index = 0

res_df <- data.frame(
  xgbAcc = 0, xgbIdx = 0, et = 0, md = 0,
  ss = 0, cs = 0, mc = 0, np = 0,
  ga = 0, es = 0, rn = 0,
  ex_t = 0
)


for (iter in 1:100) {
  print(iter)
  param <- list(objective        = "binary:logistic",
                booster          = "gbtree",
                eval_metric      = "auc",
                num_class        = 2,
                max_depth        = sample(2:7, 1),
                eta              = runif(1, .01, .3),
                gamma            = runif(1, 0.0, 0.2), 
                subsample        = runif(1, .5, .9),
                colsample_bytree = runif(1, .5, .8), 
                min_child_weight = sample(1:40, 1),
                max_delta_step   = sample(1:10, 1)
  )
  
  ex_a <- Sys.time();  
  
  cv.nround = 1500
  cv.nfold = 5
  seed.number = sample.int(10000, 1)[[1]]
  set.seed(seed.number)
  mdcv <- xgb.cv(
                 data    = dtrain,   params = param,       nthread = 6, 
                 nfold   = cv.nfold, nrounds = cv.nround,  stratified = TRUE,
                 verbose = TRUE,     early.stop.round = 8, maximize = TRUE,
                 print.every.n = 50
                 )
  
  max_auc = max(mdcv[, test.auc.mean])
  max_auc_index = which.max(mdcv[, test.auc.mean])
  
  ex_b <- Sys.time();  ex_t <- ex_b - ex_a; (ex_t)
  cat(paste('\n', ex_t,'\n')) 
  
  if (max_auc > best_auc) {
    best_auc = max_auc
    best_auc_index = max_auc_index
    best_seednumber = seed.number
    best_param = param
    print(best_auc)
  }
  
  #Store results
  res_df[iter,1]  <- max_auc
  res_df[iter,2]  <- max_auc_index
  res_df[iter,3]  <- param$eta
  res_df[iter,4]  <- param$max_depth
  res_df[iter,5]  <- param$subsample
  res_df[iter,6]  <- param$colsample_bytree
  res_df[iter,7]  <- param$min_child_weight
  res_df[iter,8]  <- param$max_delta_step
  res_df[iter,9]  <- param$gamma
  res_df[iter,10] <- cv.nround
  res_df[iter,11] <- seed.number
  res_df[iter,12] <- ex_t
  
  print(res_df)
  
}

nround = best_auc_index
set.seed(best_seednumber)
md <- xgb.train(data = dtrain, params=best_param, nrounds=nround, nthread=6)

