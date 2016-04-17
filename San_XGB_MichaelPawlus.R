#------------------------------------------------------------
# SANTANDER - Customer Satisfaction - Start: 2016-03-21
#------------------------------------------------------------

setwd("/Volumes/TOSHIBA EXT/Verbatin64/R-cosas/2016-01 - Kaggle/03_Santander")

library(xgboost)
library(Matrix)

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

train <- train[, feature.names]
test <- test[, feature.names]

save(train, test, train.y, test.id, file = "cleanData.RData")
#load("cleanData.RData")

#---------------------------------
#---------------------- XGBOOST
#---------------------------------
rn_v <- sample(1e4:1e5, size = 1); rn_v
set.seed(rn_v)

train$TARGET <- train.y


# h <- sample(nrow(train),1000)
# train_s <- sparse.model.matrix(TARGET ~ ., data = train[h,])
# test_s <- sparse.model.matrix(TARGET ~ ., data = train[-h,])
# cat("Sample data for early stopping\n")
# dval      <- xgb.DMatrix(data = test_s,  label = train.y[-h])
# dtrain    <- xgb.DMatrix(data = train_s, label = train.y[h])
# watchlist <- list(train = dtrain, val = dval)

train <- sparse.model.matrix(TARGET ~ ., data = train)
dtrain <- xgb.DMatrix(data = train, label = train.y)
watchlist <- list(train = dtrain)

param <- list(  objective           = "binary:logistic", 
                booster             = "gbtree",
                eval_metric         = "auc",
                eta                 = 0.02,
                max_depth           = 5,
                subsample           = 0.7,
                colsample_bytree    = 0.7,
                min_child_weight    = 1,
                num_paralallel_tree = 1,
                set.seed            = rn_v
)

# nrounds = 560 ??
clf <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 560, 
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
test$TARGET <- -1
test <- sparse.model.matrix(TARGET ~ ., data = test)

#--------------------------------------------------------
#-------------- FILE UPLOAD
#--------------------------------------------------------
preds <- predict(clf, test)
submission <- data.frame(ID = test.id, TARGET = preds)
cat("saving the submission file\n")
write.csv(submission, "submission.csv", row.names = F)



n.folds <- 3
cv.out <- xgb.cv(
                 params = param, data = dtrain_n, nrounds = 1500, 
                 nfold = n.folds, prediction = TRUE, stratified = TRUE, 
                 verbose = FALSE, early.stop.round = 15, maximize = TRUE,
                 print.every.n = 100 
                 #metrics = 'auc'
                 )

model.perf <- max(cv.out$dt$test.auc.mean); model.perf
best.iter <- which.max(cv.out$dt$test.auc.mean); best.iter
meta.tr <- cv.out$pred; meta.tr
