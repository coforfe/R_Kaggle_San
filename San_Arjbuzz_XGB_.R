# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages
# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats
# For example, here's several helpful packages to load in 

setwd("/Volumes/TOSHIBA EXT/Verbatin64/R-cosas/2016-01 - Kaggle/03_Santander")

library(readr) # CSV file I/O, e.g. the read_csv function
library(xgboost)

# Reading the data
 dat_train <- read.csv("train.csv", stringsAsFactors = F)
 dat_test <- read.csv("test.csv", stringsAsFactors = F)
# dat_train <- read_csv("train.csv")
# dat_test <- read_csv("test.csv")

# Mergin the test and train data
dat_test$TARGET <- NA
all_dat <- rbind(dat_train, dat_test)
names_ori <- names(all_dat)

# Removing the constant variables
train_names <- names(dat_train)[-1]
for (i in train_names)
{
  if (class(all_dat[[i]]) == "integer") 
  {
    u <- unique(all_dat[[i]])
    if (length(u) == 1) 
    {
      all_dat[[i]] <- NULL
    } 
  }
}
names_cons <- names(all_dat)

#Removing duplicate columns
train_names <- names(all_dat)[-1]
fac <- data.frame(fac = integer())    

for(i in 1:length(train_names))
{
  if(i != length(train_names))
  {
    for (k in (i+1):length(train_names)) 
    {
      if(identical(all_dat[,i], all_dat[,k]) == TRUE) 
      {
        fac <- rbind(fac, data.frame(fac = k))
      }
    }
  }
}
same <- unique(fac$fac)
all_dat <- all_dat[,-same]

#Removing hghly correlated variables
cor_v<-abs(cor(all_dat))
diag(cor_v)<-0
cor_v[upper.tri(cor_v)] <- 0
cor_f <- as.data.frame(which(cor_v > 0.85, arr.ind = T))
all_dat <- all_dat[,-unique(cor_f$row)]

# Splitting the data for model
train <- all_dat[1:nrow(dat_train), ]
test <- all_dat[-(1:nrow(dat_train)), ]

save(train, test, file = "ArjbuzzData.RData")
#load("ArjbuzzData.RData")


#Building the model
set.seed(2345688)
param <- list("objective" = "binary:logistic",booster = "gbtree",
              "eval_metric" = "auc",colsample_bytree = 0.85, subsample = 0.95)

y <- as.numeric(train$TARGET)


#AUC was highest in 310th round during cross validation
dtrain <- as.matrix(train[,-c(1,151)])
dtrain_n <- xgb.DMatrix(data = dtrain, label = y) #Cof

xgbmodel <- xgboost(data = dtrain, params = param,
                    nrounds = 310, max.depth = 5, eta = 0.03,
                    label = y, maximize = T, print.every.n = 10)

#Prediction
res <- predict(xgbmodel, newdata = data.matrix(test[,-c(1,151)]))
res <- data.frame(ID = test$ID, TARGET = res)

write.csv(res, "submission.csv", row.names = FALSE)


#------------------ cross validation
n.folds <- 3
cv.out <- xgb.cv(
  params = param, data = dtrain, nrounds = 1500, 
  nfold = n.folds, prediction = TRUE, stratified = TRUE, 
  verbose = FALSE, early.stop.round = 15, maximize = TRUE,
  print.every.n = 100 
  #,metrics = 'auc'
)

model.perf <- max(cv.out$dt$test.auc.mean); model.perf
best.iter <- which.max(cv.out$dt$test.auc.mean); best.iter
meta.tr <- cv.out$pred; meta.tr
