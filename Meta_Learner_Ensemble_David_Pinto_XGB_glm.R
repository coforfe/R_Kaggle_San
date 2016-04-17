#### SUPER LEARNER: XGBoost + Random Feature Subset Selection
##
## This script uses L xgboost models fitted from L 
## random subsets of the original features to generate 
## a NxL "level-one" data. The new dataset can be used
## to train a simple GLM model and get test AUC > 0.84
##
## Super Learner reference: 
## http://learn.h2o.ai/content/tutorials/ensembles-stacking/index.html
##
####

setwd("/Volumes/TOSHIBA EXT/Verbatin64/R-cosas/2016-01 - Kaggle/03_Santander")


### Load packages
library('readr')
library('magrittr')
library('Matrix')
library('xgboost')

### Load data
dt.train <- read_csv('train.csv', 
                            col_types = paste(rep('d', 371), collapse = ''))
dt.test <- read_csv('test.csv', 
                           col_types = paste(rep('d', 370), collapse = ''))


### Convert target to Int
label.name <- 'TARGET'
y <- as.integer(dt.train[[label.name]])

### Sample weights to "balance" classes (use the inverse of class frequency)
class.freq <- table(y) %>% prop.table
row.wt <- ifelse(y == 0, 1/class.freq[1], 1/class.freq[2])

### Remove features with less than 10% of non-zero entries
col.names <- names(dt.train) %>% setdiff(c('ID', label.name))
zero.rate <- sapply(dt.train[col.names], function(dt.col) {
  sum(dt.col == 0)/length(dt.col)
})
keep.cols <- col.names[zero.rate < 0.9]

### Set XGBoost parameters
xgb_params <- list(
  "booster"             = "gbtree",
  "objective"           = "binary:logistic",
  "eval_metric"         = "auc",
  "eta"                 = 0.0216666666666667,
  "max_depth"           = 5,
  "subsample"           = 0.690,
  "colsample_bytree"    = 0.68,
  "min_child_weight"    = 1,
  "colsample_bytree"    = 1,
  "num_paralallel_tree" = 1,
  "silent"              = 1,
  "nthread"             = 4,
  "set.seed"            = 22598
)


### Train base learners (Note that some models achieve a really great test performance: ~ 0.84)
n_models   <- 5  # ensemble size / number of base learners (10 is a better choice)
n_features <- 50 # random subset size (50 is a good choice but will take longer to train)
n_folds    <- 5  # more CV folds should increase performance (try 5 or 10)
model_perf <- numeric(n_models)
meta_tr    <- vector('list', n_models)
meta_te    <- vector('list', n_models)
for (i in 1:n_models) {
  cat(paste('\n### Model', i, '###\n'))
  
  ## Sample features
  sel_cols <- sample(keep_cols, n_features)
  x_tr <- Matrix(as.matrix(dt.train[sel_cols]), sparse = TRUE)
  dtrain <- xgb.DMatrix(x_tr, label = y, weight = row.wt)
  x_te <- Matrix(as.matrix(dt.test[sel.cols]), sparse = TRUE)
  dtest <- xgb.DMatrix(x_te)
  
  ## Generate level-one data: k-fold CV with early stopping
  cv_out <- xgb.cv(params = xgb.params, data = dtrain, nrounds = 1500, 
                   nfold = n_folds, prediction = TRUE, stratified = TRUE, 
                   verbose = FALSE, early.stop.round = 15, maximize = TRUE)
  model_perf[i] <- max(cv_out$dt$test.auc.mean)
  best_iter <- which.max(cv_out$dt$test.auc.mean)
  meta_tr[[i]] <- cv_out$pred
  
  ## Train base learner
  xgb_model <- xgb.train(data = dtrain, params = xgb_params, 
                         nrounds = best_iter);
  
  ## Generate test data
  meta_te[[i]] <- predict(xgb_model, dtest)
  
  cat(paste('\nAUC:', model_perf[i]))
}

### Save data: use "meta_train.csv" to train your model and "meta_test.csv" to
### generate your predictions
model_names <- paste('Model', 1:n_models, sep = '')
# New traning data
names(meta_tr) <- model_names
meta_tr$Id <- as.integer(dt.train$ID)
meta_tr$Wt <- row.wt
meta_tr$Target <- y
write.csv(meta.tr, 'meta_train_Cof_1_.csv', row.names = FALSE, quote = FALSE)
# New test data
names(meta_te) <- model_names
meta_te$Id <- as.integer(dt.test$ID)
write.csv(meta_te, 'meta_test_Cof_1.csv', row.names = FALSE, quote = FALSE)

### ==> Now use Logistic Regression to learn the best weights to combine the base
### learners

### EXAMPLE

# library(caret)
# dt.train <- read.csv('meta_train.csv')
# dt.train$Target <- as.factor(dt.train$Target)
# tc <- trainControl("cv", 5, savePredictions = FALSE, classProbs = TRUE,
#                    summaryFunction = twoClassSummary)
# fit <- train(Target ~ Model1 + Model2 + Model3 + Model4 + Model5,
#              data = dt.train, method = "glm", trControl = tc,
#              family = binomial, metric = 'ROC', trace = FALSE)
# fit$results$ROC
