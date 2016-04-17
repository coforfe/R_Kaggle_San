#------------------------------------------------------------
# SANTANDER - Customer Satisfaction - Start: 2016-03-21
#------------------------------------------------------------
# Starting with Michael Pawlus' code and David Pinto ensemble approach
# it's is an adaptation of both good approaches.
# let's use the best iteration found and create different models to ensemble

setwd("/Volumes/TOSHIBA EXT/Verbatin64/R-cosas/2016-01 - Kaggle/03_Santander")

library(xgboost)
library(Matrix)
library(stringr)

#---- Load Pawlus' data
load("cleanData.RData")

keep_cols <- names(train)

#----- Now David Pinto's approach with params for the best iteration
### Set XGBoost parameters
xgb_params <- list(
  "booster"             = "gbtree",
  "objective"           = "binary:logistic",
  "eval_metric"         = "auc",
  "eta"                 = 0.0216666666666667,
  "max_depth"           = 5,
  "subsample"           = 0.690,
  "colsample_bytree"    = 0.685,
  "min_child_weight"    = 1,
  "colsample_bytree"    = 1,
  "num_paralallel_tree" = 1,
  "silent"              = 1,
  "nthread"             = 4
  #"set.seed"            = 22598
)


n_models   <- 50                                # ensemble size / number of base learners (10 is a better choice)
n_features <- round(0.80 * length(keep_cols),0) # random subset size (50 is a good choice but will take longer to train)
n_folds    <- 5                                 # more CV folds should increase performance (try 5 or 10)
model_perf <- numeric(n_models)
meta_tr    <- vector('list', n_models)          # To store training models (generated in xgb.cv)
meta_te    <- vector('list', n_models)          # To sotre testing (predicted) models

for (i in 1:n_models) {
  cat(paste('\n### Model', i, '###\n'))
  rn_v <- sample(1e4:1e5, size = 1); print(rn_v)
  xgb_params$set.seed <- rn_v
  
  ## Sample features
  sel_cols <- sample(keep_cols, n_features)
  x_tr <- Matrix(as.matrix(train[sel_cols]), sparse = TRUE)
  dtrain <- xgb.DMatrix(x_tr, label = train.y)
  x_te <- Matrix(as.matrix(test[sel_cols]), sparse = TRUE)
  dtest <- xgb.DMatrix(x_te)
  
  ## Generate level-one data: k-fold CV with early stopping
  cat(paste('\n### Model', i, '------ CV ----- ###\n'))
  ex_a <- Sys.time(); 
  
  cv_out <- xgb.cv(params = xgb_params, data = dtrain, nrounds = 1500, 
                   nfold = n_folds, prediction = TRUE, stratified = TRUE, 
                   verbose = FALSE, early.stop.round = 15, maximize = TRUE)
  model_perf[i] <- max(cv_out$dt$test.auc.mean)
  best_iter <- which.max(cv_out$dt$test.auc.mean)
  meta_tr[[i]] <- cv_out$pred
  
  ex_b <- Sys.time();  ex_t <- ex_b - ex_a; (ex_t)
  cat(paste('\n', ex_t,'\n'))
  
  ## Train base learner
  cat(paste('\n### Model', i, '------  BASE ----- ###\n'))
  xgb_model <- xgb.train(data = dtrain, params = xgb_params, 
                         nrounds = best_iter);
  
  ## Generate test data
  meta_te[[i]] <- predict(xgb_model, dtest)
  
  cat(paste('\nAUC:', model_perf[i],'\n'))
}

### Save data: use "meta_train.csv" to train your model and "meta_test.csv" to
### generate your predictions
model_names <- paste('Model', 1:n_models, sep = '')
# New traning data
names(meta_tr) <- model_names
meta_tr$Target <- train.y
write.csv(meta_tr, 'meta_train_Cof_Best_.csv', row.names = FALSE, quote = FALSE)
# New test data
names(meta_te) <- model_names
meta_te$Id <- as.integer(test.id)
write.csv(meta_te, 'meta_test_Cof_Best_.csv', row.names = FALSE, quote = FALSE)


# Build the ensemble
library(caret)
dt_train <- read.csv('meta_train_Cof_Best_.csv')
dt_train$Target <- ifelse(dt_train$Target == 0, "no", "yes") 
dt_train$Target <- as.factor(dt_train$Target)

tc <- trainControl("cv", 5, savePredictions = FALSE, classProbs = TRUE,
                   summaryFunction = twoClassSummary)

fit <- train(Target ~ Model1 + Model2 + Model3 + Model4 + Model5,
             data = dt_train, method = "glm", trControl = tc,
             family = binomial, metric = 'ROC', trace = TRUE)



in_err <- fit$results$ROC
in_err

#---- Final Prediction Ensemble
dt_test <- read.csv('meta_test_Cof_Best_.csv')
pred_SAN <- predict(fit, newdata = dt_test, type = "prob")
toSubmit <- data.frame(ID = dt_test$Id, TARGET = pred_SAN$yes)

timval <- str_replace_all(Sys.time(), " |:", "_")
modtype <- fit$method
file_out <- paste("Res_XXXXX_",n_models,"_Ensembles_", modtype,"_AccTot_",in_err,"_", timval,".csv", sep = "")
write.table(toSubmit, file = file_out ,sep = "," , row.names = FALSE, col.names = TRUE, quote = FALSE)




