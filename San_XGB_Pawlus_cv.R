#------------------------------------------------------------
# SANTANDER - Customer Satisfaction - Start: 2016-03-21
#------------------------------------------------------------
# Starting with Michael Pawlus' code and data and with ideas of David Pinto
# let's iterate over different "params" to see if score improves
# taking into consideration xgb.cv (cross validation) results

setwd("/Volumes/TOSHIBA EXT/Verbatin64/R-cosas/2016-01 - Kaggle/03_Santander")

library(xgboost)
library(Matrix)

load("cleanData.RData")

#---------------------------------
#---------------------- XGBOOST
#---------------------------------
rn_v <- sample(1e4:1e5, size = 1); rn_v
set.seed(rn_v)

train$TARGET <- train.y

train <- sparse.model.matrix(TARGET ~ ., data = train)
dtrain <- xgb.DMatrix(data = train, label = train.y)
watchlist <- list(train = dtrain)

# # Original data - 0.84%
# eta                 = 0.02,
# max_depth           = 5,
# subsample           = 0.7,
# colsample_bytree    = 0.7,
# min_child_weight    = 1,
# num_paralallel_tree = 1,

xgbGrid <- expand.grid(
  et = 0.0216666666666667,
  md = 5,
  ss = 0.690,
  cs = 0.68,
  mc = 1,
  np = 1,
  nr = 1500,
  es = 15,
  rn = 22598
)

# Best... 0.842417
# xgbGrid <- expand.grid(
#   et = 0.0216666666666667,
#   md = 5,
#   ss = 0.690,
#   cs = 0.68,
#   mc = 1,
#   np = 1,
#   nr = 1500,
#   es = 15,
#   rn = 22598
# )




#rn <- c(21219, sample(21219:(21219*2) , 5))
res_df <- data.frame(
  xgbAcc = 0, xgbIdx = 0, et = 0, md = 0,
  ss = 0, cs = 0, mc = 0, np = 0,
  nr = 0, es = 0, rn = 0,
  ex_t = 0
)

rn_v <- sample(1e4:1e5, size = nrow(xgbGrid) ); rn_v
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

n_folds <- 3
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


# Now lets get the best result and get a prediction
res_df <- res_df[ order(res_df$xgbAcc, decreasing = TRUE),]
best_v <- 1
in_err <- res_df$xgbAcc[best_v]; in_err

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

pl_us <- 0
clf <- xgb.train(   
                    params              = param_best, 
                    data                = dtrain, 
                    nrounds             = ( res_df$xgbIdx[best_v] + pl_us ), 
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
load("cleanData.RData")
test$TARGET <- -1
test <- sparse.model.matrix(TARGET ~ ., data = test)

#--------------------------------------------------------
#-------------- FILE UPLOAD
#--------------------------------------------------------
preds <- predict(clf, test)
submission <- data.frame(ID = test.id, TARGET = preds)
cat("saving the submission file\n")
library(stringr)
dat_tim  <- str_replace_all(Sys.time()," |:","_")
file_out <- paste("Res_XXXX_XGB_Pawlus_cv_", n_folds, "_plus_", pl_us, "_Acc_", in_err,"_time_", dat_tim, "_.csv", sep = "" )
write.csv(submission, file = file_out, row.names = F)

file_res <- paste("res_df_Best_",in_err,"_time_", dat_tim, "_.csv", sep = "")
write.csv(res_df, file = file_res, row.names = F)

#*********************************************
#-------------- END OF PROGRAM
#*********************************************



#--------------------------------------------------------
#-------------- FILE UPLOAD
#--------------------------------------------------------

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

