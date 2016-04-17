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
train <- as.data.frame(fread("train.csv"), na.strings = c("NA", "9999999999"))
test  <- as.data.frame(fread("test.csv") , na.strings = c("NA", "9999999999"))
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
train[train == 9999999999 ] <- NA 
test[test == 9999999999 ] <- NA 
train[is.na(train)]   <- -997
test[is.na(test)]   <- -997



cat("Class of each column\n")
cl_col <- as.data.frame(mapply(class, train))
cl_col <- data.frame(var = rownames(cl_col), classe = cl_col[,1])
unique(cl_col$classe)
#All numeric, integer and some of them integer64?
#change them to integer

for (i in 1:ncol(train)) {
  cltmp <- class(train[, i])
  if (cltmp == "integer64") {
    train[,i] <- as.numeric( train[,i] )
  } else next
}
for (i in 1:ncol(test)) {
  cltmp <- class(test[, i])
  if (cltmp == "integer64") {
    test[,i] <- as.numeric( test[,i] )
  } else next
}

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

# # Correlation among god_col
# cor_set <- cor(test[, god_col])
# high_cor <- findCorrelation(cor_set, cutoff = 0.95, names = TRUE) 
# # 11 columns...
# end_col <- setdiff(god_col, high_cor)
# # Valid columns ... just 45!!.
end_col <- god_col

# Clean DataSets
tra <- train[, end_col]
tra <- cbind.data.frame( train[, 1:2], tra)
tes <- test[, end_col]




#---------------------------------
#---------------------- XGBOOST
#---------------------------------

cat("Sample data for early stopping\n")
h <- sample(nrow(train),5000)
dval      <- xgb.DMatrix(data = data.matrix(tra[h,]),label = train$TARGET[h])
dtrain    <- xgb.DMatrix(data = data.matrix(tra[-h,]),label = train$TARGET[-h])
watchlist <- list(val = dval,train = dtrain)

# # Run settings
# md <- 11
# ss <- 0.96
# cs <- 0.45
# mc <- 1
# np <- 1
# nrounds <- 1501 # CHANGE TO >1500
# early.stop.round <- 300

a <- Sys.time();a

# # et ~ 0.005 overfits.
# xgbGrid <- expand.grid(
#   et = 0.0775,
#   md = 12,
#   ss = 0.93,
#   cs = 0.45,
#   mc = 1,
#   np = 1,
#   nr = 1501,
#   es = 150,
#   rn = 0
# )

# clf = xgb.XGBClassifier(missing=np.nan, max_depth=5, n_estimators=350, learning_rate=0.03, 
#                         nthread=4, subsample=0.95, colsample_bytree=0.85, seed=4242)
#Building the model
# set.seed(88)
# param <- list("objective" = "binary:logistic",booster = "gbtree",
#               "eval_metric" = "auc",colsample_bytree = 0.85, subsample = 0.95)
# y <- as.numeric(train$TARGET)
# #AUC was highest in 310th round during cross validation
# xgbmodel <- xgboost(data = as.matrix(train[,-c(1,151)]), params = param,
#                     nrounds = 310, max.depth = 5, eta = 0.03,
#                     label = y, maximize = T)


# et ~ 0.005 overfits.
xgbGrid <- expand.grid(
  et = seq(0.01, 0.1, length.out = 3),
  md = seq(10,12, length.out = 3),
  ss = 0.96,
  cs = 0.45,
  mc = 1,
  np = 1,
  nr = 2501,
  es = 50,
  rn = 0
)


#rn <- c(21219, sample(21219:(21219*2) , 5))
res_df <- data.frame(
  xgbAcc = 0, xgbIdx = 0, et = 0, md = 0,
  ss = 0, cs = 0, mc = 0, np = 0,
  nr = 0, es = 0, rn = 0,
  ex_t = 0
)

#Param Iteration
ens_ble <- 0
con_ens <- 0

for (i in 1:nrow(xgbGrid)) {
  print(i)
  
  ex_a <- Sys.time();  
  
  param <- list(  
    objective           = "binary:logistic", 
    booster             = "gbtree",
    eval_metric         = "auc",
    eta                 = xgbGrid$et[i],
    max_depth           = xgbGrid$md[i],
    subsample           = xgbGrid$ss[i],
    colsample_bytree    = xgbGrid$cs[i],
    min_child_weight    = xgbGrid$mc[i],
    num_parallel_tree   = xgbGrid$np[i],
    set.seed            = xgbGrid$rn[i]
  )
  
  cat("Train model\n")
  ex_a <- Sys.time();  
  clf <- xgb.train(   params              = param, 
                      data                = dtrain, 
                      nrounds             = xgbGrid$nr[i],
                      early.stop.round    = xgbGrid$es[i],
                      watchlist           = watchlist,
                      maximize            = FALSE,
                      verbose             = 1,  
                      print.every.n       = 100,
                      nthread             = 4 
  )
  
  ex_b <- Sys.time();  ex_t <- ex_b - ex_a; (ex_t)
  
  Accxgb <- clf$bestScore
  Accidx <- clf$bestInd
  (Accxgb)
  (Accidx)
  
  #Store results
  res_df[i,1]  <- Accxgb
  res_df[i,2]  <- Accidx
  res_df[i,3]  <- xgbGrid$et[i]
  res_df[i,4]  <- xgbGrid$md[i]
  res_df[i,5]  <- xgbGrid$ss[i]
  res_df[i,6]  <- xgbGrid$cs[i]
  res_df[i,7]  <- xgbGrid$mc[i]
  res_df[i,8]  <- xgbGrid$np[i]
  res_df[i,9]  <- xgbGrid$nr[i]
  res_df[i,10] <- xgbGrid$es[i]
  res_df[i,11] <- xgbGrid$rn[i]
  res_df[i,12] <- ex_t
  
  cat("\n")
  print(res_df)
  cat("\n")
  
  # if (Accxgb > 0.44) { next }
  
  #--------------------------------------------------------
  #-------------- PREDICTION
  #--------------------------------------------------------
  cat("Calculate predictions\n")
  pred1 <- predict(clf,
                   data.matrix(tes),
                   ntreelimit = clf$bestInd)
  
  # if (Accxgb < 0.44) {
  #   con_ens <- con_ens + 1
  #   ens_ble <- ens_ble + pred1
  # }
  # 
  #--------------------------------------------------------
  #-------------- FILE UPLOAD
  #--------------------------------------------------------
  submission <- data.frame(ID = test$ID, TARGET = pred1)
  
  LL <- clf$bestScore
  cat(paste("Best AUC: ",LL,"\n",sep = ""))
  
  cat("Create submission file\n")
  submission <- submission[order(submission$ID),]
  
  dat_tim  <- str_replace_all(Sys.time()," |:","_")
  file_tmp <- paste("Res_xxxxx_AllCols_XGB_Acc_", Accxgb ,"_", dat_tim,"_.csv", sep = "")
  write.csv(submission,file_tmp,row.names = F)
  
  b <- Sys.time();b; b - a
  
} #for (i in 1:nrow(
#--- End of loop
Bestxgb <- min(res_df$xgbAcc)
res_csv <- paste("res_df_xgb_BestAcc_", Bestxgb ,"_",dat_tim,"_.csv", sep = "")
write_csv(res_df, res_csv)


# File Ensemble iterations
cat("Build Ensemble\n")
sub_ensemb <- data.frame(ID = test$ID, PredictedProb = ens_ble/con_ens)
sub_ensemb <- sub_ensemb[order(sub_ensemb$ID),]
dat_tim  <- str_replace_all(Sys.time()," |:","_")
file_ens <- paste("Res_xxxxx_XGB_Extended_3_GridOri_Ensemble_", con_ens, "_", dat_tim,"_.csv", sep = "")
write.csv(sub_ensemb,file_ens,row.names = F)

sub_ensemb_i <- data.frame(ID = test$ID, PredictedProb = ens_ble/i)
sub_ensemb_i <- sub_ensemb_i[order(sub_ensemb_i$ID),]
file_ens_i <- paste("Res_xxxxx_XGB_Extended_3_GridOri_Ensemble_", i, "_", dat_tim,"_.csv", sep = "")
write.csv(sub_ensemb,file_ens_i,row.names = F)

b <- Sys.time();b; b - a

#--------------------------------------------------------
#---------------------- END OF PROGRAM
#--------------------------------------------------------

#Select most important variables out of xgb model *numeric*
#The model used for this was the one that got best scoring.
var_imp <- xgb.importance( model = clf)
top_var <- names(tra)[as.numeric(var_imp$Feature[1:10])]
top_ch <- intersect(top_var, cha_col)
top_num <- setdiff(top_var, top_ch)

