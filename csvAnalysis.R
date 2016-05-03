
#------------------------------------------------------------
# SANTANDER - Customer Satisfaction - New: 2016_04_20
#------------------------------------------------------------

setwd("/Volumes/TOSHIBA EXT/Verbatin64/R-cosas/2016-01 - Kaggle/03_Santander")


#---------------------------
#Read all .csv files
#---------------------------
library(stringr)
filesTmp <- list.files(path = ".", pattern = ".csv")
filesTmp <- filesTmp[!str_detect(filesTmp, "xxxx|XXXX|res|meta|sample|test|train")]

filesTmp <- head(filesTmp,15)

dfcsv <- data.frame()
scordf <- data.frame(score = 0)

for (i in 1:length(filesTmp)) {
  print(i)
  scortmp <- word(filesTmp[i], 2, sep = fixed("_")) 
  scordf[i,1] <- scortmp
  
  dftmp <- read.csv(file = filesTmp[i], header = T) 
  if (i == 1) {
    dfcsv <- data.frame(rep(0,nrow(dftmp))) 
    dfcsv[,i] <- dftmp[,2]
    names(dfcsv)[1] <- c('V1')
  } else {dfcsv[,i] <- dftmp[,2] }
  if (i == length(filesTmp)) { ids <- dftmp[,1]}
}
#Status
dfcsv <- cbind.data.frame(ids, dfcsv)

scordf <- as.vector(scordf[,1])
scorgd <- as.numeric(scordf)
#Scores
scorgd <- ifelse(scorgd > 1, scorgd/10000, scorgd)

#------------------------------------------------------
# (2)  Model higher than 0.84228 and GEOMETRIC MEAN
# Improved!!.
# 2016_04_28 - 10 values -> 0.842414
# Watch out!!: Take 10 different values and improved the best...
#[1] 0.842316 0.842316 0.842316 0.842328 0.842328 0.842365 0.842365 0.842365 [9] 0.842365 0.842365
# 2016_04_28 - 15 values -> 0.842414 (does not improve... it includes previous value)
val_ref <- 0.842400
to_ensem <- scorgd > val_ref
df_ensem <- dfcsv[, 2:ncol(dfcsv)]
df_ensem <- df_ensem[, to_ensem]

library(psych)
mod_geme <- apply(df_ensem, 1, geometric.mean)

timval <- str_replace_all(Sys.time(), " |:", "_")
toSubmit <- data.frame(ID = dfcsv[,1], TARGET = mod_geme)

file_out <- paste("Res_xxxx_geom_mean_",val_ref,"_", timval,".csv",sep = "")
write.table(toSubmit, file = file_out, sep = "," ,
            row.names = FALSE,col.names = TRUE, quote = FALSE)

#------------------------------------------------------
# (2)  Model higher than 0.84228 and SIMPLE MEAN
# Improved!!.
val_ref <- 0.84228
to_ensem <- scorgd > val_ref
df_ensem <- dfcsv[, 2:ncol(dfcsv)]
df_ensem <- df_ensem[, to_ensem]

library(psych)
mod_geme <- apply(df_ensem, 1, mean)

timval <- str_replace_all(Sys.time(), " |:", "_")
toSubmit <- data.frame(ID = dfcsv[,1], TARGET = mod_geme)

file_out <- paste("Res_xxxx_simple_mean_",val_ref,"_", timval,".csv",sep = "")
write.table(toSubmit, file = file_out, sep = "," ,
            row.names = FALSE,col.names = TRUE, quote = FALSE)


#------------------ MODELS ----------------------------
#------------------------------------------------------
# (1) Model based on weigthed mean based on scored...
to_ensem <- str_detect(filesTmp, "gbm|xgbTree|fda|0.8407")
df_ensem <- dfcsv[, 2:ncol(dfcsv)]
df_ensem <- df_ensem[, to_ensem]
sc_ensem <- scorgd[ to_ensem ]

mod_wei <- rowSums(df_ensem * sc_ensem) / sum(sc_ensem)

timval <- str_replace_all(Sys.time(), " |:", "_")
toSubmit <- data.frame(ID = dfcsv[,1], TARGET = mod_wei)

file_out <- paste("Res_xxxx_Weight_gbm_xgb_fda_847_", timval,".csv",sep = "")
write.table(toSubmit, file = file_out, sep = "," ,
            row.names = FALSE,col.names = TRUE, quote = FALSE)

#------------------------------------------------------
# (2)  Model higher than 0.84228 and GEOMETRIC MEAN
# Improved!!.
val_ref <- 0.84228
to_ensem <- scorgd > val_ref
df_ensem <- dfcsv[, 2:ncol(dfcsv)]
df_ensem <- df_ensem[, to_ensem]

library(psych)
mod_geme <- apply(df_ensem, 1, geometric.mean)

timval <- str_replace_all(Sys.time(), " |:", "_")
toSubmit <- data.frame(ID = dfcsv[,1], TARGET = mod_geme)

file_out <- paste("Res_xxxx_geom_mean_",val_ref,"_", timval,".csv",sep = "")
write.table(toSubmit, file = file_out, sep = "," ,
            row.names = FALSE,col.names = TRUE, quote = FALSE)


#------------------------------------------------------
# (3)  Model higher than 0.8405 and GEOMETRIC MEAN
# Improved!!.

to_ensem <- scorgd > 0.8416
df_ensem <- dfcsv[, 2:ncol(dfcsv)]
df_ensem <- df_ensem[, to_ensem]

library(psych)
mod_geme <- apply(df_ensem, 1, geometric.mean)

timval <- str_replace_all(Sys.time(), " |:", "_")
toSubmit <- data.frame(ID = dfcsv[,1], TARGET = mod_geme)

file_out <- paste("Res_xxxx_geom_mean_0.8405_ncols_",ncol(df_ensem) ,"_", timval,".csv",sep = "")
write.table(toSubmit, file = file_out, sep = "," ,
            row.names = FALSE,col.names = TRUE, quote = FALSE)


#------------------------------------------------------
# (3)  Model higher than 0.8405 and GEOMETRIC MEAN
# It is better Geometric_Mean or Weighted_Geometric_Mean ? 
# Geometric_Mean and Weighted_Geometric_Mean yielded the same...

to_ensem <- scorgd > 0.8405
df_ensem <- dfcsv[, 2:ncol(dfcsv)]
df_ensem <- df_ensem[, to_ensem]
sc_ensem <- scorgd[ to_ensem ]

my_we_ge_me <- function(x, weig_val) {
   tmp_val <- exp( sum(log(x) * weig_val) / sum(weig_val))
   return(tmp_val)
}

mod_geme <- apply(df_ensem, 1, my_we_ge_me, sc_ensem)

timval <- str_replace_all(Sys.time(), " |:", "_")
toSubmit <- data.frame(ID = dfcsv[,1], TARGET = mod_geme)

file_out <- paste("Res_xxxx_weigh_geom_mean_0.8405_ncols_",ncol(df_ensem) ,"_", timval,".csv",sep = "")
write.table(toSubmit, file = file_out, sep = "," ,
            row.names = FALSE,col.names = TRUE, quote = FALSE)


#--------------------------------------
# Correlations among submissions...
to_cor <- dfcsv[, 2:ncol(dfcsv)]
library(corrplot)
M <- cor(to_cor)
corrplot(M, method = "number", col = "black", cl.pos = "n")
corrplot(M)



