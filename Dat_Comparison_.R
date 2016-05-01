
#------------------------------------------------------------
# SANTANDER - Customer Satisfaction - New: 2016_04_20
#------------------------------------------------------------

setwd("/Volumes/TOSHIBA EXT/Verbatin64/R-cosas/2016-01 - Kaggle/03_Santander")

#datIn <- read.csv("Res_0.842440_geom_mean_0.8422_2016-04-29_22_52_42.csv")

# library(tabplot)
# tableplot(datIn, decreasing = FALSE, select = c(TARGET))

# datIn_sort <- datIn[ order(datIn$TARGET),]
# dat_low <- datIn_sort[datIn_sort$TARGET > 0, ]
#Lowest values are around 0.002... yesterday were 0.001..
#Are these the candidates to be 0??.
#Let's try with values lowest than 0.0021...


# Comparison - Submission 2016_04_29 and 2016_04_28... 
dat_0 <- read.csv("Res_0.842475_Anuk_submission.csv")
dat_1 <- read.csv("Res_0.842440_Iridium_submission.csv")
dat_2 <- read.csv("Res_0.842365_ZFTurbo_submission.csv")
dat_3 <- read.csv("Res_0.842328_Ialthan_submission.csv")

datIn <- dat_0
datIn_sort <- datIn[ order(datIn$TARGET),]
dat_low <- datIn_sort[datIn_sort$TARGET > 0, ]
head(dat_low, 100)

dat_comp_Z <- cbind.data.frame(dat_0, dat_1)
names(dat_comp_Z) <- c('id_new', 'new', 'id_old', 'old')
dat_see_Z <- dat_comp_Z[dat_comp_Z$new == 0 & dat_comp_Z$old != 0, ]
head(dat_see_Z,200)

dat_comp <- cbind.data.frame(dat_1, dat_2)
names(dat_comp) <- c('id_new', 'new', 'id_old', 'old')
dat_see <- dat_comp[dat_comp$new == 0 & dat_comp$old != 0, ]
head(dat_see,100)
# New values that get 0, yesterday were 0.001 the lowest prob. values.
dat_comp_B <- cbind.data.frame(dat_2, dat_3)
names(dat_comp_B) <- c('id_new', 'new', 'id_old', 'old')
dat_see_B <- dat_comp_B[dat_comp_B$new == 0 & dat_comp_B$old != 0, ]
head(dat_see_B,100)
# New values that get 

dat_target <- dat_low
dat_021 <- dat_target[ dat_target > 0 & dat_target < 0.030]
length(dat_021)

#------------------------ CHANGE PROBABILITIES
toSubmit <- dat_0
val_filt <- 0.0018
sum(dat_0$TARGET < val_filt)
timval <- str_replace_all(Sys.time(), " |:", "_")
toSubmit$TARGET <- ifelse(toSubmit$TARGET < val_filt, 0, toSubmit$TARGET)

file_out <- paste("Res_xxxx_filtered_",val_filt,"_", timval,".csv",sep = "")
write.table(toSubmit, file = file_out, sep = "," ,
            row.names = FALSE,col.names = TRUE, quote = FALSE)


