# Source: https://stats.stackexchange.com/questions/61090/how-to-split-a-data-set-to-do-10-fold-cross-validation
# packages
source('~/Workspace/RainfallSpectralAnalysis/SpectralAnalysis/function_SetupForGraphics.R')
library(caret)        # Cross-validation
library(CAST)         # Cross-validation
library(randomForest) # Random forest
library(xgboost)      # XGBoost
library(foreach)      # Parallel computing
library(doParallel)   # Parallel computing

# excute functions in parallel
ncore = 40
cl = makePSOCKcluster(ncore)
registerDoParallel(cl)

oznet_valid = read.csv('/datasets/work/d61-af-soilmoisture/work/model_averaging/0_code/oznet_studysites.csv')

data_path = '/datasets/work/d61-af-soilmoisture/work/model_averaging/2_cleaned_data/'
out_path = '/datasets/work/d61-af-soilmoisture/work/model_averaging/3_model_fitting/'

# study period
Dates = seq(as.Date('2016-01-01'), as.Date('2019-12-31'), by='day') # training period

# now use the best model to do the prediction
oznet_valid = read.csv('/datasets/work/d61-af-soilmoisture/work/model_averaging/0_code/oznet_studysites.csv')
# exclude ya7a and ya7d because they had poor performance in cross-validation
# oznet_valid = oznet_valid[-which(oznet_valid$sitename == 'ya7a' | oznet_valid$sitename == 'ya7d'),]

whole_df_train = data.frame(matrix(nrow=0,ncol=23))

for (t in 1:nrow(oznet_valid)){
    single_df_train = read.csv(paste0(data_path, 'OzNet_', oznet_valid$sitename[t], '_cleaned_data.csv'))
    # only choose data between 2016 and 2019
    single_df_train = single_df_train[which(as.Date(single_df_train$time) >= Dates[1] & as.Date(single_df_train$time) <= Dates[length(Dates)]),]
    single_df_train$cluster = oznet_valid$cluster[t]
    whole_df_train  = rbind(whole_df_train, single_df_train)
}

# prepare the global training and test data (i.e., for the whole Australia)
# remove ndvi outliers
huge_ndvi_diff = which(abs(whole_df_train$ndvi_100m - whole_df_train$ndvi_500m) > 0.4)
if (length(huge_ndvi_diff) > 0) whole_df_train = whole_df_train[-huge_ndvi_diff,]

whole_df_train = na.omit(whole_df_train)

# specify the index of clusters
# cluster A means training on B then predicting on A
# cluster B means training on A then predicting on B
spatial_cluster_idx = c()
for (i in 1:2) spatial_cluster_idx[[i]] = which(whole_df_train$cluster != LETTERS[i])

response_train   = whole_df_train[,4]
predictors_train = whole_df_train[,c(5:9,13:20)]

# rf
set.seed(13)
rf.model = caret::train(x=predictors_train,
                        y=response_train,
                        method='rf',
                        importance=TRUE,
                        trControl = trainControl(method='cv', index=spatial_cluster_idx, allowParallel = TRUE))
# save the trained model
saveRDS(rf.model, file=paste0(out_path, 'caret/rf_model_caret_cross_cluster.rds'))

# train disimilarity index
rf.tdi = trainDI(rf.model); print(rf.tdi)
saveRDS(rf.tdi, file=paste0(out_path, 'caret/rf_tdi_caret_cross_cluster.rds'))

# xgb
set.seed(13)
xgb.model = caret::train(x=predictors_train,
                         y=response_train,
                         method='xgbTree',
                         importance=TRUE,
                         trControl = trainControl(method='cv', index=spatial_cluster_idx, allowParallel = TRUE))
# save the trained model
saveRDS(xgb.model, file=paste0(out_path, 'caret/xgb_model_caret_cross_cluster.rds'))

# train disimilarity index
xgb.tdi = trainDI(xgb.model); print(xgb.tdi)
saveRDS(xgb.tdi, file=paste0(out_path, 'caret/xgb_tdi_caret_cross_cluster.rds'))
