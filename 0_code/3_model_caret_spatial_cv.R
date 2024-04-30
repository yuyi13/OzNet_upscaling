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
    single_df_train$fold = oznet_valid$iteration[t]
    whole_df_train  = rbind(whole_df_train, single_df_train)
}

# prepare the global training and test data (i.e., for the whole Australia)
# remove ndvi outliers
huge_ndvi_diff = which(abs(whole_df_train$ndvi_100m - whole_df_train$ndvi_500m) > 0.4)
if (length(huge_ndvi_diff) > 0) whole_df_train = whole_df_train[-huge_ndvi_diff,]

whole_df_train   = na.omit(whole_df_train)

# extract some minor samples for testing
#set.seed(13)
#sample_idx = sample(1:nrow(whole_df_train), 1000)

# Standard k-fold cross-validation can lead to considerable misinterpretation in spatial-temporal modelling tasks. 
# This function can be used to prepare a Leave-Location-Out, Leave-Time-Out or Leave-Location-and-Time-Out cross-validation 
# as target-oriented validation strategies for spatial-temporal prediction tasks. 
# https://hannameyer.github.io/CAST/reference/CreateSpacetimeFolds.html
#spatial_cv = CAST::CreateSpacetimeFolds(whole_df_train, spacevar = 'sitename', timevar = NA, k = 4, class = NA, seed=13)

# specify the index of folds
# we cannot use the packaged method because that yielded different folds with previous results
spatial_cv_idx = c()
for (i in 1:4){
    spatial_cv_idx[[i]] = which(whole_df_train$fold != i)
}

response_train   = whole_df_train[,4]
predictors_train = whole_df_train[,c(5:9,13:20)]

# rf
set.seed(13)
rf.model = caret::train(x=predictors_train,
                        y=response_train,
                        method='rf',
                        importance=TRUE,
                        trControl = trainControl(method='cv', index=spatial_cv_idx, allowParallel = TRUE))
# save the trained model
saveRDS(rf.model, file=paste0(out_path, 'caret/rf_model_caret_4fold_spatial_cv.rds'))

# train disimilarity index
rf.tdi = trainDI(rf.model); print(rf.tdi)
saveRDS(rf.tdi, file=paste0(out_path, 'caret/rf_tdi_caret_4fold_spatial_cv.rds'))

# xgb
set.seed(13)
xgb.model = caret::train(x=predictors_train,
                         y=response_train,
                         method='xgbTree',
                         importance=TRUE,
                         trControl = trainControl(method='cv', index=spatial_cv_idx, allowParallel = TRUE))
# save the trained model
saveRDS(xgb.model, file=paste0(out_path, 'caret/xgb_model_caret_4fold_spatial_cv.rds'))

# train disimilarity index
xgb.tdi = trainDI(xgb.model); print(xgb.tdi)
saveRDS(xgb.tdi, file=paste0(out_path, 'caret/xgb_tdi_caret_4fold_spatial_cv.rds'))
