# packages
library(terra)        # SpatRaster
library(randomForest) # Random forest
library(xgboost)      # XGBoost
library(itertools)    # Parallel computing
library(foreach)      # Parallel computing
library(doParallel)   # Parallel computing

# execute functions in parallel
#ncore = 20
#cl = makeCluster(ncore)
#cl = makePSOCKcluster(ncore)
#registerDoParallel(cl)

model_str = commandArgs(trailingOnly=TRUE)[1]

# data paths
path2static = '/datasets/work/d61-af-soilmoisture/work/model_averaging/0_static_layers/'
path2fusion = '/datasets/work/d61-af-soilmoisture/work/model_averaging/1_downscaled_data/'
path2et     = '/datasets/work/d61-af-soilmoisture/work/model_averaging/CMRSET_ET/'
path2clim   = '/datasets/work/d61-af-soilmoisture/work/model_averaging/ANUClim_yanco/bilinear/'

# dates within hdas data period
#DOI = c(seq(as.Date('2019-09-30'), as.Date('2019-10-18'), by='day'),
#        seq(as.Date('2021-03-08'), as.Date('2021-03-26'), by='day'))

# spatial example dates
#DOI = c(as.Date('2017-02-01'), as.Date('2017-08-01'))

# entire period
DOI = seq(as.Date('2016-01-01'), as.Date('2021-12-31'), by='day')

# the specific prediction function for ranger
# source: https://stackoverflow.com/questions/46354103/image-classification-raster-stack-with-random-forest-package-ranger
f_se <- function(model, ...) predict(model, ...)$se

# the specific prediction function for xgboost
# source: https://stackoverflow.com/questions/71947124/predict-xgboost-model-onto-raster-stack-yields-error
xgbpred <- function(model, data, ...) {
    predict(model, newdata=as.matrix(data), ...)
}

###################
# global prediction
###################

out_path = paste0('/datasets/work/d61-af-soilmoisture/work/model_averaging/4_upscaled_sm/100m/global/', model_str, '/')
if (!dir.exists(out_path)) dir.create(out_path, recursive=TRUE)

# read the trained model
#select_model = readRDS(paste0('/datasets/work/d61-af-soilmoisture/work/model_averaging/3_model_fitting/final_model/', model_str, '_model_final.rds'))
select_model = readRDS(paste0('/datasets/work/d61-af-soilmoisture/work/model_averaging/3_model_fitting/caret/', model_str, '_model_caret_4fold_spatial_cv.rds'))

# static layers
rst_dem  = rast(paste0(path2static, '100m/dem_100m.tif'))
rst_awc  = rast(paste0(path2static, '100m/awc_100m.tif'))
rst_clay = rast(paste0(path2static, '100m/clay_100m.tif'))
rst_silt = rast(paste0(path2static, '100m/silt_100m.tif'))
rst_sand = rast(paste0(path2static, '100m/sand_100m.tif'))

#foreach (k = 1:length(DOI), .packages=c('terra', 'randomForest', 'xgboost')) %dopar% {
for (k in 1:length(DOI)){

    # dynamic layers
    rst_alb  = rast(paste0(path2fusion, 'albedo/ESTARFM_albedo_NBAR_cloudrm_', format(DOI[k], '%Y%m%d'), '.tif'))
    rst_ndvi = rast(paste0(path2fusion, 'ndvi/ESTARFM_NDVI_NBAR_cloudrm_', format(DOI[k], '%Y%m%d'), '.tif'))
    rst_lst  = rast(paste0(path2fusion, 'lst/ubESTARFM_LST_daytime_', format(DOI[k], '%Y%m%d'), '.tif'))
    rst_et   = rast(paste0(path2et, 'CMRSET_Landsat_ET_', format(DOI[k], '%Y_%m_01'), '.tif'))
    rst_tavg = rast(paste0(path2clim, 'tavg/ANUClimate_v2-0_tavg_daily_', format(DOI[k], '%Y%m%d'), '.tif'))
    rst_vpd  = rast(paste0(path2clim, 'vpd/ANUClimate_v2-0_vpd_daily_', format(DOI[k], '%Y%m%d'), '.tif'))
    rst_srad = rast(paste0(path2clim, 'srad/ANUClimate_v2-0_srad_daily_', format(DOI[k], '%Y%m%d'), '.tif'))
    rst_rain = rast(paste0(path2clim, 'rain/ANUClimate_v2-0_rain_daily_', format(DOI[k], '%Y%m%d'), '.tif'))

    pred_stk = c(rst_dem, rst_awc, rst_clay, rst_silt, rst_sand, 
                 rst_lst, rst_alb, rst_ndvi, rst_et, 
                 rst_tavg, rst_vpd, rst_srad, rst_rain)

    names(pred_stk) = c('dem','awc','clay','silt','sand',
                        'lst_100m','albedo_100m','ndvi_100m','et_100m',
                        'tavg','vpd','srad','rain')

    print(paste0('predicting ', format(DOI[k], '%Y-%m-%d'), '...'))
    
    # do the prediction using selected model
    if (model_str == 'rf'){
        upscaled_sm = terra::predict(pred_stk, model=select_model, na.rm=TRUE)
    } else if (model_str == 'xgb'){
        upscaled_sm = terra::predict(pred_stk, model=select_model, fun=xgbpred, na.rm=TRUE)
    }
    
    upscaled_sm[upscaled_sm < 0] = NA

    writeRaster(upscaled_sm, filename= paste0(out_path, 'Upscaled_SM_', model_str, '_daily_100m_', format(DOI[k], '%Y%m%d'), '.tif'), overwrite=TRUE)
}

################################
# prediction for cluster A and B
################################

for (j in c('A', 'B')){
#args = commandArgs(trailingOnly=TRUE)
#j = args[1]

    out_path = paste0('/datasets/work/d61-af-soilmoisture/work/model_averaging/4_upscaled_sm/100m/cluster', j, '/', model_str, '/')
    if (!dir.exists(out_path)) dir.create(out_path, recursive=TRUE)

    if (j == 'A'){
        # cluster A region
        region_cluster = ext(146.06,146.16, -34.77, -34.62)
    } else {
        # cluster B region
        region_cluster = ext(146.25,146.35, -35.02, -34.92)
    }

    # read the model for cluster
    #select_model = readRDS(paste0('/datasets/work/d61-af-soilmoisture/work/model_averaging/3_model_fitting/trained_models/', model_str, '_model_cluster_', j, '.rds'))
    select_model = readRDS(paste0('/datasets/work/d61-af-soilmoisture/work/model_averaging/3_model_fitting/caret/', model_str, '_model_caret_cross_cluster.rds'))

    # static layers
    rst_dem  = rast(paste0(path2static, '100m/dem_100m.tif'));  rst_dem  = crop(rst_dem, region_cluster)
    rst_awc  = rast(paste0(path2static, '100m/awc_100m.tif'));  rst_awc  = crop(rst_awc, region_cluster)
    rst_clay = rast(paste0(path2static, '100m/clay_100m.tif')); rst_clay = crop(rst_clay, region_cluster)
    rst_silt = rast(paste0(path2static, '100m/silt_100m.tif')); rst_silt = crop(rst_silt, region_cluster)
    rst_sand = rast(paste0(path2static, '100m/sand_100m.tif')); rst_sand = crop(rst_sand, region_cluster)

    for (k in 1:length(DOI)){

        # dynamic layers
        rst_alb  = rast(paste0(path2fusion, 'albedo/ESTARFM_albedo_NBAR_cloudrm_', format(DOI[k], '%Y%m%d'), '.tif'))
        rst_ndvi = rast(paste0(path2fusion, 'ndvi/ESTARFM_NDVI_NBAR_cloudrm_', format(DOI[k], '%Y%m%d'), '.tif'))
        rst_lst  = rast(paste0(path2fusion, 'lst/ubESTARFM_LST_daytime_', format(DOI[k], '%Y%m%d'), '.tif'))
        rst_et   = rast(paste0(path2et,     'CMRSET_Landsat_ET_', format(DOI[k], '%Y_%m_01'), '.tif'))
        rst_tavg = rast(paste0(path2clim,   'tavg/ANUClimate_v2-0_tavg_daily_', format(DOI[k], '%Y%m%d'), '.tif'))
        rst_vpd  = rast(paste0(path2clim,   'vpd/ANUClimate_v2-0_vpd_daily_', format(DOI[k], '%Y%m%d'), '.tif'))
        rst_srad = rast(paste0(path2clim,   'srad/ANUClimate_v2-0_srad_daily_', format(DOI[k], '%Y%m%d'), '.tif'))
        rst_rain = rast(paste0(path2clim,   'rain/ANUClimate_v2-0_rain_daily_', format(DOI[k], '%Y%m%d'), '.tif'))

        # crop rasters to cluster regions
        rst_alb = crop(rst_alb, region_cluster); rst_ndvi = crop(rst_ndvi, region_cluster); rst_lst = crop(rst_lst, region_cluster); rst_et = crop(rst_et, region_cluster)
        rst_tavg = crop(rst_tavg, region_cluster); rst_vpd = crop(rst_vpd, region_cluster); rst_srad = crop(rst_srad, region_cluster); rst_rain = crop(rst_rain, region_cluster)

        pred_stk = c(rst_dem, rst_awc, rst_clay, rst_silt, rst_sand, 
                     rst_lst, rst_alb, rst_ndvi, rst_et, 
                     rst_tavg, rst_vpd, rst_srad, rst_rain)

        names(pred_stk) = c('dem','awc','clay','silt','sand',
                            'lst_100m','albedo_100m','ndvi_100m','et_100m',
                            'tavg','vpd','srad','rain')

        print(paste0('predicting ', format(DOI[k], '%Y-%m-%d'), ' for cluster ', j, '...'))
    
        # do the prediction using selected model
        if (model_str == 'rf'){
            upscaled_sm = terra::predict(pred_stk, model=select_model, na.rm=TRUE)
        } else if (model_str == 'xgb'){
            upscaled_sm = terra::predict(pred_stk, model=select_model, fun=xgbpred, na.rm=TRUE)
        }
        
        upscaled_sm[upscaled_sm < 0] = NA

        writeRaster(upscaled_sm, filename= paste0(out_path, 'Upscaled_SM_', model_str, '_daily_100m_cluster_', j, format(DOI[k], '_%Y%m%d'), '.tif'), overwrite=TRUE)
    }
}
