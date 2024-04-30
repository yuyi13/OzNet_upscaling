source('~/Workspace/RainfallSpectralAnalysis/SpectralAnalysis/function_SetupForGraphics.R')
library(CAST)
library(caret)
library(terra)
library(sf)
library(gridExtra)
library(foreach)
library(doParallel)

args = commandArgs(trailingOnly=TRUE)
model_str = args[1]

# execute functions in parallel
ncore = 24
#cl = makePSOCKcluster(ncore)
cl = makeCluster(ncore)
registerDoParallel(cl)

# data paths
path2static = '/datasets/work/d61-af-soilmoisture/work/model_averaging/0_static_layers/'
path2fusion = '/datasets/work/d61-af-soilmoisture/work/model_averaging/1_downscaled_data/'
path2et     = '/datasets/work/d61-af-soilmoisture/work/model_averaging/CMRSET_ET/'
path2clim   = '/datasets/work/d61-af-soilmoisture/work/model_averaging/ANUClim_yanco/bilinear/'

out_path = paste0('/datasets/work/d61-af-soilmoisture/work/model_averaging/5_sm_aoa/', model_str, '/cr_cluster/')
if (!dir.exists(out_path)) dir.create(out_path, recursive=TRUE)

DOI = seq(as.Date('2016-01-01'), as.Date('2021-12-31'), by='day')

foreach (k = 1:length(DOI), .packages=c('terra', 'caret', 'CAST')) %dopar% {

    print(paste0('Processing date: ', format(DOI[k], '%Y%m%d')))

    # read the trained model
    if (model_str == 'rf'){
        select_model = readRDS('/datasets/work/d61-af-soilmoisture/work/model_averaging/3_model_fitting/caret/rf_model_caret_cross_cluster.rds')
        select_tdi   = readRDS('/datasets/work/d61-af-soilmoisture/work/model_averaging/3_model_fitting/caret/rf_tdi_caret_cross_cluster.rds')
    } else if (model_str == 'xgb'){
        select_model = readRDS('/datasets/work/d61-af-soilmoisture/work/model_averaging/3_model_fitting/caret/xgb_model_caret_cross_cluster.rds')
        select_tdi   = readRDS('/datasets/work/d61-af-soilmoisture/work/model_averaging/3_model_fitting/caret/xgb_tdi_caret_cross_cluster.rds')
    }

    # static layers
    rst_dem  = rast(paste0(path2static, '100m/dem_100m.tif'))
    rst_awc  = rast(paste0(path2static, '100m/awc_100m.tif'))
    rst_clay = rast(paste0(path2static, '100m/clay_100m.tif'))
    rst_silt = rast(paste0(path2static, '100m/silt_100m.tif'))
    rst_sand = rast(paste0(path2static, '100m/sand_100m.tif'))

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

    # calculate AOA using pre-determined TDI
    sm_aoa = aoa(newdata = pred_stk, trainDI = select_tdi)

    sm_aoa_stk = c(sm_aoa$DI, sm_aoa$AOA)
    writeCDF(sm_aoa_stk, paste0(out_path, 'sm_aoa_', model_str, '_', format(DOI[k], '%Y%m%d'), '.nc'),
            varname='aoa_metrics', longname='Dissimilarity index and Area of Applicability',
            overwrite=TRUE)
}

comment = 
"
    tile_list = makeTiles(x = pred_stk, y = rast(ext(pred_stk), ncols=2, nrows=2),
                          filename = '/datasets/work/d61-af-soilmoisture/work/tmp/pred_tile.tif',
                          overwrite = TRUE)

    # a parallel excution of the function
    tiles_aoa = mclapply(tile_list, function(tile){
                    pred_stk_tiles = rast(tile)
                    aoa(newdata = pred_stk_tiles, trainDI = select_tdi)
                }, mc.cores = 4)

    foreach (t = 1:length(tile_list), .packages=c('terra', 'CAST')) %dopar% {
        pred_stk_tile = rast(tile_list[t])
        aoa_tile = aoa(newdata = pred_stk_tile, trainDI = select_tdi)
        writeRaster(aoa_tile, paste0('/datasets/work/d61-af-soilmoisture/work/tmp/aoa_cr_cluster_tile', t, '.tif'), overwrite=TRUE)
    }
"
