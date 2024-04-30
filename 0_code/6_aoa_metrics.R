library(terra)
library(doParallel)

# excute functions in parallel
ncore = 40
cl = makePSOCKcluster(ncore)
registerDoParallel(cl)

path2aoa = '/datasets/work/d61-af-soilmoisture/work/model_averaging/5_sm_aoa/'
out_path = '/datasets/work/d61-af-soilmoisture/work/model_averaging/6_aoa_metrics/'

DOI = seq(as.Date('2016-01-01'), as.Date('2019-12-31'), by='day')

for (model_str in c('rf', 'xgb')){
    for (cv_type in c('spatial_cv', 'cr_cluster')){
        
        print(paste0('Processing model: ', model_str, ' and cv type: ', cv_type))

        aoa_fl   = paste0(path2aoa, model_str, '/', cv_type, '/sm_aoa_', model_str, '_', format(DOI, '%Y%m%d'), '.nc')
        comb_stk = rast(aoa_fl)
        
        di_idx   = seq(1, nlyr(comb_stk), by=2); di_stk  = comb_stk[[di_idx]]
        aoa_idx  = seq(2, nlyr(comb_stk), by=2); aoa_stk = comb_stk[[aoa_idx]]

        di_median = app(di_stk, fun=median, na.rm=TRUE, cores=cl)
        aoa_mean  = app(aoa_stk, fun=mean, na.rm=TRUE, cores=cl)
        aoa_mean[aoa_mean > 0.5] = 1; aoa_mean[aoa_mean <= 0.5] = 0

        writeRaster(di_median, paste0(out_path, 'di_median_', model_str, '_', cv_type, '.tif'), overwrite=TRUE)
        writeRaster(aoa_mean, paste0(out_path, 'aoa_major_', model_str, '_', cv_type, '.tif'), overwrite=TRUE)
    }
}
