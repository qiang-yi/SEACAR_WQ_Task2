import numpy as np
import pandas as pd
import geopandas as gpd
import arcpy
import time,sys
import arcgisscripting
from arcpy.sa import *
arcpy.env.overwriteOutput = True

# Select data in specific area, time period, parameter, and season
# s_date,e_date needs to be in format mm/dd/yyyy format
def select_aggr_area_season(df_all, year, season, area, para):
    
    df_all = df_all[(df_all['ParameterName']==para)&
          (df_all['Year']==year)&
           (df_all['Season']==season)&
            (df_all['WaterBody']==area)]
    
    df_mean = df_all.groupby(['Latitude_DD','Longitude_DD'])["ResultValue"].agg("mean").reset_index()    
    gdf = gpd.GeoDataFrame(df_mean, geometry = gpd.points_from_xy(df_mean.Longitude_DD, df_mean.Latitude_DD), crs="EPSG:4326")
    
    return df_mean, gdf

def extract_val_result(inLayer, index):
    cvResult = arcpy.CrossValidation_ga(inLayer)
    Stat = pd.DataFrame(
                {
                  "meanError": round(float(cvResult.meanError),4),
                  "meanStandardizedError": round(float(cvResult.meanStandardized),4),
                  "rootMeanSquareError": round(float(cvResult.rootMeanSquare),4)
                                              },index=[index])
    return Stat    

def interpolation(method, input_point, out_raster, 
                         z_field, out_ga_layer, extent, mask, ga_to_raster,
                         in_explanatory_rasters = None):
    
    start_time = time.time()
    print("Start the interpolation with the {} method".format(method.upper()))
    smooth_r, spatialref, c_size, parProFactor = 10000, 3086, 30, "80%"

# ---------------------------- IDW ---------------------------------
    if   method == "idw":
        with arcpy.EnvManager(extent = extent, mask = mask,outputCoordinateSystem = arcpy.SpatialReference(spatialref), cellSize = c_size, parallelProcessingFactor = parProFactor):

                 arcpy.ga.IDW(in_features = input_point, 
                 z_field = z_field, 
#                This layer is not generated  
                 out_ga_layer = out_ga_layer,
                 out_raster   = out_raster
                )
        
        ValStat = extract_val_result(out_ga_layer, method.upper())
        
        print("--- Time lapse: %s seconds ---" % (time.time() - start_time))
        
        return out_raster, ValStat
                
# ---------------------- Ordinary Kriging ---------------------------
    elif method == "ok":
        out_cv_table = out_raster.replace('.tif','_table')
        with arcpy.EnvManager(extent = extent, mask = mask,outputCoordinateSystem = arcpy.SpatialReference(spatialref), cellSize = c_size, parallelProcessingFactor = parProFactor):
            ok_out = arcpy.ga.ExploratoryInterpolation(in_features = input_point, value_field = z_field, 
                                                   out_cv_table = out_cv_table, out_geostat_layer = out_ga_layer, 
                                                   interp_methods = ['ORDINARY_KRIGING'], comparison_method = 'SINGLE', 
                                                   criterion = 'ACCURACY')
            arcpy.conversion.ExportTable(out_cv_table, out_cv_table + '.csv')
            ValStat = pd.read_csv(out_cv_table + '.csv')
            ValStat = ValStat[ValStat['DESCR'] == 'Ordinary Kriging – Default'][['DESCR','ME','ME_STD','RMSE']].rename(columns = {"RMSE": "rootMeanSquareError", "ME": "meanError",'ME_STD':'meanStandardizedError'})
            os.remove(out_cv_table + '.csv'+'.xml')
            os.remove(out_cv_table + '.csv')
            
            ValStat['DESCR'] = method.upper()
            ValStat = ValStat.set_index('DESCR')
            ValStat.index.name= None
            
            arcpy.GALayerToRasters_ga(in_geostat_layer = out_ga_layer, out_raster = out_raster, output_type = "PREDICTION", cell_size = c_size)
        
            return out_raster, ValStat

        
# ---------------------- Empirical Bayesian Kriging ---------------------------
    elif method == "ebk":
        start_time = time.time()

        with arcpy.EnvManager(extent = extent, mask = mask,outputCoordinateSystem = arcpy.SpatialReference(spatialref), cellSize = c_size, parallelProcessingFactor = parProFactor):
            arcpy.ga.EmpiricalBayesianKriging(in_features = input_point, 
                                      z_field = z_field, 
                                    # This layer is not generated  
                                      out_ga_layer = out_ga_layer,
                                      out_raster   = out_raster,
                                     # transformation_type = 'EMPIRICAL',
                                    search_neighborhood = arcpy.SearchNeighborhoodSmoothCircular(smooth_r,0.5))
            arcpy.GALayerToRasters_ga(out_ga_layer, ga_to_raster,"PREDICTION_STANDARD_ERROR", None, c_size, 1, 1, "")
        ValStat = extract_val_result(out_ga_layer, method.upper())
        print("--- Time lapse: %s seconds ---" % (time.time() - start_time))
        return out_raster, ValStat
            
# ---------------------- Regression Kriging ---------------------------
    elif method == "rk":
        start_time = time.time()
        with arcpy.EnvManager(extent = extent, mask = mask,outputCoordinateSystem = arcpy.SpatialReference(spatialref), cellSize = c_size, parallelProcessingFactor = parProFactor):
            out_surface_raster = arcpy.EBKRegressionPrediction_ga(in_features = input_point, 
                                                                   dependent_field = z_field, 
                                                                  out_ga_layer = out_ga_layer,
                                                                    out_raster = out_raster,
                                                                  in_explanatory_rasters = in_explanatory_rasters,
                                                                   transformation_type = 'EMPIRICAL',
                                                                  search_neighborhood = arcpy.SearchNeighborhoodSmoothCircular(smooth_r,0.5))
            # Convert GA layer to standard error of prediction raster
            arcpy.GALayerToRasters_ga(out_ga_layer, ga_to_raster,"PREDICTION_STANDARD_ERROR", None, c_size, 1, 1, "")
        ValStat = extract_val_result(out_ga_layer, method.upper())
        print("--- Time lapse: %s seconds ---" % (time.time() - start_time))
        return out_raster, ValStat

    

    
def interpolation_auto(method,gis_path,dataframe,managed_area,Year,Season,parameter,covariates):
    

    para_ls = ["Salinity","Total Nitrogen","Dissolved Oxygen","Turbidity","Secchi Depth","Water Temperature"]
    para_ls_ab = ["S","TN","DO","T","SD","WT"]
    # Convert full MA names to short names
    dictArea    = {
    'Guana Tolomato Matanzas': 'GTM',
    'Estero Bay': 'EB',
    'Charlotte Harbor': 'CH',
    'Biscayne Bay': 'BB',
    'Big Bend Seagrasses':'BBS'
    }
    dictPara = {
    'Salinity': 'Sal_ppt',
    'Total Nitrogen': 'TN_mgl',
    'Dissolved Oxygen': 'DO_mgl',
    'Turbidity':'Turb_ntu',
    'Secchi Depth':'Secc_m',
    'Water Temperature':'T_c'
    }
    SpatialRef = '3086'
    
    method = method
    dataframe = dataframe
    Area   = managed_area
    Year   = Year
    Season = Season
    Para   = parameter
    covariates = covariates
    fname = [dictArea[Area],Year,Season,dictPara[Para]]
    
    input_pt = gis_path+"input_point/{}_{}_{}_{}.shp".format(*fname)
    
    df,gdf= select_aggr_area_season(dataframe,Year,Season, Area, Para)
    
    try:
        gdf   = gdf.to_crs(int(SpatialRef))
        boundary_shp = gis_path+ 'managed_area_boundary/{}.shp'.format(dictArea[Area])
        gdf.to_file(input_pt,driver='ESRI Shapefile',crs="EPSG:"+SpatialRef)
        MA = gpd.read_file(gis_path + "OEAT_Waterbody_Boundaries/OEAT_Waterbody_Boundary.shp")
        boundary = MA[MA['WaterbodyA']==dictArea[Area]].to_crs(int(SpatialRef))
        boundary.to_file(boundary_shp , driver='ESRI Shapefile',crs="EPSG:"+SpatialRef)
        extent = str(boundary.geometry.total_bounds).replace('[','').replace(']','')

        if type(covariates) == str:
            in_explanatory_rasters = gis_path + "covariates/{}/{}.tif".format(covariates, dictArea[Area])
        elif type(covariates) == list:
            in_explanatory_rasters = []
            for i in range(len(covariates)):
                in_explanatory_raster = str(gis_path + "covariates/{}/{}.tif".format(covariates[i], dictArea[Area]))
                in_explanatory_rasters.append(in_explanatory_raster)

        in_features = input_pt
        out_raster = gis_path +"output_raster/{}_{}_{}_{}.tif".format(*fname)
        value_field = "ResultValu"
        out_ga_layer = gis_path +"ga_layer/{}_{}_{}_{}.lyrx".format(*fname)
        ga_to_raster = gis_path + 'standard_error_prediction/{}_{}_{}_{}_sep.tif'.format(*fname)
        in_explanatory_rasters = in_explanatory_rasters
        mask = gis_path+ 'managed_area_boundary/{}.shp'.format(dictArea[Area])


        Result,Stat = interpolation(
                        method = method, input_point = in_features, out_raster = out_raster, 
                        z_field = value_field, out_ga_layer = out_ga_layer, extent = extent, 
                        mask = mask, ga_to_raster = ga_to_raster, in_explanatory_rasters = in_explanatory_rasters)
        
        return out_raster,out_ga_layer,ga_to_raster

    except Exception:
            e = sys.exc_info()[1]
            print(Para + " in " + str(Year) + " " + Season + " caused an error:")
            print(e.args[0])
            return np.nan,np.nan,np.nan
