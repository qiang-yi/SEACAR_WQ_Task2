# -*- coding: utf-8 -*-
"""
@author: Cong, Xiang

"""
import pandas as pd
import numpy  as np
import os
import arcpy
import math
import time,sys


# Predefined abbreviation of waterbody names and parameter names
area_shortnames = {
    'Guana Tolomato Matanzas': 'GTM',
    'Estero Bay': 'EB',
    'Charlotte Harbor': 'CH',
    'Biscayne Bay': 'BB',
    'Big Bend Seagrasses':'BBS'
}

param_shortnames = {
    'Salinity': 'Sal_ppt',
    'Total Nitrogen': 'TN_mgl',
    'Dissolved Oxygen': 'DO_mgl',
    'Turbidity':'Turb_ntu',
    'Secchi Depth':'Secc_m',
    'Water Temperature':'T_c'
}


# Function to empty folder
def delete_all_files(folder_path):
    file_list = os.listdir(folder_path)
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
            

# Function to create shapefiles, preparing for interpolation
def create_shp_season(df, output_folder):
    grouped = df.groupby(['WaterBody', 'Year', 'Season', 'Parameter'])
    
    for group_keys, group_df in grouped:
        area, year, season, param = group_keys
        
        if area not in area_shortnames:
            print(f"No managed area found with name: {area}")
            continue

        if param not in param_shortnames:
            print(f"No parameter found with name: {param}")
            continue

        group_df = group_df.dropna(subset=['x', 'y'])

        if group_df.empty:
            print(f"No valid data found for area: {area_shortnames[area]}, parameter: {param_shortnames[param]}, year: {year}, and season: {season}")
            continue

        print(f"Number of data rows for {area_shortnames[area]}, {param_shortnames[param]}, {year}, {season}: {len(group_df)}")

        temp_csv_path = os.path.join(output_folder, f"temp_{area_shortnames[area]}_{param_shortnames[param]}_{year}_{season}.csv")
        group_df.to_csv(temp_csv_path, index=False)

        feature_class_name = f'SHP_{area_shortnames[area]}_{param_shortnames[param]}_{year}_{season}.shp'
        feature_class_path = os.path.join(output_folder, feature_class_name)
        spatial_reference = arcpy.SpatialReference(3086)

        if arcpy.Exists(feature_class_path):
            arcpy.Delete_management(feature_class_path)
        arcpy.management.XYTableToPoint(temp_csv_path, feature_class_path, "x", "y", coordinate_system=spatial_reference)

        os.remove(temp_csv_path)

        print(f"Shapefile for {area_shortnames[area]}: {param_shortnames[param]} for year {year} and season {season} has been saved as {feature_class_name}")
                             
# Function to interpolate water quality parameter values using RK method
def rk_interpolation(method, folder_path, waterbody, parameter, year, season, covariates, out_raster_folder,out_ga_folder,diagnostic_folder):
    area_shortnames = {
        'Guana Tolomato Matanzas': 'GTM',
        'Estero Bay': 'EB',
        'Charlotte Harbor': 'CH',
        'Biscayne Bay': 'BB',
        'Big Bend Seagrasses':'BBS'
    }
    folder = os.path.join(folder_path+r"shapefiles")
    shpName = []
    for filename in os.listdir(folder):
        if filename.endswith(".shp"):
            shpName.append(filename)        
    waterbody = waterbody
    parameter = parameter
    year      = str(year)
    season    = season
    covariates= str(covariates)
    name1 = "SHP_" + "_".join([waterbody,parameter,year,season]) 
    name  = name1 + ".shp"
    name2 = "_".join([waterbody,parameter,year,season]) + "_RK"
    
    if name in shpName:
        in_features = folder_path + "shapefiles/" + name
        shapefile_path = os.path.join(folder_path+r"shapefiles", name)
        data_count = int(arcpy.GetCount_management(shapefile_path).getOutput(0))
        dependent_field = "ResultValu"
        if "+" in covariates:
            in_explanatory_rasters = []
            covname_list = covariates.split("+")
            for i in covname_list:
                in_explanatory_raster = str(folder_path + "covariates/{}/{}.tif".format(i, waterbody))
                in_explanatory_rasters.append(in_explanatory_raster)        
        else:
            in_explanatory_rasters = str(folder_path + "covariates/{}/{}.tif".format(covariates, waterbody))

        out_ga_layer = out_ga_folder + name2 + "_ga"
        out_raster   = out_raster_folder + name2 + ".tif"
        out_diagnostic_feature_class = diagnostic_folder + name2 + "_diag.shp"

        mask = folder_path + "managed_area_boundary/"+  waterbody + ".shp"
        spatialref, c_size, parProFactor, smooth_r, extent = 3086, 30, "80%",10000, arcpy.Describe(mask).extent
        start_time = time.time()
        try:
            with arcpy.EnvManager(mask = mask,extent = extent,
                                  outputCoordinateSystem = arcpy.SpatialReference(spatialref),
                                  cellSize = c_size, 
                                  parallelProcessingFactor = parProFactor):
                out_surface_raster = arcpy.EBKRegressionPrediction_ga(in_features = in_features, 
                                                                  dependent_field = dependent_field, 
                                                                  out_ga_layer = out_ga_layer,
                                                                  out_raster = out_raster,
                                                                  in_explanatory_rasters = in_explanatory_rasters,
                                                                  out_diagnostic_feature_class = out_diagnostic_feature_class,
                                                                  transformation_type = 'EMPIRICAL',
                                                                  search_neighborhood = arcpy.SearchNeighborhoodSmoothCircular(smooth_r,0.5))

            with arcpy.da.SearchCursor(out_diagnostic_feature_class, ["RMSE","MeanError"]) as cursor:
                data_points = [row for row in cursor]
                rmse = data_points[0][0]
                ME   = data_points[0][1]
            print(f"Processing file: {name}")
            print("--- Time lapse: %s seconds ---" % (time.time() - start_time))
            return 1,rmse,ME,data_count,out_raster
        except Exception:
            e = sys.exc_info()[1]
            print(parameter + " in " + str(year) + " " + season + " caused an error:")
            print(e.args[0])
            return 0,np.nan,np.nan,data_count,np.nan
    else:
        print(f"No data for RK interpolation in {name}, skipping")
        return 0,np.nan,np.nan,0,np.nan
