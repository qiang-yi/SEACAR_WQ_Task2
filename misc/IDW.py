# -*- coding: utf-8 -*-
"""
@author: cong

"""
import pandas as pd
import os
import arcpy
import math


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
            

# Function to create shapefiles, preparing for ArcPy IDW method
def create_shp_season(df, waterbody, parameter_names, years, seasons, output_folder):
    for area in waterbody:
        if area not in area_shortnames:
            print(f"No managed area found with name: {area}")
            continue

        for year in years:
            for season in seasons:
                for param in parameter_names:
                    if param not in param_shortnames:
                        print(f"No parameter found with name: {param}")
                        continue

                    # Filter data for specific area, parameter, year, and season
                    df_filtered = df[(df['WaterBody'] == area) & 
                                     (df['ParameterName'] == param) & 
                                     (df['Year'] == year) & 
                                     (df['Season'] == season)]

                    if df_filtered.empty:
                        print(f"No data found for area: {area_shortnames[area]}, parameter: {param}, year: {year}, and season: {season}")
                        continue

                    # Print the number of data rows in the filtered DataFrame
                    print(f"Number of data rows for {area_shortnames[area]}, {param}, {year}, {season}: {len(df_filtered)}")

                    temp_csv_path = os.path.join(output_folder, f"temp_{area}_{param}_{year}_{season}.csv")
                    df_filtered.to_csv(temp_csv_path, index=False)

                    feature_class_name = f'SHP_{area_shortnames[area]}_{param_shortnames[param]}_{year}_{season}.shp'
                    feature_class_path = os.path.join(output_folder, feature_class_name)
                    spatial_reference = arcpy.SpatialReference(4152)
                    if arcpy.Exists(feature_class_path):
                        arcpy.Delete_management(feature_class_name)
                    arcpy.management.XYTableToPoint(temp_csv_path, feature_class_path, "Longitude_DD", "Latitude_DD", coordinate_system=spatial_reference)

                    os.remove(temp_csv_path)

                    print(f"Shapefile for {area_shortnames[area]}: {param_shortnames[param]} for year {year} and season {season} has been saved as {feature_class_name}")

                             
# Function to interpolate water quality parameter values using IDW method
def idw_interpolation(folder_path, output_folder, parameters, waterbodies, boundary_path, years, seasons, barrier_folder_path):
    arcpy.CheckOutExtension("Spatial")
    arcpy.env.overwriteOutput = True
    
    spatial_ref = arcpy.SpatialReference(3086)

    parameters = [param_shortnames[param] for param in parameters]
    waterbodies = [area_shortnames[area] for area in waterbodies]
    
    csv_header_written = False

    for filename in os.listdir(folder_path):
        if filename.endswith(".shp"):
            parts = filename[:-4].split("_")
            file_waterbody = parts[1]
            file_parameter = f"{parts[2]}_{parts[3]}"
            file_year = int(parts[4])
            file_season = parts[5]

            if file_waterbody in waterbodies and file_parameter in parameters and file_year in years and file_season in seasons:
                shapefile_path = os.path.join(folder_path, filename)
                print(f"Processing file: {filename}")
                
                arcpy.env.extent = None
                arcpy.env.mask = None
                boundary_found = False

                with arcpy.da.SearchCursor(boundary_path, ["WaterbodyA"]) as cursor:
                    for row in cursor:
                        if row[0] == file_waterbody:  
                            boundary_found = True
                            temp_boundary_path = os.path.join(arcpy.env.scratchFolder, "temp_boundary.shp")

                            if arcpy.Exists(temp_boundary_path):
                                arcpy.Delete_management(temp_boundary_path)
                            #print(f"Selecting boundary: {row[0]}")
                            arcpy.Select_analysis(boundary_path, temp_boundary_path, f"WaterbodyA = '{row[0]}'")
                            arcpy.DefineProjection_management(temp_boundary_path, spatial_ref)
                            extent = arcpy.Describe(temp_boundary_path).extent
                            arcpy.env.extent = extent
                            arcpy.env.mask = temp_boundary_path
                            break

                if not boundary_found:
                    print(f"No boundary found for area {file_waterbody}, skipping shapefile {filename}")
                    continue
                
                # Check if data points are sufficient for IDW interpolation
                data_count = int(arcpy.GetCount_management(shapefile_path).getOutput(0))
                if data_count < 3:
                    print(f"Not enough data for IDW interpolation in {filename}, skipping")
                    continue

                barrier_file_name = f"{file_waterbody}_Barriers.shp"

                barrier_file_path = os.path.join(barrier_folder_path, barrier_file_name)

                if not arcpy.Exists(barrier_file_path):
                    print(f"No barrier file found for {file_waterbody}, using IDW without barriers")
                    barrier_file_path = None

                z_field = "ResultValu"
                errors = []
                
                with arcpy.da.SearchCursor(shapefile_path, ["RowID_", "SHAPE@", z_field]) as cursor:
                    data_points = [row for row in cursor]
                    
                file_cross_validation_count = 0
                
                for test_point in data_points:
                    
                    temp_dataset = os.path.join(arcpy.env.scratchFolder, "temp_dataset.shp")
    
                    if arcpy.Exists(temp_dataset):
                        arcpy.Delete_management(temp_dataset)
        
                    query = f"RowID_ <> {test_point[0]}"
                    arcpy.Select_analysis(shapefile_path, temp_dataset, query)
                    
                    idw_result = arcpy.sa.Idw(temp_dataset, z_field,in_barrier_polyline_features=barrier_file_path)
                    
                    point_geometry = test_point[1]  # Get PointGeometry
                    point = point_geometry.firstPoint  # Get Point
                    location_str = f"{point.X} {point.Y}"
                    
                    raster_save_path = os.path.join(output_folder, "idw_result.tif")

                    if arcpy.Exists(raster_save_path): 
                        arcpy.Delete_management(raster_save_path)
                    idw_result.save(raster_save_path)

                    interpolated_value = arcpy.management.GetCellValue(raster_save_path, location_str, "1").getOutput(0)
                    
                    file_cross_validation_count += 1
                    
                    if interpolated_value != 'NoData':
                        error = float(interpolated_value) - float(test_point[2])
                        errors.append(error)
                    else:
                        errors.append(None)
                    
                    arcpy.Delete_management(temp_dataset)
                    
                rmse, me = calculate_rmse_me(errors)
                print(f"Calculated RMSE: {rmse}, ME: {me} for file: {filename}")
                print(f"File {filename} has completed {file_cross_validation_count} cross-validation iterations.")
                
                csv_path = os.path.join(output_folder, "rmse_me.csv")
                with open(csv_path, "a") as file:
                    if not csv_header_written:
                        file.write("Filename,RMSE,ME\n")
                        csv_header_written = True
                    file.write(f"{filename},{rmse},{me}\n")
                
                # Create IDW tif using all data points, not for cross validation, only for visualization
                idw_result_final = arcpy.sa.Idw(shapefile_path, z_field,in_barrier_polyline_features=barrier_file_path)
                out_raster_name_final = f"IDW_{file_waterbody}_{file_parameter}_{file_year}_{file_season}.tif"
                out_raster_path_final = os.path.join(output_folder, out_raster_name_final)
                idw_result_final.save(out_raster_path_final)
                #print(f"Interpolation complete for file: {filename}")
                
    arcpy.ClearWorkspaceCache_management()
    arcpy.CheckInExtension("Spatial")
    
    # Clean up temp files
    temp_files = [os.path.join(arcpy.env.scratchFolder, f) for f in os.listdir(arcpy.env.scratchFolder) if f.startswith("temp_")]
    for file in temp_files:
        try:
            if os.path.isfile(file):
                os.remove(file)
        except Exception as e:
            print(f"Error removing temp file {file}: {e}")

def calculate_rmse_me(errors):
    valid_errors = [e for e in errors if e is not None]
    if len(valid_errors) > 0: 
        mse = sum([e**2 for e in valid_errors]) / len(valid_errors)
        rmse = math.sqrt(mse)
        me = sum(valid_errors) / len(valid_errors)
    else:
        rmse = None
        me = None
    return rmse, me