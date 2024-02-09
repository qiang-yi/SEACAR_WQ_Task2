# -*- coding: utf-8 -*-
"""
@author: cong

"""
import pandas as pd
import numpy as np 
import math,time,os
import arcpy


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
            
def merge_with_lat_long(df, lat_long_df):
    lat_long_df = lat_long_df.rename(columns={'ParameterName': 'Parameter'})

    merged_df = pd.merge(df, 
                  lat_long_df[['WaterBody', 'Year', 'Season', 'Parameter', 'x', 'y','RowID','ResultValue']], 
                  on=['WaterBody', 'Year', 'Season', 'Parameter'], 
                  how='left')
    merged_df['RowID'] = merged_df['RowID'].astype('Int64')
    return merged_df

# Function to create shapefiles according to pre-defined season table, preparing for ArcPy IDW method
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
   
                    
# Function to fill NaN values in a specified column of a DataFrame with unique IDs, ensuring that each filled ID is unique.
def fill_nan_rowids(df, rowid_column):
    # Check if the rowid_column exists in the DataFrame, if not create it
    if rowid_column not in df.columns:
        # Initialize the rowid_column with unique integer IDs starting from 1
        df[rowid_column] = range(1, len(df) + 1)
    else:
        # Determine a safe starting point for new IDs, considering existing ones
        max_rowid = df[rowid_column].max()
        if np.isnan(max_rowid) or max_rowid is None:
            max_rowid = 0  # Set to 0 if there are no existing valid IDs

        # Generate a sequence of unique IDs large enough to cover all NaNs or None
        num_nans = df[df[rowid_column].isna() | df[rowid_column].isnull()].shape[0]  # Count NaNs or None
        unique_ids = range(int(max_rowid) + 1, int(max_rowid) + 1 + num_nans)

        # Assign unique IDs to NaN or None values
        nan_index = 0  # To keep track of the next unique ID to be assigned
        for i in range(len(df)):
            if np.isnan(df.at[i, rowid_column]):
                df.at[i, rowid_column] = unique_ids[nan_index]
                nan_index += 1
        return df

                 
# Function to interpolate water quality parameter values using IDW method
def idw_interpolation(df, folder_path, output_folder, boundary_path, barrier_folder_path):
    s_time =time.time() 
    arcpy.CheckOutExtension("Spatial")
    arcpy.env.overwriteOutput = True
    spatialref, c_size, parProFactor,mask = 3086, 30, "40%",''

    unique_combinations = df[['WaterBody', 'Year', 'Season', 'Parameter']].drop_duplicates()
    
    for index, row in unique_combinations.iterrows():
        water_body_full = row['WaterBody']
        year = row['Year']
        season = row['Season']
        parameter_full = row['Parameter']
        
        water_body_short = area_shortnames.get(water_body_full, water_body_full)
        parameter_short = param_shortnames.get(parameter_full, parameter_full)

        filename = f"SHP_{water_body_short}_{parameter_short}_{year}_{season}.shp"
        filepath = os.path.join(folder_path, filename)
            
        if os.path.exists(filepath):
            print(f"Processing file: {filename}")
            
            # Reset the environment settings for the next shapefile
            arcpy.env.extent = None
            arcpy.env.mask = None
            boundary_found = False
            
            # Find the boundary for the waterbody
            with arcpy.da.SearchCursor(boundary_path, ["WaterbodyA"]) as cursor:
                 for row in cursor:
                    if row[0] == water_body_short:
                        boundary_found = True
                        temp_boundary_path = os.path.join(arcpy.env.scratchFolder, "temp_boundary.shp")
                        
                        if arcpy.Exists(temp_boundary_path):
                                arcpy.Delete_management(temp_boundary_path)
                        
                        # Process mask and extent
                        arcpy.Select_analysis(boundary_path, temp_boundary_path, f"WaterbodyA = '{row[0]}'")
                        arcpy.DefineProjection_management(temp_boundary_path, arcpy.SpatialReference(spatialref))
                        extent = arcpy.Describe(temp_boundary_path).extent
                        arcpy.env.extent = extent
                        arcpy.env.mask = temp_boundary_path
                        mask = temp_boundary_path
                        break
                        
            if not boundary_found:
                    print(f"No boundary found for area {water_body_short}, skipping shapefile {filename}")
                    continue
        
            # Check if data points are sufficient for IDW interpolation
            data_count = int(arcpy.GetCount_management(filepath).getOutput(0))
            
            df_update_index = df[
                (df['WaterBody'] == water_body_full) & 
                (df['Year'] == year) & 
                (df['Season'] == season) & 
                (df['Parameter'] == parameter_full)
            ].index

            df.loc[df_update_index, 'Filename'] = filename
            df.loc[df_update_index, 'NumDataPoints'] = data_count

            if data_count < 3:
                print(f"Not enough data for IDW interpolation in {filename}, skipping")
                df.loc[df_update_index, 'RMSE'] = "NaN"
                df.loc[df_update_index, 'ME'] = "NaN"
                df.to_csv(os.path.join(output_folder, "updated_idw.csv"), index=False) 
                continue  
                 
            # Handle barrier files if they exist
            barrier_file_name = f"{water_body_short}_Barriers.shp"
            barrier_file_path = os.path.join(barrier_folder_path, barrier_file_name)
            
            if not arcpy.Exists(barrier_file_path):
                print(f"No barrier file found for {water_body_short}, using IDW without barriers")
                barrier_file_path = None
                
            # Perform IDW interpolation and cross-validation
            z_field = "ResultValu"
            errors = []
            
            with arcpy.da.SearchCursor(filepath, ["RowID_", "SHAPE@", z_field]) as cursor:
                data_points = [row for row in cursor]
                    
            file_cross_validation_count = 0
            
            for test_point in data_points:
                temp_dataset = os.path.join(arcpy.env.scratchFolder, "temp_dataset.shp")
                if arcpy.Exists(temp_dataset):
                    arcpy.Delete_management(temp_dataset)
                    
                query = f"RowID_ <> {test_point[0]}"
                arcpy.Select_analysis(filepath, temp_dataset, query)
                    
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
            e_time =time.time() 
            print(f"Calculated RMSE: {rmse}, ME: {me} for file: {filename}. Time used: {e_time - s_time}")
            print(f"File {filename} has completed {file_cross_validation_count} cross-validation iterations.")
            
            df.loc[df_update_index, 'RMSE'] = rmse
            df.loc[df_update_index, 'ME'] = me
            df.to_csv(os.path.join(output_folder, "updated_idw.csv"), index=False) 

            # Create the final IDW raster for visualization (not for cross-validation)
            with arcpy.EnvManager(outputCoordinateSystem = arcpy.SpatialReference(spatialref),
                                  cellSize = c_size, 
                                  parallelProcessingFactor = parProFactor):
                idw_result_final = arcpy.sa.Idw(filepath, z_field,in_barrier_polyline_features=barrier_file_path)
            out_raster_name_final = f"{water_body_short}_{parameter_short}_{year}_{season}_IDW.tif"
            out_raster_path_final = os.path.join(output_folder, out_raster_name_final)
            idw_result_final.save(out_raster_path_final)
            #print(f"Interpolation complete for file: {filename}")
            
        else:
            print(f"Shapefile not found for: {filename}")
            df_update_index = df[
                (df['WaterBody'] == water_body_full) & 
                (df['Year'] == year) & 
                (df['Season'] == season) & 
                (df['Parameter'] == parameter_full)
            ].index

            df.loc[df_update_index, 'Filename'] = 'NoData'
            df.loc[df_update_index, 'NumDataPoints'] = 0
            df.loc[df_update_index, 'RMSE'] = "NaN"
            df.loc[df_update_index, 'ME'] = "NaN"

            # Save updates to csv
            df.to_csv(os.path.join(output_folder, "updated_idw.csv"), index=False)
            continue
             
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