# SEACAR Water Quality Project (Task 2: FY2023-2024)

# Task 2a: Evaluating IDW and RK with Barriers

This task include:
1.	Develop Python code to create interpolation maps using RK and IDW with barriers for 6 water quality parameters in 5 waterbodies in seasons around selected storms.
2.	Develop [cross-validation algorithm]() to compute RMSE and ME for IDW with barriers.
3.	Summarizing and comparing RMSE and ME between RK and IWD in the interpolated maps.

Analyses in Task 1a are shared in:
-	[SEACAR_WQ_Exploratory_Analysis_Dis.ipynb](https://github.com/qiang-yi/SEACAR_WQ_Pilot/blob/main/Exploratory_Analysis/SEACAR_WQ_Exploratory_Analysis_Dis.ipynb): Temporal analysis of discrete data
-	[Sample_Location_Analysis_Dis.ipynb](https://github.com/qiang-yi/SEACAR_WQ_Pilot/blob/main/Exploratory_Analysis/Sample_Location_Analysis_Dis.ipynb): Spatial locations of discrete data
-	[SEACAR_WQ_Exploratory_Analysis_Con.ipynb](https://github.com/qiang-yi/SEACAR_WQ_Pilot/blob/main/Exploratory_Analysis/SEACAR_WQ_Exploratory_Analysis_Con.ipynb): Temporal analysis of continuous data: all stations
-	[Sample_Location_Analysis_Con.ipynb](https://github.com/qiang-yi/SEACAR_WQ_Pilot/blob/main/Exploratory_Analysis/Sample_Location_Analysis_Con.ipynb): Spatial locations of discrete data: all stations
-	[SEACAR_WQ_Exploratory_Analysis_Con_Stations.ipynb](https://github.com/qiang-yi/SEACAR_WQ_Pilot/blob/main/Exploratory_Analysis/SEACAR_WQ_Exploratory_Analysis_Con_Stations.ipynb): Temporal analysis of continuous data: separate by station
-	[Sample_Location_Analysis_Con_Stations.ipynb](https://github.com/qiang-yi/SEACAR_WQ_Pilot/blob/main/Exploratory_Analysis/Sample_Location_Analysis_Con_Stations.ipynb): Spatial locations of discrete data: separate by station

# Task 1b: Spatial Interpolation

### 1b.1 Regression Analysis with WQ Parameters
Ordinary least square regression (OLS) and Pearson correlation analyses have been conducted to examine the relations between the potential covariates and water quality parameters. The analyses have been conducted with data from 2016 to 2018 in all managed areas and in separate managed areas. The general procedure is:
1. 	Preprocessing, including outlier removal, daytime data selection, combine continuous and discrete data, and select data in specific managed areas and years
2. 	Aggregate data in identical locations to mean values in wet and dry seasons (tentative dry season: Nov. – Apr., wet season: May to Oct.)
3. 	Extract values from covariate rasters to water quality locations
4.	Conduct Pearson correlation and OLS regression analysis in wet and dry seasons

Regression and correlation analysis are documented in:

- [Covariates_Analysis_All.ipynb](https://github.com/qiang-yi/SEACAR_WQ_Pilot/blob/main/Covariates_Analysis/Covariates_Analysis_All.ipynb): Analysis with 2016-2018 data in all five managed areas
- [Covariates_Analysis_MA.ipynb](https://github.com/qiang-yi/SEACAR_WQ_Pilot/blob/main/Covariates_Analysis/Covariates_Analysis_MA.ipynb): Analysis with 2016-2018 data in all five managed areas
- [Correlation_Covariates.ipynb](https://github.com/qiang-yi/SEACAR_WQ_Pilot/blob/main/Covariates_Analysis/Correlation_Covariates.ipynb): Correlation between covariates with 2016-2018 data in all five managed areas

### 1b.2 Evaluation of Interpolation Methods
The following interpolation methods are selected for evaluation:
- <u>Inverse Distance Weighting (IDW)</u>: weighted average of observed data points in the neighborhood (simplest, fast)
- <u>Ordinary Kriging (OK)</u>: estimate values by fitting a theoretical variogram model (established method, proven performance)
- <u>Empirical Bayesian Kriging (EBK)</u>: estimate values by fitting a non-parametric variogram model (flexible, local model fitting, better suited for complex data pattern)
- <u>EBK Regression Prediction (Regression Kriging or RK)</u>: Extends EBK with explanatory variable that known to affect the predicted values (better suited if there are influential covariates)

The interpolation programs call functions from ArcGIS python interface (arcpy). The performance of these are evaluated through cross-validation. The purpose is to select the best performed method for batch production. The following metrics were derived to evaluate model performance:

- <u>Mean Error (ME)</u>: measures the average absolute difference between the observed and predicted values (measures biases)
- <u>Root Mean Square Error (RMSE)</u>: square root of average squared difference between observed and predicted values (measures accuracy)
- <u>Mean Standardized Error (MSE)</u> standardized by standard deviation of observed values (accuracy relative to data variability)

Performance evaluation of interpolation methods are documented in:

- [Interpolation_ArcGIS_CH.ipynb](https://github.com/qiang-yi/SEACAR_WQ_Pilot/blob/main/Method_Comparison/Interpolation_ArcGIS_CH.ipynb): Interpolation evaluation in Charlotte Harbor
- [Interpolation_ArcGIS_Estero_Bay.ipynb](https://github.com/qiang-yi/SEACAR_WQ_Pilot/blob/main/Method_Comparison/Interpolation_ArcGIS_Estero_Bay.ipynb): Interpolation evaluation in Estero Bay
- [Interpolation_ArcGIS_Big_Bend.ipynb](https://github.com/qiang-yi/SEACAR_WQ_Pilot/blob/main/Method_Comparison/Interpolation_ArcGIS_Big_Bend.ipynb): Interpolation evaluation in Big Bend
- [RK_Covariate_Assessment_CH.ipynb](https://github.com/qiang-yi/SEACAR_WQ_Pilot/blob/main/Method_Comparison/RK_Covariate_Assessment_CH.ipynb): Evaluation of regression kriging with different covariates in Charlotte Harbor
- [RK_Covariate_Assessment_EB.ipynb](https://github.com/qiang-yi/SEACAR_WQ_Pilot/blob/main/Method_Comparison/RK_Covariate_Assessment_EB.ipynb): Evaluation of regression kriging with different covariates in Estero Bay
- [RK_Covariate_Assessment_BB.ipynb](https://github.com/qiang-yi/SEACAR_WQ_Pilot/blob/main/Method_Comparison/RK_Covariate_Assessment_EB.ipynb): Evaluation of regression kriging with different covariates in Big Bend
- [RK_Covariate_Assessment_Biscayne.ipynb](https://github.com/qiang-yi/SEACAR_WQ_Pilot/blob/main/Method_Comparison/RK_Covariate_Assessment_Biscayne.ipynb): Evaluation of regression kriging with different covariates in Biscayne Bay
- [RK_Covariate_Assessment_GTM.ipynb](https://github.com/qiang-yi/SEACAR_WQ_Pilot/blob/main/Method_Comparison/RK_Covariate_Assessment_Biscayne.ipynb): Evaluation of regression kriging with different covariates in Guana Tolomato Matanzas National Estuarine Research Reserve
