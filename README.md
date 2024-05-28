# SEACAR Water Quality Project (Task 2: FY2023-2024)

## Task 2a: Evaluating IDW and RK with Barriers

This task include:
1.	Develop Python code to create interpolation maps using [RK](https://github.com/qiang-yi/SEACAR_WQ_Task2/blob/master/RK.ipynb) and [IDW](https://github.com/qiang-yi/SEACAR_WQ_Task2/blob/master/RK.ipynb) with barriers for 6 water quality parameters in 5 waterbodies in seasons around selected storms.
2.	Develop [cross-validation algorithm](https://github.com/qiang-yi/SEACAR_WQ_Task2/blob/master/IDW_Analysis.ipynb) to compute RMSE and ME for IDW with barriers.
4.	Summarizing and comparing RMSE and ME between RK and IWD in the interpolated maps.

Outcomes:
-	[Interpolated maps by IDW using all data](https://usf.box.com/s/arxm9dm0d7mibw3vsiyob9bxvet35ght)
-	[Interpolated maps by IDW using only continuous data](https://usf.box.com/s/o0neeftt00h4q2nxud552aglnektcnyc)
- [Interpolated maps by RK](https://usf.box.com/s/sk00lz3gdu9qx0hqqhofdd8eb1ap4xdf)
- [RMSE and ME of all above methods](https://usf.box.com/s/gzgmbuqr3yj5fse8qsla4ik3kg2ekjd0)
- [Analyses of RMSE and ME](https://github.com/qiang-yi/SEACAR_WQ_Task2/blob/master//RK_IDW_comparison.ipynb)

All deliverables are stored in the folder [..\Box\SEACAR_OEAT_FY22-23\SEACAR_WQ_Analysis_Pilot\Deliverables Task 2a](https://usf.box.com/s/1n84o7e05dfbooaskjw38iw6r1b4oknv)

```
├───ArcGIS_project              (ArcGIS project displaying output maps)
│   ├───Interpolated_maps
├───GIS_Data                    (Auxiliary GIS data)
├───Interpolated_maps           (Interpolated raster maps)
│   ├───idw_All
│   ├───idw_Con
│   └───rk_All
├───result                      (Comparison results of RK&IDW)
│   ├───result_v1
│   ├───result_v2
│   └───result_v3
├───shapefiles                  (Data points used for interpolation)
│   ├───shapefiles_All
│   └───shapefiles_Con
└───StandardizedOutputs         (Standardized output for Deliverables)
    ├───GIS_Data
    ├───Interpolated_maps
    └───python

```


## Task 2b: IDW using Continuous Sites and Different Time Bins

This task include:
1. Develop Python code to create interpolation maps using RK and IDW with barriers for 6 water quality parameters in 5 waterbodies in:
  - **Seasons:** Interpolation using RK and IDW for both continuous data and discrete data in the following three options:
    - Option 1: Four seasons in the year of hurricane events
        - [RK interpolated maps](https://usf.box.com/s/oqoujzr6396i0eys9hgelim19hjcdm66)
        - [IDW interpolated maps](https://usf.box.com/s/bomsn99aon61vsk1hyq2pbolnaxq3sp8)
        - Python codes for the [IDW]() and [RK]() interpolation
    - Option 2: Four seasons across two years
        - [RK interpolated maps](https://usf.box.com/s/um59gwy0xr4rqo4j3xljf9ndoadt78tc)
        - [IDW interpolated maps](https://usf.box.com/s/0ts270wy54zuoe9u7jywbkep6z74qqdg)
        - Python codes for the [IDW]() and [RK]() interpolation
    - Option 3: Two seasons (wet/dry) in two years
        - [RK interpolated maps](https://usf.box.com/s/eec00ic89joxna28brzszoce1lhorldu)
        - [IDW interpolated maps](https://usf.box.com/s/6lopa2426gyf2ajfnfqwcbwjf6v2bv9n)
        - Python codes for the [IDW]() and [RK]() interpolation.
    - [Comparison of RK vs IDW in the above options]()
  - **Monthly:** Use IDW to interpolate maps for six 30-day increments prior to the storm day, and then six 30-day increments following the storm day. Only continuous data are used.
    - [Interpolated monthly maps](https://usf.box.com/s/t1ndsi3r85xtqq4wt8eli3qizx67ysn9)
    - [Python codes for the IDW interpolation]()
  - **Weekly:** Use IDW to interpolate maps in 26 7-day increments prior to the storm day, and then 26 7-day increments following the storm day. Only continuous data are used.
    - [Interpolated weekly maps](https://usf.box.com/s/0tlcpc7o3264tcpmwow7q3z7352hovx9)
    - [Python codes for the IDW interpolation]()
  - [Analysis of RMSE & ME in the monthly and weekly maps]()

2. A preliminary gap analysis that use discrete data to validate the monthly and weekly maps interpolated using continuous data.
    - [Differences between continuous and discrete data in monthly maps](https://usf.box.com/s/y42vndozw0n9ah52jv84ykr2zv6qebxz)
    - [Differences between continuous and discrete data in weekly maps](https://usf.box.com/s/mim1oqlgtlpg0tl2u9ytsey1p6paifsw)
    - [Kernel density maps created from the differences](https://usf.box.com/s/ybvf89ccgkztn0xohl02dqofwyq6c0op)
    - [Visualization of the kernel density maps]()


  All deliverables are stored in the folder [..\Box\SEACAR_OEAT_FY22-23\SEACAR_WQ_Analysis_Pilot\Deliverables Task 2b](https://usf.box.com/s/p3e2uph06y0araw56zwm0a0oh3sgu4f9)

```
├───gap_analysis                  (Deliverables of the gap analysis)
│   └───kde_maps
│       ├───month
│       └───week
├───Interpolated_maps             (Interpolated maps in seasonal, monthly and weekly intervals)
│   ├───CrossYear_IDW_All
│   ├───CrossYear_RK_All
│   ├───FourSeasons_IDW_All
│   ├───FourSeasons_RK_All
│   ├───IDW_Month
│   ├───IDW_Week
│   ├───TwoSeasons_IDW_All
│   └───TwoSeasons_RK_All
├───rmse_me                       (RMSE and ME in the interpolated maps)
└───shapefiles                    (Point data used for the interpolation)
    ├───CrossYear_shapefiles_All
    ├───FourSeasons_All
    ├───IDW_Month
    ├───IDW_Week
    └───TwoSeasons_All
```
## Task 2f: Parameter Statistics
Calculating summary statistics () of the parameters per monitoring location and season.
- Tables of summary statistics in the five managed areas:
  - [Big Bend Seagrass]()
  - [Biscayne Bay]()
  - [Charlotte Harbor]()
  - [Estero Bay]()
  - [Guana Tolomato Matanzas]()

- Boxplots of summary statistics in the five managed areas:
  - [Big Bend Seagrass]()
  - [Biscayne Bay]()
  - [Charlotte Harbor]()
  - [Estero Bay]()
  - [Guana Tolomato Matanzas]()

- Boxplots per monitoring location can also be viewed from the [web map](https://gis.waterinstitute.usf.edu/maps/SEACAR-OEAT-WQ/), in the "SEACAR OEAT WQ Year 2/OEAT Task 2f Box Plots" layer.
