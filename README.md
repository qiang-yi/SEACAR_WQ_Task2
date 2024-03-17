# SEACAR Water Quality Project (Task 2: FY2023-2024)

## Task 2a: Evaluating IDW and RK with Barriers

This task include:
1.	Develop Python code to create interpolation maps using RK and IDW with barriers for 6 water quality parameters in 5 waterbodies in seasons around selected storms.
2.	Develop [cross-validation algorithm](https://github.com/qiang-yi/SEACAR_WQ_Task2/blob/master/IDW_Analysis.ipynb) to compute RMSE and ME for IDW with barriers.
3.	Summarizing and comparing RMSE and ME between RK and IWD in the interpolated maps.

Outcomes:
-	[Interpolated maps by IDW using all data](https://usf.box.com/s/pumjyw0qz8qg323057h0o7w0ie8u9yy6)
-	[Interpolated maps by IDW using only continuous data](https://usf.box.com/s/b99xo8qev4l48km9k8a971l4aqa04tbe)
- [Interpolated maps by RK]()
- [Comparison of ME, RMSE of the above three methods]()


## Task 2b: IDW using Continuous Sites and Different Time Bins

This task include:
1.	Develop Python code to create interpolation maps using RK and IDW with barriers for 6 water quality parameters in 5 waterbodies in:
  - **Seasons:** Run interpolations in combined seasons across years. For example, in GTM, combining samples in the falls in 2015 and 2016 to run the interpolation. This analysis will be done using both continuous data and discrete data, IDW and RK.
  - **Monthly:** Use IDW to interpolate maps for six 30-day increments prior to the storm day, and then six 30-day increments following the storm day.
  - **Weekly:** Use IDW to interpolate maps in 26 7-day increments prior to the storm day, and then 26 7-day increments following the storm day.
