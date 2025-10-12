# ai_ocean_model

Sample GODAS Potential Temperature at level 0

<img width="447" height="354" alt="image" src="https://github.com/user-attachments/assets/1188221a-7b7c-48a2-8b02-96183d8538e6" />

Sample GODAS Potential Temperature at level 40

<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/2cc40689-2fea-494f-9692-edb9cf247714" />

Sample GODAS Salinity at level 0

<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/0969d6b0-029e-4bfc-ab79-83760264166f" />

Sample GODAS Salinity at level 40

<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/157e3739-3bcf-4fea-a114-f19248486cc5" />

How does a GODAS dataset look like:

<xarray.Dataset> Size: 96MB
Dimensions:                (time: 1, level: 40, lat: 418, lon: 360)
Coordinates:
  * time                   (time) datetime64[ns] 8B 2024-02-09
  * level                  (level) int64 320B 1 2 3 4 5 6 ... 35 36 37 38 39 40
  * lat                    (lat) float64 3kB -74.5 -74.17 -73.83 ... 64.03 64.5
  * lon                    (lon) float64 3kB 0.5 1.5 2.5 ... 357.5 358.5 359.5
Data variables:
    potential_temperature  (time, level, lat, lon) float64 48MB ...
    salinity               (time, level, lat, lon) float64 48MB ...

What are the different vertical levels of GODAS ?

Temporal Coverage: Monthly values for 1980/01 - 2025/08

Spatial Coverage: 0.333 degree latitude x 1.0 degree longitude global grid (418x360)
74.5S - 64.5N, 0.5E - 359.5E

Levels: 5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 135, 145, 155, 165, 175, 185, 195, 205, 215, 225, 238, 262, 303, 366, 459, 584, 747, 949, 1193, 1479, 1807, 2174, 2579, 3016, 3483, 3972, 4478 m depth.

10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 231, 250, 282, 334, 412, 521, 665, 848, 1071, 1336, 1643, 1990, 2376, 2797, 3249, 3727, 4225, 4736 m depth (for geometric vertical velocity only).

Are there nan values in the data ?

At level = 0 which is 5 m depth, numpy is able to identify 45352 nan values using `print(np.sum(np.isnan(ds.isel(level=0).salinity.values)))`

How should normalization be done ?

I think normalization should be done with absolute min max / mean std values for the entire 3d array to preserve the vertical structure in the normlized space, but the papers on AI atmosphere and other papers treat each level as a separate variable, but we should try the idea of entire min max for normalization 

How should we treat the nan values in the data ? 

The nan values can be treated by (i) inputting large values that are not represented in the data, but that can have a tendency to pollute the coastal values, (ii) Ola model uses the coast to coast linear interpolation, (iii) We have used bilinear interpolation in the land model, lets try and go with it in the AI ocean model as well.

What are the vertical levels used by Ola paper ?

Ocean Depth: Potential temperature at the following depths in meters: [0.5, 9.8, 47.3, 97.2, 200.3, 301.7 ]

So what are the levels that we will use the data at ?

If your target is seasonal skill / ENSO

Stay focused on the upper ~300 m—that’s what Ola modeled and where most seasonal predictability resides. Either of the two sets above is appropriate; start with Ola-mimic-6 for simplicity, or Ola-plus-8 if your memory budget allows. 
arXiv

If you care about deeper heat content/decadal signals later

Add a sparse tail below 300 m, e.g. [459, 747, 1193, 1807] m, to capture deeper thermocline/intermediate waters—still keeping the upper-ocean dense. 

Levels that we will use : : [5, 15, 25, 45, 95, 155, 205, 303] m.

Indices these correspond to: 0, 1, 2, 4, 9, 15, 20, 25
