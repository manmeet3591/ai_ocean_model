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

Temporal Coverage
Monthly values for 1980/01 - 2025/08
Spatial Coverage
0.333 degree latitude x 1.0 degree longitude global grid (418x360)
74.5S - 64.5N, 0.5E - 359.5E
Levels
5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 135, 145, 155, 165, 175, 185, 195, 205, 215, 225, 238, 262, 303, 366, 459, 584, 747, 949, 1193, 1479, 1807, 2174, 2579, 3016, 3483, 3972, 4478 m depth.
10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 231, 250, 282, 334, 412, 521, 665, 848, 1071, 1336, 1643, 1990, 2376, 2797, 3249, 3727, 4225, 4736 m depth (for geometric vertical velocity only).


