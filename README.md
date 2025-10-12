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
