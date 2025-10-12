import xarray as xr
import matplotlib.pyplot as plt
path = '/scratch/08105/ms86336/godas_pentad/'

ds = xr.open_dataset(path+'godas.P.20240209.nc')

print(ds)

# ds.isel(level=0).potential_temperature.plot()
# plt.savefig('pot_temp_level_0.png')


# ds.isel(level=0).salinity.plot()
# plt.savefig('salinity_level_0.png')


# ds.isel(level=39).potential_temperature.plot()
# plt.savefig('pot_temp_level_40.png')

ds.isel(level=39).salinity.plot()
plt.savefig('salinity_level_40.png')

import numpy as np

print(np.sum(np.isnan(ds.isel(level=0).salinity.values)))
