# Sleek ERA5 variable check (ARCO Zarr) using ERA5 long names only
# Targets: [u10, v10, sw_down, lw_down, t2m, q2m, precip, sp]

import xarray as xr

ZARR = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

ds = xr.open_zarr(ZARR, consolidated=True, storage_options={"token": "anon"})
avail = set(ds.data_vars)

# ERA5 long-name targets (with humidity long-name fallbacks that are still ERA5 long names)
TARGETS = {
    "u10":  ["10m_u_component_of_wind"],
    "v10":  ["10m_v_component_of_wind"],
    "sw_down": ["surface_solar_radiation_downwards"],
    "lw_down": ["surface_thermal_radiation_downwards"],
    "t2m": ["2m_temperature"],
    "q2m": ["2m_dewpoint_temperature"],
    "precip": ["total_precipitation"],
    "sp": ["surface_pressure"],
}

def resolve(names):  # first present long name
    return next((n for n in names if n in avail), None)

resolved = {k: resolve(v) for k, v in TARGETS.items()}
found = {k: v for k, v in resolved.items() if v}
missing = [k for k, v in resolved.items() if v is None]

print("Available variables:", len(avail))
print("\nResolved (logical -> ERA5 long name):")
for k, v in resolved.items():
    print(f"  {k:7s} -> {v}")

print("\n✅ Found:")
for k, v in found.items():
    print(f"  {k}: {v}")

print("\n❌ Missing (no ERA5 long name found):", missing or "None")

# Optional: subset with found variables (keeps dataset long names)
if found:
    ds_subset = ds[list(found.values())]
    print("\nSubset contains:", list(ds_subset.data_vars))


