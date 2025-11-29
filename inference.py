import os
import json
from pathlib import Path
from copy import deepcopy

import xarray as xr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import earth2grid

# ============================================================
#                MODEL DEFINITIONS (same as training)
# ============================================================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, conv_type=nn.Conv3d, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            conv_type(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            conv_type(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, conv_type=nn.Conv3d):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            DoubleConv(in_channels, out_channels, conv_type=conv_type)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, up_channels, skip_channels, out_channels, trilinear=True):
        super().__init__()
        if trilinear:
            self.up = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(up_channels, up_channels, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv = DoubleConv(up_channels + skip_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, activation=None):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.activation == 'sigmoid':
            return torch.sigmoid(x)
        elif self.activation == 'tanh':
            return torch.tanh(x)
        return x

class DepthwiseSeparableConv3d(nn.Module):
    def __init__(self, nin, nout, kernel_size, padding, kernels_per_layer=1):
        super(DepthwiseSeparableConv3d, self).__init__()
        self.depthwise = nn.Conv3d(nin, nin * kernels_per_layer, kernel_size=kernel_size, padding=padding, groups=nin)
        self.pointwise = nn.Conv3d(nin * kernels_per_layer, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, width_multiplier=1, trilinear=True, use_ds_conv=False, out_activation=None):
        super(UNet, self).__init__()
        _channels = (32, 64, 128, 256)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.channels = [int(c * width_multiplier) for c in _channels]
        self.trilinear = trilinear
        self.convtype = DepthwiseSeparableConv3d if use_ds_conv else nn.Conv3d

        self.inc = DoubleConv(n_channels, self.channels[0], conv_type=self.convtype)
        self.down1 = Down(self.channels[0], self.channels[1], conv_type=self.convtype)
        self.down2 = Down(self.channels[1], self.channels[2], conv_type=self.convtype)
        self.down3 = Down(self.channels[2], self.channels[3], conv_type=self.convtype)

        factor = 2 if trilinear else 1

        self.up1 = Up(self.channels[3], self.channels[2], self.channels[2] // factor, trilinear)
        self.up2 = Up(self.channels[2] // factor, self.channels[1], self.channels[1] // factor, trilinear)
        self.up3 = Up(self.channels[1] // factor, self.channels[0], self.channels[0], trilinear)

        self.outc = OutConv(self.channels[0], n_classes, activation=out_activation)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits

# ============================================================
#                     SETTINGS & CONSTANTS
# ============================================================

ERA5_ZARR = 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'
GODAS_PATH = '/scratch/08105/ms86336/godas_pentad/'  # directory with godas.P.*.nc

# Healpix grid
LEVEL = 6
NSIDE = 2 ** LEVEL

# GODAS levels (indices) and variables (as in training)
GODAS_LEVELS = [0, 1, 2, 4, 9, 15, 20, 25]
GODAS_VARS = ["potential_temperature", "salinity"]

# Atmosphere logical vars (same as training)
ERA5_LOGICAL = ["u10", "v10", "sw_down", "lw_down", "t2m", "q2m", "precip", "sp"]

# Normalization JSON folder (same as training)
NORM_DIR = Path('./normalization_parts')
BEST_MODEL_PATH = "best_unet_ocean_model.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device used for inference is", DEVICE)

# ============================================================
#                       NORMALIZATION
# ============================================================

def load_json(path: Path):
    with open(path, 'r') as f:
        return json.load(f)

def norm_minmax(x, vmin, vmax):
    denom = (vmax - vmin) if (vmax - vmin) != 0 else 1.0
    return (x - vmin) / denom

def denorm_minmax(x_norm, vmin, vmax):
    return x_norm * (vmax - vmin) + vmin

# ERA5 logical min/max
ERA5_MINMAX = {v: load_json(NORM_DIR / f"era5_{v}.json") for v in ERA5_LOGICAL}

# GODAS per (var, level)
GODAS_MINMAX = {
    (var, lev): load_json(NORM_DIR / f"godas_{var}_l{lev}.json")
    for var in GODAS_VARS for lev in GODAS_LEVELS
}

# Keep channel ordering consistent with training
STATE_CHANNELS = [(var, lev) for var in GODAS_VARS for lev in GODAS_LEVELS]

# ============================================================
#                         GRIDS
# ============================================================

_regridder_cache = {}

def get_regridder_for(da: xr.DataArray):
    """
    Lat-lon -> HEALPix regridder (cached by (nlat, nlon)).
    """
    if 'latitude' in da.dims:
        if not da.latitude.values[0] < da.latitude.values[-1]:
            da = da.sortby('latitude')
    if 'lat' in da.dims:
        if not da.lat.values[0] < da.lat.values[-1]:
            da = da.sortby('lat')

    # handle either (lat, lon) or (latitude, longitude)
    lat_name = 'lat' if 'lat' in da.dims else 'latitude'
    lon_name = 'lon' if 'lon' in da.dims else 'longitude'

    nlat, nlon = da[lat_name].size, da[lon_name].size
    key = (nlat, nlon)
    if key not in _regridder_cache:
        src_grid = earth2grid.latlon.equiangular_lat_lon_grid(nlat, nlon)
        hpx_grid = earth2grid.healpix.Grid(level=LEVEL, pixel_order=earth2grid.healpix.XY())
        _regridder_cache[key] = (earth2grid.get_regridder(src_grid, hpx_grid), src_grid, hpx_grid)
    return da, _regridder_cache[key][0], _regridder_cache[key][1], _regridder_cache[key][2]

def get_latlon_back_regridder_from_godas(ds_godas):
    """
    Builds a HEALPix -> GODAS lat-lon regridder (once).
    """
    nlat = ds_godas['lat'].size
    nlon = ds_godas['lon'].size
    latlon_grid = earth2grid.latlon.equiangular_lat_lon_grid(nlat, nlon)
    hpx_grid = earth2grid.healpix.Grid(level=LEVEL, pixel_order=earth2grid.healpix.XY())
    regridder_back = earth2grid.get_regridder(hpx_grid, latlon_grid)
    return regridder_back

# ============================================================
#                 DATA: ERA5 PENTADS + GODAS
# ============================================================

print("Opening ERA5 Zarr…")
ds_era5 = xr.open_zarr(ERA5_ZARR, consolidated=True, storage_options={"token": "anon"})

print("Opening GODAS pentad files…")
ds_godas = xr.open_mfdataset(os.path.join(GODAS_PATH, 'godas.P.*.nc'))

# Resolve actual ERA5 variable names used in training
avail = set(ds_era5.data_vars)
TARGETS = {
    "u10":     ["10m_u_component_of_wind"],
    "v10":     ["10m_v_component_of_wind"],
    "sw_down": ["surface_solar_radiation_downwards"],
    "lw_down": ["surface_thermal_radiation_downwards"],
    "t2m":     ["2m_temperature"],
    "q2m":     ["2m_specific_humidity", "2m_relative_humidity", "2m_dewpoint_temperature"],
    "precip":  ["total_precipitation"],
    "sp":      ["surface_pressure"],
}

def resolve(names):
    return next((n for n in names if n in avail), None)

resolved_era5 = {k: resolve(v) for k, v in TARGETS.items()}
missing = [k for k, v in resolved_era5.items() if v is None]
if missing:
    raise RuntimeError(f"Missing expected ERA5 variables: {missing}")

era5_vars = [v for v in resolved_era5.values() if v is not None]

# Build ERA5 pentad-mean dataset aligned with GODAS time, same as training
gtime = ds_godas["time"].sortby("time")
era5_sub = ds_era5[era5_vars]
five_days = np.timedelta64(5, "D")

era5_time_start = gtime.min().values - five_days
era5_time_end   = gtime.max().values
era5_sub = era5_sub.sel(time=slice(era5_time_start, era5_time_end))

pentad_means = []
for t in gtime.values:
    end = t
    start = end - five_days
    window = era5_sub.sel(time=slice(start, end))
    if window.time.size == 0:
        template = era5_sub.isel(time=0, drop=True)
        pentad_means.append(template * np.nan)
    else:
        pentad_means.append(window.mean(dim="time"))

ds_era5_5d = xr.concat(pentad_means, dim="time")
ds_era5_5d = ds_era5_5d.assign_coords(time=gtime)

print("ERA5 pentad (5-day back):", ds_era5_5d)
print("GODAS pentad:", ds_godas)

# HEALPix -> lat/lon regridder for GODAS grid
regridder_back = get_latlon_back_regridder_from_godas(ds_godas)

# ============================================================
#                     MODEL & CHANNELS
# ============================================================

N_FORCINGS = len(ERA5_LOGICAL)
N_STATE    = len(GODAS_VARS) * len(GODAS_LEVELS)
N_CHANNELS_IN  = N_FORCINGS + N_STATE
N_CHANNELS_OUT = N_STATE

print("Input channels:", N_CHANNELS_IN)
print("Output channels:", N_CHANNELS_OUT)

model = UNet(
    n_channels=N_CHANNELS_IN,
    n_classes=N_CHANNELS_OUT,
    width_multiplier=1,
    trilinear=True,
    use_ds_conv=False,
    out_activation=None
).to(DEVICE)

print(f"Loading best model weights from {BEST_MODEL_PATH}...")
state_dict = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()

# ============================================================
#           BUILD HEALPIX INPUT AT A GIVEN PENTAD INDEX
# ============================================================

def build_input_healpix_at_index(t_idx, prev_forecast_np=None):
    """
    Build a single input volume at pentad index t_idx.

    If prev_forecast_np is not None, it overrides the GODAS state channels
    (autoregressive mode), same pattern as land model:
        input_healpix[-N_STATE:, ...] = prev_forecast_np
    """
    forcing_channels = []

    # ---- ERA5 forcings ----
    for logical_name in ERA5_LOGICAL:
        era5_varname = resolved_era5[logical_name]
        da = ds_era5_5d[era5_varname].isel(time=t_idx)

        if "latitude" in da.dims and not da.latitude.values[0] < da.latitude.values[-1]:
            da = da.sortby("latitude")
        if "lat" in da.dims and not da.lat.values[0] < da.lat.values[-1]:
            da = da.sortby("lat")

        # Fill NAs
        if "latitude" in da.dims:
            da = da.interpolate_na(dim="latitude", method="linear", fill_value="extrapolate")
            da = da.interpolate_na(dim="longitude", method="linear", fill_value="extrapolate")
        else:
            da = da.interpolate_na(dim="lat", method="linear", fill_value="extrapolate")
            da = da.interpolate_na(dim="lon", method="linear", fill_value="extrapolate")

        vmin = ERA5_MINMAX[logical_name]["min"]
        vmax = ERA5_MINMAX[logical_name]["max"]
        arr_norm = norm_minmax(da.values, vmin, vmax)

        da_tmp = da.copy(data=arr_norm)
        da_tmp, regridder, src_grid, hpx_grid = get_regridder_for(da_tmp)

        arr_torch = torch.tensor(da_tmp.values, dtype=torch.float64)
        hpx = regridder(arr_torch).reshape(12, NSIDE, NSIDE).float()
        forcing_channels.append(hpx)

    # ---- GODAS state (truth) at t_idx ----
    state_t_channels = []
    for var in GODAS_VARS:
        for lev in GODAS_LEVELS:
            da_t = ds_godas[var].isel(time=t_idx, level=lev)

            if "lat" in da_t.dims and not da_t.lat.values[0] < da_t.lat.values[-1]:
                da_t = da_t.sortby("lat")

            da_t = da_t.interpolate_na(dim="lat", method="linear", fill_value="extrapolate")
            da_t = da_t.interpolate_na(dim="lon", method="linear", fill_value="extrapolate")

            stats = GODAS_MINMAX[(var, lev)]
            vmin, vmax = stats["min"], stats["max"]
            arr_t_norm = norm_minmax(da_t.values, vmin, vmax)

            da_tmp_t = da_t.copy(data=arr_t_norm)
            da_tmp_t, regridder_godas, src_grid, hpx_grid = get_regridder_for(da_tmp_t)

            arr_t_torch = torch.tensor(da_tmp_t.values, dtype=torch.float64)
            hpx_t = regridder_godas(arr_t_torch).reshape(12, NSIDE, NSIDE).float()
            state_t_channels.append(hpx_t)

    # Stack all channels
    input_healpix = torch.stack(forcing_channels + state_t_channels, dim=0)  # (Cin, 12, NSIDE, NSIDE)

    # Optionally override state with previous forecast
    if prev_forecast_np is not None:
        # prev_forecast_np should be shape (N_STATE, 12, NSIDE, NSIDE)
        input_healpix[-N_STATE:, :, :, :] = torch.tensor(prev_forecast_np, dtype=torch.float32)

    X = input_healpix.unsqueeze(0).to(DEVICE)  # (1, Cin, 12, NSIDE, NSIDE)
    return X

# ============================================================
#                 HEALPIX -> LAT/LON BACK
# ============================================================

def healpix_state_to_latlon_dset(forecast_np, target_time, denorm=True):
    """
    Convert a single forecast (N_STATE, 12, NSIDE, NSIDE)
    into an xarray.Dataset on the GODAS lat-lon grid at 'target_time'.

    If denorm=True, convert from [0,1] back to physical units
    using GODAS_MINMAX.
    """
    nlat = ds_godas['lat'].size
    nlon = ds_godas['lon'].size

    out_list = []
    data_vars = {}

    # Regrid each channel separately
    out_ = []
    for i, (var, lev) in enumerate(STATE_CHANNELS):
        hp_field = forecast_np[i, :, :, :].flatten()
        latlon_field = regridder_back(torch.from_numpy(hp_field).double())  # (lat, lon)
        latlon_field = latlon_field.reshape(nlat, nlon)

        stats = GODAS_MINMAX[(var, lev)]
        vmin, vmax = stats["min"], stats["max"]
        if denorm:
            latlon_field = denorm_minmax(latlon_field, vmin, vmax)

        # Build variable name e.g. "potential_temperature_l0_pred"
        var_name = f"{var}_l{lev}_pred"
        data_vars[var_name] = (("lat", "lon"), latlon_field)

    ds_out = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": [target_time],
            "lat": ds_godas["lat"].values,
            "lon": ds_godas["lon"].values,
        }
    )
    return ds_out

# ============================================================
#                     EXAMPLE USAGE
# ============================================================

# ---------------------------------------------
# 1) Single-step forecast: t_idx -> t_idx+1
# ---------------------------------------------
t_idx = 0  # choose any pentad index in [0, n_times-2]

print(f"\n===== Single-step forecast at pentad index {t_idx} → {t_idx+1} =====")
with torch.no_grad():
    X = build_input_healpix_at_index(t_idx, prev_forecast_np=None)
    forecast = model(X)
    forecast_np = forecast.squeeze(0).cpu().numpy()  # (N_STATE, 12, NSIDE, NSIDE)

print("forecast_np shape:", forecast_np.shape)

# Regrid to lat/lon and build dataset for time t_idx+1
target_time = ds_godas["time"].values[t_idx + 1]
ds_pred_1 = healpix_state_to_latlon_dset(forecast_np, target_time, denorm=True)
out_file_1 = f"ocean_pred_pentad_{t_idx+1}.nc"
ds_pred_1.to_netcdf(out_file_1)
print(f"Saved single-step forecast to {out_file_1}")

# --------------------------------------------------
# 2) Roll-out forecast autoregressively over N steps
#     (like your land model for-loop)
# --------------------------------------------------
n_times = ds_godas.dims["time"]

start_idx = 0           # starting pentad index
n_steps = 10            # number of forecast steps
prev_forecast_np = None

for step in range(n_steps):
    t_idx = start_idx + step
    if t_idx >= n_times - 1:
        print(f"Reached end of available GODAS times at t_idx={t_idx}")
        break

    t_start = ds_godas["time"].values[t_idx]
    t_end   = ds_godas["time"].values[t_idx + 1]
    start_str = np.datetime_as_string(t_start, unit="D")
    end_str   = np.datetime_as_string(t_end,   unit="D")

    print(f"\n===== Rolling forecast: {start_str} → {end_str} (t_idx={t_idx}) =====")

    with torch.no_grad():
        X = build_input_healpix_at_index(t_idx, prev_forecast_np=prev_forecast_np)
        forecast = model(X)
        forecast_np = forecast.squeeze(0).cpu().numpy()  # (N_STATE, 12, NSIDE, NSIDE)

    # Save prediction (in lat/lon, denormalized)
    target_time = ds_godas["time"].values[t_idx + 1]
    ds_pred = healpix_state_to_latlon_dset(forecast_np, target_time, denorm=True)
    out_file = f"ocean_pred_roll_t{t_idx+1}.nc"
    ds_pred.to_netcdf(out_file)
    print(f"Saved rolled forecast to {out_file}, shape={forecast_np.shape}")

    # Use this forecast as state input for the next step
    prev_forecast_np = forecast_np

print("\nDone with ocean model inference.")
