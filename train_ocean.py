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

import sys
import earth2grid


# Optional: Weights & Biases (do NOT hardcode keys; use env var or interactive login)
# USE_WANDB = bool(int(os.environ.get("USE_WANDB", "0")))
USE_WANDB = True
if USE_WANDB:
    import wandb
    # wandb.login(key=os.environ.get("WANDB_API_KEY"))
    wandb.login(key="70f85253c59220a4439123cc3c97280ece560bf5")  # Replace with your API key

    wandb.init(project="healpix-ocean-model", config={
        "learning_rate": 1e-3,
        "architecture": "UNet3D",
        "epochs_per_step": 10,
        "optimizer": "Adam",
        "loss": "MSELoss"
    })

# -------------------- Model --------------------
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

# -------------------- Settings --------------------
# Data
ERA5_ZARR = 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'
GODAS_PATH = '/scratch/08105/ms86336/godas_pentad/'  # directory of godas.P.*.nc

# Healpix grid
LEVEL = 6
NSIDE = 2 ** LEVEL

# GODAS levels to use (indices, consistent with your min/max export)
GODAS_LEVELS = [0, 1, 2, 4, 9, 15, 20, 25]

# Atmosphere variables (logical names from your ERA5 mapping)
ERA5_LOGICAL = ["u10", "v10", "sw_down", "lw_down", "t2m", "q2m", "precip", "sp"]

# Where the per-variable JSONs live
NORM_DIR = Path('./normalization_parts')

# Training
EPOCHS_PER_STEP = 10
LR = 1e-3
BATCH_SIZE = 1  # per timestep, to mirror your land training loop
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device used for training is', DEVICE)

# -------------------- Normalization helpers --------------------

def load_json(path: Path):
    with open(path, 'r') as f:
        return json.load(f)

# ERA5 single-file per variable (logical)
ERA5_MINMAX = {v: load_json(NORM_DIR / f"era5_{v}.json") for v in ERA5_LOGICAL}

# GODAS per (var, level)
GODAS_VARS = ["potential_temperature", "salinity"]
GODAS_MINMAX = {
    (var, lev): load_json(NORM_DIR / f"godas_{var}_l{lev}.json")
    for var in GODAS_VARS for lev in GODAS_LEVELS
}

def norm_minmax(x, vmin, vmax):
    # guard against divide-by-zero
    denom = (vmax - vmin) if (vmax - vmin) != 0 else 1.0
    return (x - vmin) / denom

# -------------------- Grids & regridder --------------------
# We'll instantiate regridders lazily per (nlat, nlon) combo.
_regridder_cache = {}

def get_regridder_for(da: xr.DataArray):
    # Ensure latitude is ascending for earth2grid
    if 'latitude' in da.dims:
        if not da.latitude.values[0] < da.latitude.values[-1]:
            da = da.sortby('latitude')
    nlat, nlon = da.shape[-2:]
    key = (nlat, nlon)
    if key not in _regridder_cache:
        src_grid = earth2grid.latlon.equiangular_lat_lon_grid(nlat, nlon)
        hpx_grid = earth2grid.healpix.Grid(level=LEVEL, pixel_order=earth2grid.healpix.XY())
        _regridder_cache[key] = earth2grid.get_regridder(src_grid, hpx_grid)
    return da, _regridder_cache[key]

# # -------------------- Data IO --------------------
# print("Opening ERA5 Zarr…")
# ds_era5 = xr.open_zarr(ERA5_ZARR, consolidated=True, storage_options={"token": "anon"})

# print("Opening GODAS pentad files…")
# ds_godas = xr.open_mfdataset(os.path.join(GODAS_PATH, 'godas.P.*.nc'))




# print(ds_era5.time.values)
# print(ds_godas.time.values)
# sys.exit()

print("Opening ERA5 Zarr…")
ds_era5 = xr.open_zarr(ERA5_ZARR, consolidated=True, storage_options={"token": "anon"})

print("Opening GODAS pentad files…")
ds_godas = xr.open_mfdataset(os.path.join(GODAS_PATH, 'godas.P.*.nc'))

# -------------------- Pentad-average ERA5 to match GODAS --------------------
# Use only the ERA5 variables you actually need
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

# print("Resampling ERA5 to 5-day means...")
# ds_era5_5d = ds_era5[era5_vars].resample(
#     time="5D",       # 5-day bins
#     label="right",   # label bin by last day
#     closed="right"   # include right edge
# ).mean()

# print("ERA5 5-day times:", ds_era5_5d.time.values[:10])
# print("GODAS pentad times:", ds_godas.time.values[:10])


# Shortcuts
gtime = ds_godas["time"].sortby("time")  # GODAS pentad times
era5   = ds_era5[era5_vars]

# Restrict ERA5 to the overall range we actually need
five_days = np.timedelta64(5, "D")
era5_time_start = gtime.min().values - five_days
era5_time_end   = gtime.max().values
era5_sub = era5.sel(time=slice(era5_time_start, era5_time_end))

pentad_means = []

for t in gtime.values:
    end = t
    start = end - five_days

    # "Past 5 days" window: [start, end]
    # If you prefer (start, end] or [start, end) you can tweak this slice.
    window = era5_sub.sel(time=slice(start, end))

    if window.time.size == 0:
        # No ERA5 data in this window: fill with NaNs
        # Use the spatial shape from an arbitrary time slice
        template = era5_sub.isel(time=0, drop=True)
        pentad_means.append(template * np.nan)
    else:
        pentad_means.append(window.mean(dim="time"))

# Stack back into a pentad-time DataSet and assign GODAS times
ds_era5_5d = xr.concat(pentad_means, dim="time")
ds_era5_5d = ds_era5_5d.assign_coords(time=gtime)

# print("ERA5 pentad (5-day back) times:", ds_era5_5d.time.values[:10])
# print("GODAS pentad times:           ", ds_godas.time.values[:10])

print("ERA5 pentad (5-day back): ", ds_era5_5d)
print("GODAS pentad:", ds_godas)

# -------------------- Dataset helper --------------------
from torch.utils.data import Dataset, DataLoader

class HealpixSampleDataset(Dataset):
    """
    Tiny dataset wrapper around a single (X, Y) pair so we can reuse
    the same training loop pattern as the land model.
    """
    def __init__(self, X, Y):
        # X: (1, Cin, 12, NSIDE, NSIDE)
        # Y: (1, Cout, 12, NSIDE, NSIDE)
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# -------------------- Channel bookkeeping --------------------
N_FORCINGS = len(ERA5_LOGICAL)  # ERA5 forcings (2D fields)
N_STATE    = len(GODAS_VARS) * len(GODAS_LEVELS)  # T/S at levels

N_CHANNELS_IN  = N_FORCINGS + N_STATE   # forcings + state(t)
N_CHANNELS_OUT = N_STATE                # state(t+1)

print("Input channels:", N_CHANNELS_IN)
print("Output channels:", N_CHANNELS_OUT)

# -------------------- Model / optimizer --------------------
model = UNet(
    n_channels=N_CHANNELS_IN,
    n_classes=N_CHANNELS_OUT,
    width_multiplier=1,
    trilinear=True,
    use_ds_conv=False,
    out_activation=None
).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()
best_val_loss = float("inf")
best_model_path = "best_unet_ocean_model.pth"

# -------------------- Utility to build one (X_t, Y_t) sample --------------------
def build_ocean_sample_at_index(t_idx: int):
    """
    Build input/target tensors for time t_idx -> t_idx+1.
    Returns:
        X: (1, Cin, 12, NSIDE, NSIDE)
        Y: (1, Cout, 12, NSIDE, NSIDE)
    """
    # ---- ERA5 forcings at time t_idx ----
    forcing_channels = []
    for logical_name in ERA5_LOGICAL:
        era5_varname = resolved_era5[logical_name]  # actual ERA5 name
        da = ds_era5_5d[era5_varname].isel(time=t_idx)

        # Sort latitude ascending if needed
        if "latitude" in da.dims and not da.latitude.values[0] < da.latitude.values[-1]:
            da = da.sortby("latitude")

        # Fill NaNs in lat/lon
        da = da.interpolate_na(dim="latitude", method="linear", fill_value="extrapolate")
        da = da.interpolate_na(dim="longitude", method="linear", fill_value="extrapolate")

        # Normalize using per-variable JSON stats (logical name)
        vmin = ERA5_MINMAX[logical_name]["min"]
        vmax = ERA5_MINMAX[logical_name]["max"]
        arr = norm_minmax(da.values, vmin, vmax)

        # Regrid to HEALPix
        da_tmp = da.copy(data=arr)
        da_tmp, regridder = get_regridder_for(da_tmp)
        arr_torch = torch.tensor(da_tmp.values, dtype=torch.float64)  # (lat, lon)
        hpx = regridder(arr_torch).reshape(12, NSIDE, NSIDE).float()  # (12, NSIDE, NSIDE)

        forcing_channels.append(hpx)

    # ---- GODAS state at time t_idx and t_idx+1 ----
    state_t_channels   = []
    state_tp1_channels = []

    for var in GODAS_VARS:           # "potential_temperature", "salinity"
        for lev in GODAS_LEVELS:     # indices into GODAS "level" dim (consistent with your JSONs)
            da_t   = ds_godas[var].isel(time=t_idx,   level=lev)
            da_t1  = ds_godas[var].isel(time=t_idx+1, level=lev)

            # Sort latitude ascending if needed (GODAS uses 'lat', 'lon')
            if "lat" in da_t.dims and not da_t.lat.values[0] < da_t.lat.values[-1]:
                da_t  = da_t.sortby("lat")
                da_t1 = da_t1.sortby("lat")

            # Fill NaNs
            da_t  = da_t.interpolate_na(dim="lat",  method="linear", fill_value="extrapolate")
            da_t  = da_t.interpolate_na(dim="lon",  method="linear", fill_value="extrapolate")
            da_t1 = da_t1.interpolate_na(dim="lat", method="linear", fill_value="extrapolate")
            da_t1 = da_t1.interpolate_na(dim="lon", method="linear", fill_value="extrapolate")

            stats = GODAS_MINMAX[(var, lev)]
            vmin, vmax = stats["min"], stats["max"]

            arr_t  = norm_minmax(da_t.values,  vmin, vmax)
            arr_t1 = norm_minmax(da_t1.values, vmin, vmax)

            da_tmp_t  = da_t.copy(data=arr_t)
            da_tmp_t1 = da_t1.copy(data=arr_t1)

            da_tmp_t,  regridder_godas = get_regridder_for(da_tmp_t)
            # same grid => cache gives same regridder for t+1 as well

            arr_t_torch  = torch.tensor(da_tmp_t.values,  dtype=torch.float64)
            arr_t1_torch = torch.tensor(da_tmp_t1.values, dtype=torch.float64)

            hpx_t  = regridder_godas(arr_t_torch).reshape(12, NSIDE, NSIDE).float()
            hpx_t1 = regridder_godas(arr_t1_torch).reshape(12, NSIDE, NSIDE).float()

            state_t_channels.append(hpx_t)
            state_tp1_channels.append(hpx_t1)

    # Stack into channel dimension
    X_channels = forcing_channels + state_t_channels
    Y_channels = state_tp1_channels

    X = torch.stack(X_channels, dim=0)  # (Cin, 12, NSIDE, NSIDE)
    Y = torch.stack(Y_channels, dim=0)  # (Cout, 12, NSIDE, NSIDE)

    # Add batch dimension
    X = X.unsqueeze(0)  # (1, Cin, 12, NSIDE, NSIDE)
    Y = Y.unsqueeze(0)  # (1, Cout, 12, NSIDE, NSIDE)

    return X, Y

# -------------------- Training loop (pentad-by-pentad, like land model) --------------------
n_times = ds_godas.dims["time"]
print("Number of GODAS pentads:", n_times)

for t_idx in range(n_times - 1):

     # ---- Get the start/end dates for this pentad transition ----
    t_start = ds_godas["time"].values[t_idx]
    t_end   = ds_godas["time"].values[t_idx + 1]

    # Format as 'YYYY-MM-DD'
    start_str = np.datetime_as_string(t_start, unit="D")
    end_str   = np.datetime_as_string(t_end,   unit="D")

    print(f"\n🌊 Training on {start_str} → {end_str}")
    # print(f"\n🌊 Training on pentad {t_idx} → {t_idx + 1}")

    # Build one (X, Y) sample for this step
    X, Y = build_ocean_sample_at_index(t_idx)

    # Move to device
    X = X.to(DEVICE)
    Y = Y.to(DEVICE)

    dataset = HealpixSampleDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for epoch in range(1, EPOCHS_PER_STEP + 1):
        # ---- Train ----
        model.train()
        train_loss = 0.0

        for xb, yb in dataloader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(xb)
            loss = criterion(y_pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(dataloader)

        # ---- Eval on same sample ----
        model.eval()
        with torch.no_grad():
            eval_loss = 0.0
            for xb, yb in dataloader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                y_eval = model(xb)
                eval_loss += criterion(y_eval, yb).item()
            avg_eval_loss = eval_loss / len(dataloader)

        print(
            f"[Pentad {t_idx}] Epoch {epoch:02d} - "
            f"Train Loss: {avg_train_loss:.6f} | Eval Loss: {avg_eval_loss:.6f}"
        )

        if USE_WANDB:
            wandb.log({
                "pentad": t_idx,
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "eval_loss": avg_eval_loss,
            })

        # Track global best
        if avg_eval_loss < best_val_loss:
            best_val_loss = avg_eval_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ Saved best ocean model with eval_loss: {avg_eval_loss:.6f}")

    # Clean up
    del X, Y, dataset, dataloader
    torch.cuda.empty_cache()
