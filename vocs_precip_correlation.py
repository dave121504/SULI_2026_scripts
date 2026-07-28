#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ============================================================
# FILE PATHS
# ============================================================

netcdf_path = (
    '/Users/dave121504/Downloads/ARM_BNF_STAMP_data/'
    'S10_precip/bnfstamppcpS10*.nc'
)

csv_path = (
    '/Users/dave121504/Downloads/'
    'SoilVOC_STAMP(VOCs) (2).csv'
)

output_path = (
    '/Users/dave121504/Downloads/'
    'S10_daily_precip_VOC_timeseries.png'
)


# ============================================================
# READ PRECIPITATION NETCDF FILES
# ============================================================

files_s10 = sorted(glob.glob(netcdf_path))

if len(files_s10) == 0:
    raise FileNotFoundError(
        f'No NetCDF files were found using:\n{netcdf_path}'
    )

s10_precip = xr.open_mfdataset(
    files_s10,
    combine='by_coords'
)

print(f'Number of NetCDF files opened: {len(files_s10)}')


# ============================================================
# CALCULATE DAILY TOTAL PRECIPITATION
# ============================================================

precip = s10_precip['precip']

daily_precip = precip.resample(time='1D').sum(
    skipna=True,
    min_count=1
)

# Remove days without valid precipitation data
daily_precip = daily_precip.dropna(dim='time')

precip_units = precip.attrs.get('units', 'mm')


# ============================================================
# READ VOC CSV
# ============================================================

vocs = pd.read_csv(csv_path)
vocs.columns = vocs.columns.str.strip()

# Retry while skipping the first row if the CSV has metadata
if 'Sample Date' not in vocs.columns:
    vocs = pd.read_csv(csv_path, skiprows=[0])
    vocs.columns = vocs.columns.str.strip()

vocs['Sample Date'] = pd.to_datetime(
    vocs['Sample Date'],
    errors='coerce'
).dt.normalize()

vocs['sum_biogenic'] = pd.to_numeric(
    vocs['sum_biogenic'],
    errors='coerce'
)

vocs = (
    vocs[['Sample Date', 'sum_biogenic']]
    .dropna()
    .sort_values('Sample Date')
)

print(f'Number of VOC samples plotted: {len(vocs)}')
# Date range covered by the VOC measurements
voc_start = vocs['Sample Date'].min()
voc_end = vocs['Sample Date'].max()

#print(f'VOC date range: {voc_start:%Y-%m-%d} to {voc_end:%Y-%m-%d}')

# Limit precipitation to the VOC measurement period
daily_precip = daily_precip.sel(
    time=slice(voc_start, voc_end)
)

# ============================================================
# CREATE PLOT
# ============================================================

fig, precip_ax = plt.subplots(figsize=(18, 8))


# Daily precipitation on the left axis
precip_line = precip_ax.plot(
    daily_precip['time'],
    daily_precip,
    color='royalblue',
    linewidth=1.5,
    label='Daily total precipitation'
)

precip_ax.set_xlabel(
    'Date',
    fontsize=14,
    fontweight='bold'
)

precip_ax.set_ylabel(
    f'Daily Total Precipitation ({precip_units})',
    color='royalblue',
    fontsize=14,
    fontweight='bold'
)

precip_ax.tick_params(
    axis='x',
    labelrotation=45,
    labelsize=11
)

precip_ax.tick_params(
    axis='y',
    labelcolor='royalblue',
    labelsize=12
)

precip_ax.grid(
    linestyle='--',
    alpha=0.3
)


# VOC concentration on the right axis
voc_ax = precip_ax.twinx()

voc_points = voc_ax.scatter(
    vocs['Sample Date'],
    vocs['sum_biogenic'],
    color='darkorange',
    edgecolor='black',
    linewidth=0.5,
    s=65,
    zorder=5,
    label='Biogenic VOC concentration'
)

voc_ax.set_ylabel(
    'Biogenic VOC Concentration (ppbv)',
    color='darkorange',
    fontsize=14,
    fontweight='bold'
)

voc_ax.tick_params(
    axis='y',
    labelcolor='darkorange',
    labelsize=12
)


# ============================================================
# DATE FORMATTING
# ============================================================

# Limit the shared x-axis to the VOC sampling period
precip_ax.set_xlim(voc_start-2, voc_end+2)

# Display one tick for each month
precip_ax.xaxis.set_major_locator(
    mdates.MonthLocator(interval=1)
)

precip_ax.xaxis.set_major_formatter(
    mdates.DateFormatter('%b %Y')
)

precip_ax.tick_params(
    axis='x',
    labelrotation=45,
    labelsize=11
)

# ============================================================
# TITLE AND LEGEND
# ============================================================

precip_ax.set_title(
    'S10 Daily Precipitation and Biogenic VOC Concentration',
    fontsize=17,
    fontweight='bold',
    pad=15
)

handles = precip_line + [voc_points]
labels = [handle.get_label() for handle in handles]

precip_ax.legend(
    handles,
    labels,
    loc='upper left',
    fontsize=11,
    framealpha=0.9
)

fig.tight_layout()

#fig.savefig(
    #output_path,
    #dpi=300,
    #bbox_inches='tight'
#)

plt.show()

s10_precip.close()