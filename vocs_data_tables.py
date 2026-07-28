#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import linregress


# ============================================================
# FILE PATHS
# ============================================================

netcdf_path = (
    '/Users/dave121504/Downloads/'
    'ARM_BNF_chemical_correlations/bnfstampS10*.nc'
)

csv_path = (
    '/Users/dave121504/Downloads/'
    'SoilVOC_STAMP(VOCs).csv'
)

output_path = (
    '/Users/dave121504/Downloads/'
    'S10_east_VOC_depth_correlations_full.png'
)


# ============================================================
# READ NETCDF FILES
# ============================================================

files_s10 = sorted(glob.glob(netcdf_path))

if len(files_s10) == 0:
    raise FileNotFoundError(
        f'No NetCDF files were found using:\n{netcdf_path}'
    )

s10 = xr.open_mfdataset(
    files_s10,
    combine='by_coords'
)

print(f'Number of NetCDF files opened: {len(files_s10)}')
print(f'Available depths: {s10["depth"].values}')


# ============================================================
# SELECT SOIL MOISTURE AND TEMPERATURE
# No QC filtering is applied
# ============================================================

depths = [5, 20, 50]

soil_variables = {}

for depth in depths:

    soil_variables[f'Soil Moisture_{depth}cm'] = (
        s10['loam_soil_water_content_east']
        .sel(depth=depth, drop=True)
    )

    soil_variables[f'Soil Temp_{depth}cm'] = (
        s10['soil_temperature_east']
        .sel(depth=depth, drop=True)
    )


# Combine the selected variables into one dataset
soil = xr.Dataset(soil_variables)


# ============================================================
# CALCULATE DAILY MEANS
# ============================================================

soil_daily = (
    soil
    .resample(time='1D')
    .mean()
    .to_dataframe()
    .reset_index()
)

soil_daily['Sample Date'] = (
    pd.to_datetime(soil_daily['time'])
    .dt.normalize()
)


# ============================================================
# READ VOC DATA FROM CSV
# ============================================================

vocs = pd.read_csv(
    csv_path,
    skiprows=[0]
)

vocs['Sample Date'] = pd.to_datetime(
    vocs['Sample Date'],
    format='%m/%d/%Y',
    errors='coerce'
)

vocs['Sum (ppbv)'] = pd.to_numeric(
    vocs['Sum (ppbv)'],
    errors='coerce'
)

# Retain only the columns needed from the CSV
vocs = vocs[
    ['Sample Date', 'Sum (ppbv)']
].dropna().copy()

vocs['Sample Date'] = (
    vocs['Sample Date']
    .dt.normalize()
)


# ============================================================
# MATCH SOIL VALUES TO VOC SAMPLING DATES
# ============================================================

soil_columns = [
    'Sample Date',
    'Soil Moisture_5cm',
    'Soil Temp_5cm',
    'Soil Moisture_20cm',
    'Soil Temp_20cm',
    'Soil Moisture_50cm',
    'Soil Temp_50cm'
]

paired_data = pd.merge(
    vocs,
    soil_daily[soil_columns],
    on='Sample Date',
    how='inner',
    validate='one_to_one'
)

paired_data = (
    paired_data
    .sort_values('Sample Date')
    .reset_index(drop=True)
)

print(f'Number of VOC dates: {len(vocs)}')
print(f'Number of matched dates: {len(paired_data)}')

# ============================================================
# PRINT 20 CM AND 50 CM SOIL VALUES
# ============================================================

print('\n20 cm and 50 cm soil values on VOC sampling dates:\n')

print(
    paired_data[
        [
            'Sample Date',
            'Soil Moisture_5cm',
            'Soil Temp_5cm',
            'Soil Moisture_20cm',
            'Soil Temp_20cm',
            'Soil Moisture_50cm',
            'Soil Temp_50cm',
            'Sum (ppbv)'
        ]
    ].to_string(
        index=False,
        formatters={
            'Sample Date':
                lambda date: date.strftime('%Y-%m-%d'),
                
            'Soil Moisture_5cm':
                lambda value: f'{value:.2f}%',
                
            'Soil Temp_5cm':
                lambda value: f'{value:.2f} C',

            'Soil Moisture_20cm':
                lambda value: f'{value:.2f}%',

            'Soil Temp_20cm':
                lambda value: f'{value:.2f} °C',

            'Soil Moisture_50cm':
                lambda value: f'{value:.2f}%',

            'Soil Temp_50cm':
                lambda value: f'{value:.2f} °C',

            'Sum (ppbv)':
                lambda value: f'{value:.2f}'
        },
        na_rep='Missing'
    )
)
# ============================================================
# PRINT TEMPERATURE VALUES AT ALL THREE DEPTHS
# ============================================================

print('\nSoil temperatures on VOC sampling dates:\n')

print(
    paired_data[
        [
            'Sample Date',
            'Soil Temp_5cm',
            'Soil Temp_20cm',
            'Soil Temp_50cm',
            'Sum (ppbv)'
        ]
    ].to_string(
        index=False,
        formatters={
            'Sample Date':
                lambda date: date.strftime('%Y-%m-%d'),

            'Soil Temp_5cm':
                lambda value: f'{value:.2f} °C',
                

            'Soil Temp_20cm':
                lambda value: f'{value:.2f} °C',

            'Soil Temp_50cm':
                lambda value: f'{value:.2f} °C',

            'Sum (ppbv)':
                lambda value: f'{value:.2f}'
        },
        na_rep='Missing'
    )
)
