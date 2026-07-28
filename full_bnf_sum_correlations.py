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
    'S10_south_VOC_depth_correlations_full.png'
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
        s10['loam_soil_water_content_south']
        .sel(depth=depth, drop=True)
    )

    soil_variables[f'Soil Temp_{depth}cm'] = (
        s10['soil_temperature_south']
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


# ============================================================
# CREATE 3-ROW, 2-COLUMN FIGURE
# Rows: 5, 20, and 50 cm
# Left column: soil moisture
# Right column: soil temperature
# ============================================================

fig, axes = plt.subplots(
    nrows=3,
    ncols=2,
    figsize=(18, 20),
    sharey=True
)

panel_letters = [
    ['a', 'b'],
    ['c', 'd'],
    ['e', 'f']
]

moisture_colors = [
    'royalblue',
    'mediumblue',
    'navy'
]

temperature_colors = [
    'darkorange',
    'orangered',
    'firebrick'
]


# ============================================================
# CREATE EACH CORRELATION PLOT
# ============================================================

for row, depth in enumerate(depths):

    moisture_column = f'Soil Moisture_{depth}cm'
    temperature_column = f'Soil Temp_{depth}cm'

    plot_columns = [
        moisture_column,
        temperature_column
    ]

    x_labels = [
        f'{depth} cm Soil Moisture (%)',
        f'{depth} cm Soil Temperature (°C)'
    ]

    colors = [
        moisture_colors[row],
        temperature_colors[row]
    ]

    variable_names = [
        'Soil Moisture',
        'Soil Temperature'
    ]

    for column in range(2):

        ax = axes[row, column]
        x_column = plot_columns[column]

        # Use dates with both a soil value and VOC value
        panel_data = paired_data[
            [x_column, 'Sum (ppbv)']
        ].dropna()

        x = panel_data[x_column].to_numpy()
        y = panel_data['Sum (ppbv)'].to_numpy()

        if len(panel_data) < 2 or len(np.unique(x)) < 2:

            ax.text(
                0.5,
                0.5,
                'Insufficient data',
                transform=ax.transAxes,
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=14
            )

            ax.set_xlabel(
                x_labels[column],
                fontsize=12,
                fontweight='bold'
            )

            ax.set_ylabel(
                'VOC Concentration (ppbv)',
                fontsize=12,
                fontweight='bold'
            )

            continue

        # Calculate linear regression
        regression = linregress(x, y)

        r_value = regression.rvalue
        r_squared = r_value**2
        p_value = regression.pvalue

        # Create regression line
        x_line = np.linspace(
            x.min(),
            x.max(),
            100
        )

        y_line = (
            regression.slope * x_line
            + regression.intercept
        )

        # Scatterplot
        ax.scatter(
            x,
            y,
            color=colors[column],
            edgecolor='black',
            s=75,
            alpha=0.75,
            label='VOC sampling dates'
        )

        # Regression line
        ax.plot(
            x_line,
            y_line,
            color='black',
            linewidth=2,
            linestyle='--',
            label='Linear regression'
        )

        ax.set_xlabel(
            x_labels[column],
            fontsize=12,
            fontweight='bold'
        )

        ax.set_ylabel(
            'VOC Concentration (ppbv)',
            fontsize=12,
            fontweight='bold'
        )

        ax.set_title(
            f'{panel_letters[row][column]}) '
            f'{depth} cm {variable_names[column]} vs. VOCs',
            fontsize=14,
            fontweight='bold'
        )

        ax.text(
            0.03,
            0.95,
            f'$R^2$ = {r_squared:.3f}\n'
            f'$r$ = {r_value:.3f}\n'
            f'$p$ = {p_value:.3f}\n'
            f'n = {len(panel_data)}',
            transform=ax.transAxes,
            verticalalignment='top',
            fontsize=11,
            bbox={
                'facecolor': 'white',
                'edgecolor': 'gray',
                'alpha': 0.85
            }
        )

        ax.grid(alpha=0.25)
        ax.legend(loc='best')


# ============================================================
# FINAL FORMATTING AND SAVE
# ============================================================

fig.suptitle(
    'Soil Moisture and Temperature Correlations - south '
    'with VOC Concentration at S10',
    fontsize=20,
    fontweight='bold'
)

plt.tight_layout(
    rect=[0, 0, 1, 0.97],
    h_pad=3,
    w_pad=2
)

# Save before displaying the figure
# %%


plt.savefig(
    output_path,
    dpi=300,
    bbox_inches='tight',
    facecolor='white'
)



s10.close()
