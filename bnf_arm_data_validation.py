#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 13:33:02 2026

@author: dave121504
"""
# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from datetime import timedelta
from netCDF4 import Dataset, date2num, num2date
import cftime
import glob
import os
import csv
import datetime
from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.dates as md
import metpy.calc as mpcalc
from metpy.units import units
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from pathlib import Path

df= pd.read_csv('/users/dave121504/bnf_smos_s10.txt', delimiter = ',', na_values=['NAN', 'NaN', 'nan', 'NAN ', 'M'], keep_default_na = True)
cleaned_df = df.dropna()
cleaned_df['Timestamp'] = pd.to_datetime(cleaned_df['Timestamp'])
time = cleaned_df['Timestamp']

smos_moisture = cleaned_df['Soil_Moisture']

files_s40 = sorted(glob.glob('/Users/dave121504/ARM Project/BNF_files/s40/bnfstampS40*.nc'))
s40 = xr.open_mfdataset(files_s40, combine='by_coords')
files_s40_precip = sorted(glob.glob('/Users/dave121504/ARM Project/BNF_files/s40_precip/bnfstamp*.nc'))
s40_precip = xr.open_mfdataset(files_s40_precip, combine='by_coords')

s40_p = s40_precip['precip']
#.resample(time='1D').sum()
files_s10 = sorted(glob.glob('/Users/dave121504/ARM Project/BNF_files/s10/bnfstampS10*.nc'))
s10 = xr.open_mfdataset(files_s10, combine='by_coords')

files_s13 = sorted(glob.glob('/Users/dave121504/ARM Project/BNF_files/s13/bnfstampS13*.nc'))
s13 = xr.open_mfdataset(files_s13, combine='by_coords')

files_s14 = sorted(glob.glob('/Users/dave121504/ARM Project/BNF_files/s14/bnfstampS14*.nc'))
s14 = xr.open_mfdataset(files_s14, combine='by_coords')

files_s20 = sorted(glob.glob('/Users/dave121504/ARM Project/BNF_files/s20/bnfstampS20*.nc'))
s20 = xr.open_mfdataset(files_s20, combine='by_coords')

files_s30 = sorted(glob.glob('/Users/dave121504/ARM Project/BNF_files/s30/bnfstampS30*.nc'))
s30 = xr.open_mfdataset(files_s30, combine='by_coords')

s40_moisture = s40['loam_soil_water_content_east'].isel(depth = 0)
s10_moisture = s10['loam_soil_water_content_east'].isel(depth = 0)
s13_moisture = s13['loam_soil_water_content_east'].isel(depth = 0)
s14_moisture = s14['loam_soil_water_content_east'].isel(depth = 0)
s20_moisture = s20['loam_soil_water_content_east'].isel(depth = 0)
s30_moisture = s30['loam_soil_water_content_east'].isel(depth = 0)

# %%

scan_data = pd.read_csv('/users/dave121504/suddith_farms_scan_site.txt', delimiter = ',', na_values=['NAN', 'NaN', 'nan', 'NAN ', 'M'], keep_default_na = True)
cleaned_scan = scan_data.dropna()
cleaned_scan['Date'] = pd.to_datetime(cleaned_scan['Date'])
scan_time = cleaned_scan['Date']
scan_moisture = cleaned_scan['Soil Moisture Percent -2in (pct) Mean of Hourly Values']
# %%



fig, ax = plt.subplots(1, 1, figsize = (18, 12))

ax.plot(time, smos_moisture*100, color = 'red', linewidth = 2, label = 'SMOS')
ax.plot(s40['time'], s40_moisture, color = 'blue', linewidth = 0.25, label = 'BNF Sites')
ax.plot(s10['time'], s10_moisture, color = 'blue', linewidth = 0.25)#, label = 'BNF s10')
ax.plot(s13['time'], s13_moisture, color = 'blue', linewidth = 0.25)#, label = 'BNF s13')
ax.plot(s14['time'], s14_moisture, color = 'blue', linewidth = 0.25)#,label = 'BNF s14')
ax.plot(s30['time'], s30_moisture, color = 'blue', linewidth = 0.25)#, label = 'BNF s30')
ax.plot(scan_time, scan_moisture, color = 'green', linewidth = 2, label = 'SCAN site')
ax.legend(loc = 'lower right', fontsize = 8, framealpha = 0.4)
ax.set_ylabel('Soil Moisture (%)', fontsize = 10)
ax.set_title('(a) Soil Moisture between SMOS and BNF Sites', fontsize = 12, fontweight = 'bold')
ax.tick_params(axis = 'x', labelrotation = 45, labelsize = 6)
ax.set_ylim(0, (smos_moisture.max()*100))

#p_plot = ax.twinx()
#p_plot.set_ylim(s40_p.max() * 2, 0)
#p_plot.plot(s40_p['time'], s40_p, linewidth = 0.8, color = 'black', label = 'precip')
            #.resample(time='1D').mean(), s40_p, linewidth = 0.8, color = 'black', label = 'precip')
#p_plot.set_ylabel('Precip (mm)', fontsize = 10)