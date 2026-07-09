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

#change this to the correct directory based on where soilvue data is going to be pulled from
df= pd.read_csv('/users/dave121504/SEBS_E33_sixty_min.dat', header = 1, skiprows = [2,3], na_values=['NAN', 'NaN', 'nan', 'NAN '], keep_default_na = True)
pcp = pd.read_csv('/users/dave121504/SEBS_E33_precip_vars.dat', header = 1, skiprows = [2,3], na_values=['NAN', 'NaN', 'nan', 'NAN '], keep_default_na = True )
#some NAN's, so I clean it here
cleaned_df = df.dropna()
cleaned_df['TIMESTAMP'] = pd.to_datetime(cleaned_df['TIMESTAMP'])
time = cleaned_df['TIMESTAMP']

cleaned_pcp = pcp.dropna()
cleaned_pcp['TIMESTAMP'] = pd.to_datetime(cleaned_pcp['TIMESTAMP'])
time_pcp = cleaned_pcp['TIMESTAMP']

fig, axes = plt.subplots(2,2, figsize = (18, 12))

#Soil temp plots
axes[0,0].plot(time, cleaned_df['T_5cm'], color = 'red', linewidth = 1, label = '5 cm')    
axes[0,0].plot(time, cleaned_df['T_10cm'], color = 'darkorange', linewidth = 1, label = '10 cm')
axes[0,0].plot(time, cleaned_df['T_20cm'], color = 'yellow', linewidth = 1, label = '20 cm')
axes[0,0].plot(time, cleaned_df['T_30cm'], color = 'seagreen', linewidth = 1, label = '30 cm')
axes[0,0].plot(time, cleaned_df['T_40cm'], color = 'blue', linewidth = 1, label = '40 cm')
axes[0,0].plot(time, cleaned_df['T_50cm'], color = 'darkviolet', linewidth = 1, label = '50 cm')
axes[0,0].plot(time, cleaned_df['T_60cm'], color = 'deeppink', linewidth = 1, label = '60 cm')
axes[0,0].plot(time, cleaned_df['T_75cm'], color = 'lightgreen', linewidth = 1, label = '75 cm')
axes[0,0].plot(time, cleaned_df['T_100cm'], color = 'darkcyan', linewidth = 1, label = '100 cm')
axes[0,0].legend(loc = 'lower right', fontsize = 6, framealpha = 0.4, ncols=2)
axes[0,0].set_ylabel('Temperature (C)', fontsize = 10)
axes[0,0].set_title('(a) Soil Temperature (C)', fontsize = 12, fontweight = 'bold')
axes[0,0].tick_params(axis = 'x', labelrotation = 45, labelsize = 6)

#Volumetric Water Content plots
axes[0,1].plot(time, cleaned_df['VWC_5cm']*100, color = 'red', linewidth = 1, label = '5 cm')
axes[0,1].plot(time, cleaned_df['VWC_10cm']*100, color = 'darkorange', linewidth = 1, label = '10 cm')
axes[0,1].plot(time, cleaned_df['VWC_20cm']*100, color = 'yellow', linewidth = 1, label = '20 cm')
axes[0,1].plot(time, cleaned_df['VWC_30cm']*100, color = 'seagreen', linewidth = 1, label = '30 cm')
axes[0,1].plot(time, cleaned_df['VWC_40cm']*100, color = 'blue', linewidth = 1, label = '40 cm')
axes[0,1].plot(time, cleaned_df['VWC_50cm']*100, color = 'darkviolet', linewidth = 1, label = '50 cm')
axes[0,1].plot(time, cleaned_df['VWC_60cm']*100, color = 'deeppink', linewidth = 1, label = '60 cm')
axes[0,1].plot(time, cleaned_df['VWC_75cm']*100, color = 'lightgreen', linewidth = 1, label = '75 cm')
axes[0,1].plot(time, cleaned_df['VWC_100cm']*100, color = 'darkcyan', linewidth = 1, label = '100 cm')
axes[0,1].legend(loc = 'lower right', fontsize = 6, framealpha = 0.4, ncols=2)
axes[0,1].set_ylabel('Volumetric Water Content (%)', fontsize = 10)
axes[0,1].set_title('(b) Volumetric Water Content (%)', fontsize = 12, fontweight = 'bold')
axes[0,1].tick_params(axis = 'x', labelrotation = 45, labelsize = 6)
axes[0,1].set_ylim(0, (cleaned_df['VWC_5cm']*100).max() * 1.5)

#precip plot on the VWC plots 
pcp_plot = axes[0,1].twinx()
precip_time = mdates.date2num(pd.to_datetime(cleaned_pcp['TIMESTAMP'].values))
precip_vals = cleaned_pcp['tb_rain_mm_8in_Tot'].values

pcp_plot.plot(precip_time, precip_vals, color = 'black', linewidth = 1.5, alpha = 0.6, label = 'Precip')
pcp_plot.set_ylim(precip_vals.max() * 4, 0)
pcp_plot.set_ylabel('Precipitation (mm)',  fontsize = 8)
pcp_plot.legend(loc = 'lower left', fontsize = 6, framealpha = 0.4)



#Permittivity Plots
axes[1,0].plot(time, cleaned_df['Ka_5cm'], color = 'red', linewidth = 1, label = '5 cm')
axes[1,0].plot(time, cleaned_df['Ka_10cm'], color = 'darkorange', linewidth = 1, label = '10 cm')
axes[1,0].plot(time, cleaned_df['Ka_20cm'], color = 'yellow', linewidth = 1, label = '20 cm')
axes[1,0].plot(time, cleaned_df['Ka_30cm'], color = 'seagreen', linewidth = 1, label = '30 cm')
axes[1,0].plot(time, cleaned_df['Ka_40cm'], color = 'blue', linewidth = 1, label = '40 cm')
axes[1,0].plot(time, cleaned_df['Ka_50cm'], color = 'darkviolet', linewidth = 1, label = '50 cm')
axes[1,0].plot(time, cleaned_df['Ka_60cm'], color = 'deeppink', linewidth = 1, label = '60 cm')
axes[1,0].plot(time, cleaned_df['Ka_75cm'], color = 'lightgreen', linewidth = 1, label = '75 cm')
axes[1,0].plot(time, cleaned_df['Ka_100cm'], color = 'darkcyan', linewidth = 1, label = '100 cm')
axes[1,0].legend(loc = 'lower right', fontsize = 6, framealpha = 0.4, ncols=2)
#axes[0,0].set_ylabel('Permittivity', fontsize = 10)
axes[1,0].set_title('(c) Permittivity', fontsize = 12, fontweight = 'bold')
axes[1,0].tick_params(axis = 'x', labelrotation = 45, labelsize = 6)

#Bulk Electrical Conductivity Plots
axes[1,1].plot(time, cleaned_df['BulkEC_5cm'], color = 'red', linewidth = 1, label = '5 cm')
axes[1,1].plot(time, cleaned_df['BulkEC_10cm'], color = 'darkorange', linewidth = 1, label = '10 cm')
axes[1,1].plot(time, cleaned_df['BulkEC_20cm'], color = 'yellow', linewidth = 1, label = '20 cm')
axes[1,1].plot(time, cleaned_df['BulkEC_30cm'], color = 'seagreen', linewidth = 1, label = '30 cm')
axes[1,1].plot(time, cleaned_df['BulkEC_40cm'], color = 'blue', linewidth = 1, label = '40 cm')
axes[1,1].plot(time, cleaned_df['BulkEC_50cm'], color = 'darkviolet', linewidth = 1, label = '50 cm')
axes[1,1].plot(time, cleaned_df['BulkEC_60cm'], color = 'deeppink', linewidth = 1, label = '60 cm')
axes[1,1].plot(time, cleaned_df['BulkEC_75cm'], color = 'lightgreen', linewidth = 1, label = '75 cm')
axes[1,1].plot(time, cleaned_df['BulkEC_100cm'], color = 'darkcyan', linewidth = 1, label = '100 cm')
axes[1,1].legend(loc = 'lower right', fontsize = 6, framealpha = 0.4, ncols=2)
axes[1,1].set_ylabel('Bulk Electrical Conductivity (dS/m)', fontsize = 10)
axes[1,1].set_title('(d) Bulk Electrical Conductivity (dS/m)', fontsize = 12, fontweight = 'bold')
axes[1,1].tick_params(axis = 'x', labelrotation = 45, labelsize = 6)

#grids for visuals
axes[0,0].grid(alpha = 0.3)
axes[0,1].grid(alpha = 0.3)
axes[1,0].grid(alpha = 0.3)
axes[1,1].grid(alpha = 0.3)

#this is for the title dates so the plot will adjust based on the range of the data 
initial_time = time.iloc[0]
final_time = time.iloc[-1]
fig.suptitle(f'SoilVue Data Plots from {initial_time} UTC to {final_time} UTC', fontweight = 'bold', fontsize = 20)          
plt.subplots_adjust(hspace=0.3, wspace = 0.2)

#change the savefig to the actual directory, maybe change it to date tag the files when saving 
output_dir = Path('/Users/dave121504/Documents')
file_path = output_dir / 'soilvue_data_plots.png'
plt.savefig(file_path, dpi = 300, bbox_inches = 'tight')