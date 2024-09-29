import streamlit as st
from netCDF4 import Dataset, num2date
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import time

# Load the data
data = Dataset(r'E:\NAMAN\SIH Nautical\Datasets\Wavewatch_III_25_28_2024_to_03_09_2024.nc')  # Replace with your .nc file location

# Get latitude, longitude, time, and wind speed
lats = data.variables['LAT'][:]
lons = data.variables['LON'][:]
time_data = data.variables['TIME'][:]  # Assuming 'TIME' is in hours since a reference date
time_units = data.variables['TIME'].units  # Time units (e.g., "hours since YYYY-MM-DD hh:mm:ss")
ws = data.variables['WS'][:]  # Wind speed data

# Convert time to human-readable format
time_dates = num2date(time_data, units=time_units)

# Create a Basemap instance
mp = Basemap(projection='merc',
             llcrnrlon=61.4309660, llcrnrlat=-2.4438284,
             urcrnrlon=93.4210174, urcrnrlat=27.3475395,
             resolution='l')  # Use 'i' for intermediate resolution

lon, lat = np.meshgrid(lons, lats)
x, y = mp(lon, lat)

# Streamlit interface
st.title("Wind Speed Visualization")

# Slider for selecting time frame
selected_frame = st.slider("Select time frame", 0, len(time_data) - 1, 0)

# Button to start animation
animate = st.button("Start Animation")

# Create a placeholder for the plot
plot_placeholder = st.empty()

# Function to plot a single frame
def plot_frame(frame):
    fig, ax = plt.subplots()
    mp.drawcoastlines()
    mp.drawcountries()
    mp.drawstates()

    current_ws = np.squeeze(ws[frame, :, :])
    if np.ma.is_masked(current_ws):
        current_ws = np.ma.filled(current_ws, fill_value=np.nan)

    c_scheme = mp.pcolor(x, y, current_ws, cmap='jet', shading='auto')
    plt.colorbar(c_scheme, ax=ax, label='Wind Speed (m/s)')
    ax.set_title(f'Wind Speed on {time_dates[frame].strftime("%Y-%m-%d %H:%M:%S")}')
    return fig

# Animation logic
if animate:
    for frame in range(len(time_data)):
        fig = plot_frame(frame)
        plot_placeholder.pyplot(fig)  # Update the plot in the same placeholder
        time.sleep(0.1)  # Adjust the speed of the animation
else:
    # Plot the selected frame using the slider
    fig = plot_frame(selected_frame)
    plot_placeholder.pyplot(fig)  # Display the plot in the placeholder
