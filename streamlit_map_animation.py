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

# Create a placeholder for the plot
plot_placeholder = st.empty()

# Initialize session state for animation control
if 'animate_flag' not in st.session_state:
    st.session_state.animate_flag = False

# Button to control the animation
start_animation = st.button("Start Animation")
stop_animation = st.button("Stop Animation")

# Update session state based on button clicks
if start_animation:
    st.session_state.animate_flag = True
if stop_animation:
    st.session_state.animate_flag = False

# Function to plot a single frame, focusing on India
def plot_frame(frame):
    fig, ax = plt.subplots(figsize=(8, 6))  # Set a fixed figure size
    mp = Basemap(projection='merc',
                 llcrnrlon=68.0, llcrnrlat=6.0,
                 urcrnrlon=98.0, urcrnrlat=38.0,
                 resolution='l', ax=ax)  # Set map boundaries to show only India
    mp.drawcoastlines()
    mp.drawcountries()
    mp.drawstates()

    current_ws = np.squeeze(ws[frame, :, :])
    if np.ma.is_masked(current_ws):
        current_ws = np.ma.filled(current_ws, fill_value=np.nan)

    c_scheme = mp.pcolor(x, y, current_ws, cmap='jet', shading='auto', latlon=True)

    # Add colorbar with fixed position and size
    cbar = plt.colorbar(c_scheme, ax=ax, orientation='horizontal', pad=0.05)
    cbar.set_label('Wind Speed (m/s)')

    ax.set_title(f'Wind Speed on {time_dates[frame].strftime("%Y-%m-%d %H:%M:%S")}')
    return fig

# Get latitude and longitude for Basemap
lon, lat = np.meshgrid(lons, lats)
x, y = lon, lat

# Slider for selecting time frame
selected_frame = st.slider("Select time frame", 0, len(time_data) - 1, 0)

# Animation logic using session state to prevent page reloads
if st.session_state.animate_flag:
    for frame in range(len(time_data)):
        if not st.session_state.animate_flag:
            break  # Stop animation if the flag is set to False
        fig = plot_frame(frame)
        plot_placeholder.pyplot(fig)  # Update the plot in the same placeholder
        time.sleep(0.1)  # Adjust the speed of the animation
else:
    # Plot the selected frame using the slider
    fig = plot_frame(selected_frame)
    plot_placeholder.pyplot(fig)  # Display the plot in the placeholder
