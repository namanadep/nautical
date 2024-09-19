import netCDF4 as nc

# defining the path to file
filePath = 'Datasets\Wavewatch_III_25_28_2024_to_03_09_2024.nc'

# using the Dataset() function
dSet = nc.Dataset(filePath)

# printing the variables of the dataset
for variable in dSet.variables.values():
   print(variable)