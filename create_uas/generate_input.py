import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import sys


gas = sys.argv[1]
print(f"gas = {gas}")

# lat
# lon
# time
# lev

# float latitude(lat)
# float longitude(lon)
# short datetime_utc(time)
# short datetime_lt(time)
# float pressure(lev, lat, lon)
# float height(lev, lat, lon)
# float temperature(lev, lat, lon)
# float h2o(lev, lat, lon)
# float co2(lev, lat, lon)
# float ch4(lev, lat, lon)
# float albedo_2000nm(lat, lon)

# Here, Nlat and Nlon do not matter
# Instead we are interested in different viewing geometries
# and different enhancements.
# All simulations will be performed for lat=0, lon=0
# We still need multiple points for the angles and the enhancements
# These will be calculated for "different" lats and lons
Nenhancements = 4
Ngeometries = 6

Nlat = Nenhancements
Nlon = Ngeometries
Ntime = 6
Nlev = 60

data = xr.Dataset()

data["latitude"] = xr.DataArray(
    data=Nlat*[0],
    dims=("lat")
).astype("float32")

data["longitude"] = xr.DataArray(
    data=Nlon*[0],
    dims=("lon")
).astype("float32")

data["datetime_utc"] = xr.DataArray(
    data=Ntime*[0],
    dims=("time")
)

data["datetime_lt"] = xr.DataArray(
    data=Ntime*[0],
    dims=("time")
)

# generate equally spaced pressure levels
p_surf = 1013.25  # hPa
p_space = 0.02  # hPa (chosen for RemoTeC by convention)
pressure = np.linspace(p_space, p_surf, Nlev)
pressure_grid = np.tile(pressure, (Nlat, Nlon, 1))
pressure_grid = np.moveaxis(pressure_grid, (0, 1, 2), (1, 2, 0))
data["pressure"] = xr.DataArray(
    data=pressure_grid,
    dims=("lev", "lat", "lon")
).astype("float32")

# generate height depending on pressure levels
z_surf = 0  # m (by definition)
scale_height = 8500  # m (p = p0 * exp(-z/scale))
height = z_surf + scale_height * np.log(p_surf/pressure)
height_grid = np.tile(height, (Nlat, Nlon, 1))
height_grid = np.moveaxis(height_grid, (0, 1, 2), (1, 2, 0))
data["height"] = xr.DataArray(
    data=height_grid,
    dims=("lev", "lat", "lon")
).astype("float32")

# generate albedo
albedo = 1
albedo_grid = np.tile(albedo, (Nlat, Nlon))
data["albedo_2000nm"] = xr.DataArray(
    data=albedo_grid,
    dims=("lat", "lon")
).astype("float32")

# generate temperature curve
# stolen from a turkmenistan pixel
temperature = [191.26947, 220.74019, 214.52698, 212.62027, 209.04094, 206.58559, 206.50642, 208.73889, 210.88342, 212.91727, 214.74251, 216.69302, 218.56128, 220.99455, 223.04814, 224.96338, 227.85973, 231.08394, 234.2546, 237.07098, 239.59401, 242.1424, 244.72778, 247.26695, 249.75653, 252.24036, 254.73994, 257.22632, 259.63913, 261.93005, 264.08087, 266.12656, 268.04312, 269.72568, 271.13547, 272.33545, 273.5517, 274.84506, 276.19662, 277.5877, 279.00128, 280.4308, 281.89957, 283.41507, 284.9685, 286.53674, 288.10297, 289.59775, 290.9289, 292.03818, 292.92377, 293.6516, 294.41013, 295.1759, 295.86624, 295.96704, 294.80182, 295.11316, 296.58817, 298.3027]
temperature_grid = np.tile(temperature, (Nlat, Nlon, 1))
temperature_grid = np.moveaxis(temperature_grid, (0, 1, 2), (1, 2, 0))
data["temperature"] = xr.DataArray(
    data=temperature_grid,
    dims=("lev", "lat", "lon")
).astype("float32")

# generate h2o curve (constant through atmosphere, made up)
h2o = Nlev*[0.01]
h2o_grid = np.tile(h2o, (Nlat, Nlon, 1))
h2o_grid = np.moveaxis(h2o_grid, (0, 1, 2), (1, 2, 0))
data["h2o"] = xr.DataArray(
    data=h2o_grid,
    dims=("lev", "lat", "lon")
).astype("float32")

# generate co2 curve (constant through atmosphere, made up)
co2 = Nlev*[420e-6]
co2_grid = np.tile(co2, (Nlat, Nlon, 1))
co2_grid = np.moveaxis(co2_grid, (0, 1, 2), (1, 2, 0))
data["co2"] = xr.DataArray(
    data=co2_grid,
    dims=("lev", "lat", "lon")
).astype("float32")

# generate ch4 curve (constant through atmosphere, made up)
ch4 = Nlev*[1912e-9]
ch4_grid = np.tile(ch4, (Nlat, Nlon, 1))
ch4_grid = np.moveaxis(ch4_grid, (0, 1, 2), (1, 2, 0))
data["ch4"] = xr.DataArray(
    data=ch4_grid,
    dims=("lev", "lat", "lon")
).astype("float32")

# enhancements up to 20% are handled for the UAS
# higher enhancements can be found anyway, but the
# calculation of the UAS only accounts for this range.
enhancements = np.linspace(1, 1.2, Nenhancements)
enhancements = np.tile(enhancements, (Ngeometries, 1))
enhancements = np.moveaxis(enhancements, (0, 1), (1, 0))
data["enhancements"] = xr.DataArray(
    data=enhancements,
    dims=("lat", "lon")
).astype("float32")
# viewing geometry is usually handled through vza and sza.
# this can be handled through the airmass factor.
# COL will be multiplied by AMF_down for half of the light path (downwelling)
# and multiplied by AMF_up for half of the light path (upwelling).
# in total, we can multiply the lightpath by the AVERAGE air mass factor.
# here, we don't multiply the light path, but instead we multiply the
# concentrations (it's the same, mathematically)
mean_amf = np.linspace(1, 1/np.cos(np.deg2rad(70)), Ngeometries)
mean_amf = np.tile(mean_amf, (Nenhancements, 1))
data["mean_amf"] = xr.DataArray(
    data=mean_amf,
    dims=("lat", "lon")
).astype("float32")

if gas == "ch4":
    data["ch4"] = data.ch4 * data.enhancements
    data["ch4"] = data.ch4 * data.mean_amf
    data.ch4.attrs["comment"] = "has been multiplied with "\
                                + "enhancement and mean_amf"

if gas == "co2":
    data["co2"] = data.co2 * data.enhancements
    data["co2"] = data.co2 * data.mean_amf
    data.co2.attrs["comment"] = "has been multiplied with "\
                                + "enhancement and mean_amf"

for var in data.data_vars:
    data[var].encoding.update({"_FillValue": None})

data.to_netcdf(
    f"INPUT/{gas}.nc",
    mode="w",
    format="NETCDF4"
)
