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
# according to ISA Standard Atmosphere 1976
isa_altitude = np.array([0, 11019, 20063, 32162, 47350, 51412, 71802, 86000, 1000000])
isa_temperature = np.array([15, -56.5, -56.5, -44.5, -2.5, -2.5, -58.5, -86.204, -86.204])
isa_temperature = isa_temperature + 273.15
temperature = np.interp(height, isa_altitude, isa_temperature)
temperature_grid = np.tile(temperature, (Nlat, Nlon, 1))
temperature_grid = np.moveaxis(temperature_grid, (0, 1, 2), (1, 2, 0))
data["temperature"] = xr.DataArray(
    data=temperature_grid,
    dims=("lev", "lat", "lon")
).astype("float32")

# generate h2o curve (constant through atmosphere, made up)
h2o = Nlev*[0.003]
h2o_grid = np.tile(h2o, (Nlat, Nlon, 1))
h2o_grid = np.moveaxis(h2o_grid, (0, 1, 2), (1, 2, 0))
data["h2o"] = xr.DataArray(
    data=h2o_grid,
    dims=("lev", "lat", "lon")
).astype("float32")

# generate co2 curve (constant through atmosphere, made up)
co2 = Nlev*[425e-6]
co2_grid = np.tile(co2, (Nlat, Nlon, 1))
co2_grid = np.moveaxis(co2_grid, (0, 1, 2), (1, 2, 0))
data["co2"] = xr.DataArray(
    data=co2_grid,
    dims=("lev", "lat", "lon")
).astype("float32")

# generate ch4 curve (constant through atmosphere, made up)
ch4 = Nlev*[1930e-9]
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
# the AMF describes both of these angles in one quantity.
# COL will be multiplied by AMF_down for the downwelling part of the light path
# and multiplied by AMF_up for the upwelling part of the light path.
# In total, the light path experiences a factor of (AMF_down + AMF_up) and only
# the sum matters, not the individual air mass factors. Air mass factors for
# a total air mass factor being a light path with SZA and VZA both being 70 degrees
# are handled here.
# hack:
# In RemoTeC, changing the light path requires modification of syn_create.nml.
# This is tedious. Instead of doing this, the concentration is multiplied by the
# total air mass factor (halved, because RemoTeC calculates upwelling and downwelling
# direction separately). Mathematically, this is the same as changing the AMF.
total_amf = np.linspace(2, 2 * 1/np.cos(np.deg2rad(70)), Ngeometries)
total_amf = np.tile(total_amf, (Nenhancements, 1))
data["total_amf"] = xr.DataArray(
    data=total_amf,
    dims=("lat", "lon")
).astype("float32")

# Note: Halving the total_amf for multiplication, RemoTeC will calculate up and down
# separately and add them together, which gives a factor 2
if gas == "ch4":
    data["ch4"] = data.ch4 * data.enhancements * data.total_amf/2
    data.ch4.attrs["comment"] = "has been multiplied with "\
                                + "enhancement and total_amf/2"

if gas == "co2":
    data["co2"] = data.co2 * data.enhancements * data.total_amf/2
    data.co2.attrs["comment"] = "has been multiplied with "\
                                + "enhancement and total_amf/2"

for var in data.data_vars:
    data[var].encoding.update({"_FillValue": None})

data.to_netcdf(
    f"INPUT/{gas}.nc",
    mode="w",
    format="NETCDF4"
)
