import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import sys

# run co2 and run ch4 scenarios
# - use respective INPUT/x.nc files
# - use respective INI/settings_RTC_create_x.nml files
# - rename output to SYNTH_SPECTRA/L1B_x.nc and same for ATM

gas = sys.argv[1]

lst = np.loadtxt("LST/full.lst", dtype="str")
inp = xr.open_dataset(f"INPUT/{gas}.nc")
l1b = xr.open_dataset(f"SYNTH_SPECTRA/L1B_{gas}.nc", group="BAND1")

print("obs lon lat      conc  enh  amf")
i = 0

amf_list = []
uas_list = []

for obs, elem in enumerate(lst):
    i += 1

    lon, lat = elem.split("_")[-2:]
    lon = int(lon[1:]) - 1
    lat = int(lat[1:]) - 1
    if gas == "ch4":
        conc = inp.ch4.isel(lon=lon, lat=lat, lev=-1).values
    if gas == "co2":
        conc = inp.co2.isel(lon=lon, lat=lat, lev=-1).values
    enh = inp.enhancements.isel(lon=lon, lat=lat).values
    amf = inp.total_amf.isel(lon=lon, lat=lat).values

    print(f"{obs: 3}", f"{lon: 3}", f"{lat: 3}", f"{conc:.3E}", f"{enh:.2f}", f"{amf:.2f}")

    # get spectrum for bg and for this pixel
    wavelength = l1b.wavelength.values
    radiance_pix = l1b.radiance.isel(nobs=obs).values

    # new amf: we are done for this specific amf and continue to the next
    # save the amf to amf_list and save the uas to uas_list
    # except for the first observation, as everything is empty
    if amf not in amf_list:
        # new amf is now handled
        amf_list.append(amf)

        # first entry should be background for this amf
        if not enh == 1:
            sys.exit("ERROR: first pixel for new amf should have enh == 1")
        radiance_bg = radiance_pix

        # save the previous uas to the uas_list and reset local_uas_list
        # this does not have to be done if obs == 1 as no previous list exists
        if obs == 0:
            local_uas_list = []
            continue

        uas_list.append(np.mean(local_uas_list, axis=0))
        local_uas_list = []
        continue

    # calculate enhancement in ppm
    if gas == "ch4":
        conc_bg = inp.ch4.isel(lon=lon, lat=0, lev=-1).values
        conc_pix = inp.ch4.isel(lon=lon, lat=lat, lev=-1).values
        conc_enh = (conc_pix - conc_bg)/(amf/2)*1e6  # ppm
    if gas == "co2":
        conc_bg = inp.co2.isel(lon=lon, lat=0, lev=-1).values
        conc_pix = inp.co2.isel(lon=lon, lat=lat, lev=-1).values
        conc_enh = (conc_pix - conc_bg)/(amf/2)*1e6  # ppm

    # get residual spectrum
    uas = radiance_pix - radiance_bg
    # normalize by enhancement in ppm
    uas = uas / conc_enh
    # normalize by background spectrum
    uas = np.divide(uas, radiance_bg)

    local_uas_list.append(uas)

# uas for last amf was not entered yet
uas_list.append(np.mean(local_uas_list, axis=0))

# done
# plt.title(gas)
# plt.xlabel("wavelength / nm")
# plt.ylabel("unit absorption spectrum / ppm$^{-1}$")
# for i in range(len(amf_list)):
#     plt.plot(wavelength, uas_list[i], label=f"{amf_list[i]:.2f}")
# plt.legend(title="amf")
# plt.show()

# save as netcdf
outpath = f"uas/uas_{gas}.nc"

uas_data = xr.Dataset()

uas_data = uas_data.expand_dims(
    dim={
        "amf": amf_list,
        "wavelength": wavelength.astype("float32")
    }
)

uas_data["uas"] = xr.DataArray(
    data=uas_list,
    dims=("amf", "wavelength")
).astype("float32")

for var in uas_data.data_vars:
    uas_data[var].encoding.update({"_FillValue": None})

uas_data.to_netcdf(
    outpath,
    mode="w",
    format="NETCDF4"
)
