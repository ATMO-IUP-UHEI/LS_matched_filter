import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

co2 = xr.open_dataset("uas_co2.nc").interp(amf=1)
ch4 = xr.open_dataset("uas_ch4.nc").interp(amf=1)

fig, axs = plt.subplots(2, 1)

axs[0].set_title("ch4")
axs[0].set_xlabel("wavelength / nm")
axs[0].set_ylabel("unit absorption spectrum / ppm$^{-1}$")
axs[0].plot(ch4.wavelength, ch4.uas, color="black")

axs[1].set_title("co2")
axs[1].set_xlabel("wavelength / nm")
axs[1].set_ylabel("unit absorption spectrum / ppm$^{-1}$")
axs[1].plot(co2.wavelength, co2.uas, color="black")

plt.show(block=True)
