import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

co2 = xr.open_dataset("uas_co2_win_1.nc").interp(amf=2)
ch4 = xr.open_dataset("uas_ch4_win_1.nc").interp(amf=2)
ch4_hires = xr.open_dataset("../../create_uas/uas/uas_ch4_fwhm03.nc").interp(amf=2)
l1b = xr.open_dataset("~/sds/data/scenarios/enmap/turkmenistan_20221002/SYNTH_SPECTRA/L1B_DATA.nc", group="BAND01")

fig, axs = plt.subplots(ncols=2)

axs[0].set_title("CH$_4$")
axs[0].set_xlabel("wavelength / nm")
axs[0].set_ylabel("unit absorption spectrum / ppm$^{-1}$")
axs[0].axvspan(2110, 2400, color="red", alpha=.15)
axs[0].plot(ch4.wavelength, ch4.uas, color="black", label="lores")
# axs[0].plot(ch4_hires.wavelength, ch4_hires.uas, color="black", alpha=.2, label="hires")
# axs[0].plot(l1b.wavelength, ch4.uas.interp(wavelength=l1b.wavelength), color="red", label="enmap grid")
axs[0].set_xlim(2090, 2460)
axs[0].set_ylim(-0.34, + 0.02)
axs[0].grid()
axs[0].legend()

axs[1].set_title("CO$_2$")
axs[1].set_xlabel("wavelength / nm")
axs[1].set_ylabel("unit absorption spectrum / ppm$^{-1}$")
axs[1].axvspan(1982, 2092, color="red", alpha=.15)
axs[1].plot(co2.wavelength, co2.uas, color="black", label="lores")
axs[1].legend()

plt.tight_layout()
plt.show(block=True)
