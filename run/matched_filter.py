import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import sys
import os


def main():
    rundir = os.getcwd()
    scriptdir = os.path.dirname(__file__)

    data = f"{rundir}/SYNTH_SPECTRA/L1B_DATA.nc"
    uas_ch4 = f"{scriptdir}/uas/uas_ch4.nc"
    uas_co2 = f"{scriptdir}/uas/uas_co2.nc"

    l1b_root = xr.open_dataset(data)
    l1b_band = xr.open_dataset(data, group="BAND01")
    uas_data_ch4 = xr.open_dataset(uas_ch4)
    uas_data_co2 = xr.open_dataset(uas_co2)

    print("performing mf for co2")
    alpha_co2 = linewise_mf(l1b_root, l1b_band, uas_data_co2)
    print("performing mf for ch4")
    alpha_ch4 = linewise_mf(l1b_root, l1b_band, uas_data_ch4)

    mtf_out_data = xr.Dataset()
    mtf_out_data["latitude"] = l1b_root.latitude
    mtf_out_data["longitude"] = l1b_root.longitude
    mtf_out_data["x_type0002_enh"] = xr.DataArray(
        data=alpha_co2, dims=("frame", "line")).astype("float32")
    mtf_out_data["x_type0006_enh"] = xr.DataArray(
        data=alpha_ch4, dims=("frame", "line")).astype("float32")

    for var in mtf_out_data.data_vars:
        mtf_out_data[var].encoding.update({"_FillValue": None})

    mtf_out_data.to_netcdf("CONTRL_OUT/MTF_OUT_DATA.nc", mode="w",
                           format="NETCDF4")


def scene_mf(l1b_root, l1b_band, uas_data):
    Nframes = l1b_root.sizes["frame"]
    Nlines = l1b_root.sizes["line"]

    sza = np.deg2rad(l1b_root.solar_zenith_angle)
    vza = np.deg2rad(l1b_root.viewing_zenith_angle)
    mean_amf = 0.5 * (1/np.cos(sza) + 1/np.cos(vza))
    mean_amf = mean_amf.mean(dim=("frame", "line"))

    s = uas_data.uas.interp(amf=mean_amf)
    s = s.expand_dims(dim={"frame": Nframes, "line": Nlines})
    s = s.drop_vars("amf")

    # interpolate x onto wavelength grid of uas
    l1b_band = l1b_band.assign_coords(
        {"channel": l1b_band.wavelength})
    l1b_band = l1b_band.drop_vars(
        {"wavelength"})
    l1b_band = l1b_band.rename(
        {"channel": "wavelength"})
    l1b_band = l1b_band.interp(
        wavelength=uas_data.wavelength)

    # normalize everything for easier numerical calculations
    # normalizing x will also normalize mu and t and cov
    # for alpha, this norm cancels out
    # for alpha_err, it has to be corrected for
    norm = 1e11

    # get x and get mu on the same grid as x
    x = l1b_band.radiance / norm
    x = x.stack(pixel=("frame", "line"))

    mu = x.mean(dim="pixel")
    mu = mu.expand_dims(dim={"frame": Nframes, "line": Nlines})

    # get t on the same grid as x
    t = mu * s

    # get cov on the same frame/line grid as x
    # but with the dimension wavelength/wavelength
    # also get cov_inv
    cov = xr.apply_ufunc(
        lambda x: np.cov(x, rowvar=False),
        x,
        input_core_dims=[["pixel", "wavelength"]],
        output_core_dims=[["wavelength1", "wavelength2"]],
        vectorize=True,
    )
    cov_inv = xr.apply_ufunc(
        lambda x: np.linalg.pinv(x, rcond=1e-5),
        cov,
        input_core_dims=[["wavelength1", "wavelength2"]],
        output_core_dims=[["wavelength1", "wavelength2"]],
        vectorize=True,
    )
    cov = cov.expand_dims(dim={"frame": Nframes, "line": Nlines})
    cov_inv = cov_inv.expand_dims(dim={"frame": Nframes, "line": Nlines})
    print(np.linalg.cond(cov[0, 0, :, :]))

    # print(f"{x.dims=}")
    # print(f"{mu.dims=}")
    # print(f"{t.dims=}")
    # print(f"{cov.dims=}")
    # print(f"{cov_inv.dims=}")

    # plt.plot(x[0, 0, :], label="x")
    # plt.plot(mu[0, 0, :], label="mu")
    # plt.legend()
    # plt.show()

    # plt.plot(x[0, 0, :] - mu[0, 0, :], label="diff")
    # plt.plot(t[0, 0, :], label="t")
    # plt.legend()
    # plt.show()

    # plt.imshow(cov[0, 0, :, :])
    # plt.colorbar()
    # plt.show()

    # plt.imshow(cov_inv[0, 0, :, :])
    # plt.colorbar()
    # plt.show()

    num = np.einsum("fli,flij,flj->fl", (x-mu), cov_inv, t)
    den = np.einsum("fli,flij,flj->fl", t, cov_inv, t)
    alpha = num/den

    return alpha


def linewise_mf(l1b_root, l1b_band, uas_data):
    Nframes = l1b_root.sizes["frame"]

    sza = np.deg2rad(l1b_root.solar_zenith_angle)
    vza = np.deg2rad(l1b_root.viewing_zenith_angle)
    mean_amf = 0.5 * (1/np.cos(sza) + 1/np.cos(vza))
    mean_amf = mean_amf.mean(dim="frame")

    s = uas_data.uas.interp(amf=mean_amf)
    s = s.expand_dims(dim={"frame": Nframes})
    s = s.drop_vars("amf")

    # interpolate x onto wavelength grid of uas
    l1b_band = l1b_band.assign_coords(
        {"channel": l1b_band.wavelength})
    l1b_band = l1b_band.drop_vars(
        {"wavelength"})
    l1b_band = l1b_band.rename(
        {"channel": "wavelength"})
    l1b_band = l1b_band.interp(
        wavelength=uas_data.wavelength)

    # normalize everything for easier numerical calculations
    # normalizing x will also normalize mu and t and cov
    # for alpha, this norm cancels out
    # for alpha_err, it has to be corrected for
    norm = 1e11

    # get x and get mu on the same grid as x
    x = l1b_band.radiance / norm

    mu = x.mean(dim="frame")
    mu = mu.expand_dims(dim={"frame": Nframes})

    # get t on the same grid as x
    t = mu * s

    # get cov on the same frame/line grid as x
    # but with the dimension wavelength/wavelength
    # also get cov_inv
    cov = xr.apply_ufunc(
        lambda x: np.cov(x, rowvar=False),
        x,
        input_core_dims=[["frame", "wavelength"]],
        output_core_dims=[["wavelength1", "wavelength2"]],
        vectorize=True,
    )
    cov_inv = xr.apply_ufunc(
        lambda x: np.linalg.pinv(x, rcond=1e-5),
        cov,
        input_core_dims=[["wavelength1", "wavelength2"]],
        output_core_dims=[["wavelength1", "wavelength2"]],
        vectorize=True,
    )
    cov = cov.expand_dims(dim={"frame": Nframes})
    cov_inv = cov_inv.expand_dims(dim={"frame": Nframes})
    print(np.linalg.cond(cov[0, 0, :, :]))

    # print(f"{x.dims=}")
    # print(f"{mu.dims=}")
    # print(f"{t.dims=}")
    # print(f"{cov.dims=}")
    # print(f"{cov_inv.dims=}")

    # plt.plot(x[0, 0, :], label="x")
    # plt.plot(mu[0, 0, :], label="mu")
    # plt.legend()
    # plt.show()

    # plt.plot(x[0, 0, :] - mu[0, 0, :], label="diff")
    # plt.plot(t[0, 0, :], label="t")
    # plt.legend()
    # plt.show()

    # plt.imshow(cov[0, 0, :, :])
    # plt.colorbar()
    # plt.show()

    # plt.imshow(cov_inv[0, 0, :, :])
    # plt.colorbar()
    # plt.show()

    num = np.einsum("fli,flij,flj->fl", (x-mu), cov_inv, t)
    den = np.einsum("fli,flij,flj->fl", t, cov_inv, t)
    alpha = num/den

    return alpha


if __name__ == "__main__":
    main()
