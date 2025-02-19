import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import sys
import os


def main():
    # matched filter method: linewise, whole_scene
    matched_filter_method = "linewise"

    root_data = xr.open_dataset("SYNTH_SPECTRA/L1B_DATA.nc")

    # run matched filter
    win = [[1975, 2095]]
    alpha_co2 = matched_filter("co2", win, matched_filter_method)

    win = [[2120, 2430]]
    alpha_ch4 = matched_filter("ch4", win, matched_filter_method)

    # plt.xlabel("Longitude / degree")
    # plt.ylabel("Latitude / degree")
    # plt.pcolor(
    #     root_data.longitude, root_data.latitude, alpha_co2,
    #     cmap="inferno_r"
    # )
    # plt.colorbar(label="XCO2 enhancement / ppm")
    # plt.show()

    # plt.xlabel("Longitude / degree")
    # plt.ylabel("Latitude / degree")
    # plt.pcolor(
    #     root_data.longitude, root_data.latitude, alpha_ch4,
    #     cmap="inferno_r"
    # )
    # plt.colorbar(label="XCH4 enhancement / ppm")
    # plt.show()

    write_output(root_data, alpha_co2, alpha_ch4)


def matched_filter(gas, win, matched_filter_method):
    print(
        f"running matched filter version \"{matched_filter_method}\" "
        f"for gas {gas} with {len(win)} fit windows with boundaries {win}."
    )
    wl, x, s = get_variables(gas, win, matched_filter_method)
    alpha, x, mu, t, cov, cov_inv \
        = run_matched_filter(wl, x, s, matched_filter_method)
    # output_debug_info(alpha, x, mu, t, cov, cov_inv)

    return alpha


def get_variables(gas, win, matched_filter_method):
    rundir = os.getcwd()
    scriptdir = os.path.dirname(__file__)

    root_data = xr.open_dataset(
        f"{rundir}/SYNTH_SPECTRA/L1B_DATA.nc")
    band_data = xr.open_dataset(
        f"{rundir}/SYNTH_SPECTRA/L1B_DATA.nc", group="BAND01")
    uas_data = xr.open_dataset(f"{scriptdir}/uas/uas_{gas}.nc")
    Nframes = root_data.sizes["frame"]
    Nlines = root_data.sizes["line"]

    # wavelength grid is given by the data, select range between our fit range.
    # easiest if the data has a coordinate "wavelength" instead of channel
    band_data = band_data.assign_coords({"channel": band_data.wavelength})
    band_data = band_data.drop_vars({"wavelength"})
    band_data = band_data.rename({"channel": "wavelength"})
    band_data = band_data.sel(wavelength=slice(win[0][0], win[0][1]))

    # extract necessary variables for matched filter
    # wl = wavelength grid on which fit is performed
    # x = spectra for each spatial pixel
    # s = unit absorption spectrum
    wl = band_data.wavelength.values
    x = band_data.radiance

    # unit absorption spectrum is given for different background spectra. The
    # correct background spectrum is calculated from the mean_amf of the scene.
    sza = np.deg2rad(root_data.solar_zenith_angle)
    vza = np.deg2rad(root_data.viewing_zenith_angle)
    mean_amf = 0.5 * (1/np.cos(sza) + 1/np.cos(vza))
    match matched_filter_method:
        case "linewise":
            mean_amf = mean_amf.mean(dim="frame")
            s = uas_data.uas.interp(amf=mean_amf)
            s = s.expand_dims(dim={"frame": Nframes})
            s = s.drop_vars("amf")
        case "whole_scene":
            mean_amf = mean_amf.mean(dim=("frame", "line"))
            s = uas_data.uas.interp(amf=mean_amf)
            s = s.expand_dims(dim={"frame": Nframes, "line": Nlines})
            s = s.drop_vars("amf")
    # get s on the spectral grid of the measurement
    s = s.interp(wavelength=wl)

    return wl, x, s


def run_matched_filter(wl, x, s, matched_filter_method):
    match matched_filter_method:
        case "linewise":
            mf = mf_linewise
        case "whole_scene":
            mf = mf_whole_scene

    alpha, x, mu, t, cov, cov_inv = mf(wl, x, s)

    return alpha, x, mu, t, cov, cov_inv


def mf_linewise(wl, x, s):
    Nframes = x.sizes["frame"]

    # normalize everything for easier numerical calculations
    # normalizing x will also normalize mu and t and cov
    # for alpha, this norm cancels out
    # for alpha_err, it has to be corrected for
    norm = 1e11
    x = x / norm

    # get mu on the same grid as x
    mu = x.mean(dim="frame")
    mu = mu.expand_dims(dim={"frame": Nframes})

    # get t on the same grid as x
    t = mu * s

    # get cov on the same grid as x
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
    print(f"condition number: {np.linalg.cond(cov[0, 0, :, :])}")

    # calculate alpha on the same grid as x
    num = np.einsum("fli,flij,flj->fl", (x-mu), cov_inv, t)
    den = np.einsum("fli,flij,flj->fl", t, cov_inv, t)
    alpha = num/den

    return alpha, x, mu, t, cov, cov_inv


def mf_whole_scene(wl, x, s):
    Nframes = x.sizes["frame"]
    Nlines = x.sizes["line"]

    # for calculations, stack both spatial dimensions of x onto one
    x = x.stack(pixel=("frame", "line"))

    # normalize everything for easier numerical calculations
    # normalizing x will also normalize mu and t and cov
    # for alpha, this norm cancels out
    # for alpha_err, it has to be corrected for
    norm = 1e11
    x = x / norm

    # get mu on the same grid as x
    mu = x.mean(dim="pixel")
    mu = mu.expand_dims(dim={"frame": Nframes, "line": Nlines})

    # get t on the same grid as x
    t = mu * s

    # get cov on the same grid as x
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
    print(f"condition number: {np.linalg.cond(cov[0, 0, :, :])}")

    # calculate alpha on the same grid as x
    num = np.einsum("fli,flij,flj->fl", (x-mu), cov_inv, t)
    den = np.einsum("fli,flij,flj->fl", t, cov_inv, t)
    alpha = num/den

    return alpha, x, mu, t, cov, cov_inv


def output_debug_info(alpha, x, mu, t, cov, cov_inv):
    # print(f"{x.dims=}")
    # print(f"{mu.dims=}")
    # print(f"{t.dims=}")
    # print(f"{cov.dims=}")
    # print(f"{cov_inv.dims=}")

    plt.imshow(alpha, origin="lower", cmap="inferno_r")
    plt.colorbar()
    plt.show()

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


def write_output(root_data, alpha_co2, alpha_ch4):
    mtf_out_data = xr.Dataset()
    mtf_out_data["latitude"] = root_data.latitude
    mtf_out_data["longitude"] = root_data.longitude
    mtf_out_data["x_type0002_enh"] = xr.DataArray(
        data=alpha_co2, dims=("frame", "line")).astype("float32")
    mtf_out_data["x_type0006_enh"] = xr.DataArray(
        data=alpha_ch4, dims=("frame", "line")).astype("float32")

    for var in mtf_out_data.data_vars:
        mtf_out_data[var].encoding.update({"_FillValue": None})

    mtf_out_data.to_netcdf("CONTRL_OUT/MTF_OUT_DATA.nc", mode="w",
                           format="NETCDF4")


if __name__ == "__main__":
    main()
