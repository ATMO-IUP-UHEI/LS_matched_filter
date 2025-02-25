import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import sys
import os


def main():
    # matched filter method: linewise, whole_scene
    settings = {
        "uas_grid": "mean_amf",  # mean_amf, airmass
        "method": "linewise",  # linewise, whole_scene
        "iterative": True,  # bool
    }

    root_data = xr.open_dataset("SYNTH_SPECTRA/L1B_DATA.nc")

    # run matched filter
    win = [[1975, 2095]]
    alpha_co2 = matched_filter("co2", win, settings)

    win = [[2120, 2430]]
    alpha_ch4 = matched_filter("ch4", win, settings)

    plt.xlabel("Longitude / degree")
    plt.ylabel("Latitude / degree")
    plt.pcolor(
        root_data.longitude, root_data.latitude, alpha_co2,
        cmap="inferno_r"
    )
    plt.colorbar(label="XCO2 enhancement / ppm")
    plt.show()

    plt.xlabel("Longitude / degree")
    plt.ylabel("Latitude / degree")
    plt.pcolor(
        root_data.longitude, root_data.latitude, alpha_ch4,
        cmap="inferno_r"
    )
    plt.colorbar(label="XCH4 enhancement / ppm")
    plt.show()

    write_output(root_data, alpha_co2, alpha_ch4)


def matched_filter(gas, win, settings):
    print(
        f"running matched filter with settings {settings} "
        f"for gas {gas} with {len(win)} fit windows with boundaries {win}."
    )

    # get wavelength grid, radiances on the spatial grid, and the unit absorption spectrum
    wl, x, s = get_variables(gas, win, settings)

    # normalize radiances for numerical reasons. t (the target signature) will be normalized
    # automatically, since it is calculated from mu*s, where mu is calculated from x.
    norm = 1e11
    x = x / norm

    # calculate statistical properties mu (mean radiance) and cov (covariance matrix)
    mu, cov, cov_inv = statistical_properties(x, settings)

    # get target signature
    t = mu * s

    # run matched filter
    alpha = run_matched_filter(x, mu, cov_inv, t)

    # output_debug_info(alpha, x, mu, t, cov, cov_inv)

    return alpha


def get_variables(gas, win, settings):
    match settings["uas_grid"]:
        case "mean_amf":
            pass
        case "airmass":
            sys.exit("uas_grid airmass not implemented.")
        case _:
            sys.exit(f"uas_grid {settings['uas_grid']} not implemented.")

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

    match settings["method"]:
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

        case _:
            sys.exit(f"method {settings['method']} not implemented.")

    # get s on the spectral grid of the measurement
    s = s.interp(wavelength=wl)

    return wl, x, s


def statistical_properties(x, settings):
    match settings["method"]:
        case "whole_scene":
            Nframes = x.sizes["frame"]
            Nlines = x.sizes["line"]

            # for calculations, stack both spatial dimensions of x onto one
            x = x.stack(pixel=("frame", "line"))

            # get mu on the same grid as x
            mu = x.mean(dim="frame")
            mu = mu.expand_dims(dim={"frame": Nframes})

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

        case "linewise":
            Nframes = x.sizes["frame"]

            # get mu on the same grid as x
            mu = x.mean(dim="frame")
            mu = mu.expand_dims(dim={"frame": Nframes})

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

        case _:
            sys.exit(f"method {settings['method']} not implemented.")

    return mu, cov, cov_inv


def run_matched_filter(x, mu, cov_inv, t):
    # calculate alpha on the same grid as x
    num = np.einsum("fli,flij,flj->fl", (x-mu), cov_inv, t)
    den = np.einsum("fli,flij,flj->fl", t, cov_inv, t)
    alpha = num/den

    return alpha


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
