import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import sys
import os


def main():
    settings = {
        "uas_grid": "mean_amf",  # mean_amf, airmass
        "method": "linewise",  # linewise, whole_scene
        "iterative": True,  # bool
        "iterations": 3,  # int
    }

    root_data = xr.open_dataset("SYNTH_SPECTRA/L1B_DATA.nc")

    # run matched filter
    win = [[1975, 2095]]
    alpha_co2, mask_co2 = matched_filter("co2", win, settings)

    win = [[2120, 2430]]
    alpha_ch4, mask_ch4 = matched_filter("ch4", win, settings)

    # plot_alpha(root_data, alpha_co2, alpha_ch4)
    plot_mask(root_data, mask_co2, mask_ch4)

    # write_output(root_data, alpha_co2, alpha_ch4)


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

    match settings["iterative"]:
        case True:
            max_iter = settings["iterations"]+1
        case False:
            max_iter = 1

    # create mask that is applied for the purposes of calculating statistics
    mask = np.zeros(shape=(x.shape[0], x.shape[1], 1))  # one dimension for spectral grid

    for iter in range(1, max_iter):
        print(f"{iter=}")

        if iter == 1:
            # filter x for anomalous pixels
            anom = np.array(
                (x.sum(dim="wavelength") < 0.1 * x.median(dim={"frame", "line"}).sum("wavelength")) |\
                (x.sum(dim="wavelength") > 2 * x.median(dim={"frame", "line"}).sum("wavelength"))
            )

            mask[anom] = iter
        else:
            # filter x for unrealistic alpha
            anom = np.array(
                (alpha.values < -10) |\
                (alpha.values > +10)
            )

            # set mask to iteration number only where currently unmasked
            mask[anom[..., np.newaxis] & (mask==0)] = iter

        # calculate statistical properties mu (mean radiance) and cov (covariance matrix)
        # also output inverse of covariance matrix and condition number of inversion
        x_masked = x.where(mask == 0, np.nan)
        mu, cov, cov_inv = statistical_properties(x_masked, settings)

        # get target signature
        t = mu * s

        # run matched filter
        alpha = run_matched_filter(x, mu, cov_inv, t)

        plot_debug(gas, iter, alpha, x, mu, t, cov, cov_inv, mask)

    return alpha, mask[:, :, 0]


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


def statistical_properties(x_masked, settings):
    match settings["method"]:
        case "whole_scene":
            Nframes = x_masked.sizes["frame"]
            Nlines = x_masked.sizes["line"]

            # for calculations, stack both spatial dimensions of x onto one
            x_masked = x_masked.stack(pixel=("frame", "line"))

            # get mu on the same grid as x
            mu = x_masked.mean(dim="pixel")
            mu = mu.expand_dims(dim={"frame": Nframes, "line": Nlines})

            # get cov on the same grid as x
            # but with the dimension wavelength/wavelength
            # also get cov_inv
            cov = xr.apply_ufunc(
                lambda x: covariance_skipna(x),
                x_masked,
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

        case "linewise":
            Nframes = x_masked.sizes["frame"]

            # get mu on the same grid as x
            mu = x_masked.mean(dim="frame", skipna=True)
            mu = mu.expand_dims(dim={"frame": Nframes})

            # get cov on the same grid as x
            # but with the dimension wavelength/wavelength
            # also get cov_inv
            cov = xr.apply_ufunc(
                lambda x: covariance_skipna(x),
                x_masked,
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

        case _:
            sys.exit(f"method {settings['method']} not implemented.")

    return mu, cov, cov_inv


def covariance_skipna(x_masked):
    x_filtered = x_masked[~np.isnan(x_masked).all(axis=1)]
    cov = np.cov(x_filtered, rowvar=False)

    return cov


def run_matched_filter(x, mu, cov_inv, t):
    # calculate alpha on the same grid as x
    num = np.einsum("fli,flij,flj->fl", (x-mu), cov_inv, t)
    den = np.einsum("fli,flij,flj->fl", t, cov_inv, t)
    alpha = num/den

    alpha = xr.DataArray(alpha, dims={"frame", "line"})

    return alpha


def plot_debug(gas, iter, alpha, x, mu, t, cov, cov_inv, mask):
    frame = 0
    line = 0

    # calculate condition number
    ncond = np.linalg.cond(cov[frame, line, :, :])
    print(f"condition number = {ncond}")

    # print(f"{x.dims=}")
    # print(f"{mu.dims=}")
    # print(f"{t.dims=}")
    # print(f"{cov.dims=}")
    # print(f"{cov_inv.dims=}")

    # plt.title(f"alpha for {gas}, {iter=}, pixel=[{frame}, {line}]")
    # plt.imshow(alpha, origin="lower", cmap="inferno_r")
    # plt.colorbar()
    # plt.show()

    # plt.title(f"example spectrum for {gas}, {iter=}, pixel=[{frame}, {line}]")
    # plt.plot(x[frame, line, :], label="x")
    # plt.plot(mu[frame, line, :], label="mu")
    # plt.legend()
    # plt.show()

    # plt.title(f"example target signature and residual for {gas}, {iter=}, pixel=[{frame}, {line}]")
    # plt.plot(x[frame, line, :] - mu[frame, line, :], label="diff")
    # plt.plot(t[frame, line, :], label="t")
    # plt.legend()
    # plt.show()

    # plt.title(f"covariance matrix for {gas}, {iter=}, pixel=[{frame}, {line}]\ncondition number = {ncond}")
    # plt.imshow(cov[frame, line, :, :])
    # plt.colorbar()
    # plt.show()

    # plt.title(f"inverse of covariance matrix for {gas}, {iter=}, pixel=[{frame}, {line}]\ncondition number = {ncond}")
    # plt.imshow(cov_inv[frame, line, :, :])
    # plt.colorbar()
    # plt.show()

    # plt.title(f"mask, {iter=}")
    # plt.imshow(mask[..., 0], origin="lower")
    # plt.colorbar()
    # plt.show()


def plot_alpha(root_data, alpha_co2, alpha_ch4):
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


def plot_mask(root_data, mask_co2, mask_ch4):
    plt.xlabel("Longitude / degree")
    plt.ylabel("Latitude / degree")
    plt.pcolor(
        root_data.longitude, root_data.latitude, mask_co2,
        cmap="bone_r"
    )
    plt.colorbar()
    plt.show()

    plt.xlabel("Longitude / degree")
    plt.ylabel("Latitude / degree")
    plt.pcolor(
        root_data.longitude, root_data.latitude, mask_ch4,
        cmap="bone_r"
    )
    plt.colorbar()
    plt.show()


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
