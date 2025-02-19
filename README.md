# Matched Filter

This package allows running a matched filter.
It also provides a tool to generate unit absorption spectra using RemoTeC.

## create uas

This package allows generating unit absorption spectra.
First, generate input for the forward RemoTeC run.
The RemoTeC version is suited for 2D satellite images with two spatial axes.
Here, we will generate data for which the location does not matter.
Instead, we will abuse the two axes for our own purpose: Generating spectra of radiances for different background concentrations, and for enhancements above each of the background concentrations.
The location will be set to a dummy location (lat=lon=0).

The script generate_input.py is used to create the input files INPUT/ch4.nc and INPUT/co2.nc, which will later be used by RemoTeC after being renamed to INPUT/RTC_INP_DATA.nc.

Next, LST/full.lst has to be adjusted.
The X dimension goes from 1 to Ngeometries.
For each X, the Y dimension goes from 1 to Nenhancements.
TODO: do this automatically with a script

Next, INI/syn_create.nml may be adjusted.
No noise is needed here, so you don't have to touch it.

Next, INI/settings_RTC_create_co2.nml and INI/settings_RTC_create_ch4.nml may be adjusted, for example by settings wave_start and wave_stop.

All files are now ready and RemoTeC_create can be run.
To do this, copy the input files to the corresponding filenames (copy_input.sh {gas}):
- INPUT/{gas}.nc -> RTC_INP_DATA.nc
- INI/settings_RTC_create_{gas}.nml -> INI/settings_RTC_create_000001.nml
- INI/syn_create.nml -> INI/syn_create_000001.nml

Run RemoTeC_create using
- /path/to/RemoTeC_create 1 LST/full.lst

This creates SYNTH_SPECTRA/ATM_000001.nc and SYNTH_SPECTRA/L1B_000001.nc
Copy them (copy_output.sh {gas})
- SYNTH_SPECTRA/ATM_000001.nc -> SYNTH_SPECTRA/ATM_{gas}.nc
- SYNTH_SPECTRA/L1B_000001.nc -> SYNTH_SPECTRA/L1B_{gas}.nc

Clean up after yourself (clean.sh)

Now you can convert the output files into a uas file using create_uas.py.
What this basically does is to reformat the RemoTeC output into nicer axes (dummy lat lon -> background and enhancement).
It then combines the information for each enhancement into one uas for this background concentration.
The background concentration is determined via the mean AMF.
TODO: use total air column for this).
The uas will be placed into uas/uas_{gas}.nc
Copy it into the directory of the matched filter.

## run

run the matched filter from the rundir of the scenario.
settings can be set inside of matched_filter.py.
This includes the fit range and the matched filter version (linewise vs whole scene).
